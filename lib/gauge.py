import tensorflow as tf
import field
import fieldio
import lattice as l

class Transporter:
    """
    Transporter is a Lattice with a directed path.
    When applied to a field, it transports a site value from the end of the path, to the beginning.
    The relative path defines the behavior of the transporter applied on a color vector.
    The absolute location of the path affects how the transporter connects to other transporters.
    """
    def __init__(self, lattice, path, origin=None, forward=True):
        if not isinstance(lattice, l.Lattice):
            raise ValueError(f'lattice must be an instance of Lattice but got {lattice.__class__}.')
        if not isinstance(path, field.Path):
            raise ValueError(f'path must be an instance of Path but got {path.__class__}.')
        site = lattice.site_shape()
        if len(site)!=2 or site[0]!=site[1]:
            raise ValueError(f'lattice sites are not square matrix, got shape {site}')
        # The value of lattice is always the transporter from the start of the path to the end.
        # forward means the effect of the application of the transporter, transport the applicant
        # from the end of the path to the start.  Not forward means the reverse.
        self.lattice = lattice
        self.nc = site[0]
        self.path = path
        self.origin = field.Coord((0,)*lattice.nd) if origin is None else origin # where the path starts relative to the residing site
        self.forward = forward
    def __call__(self, field):
        """
        Returns the transported field, optionally shift it.
        If field is a Transporter, the result is a new Transporter connecting us and field,
        which is shifted automatically, effectively extending our path.
        If field is not a Transporter, we assume it is a color vector, the result is the vector
        transported via the path from its end to its start.
        Eg.
            UV = Transporter(U, field.Path(4, [1])).adjoint()(Transporter(V, field.Path(4, [2])))
        the result UV has a path [-1,2] starting from (1,0).
        The actual adjoint of the tensor data will be carried out during the multiplication.
        """
        # relative to the site of the object
        path_start = self.origin
        path_end = self.origin + self.path.deltaX()
        if self.forward:
            field_shift = path_end
        else:
            field_shift = path_start

        if isinstance(field, Transporter):
            if not self.lattice.is_compatible(field.lattice):
                raise ValueError(f'incompatible lattices')
            field_path_start = field.origin
            field_path_end = field.origin + field.path.deltaX()
            # change the shifts according to the transporters
            if field.forward:
                field_shift -= field_path_start
                rpath = field.path
            else:
                field_shift -= field_path_end
                rpath = field.path.adjoint()
            f = field.lattice
            f = f.shifts(field_shift)
            f = l.matmul(self.lattice, f, adjoint_l=not self.forward, adjoint_r=not field.forward)
            if self.forward:
                lpath = self.path
                neworigin = path_start
            else:
                lpath = self.path.adjoint()
                neworigin = path_end
            return Transporter(f, lpath+rpath, origin=neworigin)
        elif isinstance(field, l.Lattice):
            if not self.lattice.is_compatible(field):
                raise ValueError(f'incompatible lattices')
            if self.forward:
                result_shift = -path_start
            else:
                result_shift = -path_end
            f = field.shifts(field_shift)
            f = l.matvec(self.lattice, f, adjoint=not self.forward)
            f = f.shifts(result_shift)
            return f
        else:
            raise ValueError(f'cannot apply on field: {field}')
    def wrap(self, lattice, **kwargs):
        dkwargs = {'path':self.path, 'origin':self.origin, 'forward':self.forward, **kwargs}
        return self.__class__(lattice, **dkwargs)
    def unwrap(self):
        return self.lattice
    def from_tensors(self, tensors):
        return self.wrap(self.unwrap().from_tensors(tensors))
    def to_tensors(self):
        return self.unwrap().to_tensors()
    def hypercube_partition(self):
        return self.wrap(self.lattice.hypercube_partition())
    def combine_hypercube(self):
        return self.wrap(self.lattice.combine_hypercube())
    def adjoint(self):
        "Reverse the direction of the Transporter, does not change tensor data."
        return Transporter(self.lattice, self.path, origin=self.origin, forward=not self.forward)
    def checkSU_(self):
        """Returns the average and maximum of the sum of deviations of x^dag x and det(x) from unitarity."""
        ds = self.adjoint()(self)
        d = (ds - ds.unit()).unwrap().norm2(scope='site') + (-1 + self.det()).norm2(scope='site')
        a = d.reduce_mean()
        b = d.reduce_max()
        return a,b
    def checkSU(self):
        a,b = self.checkSU_()
        nc = self.nc
        c = 2*(nc*nc+1)
        return tf.math.sqrt(a/c),tf.math.sqrt(b/c)
    def projectSU(self):
        return self.wrap(self.lattice.projectSU())
    def unit(self):
        return self.wrap(self.lattice.unit())
    def det(self):
        return self.lattice.det()
    def same_points(self, other):
        if self.forward==other.forward:
            return self.origin==other.origin and self.origin+self.path.deltaX()==other.origin+other.path.deltaX()
        else:
            return self.origin==other.origin+other.path.deltaX() and self.origin+self.path.deltaX()==other.origin
    def is_compatible(self, other):
        return isinstance(other, self.__class__) and self.__class__==other.__class__ and self.nc==other.nc and self.same_points(other)
    def if_compatible(self, other, f, e=None):
        if self.is_compatible(other):
            return self.wrap(f(self.lattice,other.lattice))
        elif isinstance(other, self.__class__) and self.__class__==other.__class__:
            raise ValueError(f'incompatible transporters: {(self.path, self.origin, self.forward)} vs {(other.path, other.origin, other.forward)}')
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            return self.wrap(e(self.lattice))
    def __add__(self, other):
        return self.if_compatible(other, lambda a,b:a+b, lambda a:a+other)
    def __radd__(self, other):
        return self.if_compatible(other, lambda a,b:b+a, lambda a:other+a)
    def __sub__(self, other):
        return self.if_compatible(other, lambda a,b:a-b, lambda a:a-other)
    def __rsub__(self, other):
        return self.if_compatible(other, lambda a,b:b-a, lambda a:other-a)
    def __mul__(self, other):
        return self.if_compatible(other, lambda a,b:a*b, lambda a:a*other)
    def __rmul__(self, other):
        return self.if_compatible(other, lambda a,b:b*a, lambda a:other*a)
    def __truediv__(self, other):
        return self.if_compatible(other, lambda a,b:a/b, lambda a:a/other)
    def __rtruediv__(self, other):
        return self.if_compatible(other, lambda a,b:b/a, lambda a:other/a)
    def __neg__(self):
        return self.wrap(-self.data)
    def __pos__(self):
        return self.wrap(+self.data)

class Transporters:
    """
    A list of Transporters that have the same nc.
    """
    def __init__(self, transporters):
        nc = transporters[0].nc
        for i,t in enumerate(transporters):
            if not isinstance(t, Transporter):
                raise ValueError(f'lattice must be a Transporter, but got {t.__class__}')
            if t.nc!=nc:
                raise ValueError(f'incompatible transporters with: {(t.nc, t.lattice.nd, t.path, t.origin, t.forward)}')
        self.data = transporters
    def __call__(self, field):
        if isinstance(field, Transporters):
            if self.data[0].nc!=field.data[0].nc or len(self)!=len(field):
                raise ValueError(f'mismatched transporters {self.data[0].nc, len(self)} vs. {field.data[0].nc, len(field)}')
            return field.wrap([a(b) for a,b in zip(self.data,field.data)])
        elif isinstance(field, Transporter):
            return Transporters([a(field) for a in self.data])
        elif isinstance(field, l.Lattice):
            return Transporters([a(field) for a in self.data])
        else:
            raise ValueError(f'unimplemented for field {field.__class__}')
    def wrap(self, transporters):
        return self.__class__(transporters)
    def unwrap(self):
        return self.data
    def from_tensors(self, tensors):
        return self.wrap([d.from_tensors(t) for d,t in zip(self.data,tensors)])
    def to_tensors(self):
        return [d.to_tensors() for d in self.data]
    def hypercube_partition(self):
        return self.wrap([d.hypercube_partition() for d in self.data])
    def combine_hypercube(self):
        return self.wrap([d.combine_hypercube() for d in self.data])
    def adjoint(self):
        return self.wrap([d.adjoint() for d in self.data])
    def unit(self):
        return self.wrap([d.unit() for d in self.data])
    def checkSU(self):
        a,b = zip(*[d.checkSU_() for d in self.data])
        a,b = tf.math.reduce_mean(a,axis=0),tf.math.reduce_max(b,axis=0)
        nc = self.data[0].nc
        c = 2*(nc*nc+1)
        return tf.math.sqrt(a/c),tf.math.sqrt(b/c)
    def projectSU(self):
        return self.wrap([d.projectSU() for d in self.data])
    def is_compatible(self, other):
        return isinstance(other, self.__class__) and self.__class__==other.__class__
    def if_compatible(self, other, f, e=None):
        if self.is_compatible(other):
            return self.wrap([f(a,b) for a,b in zip(self.data,other.data)])
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            return self.wrap([e(a) for a in self.data])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        return self.data[key]
    def __add__(self, other):
        return self.if_compatible(other, lambda a,b:a+b, lambda a:a+other)
    def __radd__(self, other):
        return self.if_compatible(other, lambda a,b:b+a, lambda a:other+a)
    def __sub__(self, other):
        return self.if_compatible(other, lambda a,b:a-b, lambda a:a-other)
    def __rsub__(self, other):
        return self.if_compatible(other, lambda a,b:b-a, lambda a:other-a)
    def __mul__(self, other):
        return self.if_compatible(other, lambda a,b:a*b, lambda a:a*other)
    def __rmul__(self, other):
        return self.if_compatible(other, lambda a,b:b*a, lambda a:other*a)
    def __truediv__(self, other):
        return self.if_compatible(other, lambda a,b:a/b, lambda a:a/other)
    def __rtruediv__(self, other):
        return self.if_compatible(other, lambda a,b:b/a, lambda a:other/a)
    def __neg__(self):
        return self.wrap([-d for d in self.data])
    def __pos__(self):
        return self.wrap([+d for d in self.data])

class Gauge(Transporters):
    """
    Gauge is nd Transporters (X,Y,Z,...),
    each connects a neighboring site in the forward direction of that dimension.
    """
    def __init__(self, transporters):
        nd = len(transporters)
        nc = transporters[0].nc
        for i,t in enumerate(transporters):
            if not isinstance(t, Transporter):
                raise ValueError(f'lattice must be a Transporter, but got {t.__class__}')
            if not (t.nc==nc and t.lattice.nd==nd and len(t.path)==1 and t.path[0]==i+1 and t.origin==field.Coord((0,)*nd) and t.forward):
                raise ValueError(f'incompatible transporters with: {(t.nc, t.lattice.nd, t.path, t.origin, t.forward)}')
        super(Gauge, self).__init__(transporters)

def from_tensor(tensor, nd=4, batch_dim=-1):
    """
    Assume dir dim the outer most dimension of the tensor, aside from optional batch_dim==0.
    """
    if batch_dim==0:
        ts = [Transporter(l.Lattice(tensor[:,i], nd=nd, batch_dim=batch_dim), field.Path(nd, i+1)) for i in range(nd)]
    else:
        ts = [Transporter(l.Lattice(tensor[i], nd=nd, batch_dim=batch_dim), field.Path(nd, i+1)) for i in range(nd)]
    return Gauge(ts)

def from_tensors(tensors, batch_dim=-1):
    """
    Assume a sequence of tensors, each represents one direction, ordered as X,Y,Z,...
    """
    nd = len(tensors)
    ts = [Transporter(l.Lattice(tensor, nd=nd, batch_dim=batch_dim), field.Path(nd, i+1)) for i,tensor in enumerate(tensors)]
    return Gauge(ts)

def from_hypercube_parts(parts, **kwargs):
    """pass kwargs to l.from_hyper_cube_parts"""
    nd = len(parts)
    return Gauge([Transporter(l.from_hypercube_parts(p, **kwargs), field.Path(nd, i+1)) for i,p in enumerate(parts)])

def readGauge(file):
    t,d = fieldio.readLattice(file)
    nd = len(d)
    print(f'read gauge conf, size {d}')
    return Gauge([Transporter(l.Lattice(t[i],nd), field.Path(nd, i+1)) for i in range(nd)])

def unit(dims, nbatch=0, nc=3):
    """
    dims: [X,Y,Z,T]
    """
    nd = len(dims)
    dim_shape = reversed(dims)
    bs = tuple(dim_shape)
    bd = -1
    if nbatch>0:
        bs = (nbatch,)+bs
        bd = 0
    return Gauge([Transporter(l.Lattice(tf.eye(nc, nc, batch_shape=bs, dtype=tf.complex128),nd=nd,batch_dim=bd), field.Path(nd, i+1)) for i in range(nd)])

def plaqutteField(gauge):
    ps = []
    for mu in range(1,len(gauge)):
        for nu in range(0,mu):
            Umunu = gauge[mu](gauge[nu])
            Unumu = gauge[nu](gauge[mu])
            ps.append(Umunu(Unumu.adjoint()))
    return ps

def plaquette(gauge):
    nc = gauge[0].nc
    return [l.reduce_mean(l.trace(p.lattice))/nc for p in plaqutteField(gauge)]

C1Symanzik = -1.0/12.0  # tree-level
C1Iwasaki = -0.331
C1DBW2 = -1.4088

def action(gauge, beta, c1=0):
    pass

def setBC(gauge):
    """
    Output: new gauge with antiperiodic BC in T dir for fermions.
    Input: Gauge, gauge[-1] is the T dir
    """
    return Gauge(gauge[:-1]+[setBC_T(gauge[-1])])

def setBC_T(gaugeT, T_dim=None):
    """
    Output: new transporter with antiperiodic BC in T dir for fermions, flipping the sign on the last T slice.
    Input: transporter in T dir, shape: [batch] T Z ... X ...
    """
    if isinstance(gaugeT, Transporter):
        return gaugeT.wrap(setBC_T(gaugeT.unwrap()))
    elif isinstance(gaugeT, l.Lattice):
        corners = gaugeT.subset.to_corner_indices()
        if gaugeT.batch_dim==0:
            td = 1
        else:
            td = 0
        if len(corners)>1 or corners[0]&0b1000!=0:
            return gaugeT.wrap(setBC_T(gaugeT.unwrap(), T_dim=td))
        else:  # single corner and not T boundary
            return gaugeT
    elif isinstance(gaugeT, list):
        return [setBC_T(g, T_dim=T_dim) for g in gaugeT]
    elif isinstance(gaugeT, tuple):
        return tuple([setBC_T(g, T_dim=T_dim) for g in gaugeT])
    if T_dim is None:
        raise ValueError(f'unknown T_dim with gaugeT {gaugeT}')
    shape = gaugeT.shape
    tb = shape[T_dim]-1
    if T_dim==0:
        tbgauge = (-1.0) * gaugeT[tb]
        ix = [[tb]]
        g = tf.tensor_scatter_nd_update(gaugeT, ix, tf.expand_dims(tbgauge,0))
    elif T_dim==1:
        bn = shape[0]
        tbgauge = (-1.0) * gaugeT[:,tb]
        ix = [[i,tb] for i in range(bn)]
        g = tf.tensor_scatter_nd_update(gaugeT, ix, tbgauge)
    else:
        raise ValueError(f'unsupported T_dim: {T_dim}')
    return g

if __name__=='__main__':
    import sys
    import fieldio

    gconf = readGauge(sys.argv[1])
    nd = len(gconf)
    print('Plaq:')
    for p in plaquette(gconf):
        print(p)

    def tripleLat(lat):
        t = lat.unwrap()
        return lat.wrap(tf.stack([t,t,t], axis=0), nd=nd, batch_dim=0)
    batchedGconf = Gauge([t.wrap(tripleLat(t.unwrap())) for t in gconf])
    print('BatchedPlaq:')
    for p in plaquette(batchedGconf):
        print(p)

    gaugeEO = gconf.hypercube_partition()
    print(gaugeEO[0].lattice[0].shape)
    print('Plaq from EO:')
    for p in plaquette(gaugeEO):
        print(p)

    batchedGaugeEO = batchedGconf.hypercube_partition()
    print(batchedGaugeEO[0].lattice[0].shape)
    print('Plaq from EO:')
    for p in plaquette(batchedGaugeEO):
        print(p)
