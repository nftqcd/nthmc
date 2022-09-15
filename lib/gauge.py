import tensorflow as tf
import field,fieldio,group
import lattice as l

class LatticeWrapper:
    """
    Wraps a Lattice object, so it works on any forms of subsets.
    """
    def __init__(self, lattice):
        if not isinstance(lattice, l.Lattice):
            raise ValueError(f'lattice must be an instance of Lattice but got {lattice.__class__}.')
        self.lattice = lattice
    def shifts(self, site):
        "Shift the transporter from site."
        return self.wrap(self.lattice.shifts(site), origin=self.origin+site)
    def shift(self, direction, length):
        xs = [0]*self.lattice.nd
        xs[direction] = length
        return self.wrap(self.lattice.shift(direction, length), origin=self.origin+field.Coord(xs))
    def full_volume(self):
        return self.lattice.full_volume()
    def site_shape(self):
        return self.lattice.site_shape()
    def get_site(self, *xs):
        return self.lattice.get_site(*xs)
    def get_batch(self, b):
        return self.wrap(self.lattice.get_batch(b))
    def batch_update(self, cond, other):
        return self.wrap(self.lattice.batch_update(cond, other.lattice))
    def wrap(self, lattice, **kwargs):
        return self.__class__(lattice, **kwargs)
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
    def typecast(self, x):
        return self.lattice.typecast(x)
    def norm2(self, *args, **kwargs):
        return self.lattice.norm2(*args, **kwargs)
    def is_compatible(self, other):
        return isinstance(other, self.__class__) and self.__class__==other.__class__
    def if_compatible(self, other, f, e=None):
        if self.is_compatible(other):
            return self.wrap(f(self.lattice,other.lattice))
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
        return self.wrap(-self.lattice)
    def __pos__(self):
        return self.wrap(+self.lattice)

class Transporter(LatticeWrapper):
    """
    Transporter is a Lattice with a directed path.
    When applied to a field, it transports a site value from the end of the path, to the beginning.
    The relative path defines the behavior of the transporter applied on a color vector.
    The absolute location of the path affects how the transporter connects to other transporters.
    """
    def __init__(self, lattice, path, origin=None, forward=True):
        super(Transporter, self).__init__(lattice=lattice)
        if not isinstance(path, field.Path):
            raise ValueError(f'path must be an instance of Path but got {path.__class__}.')
        site = lattice.site_shape()
        if len(site)!=2 or site[0]!=site[1]:
            raise ValueError(f'lattice sites are not square matrix, got shape {site}')
        # The value of lattice is always the transporter from the start of the path to the end.
        # forward means the effect of the application of the transporter, transport the applicant
        # from the end of the path to the start.  Not forward means the reverse.
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
            UV = Transporter(U, field.Path(1)).adjoint()(Transporter(V, field.Path(2)))
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
            # print(f'field_shift {field_shift}')
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
    def projectTangent(self):
        return self.wrap(self.lattice.projectTangent())
    def zeroTangentVector(self):
        c = self.nc
        return TangentVector(self.lattice.zeros(new_site_shape=[c*c-1], dtype=tf.float64))
    def randomTangent(self, rng):
        return self.wrap(self.lattice.randomTangent(rng), path=field.Path(), origin=field.Coord((0,)*self.lattice.nd))
    def randomTangentVector(self, rng):
        c = self.nc
        return TangentVector(self.lattice.randomNormal(rng, new_site_shape=[c*c-1], dtype=tf.float64))
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
            if self.forward!=other.forward:
                raise ValueError(f'unimplemented')
            if self.path == other.path:
                newp = self.path
            else:  # lose the exact ordered path
                newp = field.Path(self.path.deltaX())
            return self.wrap(f(self.lattice,other.lattice), path=newp)
        elif isinstance(other, self.__class__) and self.__class__==other.__class__:
            raise ValueError(f'incompatible transporters: {(self.path, self.origin, self.forward)} vs {(other.path, other.origin, other.forward)}')
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            if not self.forward:
                raise ValueError(f'unimplemented')
            return self.wrap(e(self.lattice))

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
    def full_volume(self):
        return self.data[0].full_volume()
    def site_shape(self):
        return self.data[0].site_shape()
    def get_site(self, *xs):
        return [t.get_site(*xs) for t in self.data]
    def get_batch(self, b):
        return self.wrap([t.get_batch(b) for t in self.data])
    def batch_update(self, cond, other):
        return self.wrap([t.batch_update(cond, o) for t,o in zip(self.data,other.data)])
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
        return Transporters([d.adjoint() for d in self.data])
    def unit(self):
        return self.wrap([d.unit() for d in self.data])
    def typecast(self, x):
        return self.data[0].typecast(x)
    def checkSU(self):
        a,b = zip(*[d.checkSU_() for d in self.data])
        a,b = tf.math.reduce_mean(a,axis=0),tf.math.reduce_max(b,axis=0)
        nc = self.data[0].nc
        c = 2*(nc*nc+1)
        return tf.math.sqrt(a/c),tf.math.sqrt(b/c)
    def projectSU(self):
        return self.wrap([d.projectSU() for d in self.data])
    def projectTangent(self):
        return Tangent([d.projectTangent() for d in self.data])
    def zeroTangentVector(self):
        return TangentVectors([t.zeroTangentVector() for t in self.data])
    def norm2(self, scope='lattice'):
        if scope=='lattice':
            return l.reduce_sum([t.norm2(scope=scope) for t in self.data])
        elif scope=='site':
            return l.LatticeList([t.norm2(scope=scope) for t in self.data])
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
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
    def randomTangent(self, rng):
        return Tangent([t.randomTangent(rng) for t in self.data])
    def randomTangentVector(self, rng):
        return TangentVectors([t.randomTangentVector(rng) for t in self.data])

class Tangent(Transporters):
    """
    Tangent bundle is represented as nd Transporters (X,Y,Z,...),
    each has a zero-distance path corresponding to the tangent space of the gauge
    links in the forward direction of that dimension.
    """
    def __init__(self, transporters):
        nd = len(transporters)
        nc = transporters[0].nc
        zeroCoord = field.Coord((0,)*nd)
        for i,t in enumerate(transporters):
            if not isinstance(t, Transporter):
                raise ValueError(f'lattice must be a Transporter, but got {t.__class__}')
            if not (t.nc==nc and t.lattice.nd==nd and t.path.deltaX()==zeroCoord and t.origin==zeroCoord and t.forward):
                raise ValueError(f'incompatible transporters with: {(t.nc, t.lattice.nd, t.path, t.origin, t.forward)}')
        super(Tangent, self).__init__([t.wrap(t.unwrap(), path=field.Path()) for t in transporters])
    def energy(self):
        nc = self.data[0].nc
        ts = [t.norm2(scope='site') for t in self.data]
        c = ts[0].typecast(nc*nc-1)
        return 0.5*l.reduce_sum([t-c for t in ts])
    def exp(self):
        return Transporters([t.wrap(t.unwrap().exp()) for t in self.data])
    def to_tangentVector(self):
        return TangentVectors([TangentVector(t.unwrap().to_su3vector()) for t in self.data])

class TangentVector(LatticeWrapper):
    def __init__(self, lattice):
        super(TangentVector, self).__init__(lattice=lattice)
        site = lattice.site_shape()
        if len(site)!=1:
            raise ValueError(f'lattice sites are not vectors, got shape {site}')
        # The value of lattice is always the transporter from the start of the path to the end.
        # forward means the effect of the application of the transporter, transport the applicant
        # from the end of the path to the start.  Not forward means the reverse.
        self.group_dim = site[0]
    def to_transporter(self):
        if self.group_dim==8:
            return Transporter(self.lattice.to_su3matrix(), field.Path())
        else:
            raise ValueError(f'unimplemented for group_dim {self.group_dim}')

class TangentVectors:
    def __init__(self, tangentvectors):
        for i,t in enumerate(tangentvectors):
            if not isinstance(t, TangentVector):
                raise ValueError(f'lattice must be a TangentVector, but got {t.__class__}')
        self.data = tangentvectors
    def full_volume(self):
        return self.data[0].full_volume()
    def site_shape(self):
        return self.data[0].site_shape()
    def get_site(self, *xs):
        return [t.get_site(*xs) for t in self.data]
    def get_batch(self, b):
        return self.wrap([t.get_batch(b) for t in self.data])
    def batch_update(self, cond, other):
        return self.wrap([t.batch_update(cond, o) for t,o in zip(self.data,other.data)])
    def wrap(self, tangentvectors):
        return self.__class__(tangentvectors)
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
    def typecast(self, x):
        return self.data[0].typecast(x)
    def norm2(self, scope='lattice'):
        if scope=='lattice':
            return l.reduce_sum([t.norm2(scope=scope) for t in self.data])
        elif scope=='site':
            return l.LatticeList([t.norm2(scope=scope) for t in self.data])
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
    def energy(self):
        nd = self.data[0].group_dim
        ts = [t.norm2(scope='site') for t in self.data]
        c = ts[0].typecast(nd)
        return 0.5*l.reduce_sum([t-c for t in ts])
    def exp(self):
        return Tangent([t.to_transporter() for t in self.data]).exp()
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

def from_tensor(tensor, nd=4, batch_dim=-1):
    """
    Assume dir dim the outer most dimension of the tensor, aside from optional batch_dim==0.
    """
    if batch_dim==0:
        ts = [Transporter(l.Lattice(tensor[:,i], nd=nd, batch_dim=batch_dim), field.Path(i+1)) for i in range(nd)]
    else:
        ts = [Transporter(l.Lattice(tensor[i], nd=nd, batch_dim=batch_dim), field.Path(i+1)) for i in range(nd)]
    return Gauge(ts)

def from_tensors(tensors, batch_dim=-1):
    """
    Assume a sequence of tensors, each represents one direction, ordered as X,Y,Z,...
    """
    nd = len(tensors)
    ts = [Transporter(l.Lattice(tensor, nd=nd, batch_dim=batch_dim), field.Path(i+1)) for i,tensor in enumerate(tensors)]
    return Gauge(ts)

def from_hypercube_parts(parts, **kwargs):
    """pass kwargs to l.from_hyper_cube_parts"""
    nd = len(parts)
    return Gauge([Transporter(l.from_hypercube_parts(p, **kwargs), field.Path(i+1)) for i,p in enumerate(parts)])

def readGauge(file):
    t,d = fieldio.readLattice(file)
    nd = len(d)
    print(f'read gauge conf, size {d}')
    return Gauge([Transporter(l.Lattice(tf.constant(t[i]),nd), field.Path(i+1)) for i in range(nd)])

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
    return Gauge([Transporter(l.Lattice(tf.eye(nc, nc, batch_shape=bs, dtype=tf.complex128),nd=nd,batch_dim=bd), field.Path(i+1)) for i in range(nd)])

def random(rng, dims, nbatch=0, nc=3):
    """
    dims: [X,Y,Z,T]
    """
    if nc!=3:
        raise ValueError(f'unimplemented for nc={nc}')
    nd = len(dims)
    dim_shape = reversed(dims)
    bs = tuple(dim_shape)+(nc,nc)
    bd = -1
    if nbatch>0:
        bs = (nbatch,)+bs
        bd = 0
    return Gauge([Transporter(l.Lattice(group.SU3.random(bs, rng),nd=nd,batch_dim=bd), field.Path(i+1)) for i in range(nd)])

def plaquetteField(gauge, computeRect=False):
    ps = []
    rs = []
    for mu in range(1,len(gauge)):
        for nu in range(0,mu):
            Umunu = gauge[mu](gauge[nu])
            Unumu = gauge[nu](gauge[mu])
            ps.append(Umunu(Unumu.adjoint()))
            if computeRect:
                Uu = gauge[nu].adjoint()(Umunu)
                Ur = gauge[mu].adjoint()(Unumu)
                Ul = Umunu(gauge[mu].adjoint())
                Ud = Unumu(gauge[nu].adjoint())
                rs.append(Ur(Ul.adjoint()))
                rs.append(Uu(Ud.adjoint()))
    return ps, rs

def plaquette(gauge):
    nc = gauge[0].nc
    ps,_ = plaquetteField(gauge)
    return [l.reduce_mean(l.trace(p.lattice))/nc for p in ps]

def plaq(gauge):
    nc = gauge[0].nc
    ps,_ = plaquetteField(gauge)
    return sum([l.reduce_mean(l.trace(p.lattice.real()))/nc for p in ps])/len(ps)

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
