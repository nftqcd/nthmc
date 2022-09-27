import math
import group
import tensorflow as tf

class SubSet:
    """
    Represent a subset of an 4-D lattice sites.
    Currently it has a granularity of hypercube corners.
    """
    NP = 1<<4  # vertices in a 4D hypercube
    def __init__(self, subset=0):
        """
        subset: either 'all', 'even', 'odd', or a bit pattern with 1<<corner_index for each corner.
        corner_index counts from 0 to 15, with order X, Y, Z, T.
        The bit pattern of the index is 0bTZYX.
        """
        if subset=='all':
            self.s = SubSet.even_set + SubSet.odd_set
        elif subset=='even':
            self.s = SubSet.even_set
        elif subset=='odd':
            self.s = SubSet.odd_set
        elif subset>=0 and subset<1<<SubSet.NP:
            self.s = subset
        else:
            raise ValueError(f'unsupported subset {subset}')
    def is_empty(self):
        return self.s == 0
    def union(self, x):
        return SubSet(self.s | x.s)
    def intersection(self, x):
        return SubSet(self.s & x.s)
    def to_corner_indices(self):
        return [i for i in range(SubSet.NP) if self.s>>i&1!=0]
    def to_corner_index(self):
        ix = self.to_corner_indices()
        if len(ix)!=1:
            raise ValueError(f'subset {self} contains more than one corner')
        else:
            return ix[0]
    def shift(self, direction, length):
        if length%2==1:
            d = 1<<direction
            return SubSet(sum([1<<(i^d) for i in self.to_corner_indices()]))
        else:
            return self
    def __repr__(self):
        return f'SubSet({bin(self.s)})'
    def __hash__(self):
        return hash(self.s)
    def __lt__(self, x):
        if isinstance(x, SubSet):
            return self.s < x.s
        else:
            return False
    def __le__(self, x):
        if isinstance(x, SubSet):
            return self.s <= x.s
        else:
            return False
    def __eq__(self, x):
        if isinstance(x, SubSet):
            return self.s == x.s
        else:
            return False
    def __ne__(self, x):
        if isinstance(x, SubSet):
            return self.s != x.s
        else:
            return False
    def __gt__(self, x):
        if isinstance(x, SubSet):
            return self.s > x.s
        else:
            return False
    def __ge__(self, x):
        if isinstance(x, SubSet):
            return self.s >= x.s
        else:
            return False
    def __contains__(self, subset):
        if not isinstance(subset, SubSet):
            raise ValueError(f'subset {subset} is not a SubSet')
        return subset.s == subset.s&self.s
    def __add__(self, x):
        if not isinstance(x, SubSet):
            raise ValueError(f'operand {x} is not a SubSet')
        return self.union(x)
    def __sub__(self, x):
        if not isinstance(x, SubSet):
            raise ValueError(f'operand {x} is not a SubSet')
        return SubSet(self.s - (self.s & x.s))
    def from_corner_index(i):
        return SubSet(1<<i)
    def from_coord(xs):
        "xs: [X,Y,Z,...]"
        ix = 0
        for i,n in enumerate(xs):
            ix |= (n%2)<<i
        return SubSet(1<<ix)
    is_even_index = lambda x: (1<<x) & SubSet.even_set != 0
    even_set = sum([1<<i for i in (0b0, 0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100, 0b1111)])
    odd_set = sum([1<<i for i in (0b1, 0b10, 0b100, 0b111, 0b1000, 0b1011, 0b1101, 0b1110)])

SubSetAll = SubSet('all')
SubSetEven = SubSet('even')
SubSetOdd = SubSet('odd')

# generic functions, broadcast/reduce for list or tuple, use method if Lattice, else use TF

def shift(lat, direction, length, subset=SubSetAll, axis=None):
    """
    Output: circular shifted lattice along direction with length
    Input:
        direction: 0,1,2,... corresponding to x,y,z,..., not used for tensors directly.
        length: +/- denotes shift from forward/backward direction corresponding to tf.roll with -/+ shifts rolling backward/forward.
        axis: used in tensors, and does not pass to Lattice object.
    """
    if direction<0:
        raise ValueError(f'direction less than 0: {direction}')
    if length==0:
        return lat
    elif isinstance(lat, Lattice):
        return lat.shift(direction, length, subset=subset)
    elif isinstance(lat, list):
        return [shift(l, direction, length, subset=subset, axis=axis) for l in lat]
    elif isinstance(lat, tuple):
        return tuple([shift(l, direction, length, subset=subset, axis=axis) for l in lat])
    elif subset==SubSetAll:
        return tf.roll(lat, shift=-length, axis=axis)
    else:
        try:
            corner = subset.to_corner_index()
        except ValueError as err:
            raise ValueError(f'unsupported subset {subset}: {err}')
        if corner&1<<direction==0:
            return tf.roll(lat, shift=-length//2, axis=axis)
        else:
            return tf.roll(lat, shift=-(length-1)//2, axis=axis)

def zeros(lat, new_site_shape=None, site_shape_len=None, dtype=None, **kwargs):
    """Pass kwargs to Lattice.zeros, allow Lattice to overwrite site_shape_len."""
    if isinstance(lat, Lattice):
        return lat.zeros(new_site_shape=new_site_shape, dtype=dtype, **kwargs)
    elif len(kwargs)>0:
        raise ValueError(f'unsupported kwargs {kwargs}')
    elif isinstance(lat, list):
        return [zeros(x, new_site_shape=new_site_shape, site_shape_len=site_shape_len, dtype=dtype) for x in lat]
    elif isinstance(lat, tuple):
        return tuple([zeros(x, new_site_shape=new_site_shape, site_shape_len=site_shape_len, dtype=dtype) for x in lat])
    else:
        if dtype is None:
            dtype = lat.dtype
        if new_site_shape is None:
            return tf.zeros(lat.shape, dtype=dtype)
        elif site_shape_len is None:
            raise ValueError(f'unknown site_shape_len, required for new_site_shape={new_site_shape}')
        else:
            return tf.zeros(lat.shape[:-site_shape_len]+tuple(new_site_shape), dtype=dtype)

def unit(lat, site_shape_len=None, **kwargs):
    """Pass kwargs to Lattice.unit, allow Lattice to overwrite site_shape_len."""
    if isinstance(lat, Lattice):
        return lat.unit(**kwargs)
    elif len(kwargs)>0:
        raise ValueError(f'unsupported kwargs {kwargs}')
    elif isinstance(lat, list):
        return [unit(x, site_shape_len=site_shape_len) for x in lat]
    elif isinstance(lat, tuple):
        return tuple([unit(x, site_shape_len=site_shape_len) for x in lat])
    else:
        if site_shape_len is None:
            raise ValueError('unknown site_shape_len')
        if site_shape_len!=2:
            raise ValueError(f'unimplemented for site shape {lat.shape[-site_shape_len:]}')
        return tf.eye(*lat.shape[-site_shape_len:], batch_shape=lat.shape[:-site_shape_len], dtype=lat.dtype)

def typecast(lat, x):
    """makes x the same dtype as lat"""
    if isinstance(lat, Lattice):
        return lat.typecast(x)
    elif isinstance(lat, (tuple, list)):
        return typecast(lat[0], x)
    else:
        if tf.is_tensor(x):
            return tf.cast(x, dtype=lat.dtype)
        else:
            return tf.convert_to_tensor(x, dtype=lat.dtype)

def lattice_map(lat, functor, tfunctor, *args, **kwargs):
    if isinstance(lat, Lattice):
        return lat.map(functor, *args, **kwargs)
    if isinstance(lat, list):
        return [functor(x, *args, **kwargs) for x in lat]
    elif isinstance(lat, tuple):
        return tuple([functor(x, *args, **kwargs) for x in lat])
    else:
        return tfunctor(lat, *args, **kwargs)

def real(lat):
    return lattice_map(lat, real, tf.math.real)

def imag(lat):
    return lattice_map(lat, imag, tf.math.imag)

def trace(lat):
    """
    Output: trace of the site over lattice
    Input:
        lattice: assuming matrix valued sites, and batch_dim never in the matrix
    """
    return lattice_map(lat, trace, tf.linalg.trace)

def det(lat):
    """
    Output: determinant of the site over lattice
    Input:
        lattice: assuming matrix valued sites, and batch_dim never in the matrix
    """
    return lattice_map(lat, det, tf.linalg.det)

def exp(lat):
    return lattice_map(lat, exp, group.exp)

def projectSU(lat):
    return lattice_map(lat, projectSU, group.projectSU)

def projectTangent(lat):
    return lattice_map(lat, projectTangent, group.projectTAH)

def randomTangent_(lat, rng):
    if lat.shape[-1]!=3:
        raise ValueError(f'unimplemented for nc={lat.shape[-1]}')
    return group.randTAH3(lat.shape[:-2], rng)

def randomTangent(lat, rng):
    return lattice_map(lat, randomTangent, randomTangent_, rng)

def randomNormal_(lat, rng, new_site_shape=None, site_shape_len=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = lat.dtype
    if new_site_shape is None:
        if dtype==tf.complex128:
            return tf.dtypes.complex(
                rng.normal(lat.shape, dtype=tf.float64, **kwargs),
                rng.normal(lat.shape, dtype=tf.float64, **kwargs))
        elif dtype==tf.complex64:
            return tf.dtypes.complex(
                rng.normal(lat.shape, dtype=tf.float32, **kwargs),
                rng.normal(lat.shape, dtype=tf.float32, **kwargs))
        else:
            return rng.normal(lat.shape, dtype=dtype, **kwargs)
    elif site_shape_len is None:
        raise ValueError(f'unknown site_shape_len, required for new_site_shape={new_site_shape}')
    else:
        sp = lat.shape[:-site_shape_len]+tuple(new_site_shape)
        if dtype==tf.complex128:
            return tf.dtypes.complex(
                rng.normal(sp, dtype=tf.float64, **kwargs),
                rng.normal(sp, dtype=tf.float64, **kwargs))
        elif dtype==tf.complex64:
            return tf.dtypes.complex(
                rng.normal(sp, dtype=tf.float32, **kwargs),
                rng.normal(sp, dtype=tf.float32, **kwargs))
        else:
            return rng.normal(sp, dtype=dtype, **kwargs)

def randomNormal(lat, rng, new_site_shape=None, dtype=None, **kwargs):
    return lattice_map(lat, randomNormal, randomNormal_, rng, new_site_shape=new_site_shape, dtype=dtype, **kwargs)

def to_su3matrix(lat):
    return lattice_map(lat, to_su3matrix, group.su3fromvec)

def to_su3vector(lat):
    return lattice_map(lat, to_su3vector, group.su3vec)

def reduce(lat, functor, tfunctor, transform=None, scope='lattice', exclude=None):
    if isinstance(lat, Lattice):
        return lat.reduce(functor, scope=scope, exclude=exclude)
    elif isinstance(lat, (tuple, list)):
        if scope=='site':
            if isinstance(lat, tuple):
                return tuple([functor(x, scope=scope, exclude=exclude) for x in lat])
            else:
                return [functor(x, scope=scope, exclude=exclude) for x in lat]
        else:
            return tfunctor([functor(x, scope=scope, exclude=exclude) for x in lat], axis=0)
    else:
        # tf.print('reduce',scope,exclude,lat.shape)
        if transform is not None:
            lat = transform(lat)
        if exclude is None:
            return tfunctor(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tfunctor(lat, axis=axis)

def reduce_sum(lat, scope='lattice', exclude=None):
    return reduce(lat, reduce_sum, tf.math.reduce_sum, scope=scope, exclude=exclude)

def reduce_mean(lat, scope='lattice', exclude=None):
    return reduce(lat, reduce_mean, tf.math.reduce_mean, scope=scope, exclude=exclude)

def reduce_max(lat, scope='lattice', exclude=None):
    return reduce(lat, reduce_max, tf.math.reduce_max, scope=scope, exclude=exclude)

def norm2(lat, scope='lattice', exclude=None):
    # tf.print('norm2',scope,exclude)
    def transform(x):
        if x.dtype==tf.complex128 or x.dtype==tf.complex64:
            x = tf.abs(x)
        return tf.math.square(x)
    return reduce(lat, norm2, tf.math.reduce_sum, transform=transform, scope=scope, exclude=exclude)

def redot(x, y, scope='lattice', exclude=None):
    if isinstance(x, Lattice):
        return x.redot(y, scope=scope, exclude=exclude)
    elif isinstance(y, Lattice):
        return y.rredot(x, scope=scope, exclude=exclude)
    elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        if scope=='site':
            if isinstance(x,tuple):
                return tuple([redot(a,b, scope=scope, exclude=exclude) for a,b in zip(x,y)])
            else:
                return [redot(a,b, scope=scope, exclude=exclude) for a,b in zip(x,y)]
        else:
            return tf.math.reduce_sum([redot(a,b, scope=scope, exclude=exclude) for a,b in zip(x,y)], axis=0)
    elif isinstance(x, (tuple, list)):
        if scope=='site':
            if isinstance(x,tuple):
                return tuple([redot(a,y, scope=scope, exclude=exclude) for a in x])
            else:
                return [redot(a,y, scope=scope, exclude=exclude) for a in x]
        else:
            return tf.math.reduce_sum([redot(a,y, scope=scope, exclude=exclude) for a in x], axis=0)
    elif isinstance(y, (tuple, list)):
        if scope=='site':
            if isinstance(y,tuple):
                return tuple([redot(x,b, scope=scope, exclude=exclude) for b in y])
            else:
                return [redot(x,b, scope=scope, exclude=exclude) for b in y]
        else:
            return tf.math.reduce_sum([redot(x,b, scope=scope, exclude=exclude) for b in y], axis=0)
    else:
        lat = tf.math.real(tf.math.conj(x)*y)
        if exclude is None:
            return tf.math.reduce_sum(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tf.math.reduce_sum(lat, axis=axis)

def matmul(l, r, adjoint_l=False, adjoint_r=False):
    """
    Output: lattice of site-wide matrix-matrix multiplications
    Input:
        l,r: left and right lattice fields
        adjoint_l,adjoint_r: take the adjoint of l/r before multiplication
    """
    if isinstance(l, Lattice) and isinstance(r, Lattice):
        return l.matmul(r, adjoint_l=adjoint_l, adjoint_r=adjoint_r)
    elif isinstance(l, list) and isinstance(r, list):
        if len(l)==len(r):
            return [matmul(a, b, adjoint_l=adjoint_l, adjoint_r=adjoint_r) for a,b in zip(l,r)]
        else:
            raise ValueError(f'unequal length {len(l)} and {len(r)}.')
    elif isinstance(l, tuple) and isinstance(r, tuple):
        if len(l)==len(r):
            return tuple([matmul(a, b, adjoint_l=adjoint_l, adjoint_r=adjoint_r) for a,b in zip(l,r)])
        else:
            raise ValueError(f'unequal length {len(l)} and {len(r)}.')
    elif isinstance(l, (Lattice, list, tuple)) or isinstance(r, (Lattice, list, tuple)):
        raise ValueError(f'unsupported object classes {l.__class__} and {r.__class__}')
    else:
        return tf.linalg.matmul(l, r, adjoint_a=adjoint_l, adjoint_b=adjoint_r)

def matvec(m, v, adjoint=False):
    """
    Output: lattice of site-wide matrix-vector multiplications
    Input:
        m,v: matrix and vector lattice fields
        adjoint: take the adjoint of m before multiplication
    """
    if isinstance(m, Lattice) and isinstance(v, Lattice):
        return m.matvec(v, adjoint=adjoint)
    elif isinstance(m, list) and isinstance(v, list):
        if len(m)==len(v):
            return [matvec(a, b, adjoint=adjoint) for a,b in zip(m,v)]
        else:
            raise ValueError(f'unequal length {len(m)} and {len(v)}.')
    elif isinstance(m, tuple) and isinstance(v, tuple):
        if len(m)==len(v):
            return tuple([matvec(a, b, adjoint=adjoint) for a,b in zip(m,v)])
        else:
            raise ValueError(f'unequal length {len(m)} and {len(v)}.')
    elif isinstance(m, (Lattice, list, tuple)) or isinstance(v, (Lattice, list, tuple)):
        raise ValueError(f'unsupported object classes {m.__class__} and {v.__class__}')
    else:
        return tf.linalg.matvec(m, v, adjoint_a=adjoint)

# base class for all

class Lattice:
    """
    Wrapper for tf.Tensor with enough information for lattice objects.
    """
    def __init__(self, tensor, nd=4, batch_dim=-1, subset=SubSetAll):
        """
        tensor: a single tf.Tensor representing the lattice data.
            shape: Dims... Site ...
                Dims from T to X, with X direction fastest
                batch_dim, if exists, can be before or after Dims
        nd: number of dimensions in the lattice.
        batch_dim: the location of the batch dimension.  -1 means no batch dimension.
        subset: even, odd, all.

        The shape of the tensor begins with nd integers,
        with possible batch_dim denotes the batch dimension,
        before or after the nd dimensions.
        """
        if batch_dim>0 and batch_dim<nd:
            raise ValueError(f'expecting consecutive {nd} dimensions in shape, but batch_dim {batch_dim} breaks it.')
        if not isinstance(subset, SubSet):
            raise ValueError(f'subset ({subset.__class__}) is not a SubSet')
        self.data = tensor
        self.nd = nd
        self.batch_dim = batch_dim
        self.subset = subset
    def full_volume(self):
        shape = self.data.shape
        nd = self.nd
        if self.batch_dim==0:
            lat = shape[1:nd+1]
        else:  # self.batch_dim<0 or self.batch_dim>=nd:
            lat = shape[:nd]
        return math.prod(lat)*SubSet.NP/len(self.subset.to_corner_indices())
    def site_shape(self):
        shape = self.data.shape
        nd = self.nd
        if self.batch_dim<0:
            return shape[nd:]
        elif self.batch_dim<=nd:
            return shape[nd+1:]
        else:
            raise ValueError(f'got shape {shape} with nd {nd} and batch_dim {self.batch_dim}')
    def get_site(self, *xs):
        "xs: X,Y,Z,..."
        ss = SubSet.from_coord(xs)
        if ss not in self.subset:
            raise ValueError(f'coord {xs} not in subset {self.subset}')
        nd = self.nd
        if len(xs)>nd:
            raise ValueError(f'index {xs} longer than nd')
        ix = [x//2 for x in xs]
        begin = [0]*nd
        size = [-1]*nd
        axis = []
        for i,x in enumerate(ix):
            a = nd-1-i
            begin[a] = x
            size[a] = 1
            axis.append(a)
        if self.batch_dim==0:
            begin = [0]+begin
            size = [-1]+size
            axis = [a+1 for a in axis]
        l = len(self.data.shape)-len(begin)
        begin = begin+[0]*l
        size = size+[-1]*l
        return tf.squeeze(tf.slice(self.data, begin=begin, size=size), axis=axis)
    def batch_size(self):
        bd = self.batch_dim
        if bd<0:
            return 0
        else:
            return self.data.shape[bd]
    def get_batch(self, b):
        if self.batch_dim<0:
            return self
        elif self.batch_dim==0:
            return self.wrap(self.data[b], batch_dim=-1)
        elif self.batch_dim==4:
            return self.wrap(self.data[:,:,:,:,b], batch_dim=-1)
        else:
            raise ValueError(f'unimplemented for batch_dim {self.batch_dim}')
    def batch_update(self, cond, other):
        if self.batch_dim<0:
            return self.wrap(tf.cond(cond, lambda:other.data, lambda:self.data))
        else:
            c = tf.reshape(cond, (1,)*d + cond.shape + (1,)*(len(self.data.shape)-1-d))
            return self.wrap(tf.where(c, other.data, self.data))
    def get_subset(self, subset):
        return self.hypercube_partition().get_subset(subset)
    def if_compatible(self, other, f, e=None, keep_unmatched_subset=False):
        """
        If compatible, run f on data, else run e on our data.
        Result drops unmatched subset, unless keep_unmatched_subset is True.
        Always raise ValueError if subsets do not overlap.
        """
        if self.is_compatible(other):
            if self.subset==other.subset:
                return self.wrap(f(self.data,other.data))
            elif self.subset in other.subset:
                res = self.wrap(f(self.data,other.get_subset(self.subset).unwrap()))
                if keep_unmatched_subset:
                    res = combine_subsets(other.get_subset(other.subset-self.subset), res)
                return res
            elif other.subset in self.subset:
                res = other.wrap(f(self.get_subset(other.subset).unwrap(),other.data))
                if keep_unmatched_subset:
                    res = combine_subsets(self.get_subset(self.subset-other.subset), res)
                return res
            else:
                ss = self.subset.intersection(other.subset)
                if not ss.is_empty():
                    res = self.wrap(f(self.get_subset(ss).unwrap(),other.get_subset(ss).unwrap()), subset=ss)
                    if keep_unmatched_subset:
                        res = combine_subsets(self.get_subset(self.subset-ss), other.get_subset(other.subset-ss), res)
                    return res
                else:
                    raise ValueError(f'subsets have no intersection {self} vs {other}')
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            return self.wrap(e(self.data))
    def hypercube_partition(self):
        batch_dim = self.batch_dim
        nd = self.nd
        subset = self.subset
        if subset!=SubSetAll:
            raise ValueError(f'only supports subset SubSetAll but got {subset}')
        shape = self.data.shape
        if batch_dim==0:
            batch_size = shape[0]
            lat_shape = shape[1:1+nd]
            part_shape = shape[:1]+tuple([s//2 for s in lat_shape])+shape[1+nd:]
        else:
            batch_size = 0
            lat_shape = shape[:nd]
            part_shape = tuple([s//2 for s in lat_shape])+shape[nd:]
        mask = hypercube_mask(lat_shape,batch_size)
        latparts = tf.dynamic_partition(self.data, mask, 1<<nd)
        return LatticeHypercubeParts(
            [self.wrap(tf.reshape(p, part_shape), subset=SubSet.from_corner_index(i)) for i,p in enumerate(latparts)],
            nd = nd,
            batch_dim = batch_dim,
            subset = subset)

    # generic methods follows

    def shifts(self, site):
        """
        Shift value from site.
        site: list or tuple with site[dir] the index at the direction
              with dir: 0,1,2,... corresponding to x,y,z,...
        """
        v = self
        for dir,len in enumerate(site):
            v = v.shift(dir, len)
        return v
    def wrap(self, tensor, **kwargs):
        dkwargs = {'nd':self.nd, 'batch_dim':self.batch_dim, 'subset':self.subset, **kwargs}
        return self.__class__(tensor, **dkwargs)
    def unwrap(self):
        return self.data
    def from_tensors(self, tensors):
        d = self.unwrap()
        if isinstance(d, Lattice):
            tensors = d.from_tensors(tensors)
        elif isinstance(d, (list,tuple)):
            tensors = [a.from_tensors(t) for a,t in zip(d,tensors)]
        return self.wrap(tensors)
    def to_tensors(self):
        d = self.unwrap()
        if isinstance(d, Lattice):
            d = d.to_tensors()
        elif isinstance(d, (list,tuple)):
            d = [a.to_tensors() for a in d]
        return d

    # generic methods recurse call generic functions

    def shift(self, direction, length, subset=SubSetAll):  # no axis param here, we decide axis
        # print(f'shift dir {direction} len {length}')
        axis = self.nd-direction-1
        if self.batch_dim==0:
            axis = axis+1
        if self.subset in subset:
            shiftedsubset = self.subset.shift(direction, length)
            return self.wrap(shift(self.unwrap(), direction, length, subset=self.subset, axis=axis), subset=shiftedsubset)
        else:
            raise ValueError(f'subset mismatch in shift: {self.subset} is not in the requested {subset}')
    def zeros(self, new_site_shape=None, dtype=None, site_shape_len=None, **kwargs):
        "kwargs pass to self.wrap"
        return self.wrap(zeros(self.unwrap(), new_site_shape=new_site_shape, site_shape_len=len(self.site_shape()), dtype=dtype), **kwargs)
    def unit(self, **kwargs):
        "kwargs pass to self.wrap"
        return self.wrap(unit(self.unwrap(), site_shape_len=len(self.site_shape())), **kwargs)
    def typecast(self, x):
        return typecast(self.unwrap(), x)
    def map(self, functor, *args, **kwargs):
        "args/kwargs are extra argument to functor"
        return self.wrap(functor(self.unwrap(), *args, **kwargs))
    def real(self):
        return self.map(real)
    def imag(self):
        return self.map(imag)
    def trace(self):
        return self.map(trace)
    def det(self):
        return self.map(det)
    def exp(self):
        return self.map(exp)
    def projectSU(self):
        return self.map(projectSU)
    def projectTangent(self):
        return self.map(projectTangent)
    def randomTangent(self, rng):
        return self.map(randomTangent, rng)
    def randomNormal(self, rng, new_site_shape=None, dtype=None, site_shape_len=None, **kwargs):
        return self.map(randomNormal, rng, new_site_shape=new_site_shape, dtype=dtype, site_shape_len=len(self.site_shape()), **kwargs)
    def to_su3matrix(self):
        return self.map(to_su3matrix)
    def to_su3vector(self):
        return self.map(to_su3vector)
    def reduce(self, functor, scope='lattice', exclude=None):
        if scope=='lattice':
            if exclude is None:
                exclude = (self.batch_dim,)
            # tf.print('Lattice.reduce',scope,exclude)
            return functor(self.unwrap(), scope=scope, exclude=exclude)
        elif scope=='site':
            if exclude is None:
                exclude = tuple(range(self.nd))
                if self.batch_dim in exclude:
                    exclude += (self.nd,)
                else:
                    exclude += (self.batch_dim,)
            return self.wrap(functor(self.unwrap(), scope=scope, exclude=exclude))
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
    def reduce_sum(self, scope='lattice', exclude=None):
        return self.reduce(reduce_sum, scope=scope, exclude=exclude)
    def reduce_mean(self, scope='lattice', exclude=None):
        return self.reduce(reduce_mean, scope=scope, exclude=exclude)
    def reduce_max(self, scope='lattice', exclude=None):
        return self.reduce(reduce_max, scope=scope, exclude=exclude)
    def norm2(self, scope='lattice', exclude=None):
        return self.reduce(norm2, scope=scope, exclude=exclude)
    def redot(self, other, scope='lattice', exclude=None):
        if self.is_compatible(other):
            other = other.unwrap()
        if scope=='lattice':
            if exclude is None:
                exclude = (self.batch_dim,)
            return redot(self.unwrap(), other, scope=scope, exclude=exclude)
        elif scope=='site':
            if exclude is None:
                exclude = tuple(range(self.nd))
                if self.batch_dim in exclude:
                    exclude += (self.nd,)
                else:
                    exclude += (self.batch_dim,)
            return self.wrap(redot(self.unwrap(), other, scope=scope, exclude=exclude))
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
    def rredot(self, other, scope='lattice', exclude=None):
        if self.is_compatible(other):
            other = other.unwrap()
        if scope=='lattice':
            if exclude is None:
                exclude = (self.batch_dim,)
            return redot(other, self.unwrap(), scope=scope, exclude=exclude)
        elif scope=='site':
            if exclude is None:
                exclude = tuple(range(self.nd))
                if self.batch_dim in exclude:
                    exclude += (self.nd,)
                else:
                    exclude += (self.batch_dim,)
            return self.wrap(redot(other, self.unwrap(), scope=scope, exclude=exclude))
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
    def matmul(self, mat, adjoint_l=False, adjoint_r=False):
        return self.if_compatible(mat,
          lambda a,b:matmul(a,b,adjoint_l=adjoint_l,adjoint_r=adjoint_r),
          lambda a:matmul(a,mat,adjoint_l=adjoint_l,adjoint_r=adjoint_r))
    def matvec(self, vec, adjoint=False):
        return self.if_compatible(vec,
          lambda a,b:matvec(a,b,adjoint=adjoint),
          lambda a:matvec(a,mat,adjoint=adjoint))

    def is_compatible(self, other):
        return isinstance(other, self.__class__) and self.__class__==other.__class__ and self.nd==other.nd and self.batch_dim==other.batch_dim

    # builtin generics

    def __str__(self):
        return f'{self.__class__.__name__}{(self.nd,self.batch_dim,self.subset)}'
    def __repr__(self):
        return f'{self.__class__.__name__}#{hex(id(self))}{(self.nd,self.batch_dim,self.subset)}'
    def __getitem__(self, key):
        return self.data[key]
    def __add__(self, other):
        return self.if_compatible(other, lambda a,b:a+b, lambda a:a+other, keep_unmatched_subset=True)
    def __radd__(self, other):
        return self.if_compatible(other, lambda a,b:b+a, lambda a:other+a, keep_unmatched_subset=True)
    def __sub__(self, other):
        return self.if_compatible(other, lambda a,b:a-b, lambda a:a-other, keep_unmatched_subset=True)
    def __rsub__(self, other):
        return self.if_compatible(other, lambda a,b:b-a, lambda a:other-a, keep_unmatched_subset=True)
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

class LatticeList(Lattice):
    "A list of Lattice"
    def __init__(self, list, nd=None, batch_dim=None, subset=None, **kwargs):
        vol = None
        listsubset = None
        for lat in list:
            if not isinstance(lat, Lattice):
                raise ValueError(f'lat ({lat.__class__}) is not a Lattice')
            if nd is None:
                nd = lat.nd
            elif nd!=lat.nd:
                raise ValueError(f'incompatible nd {nd} from lattice {lat}')
            if batch_dim is None:
                batch_dim = lat.batch_dim
            elif batch_dim!=lat.batch_dim:
                raise ValueError(f'incompatible batch_dim {batch_dim} from lattice {lat}')
            if vol is None:
                vol = lat.full_volume()
            elif vol!=lat.full_volume():
                raise ValueError(f'incompatible volume {vol} from lattice {lat}')
            if listsubset is None:
                listsubset = lat.subset
            else:
                listsubset = listsubset.union(lat.subset)
        if subset is not None and subset!=listsubset:
            raise ValueError(f'specified {subset} and subset from parts {listsubset} mismatch')
        super(LatticeList, self).__init__(sorted(list,key=lambda x:x.subset), nd=nd, batch_dim=batch_dim, subset=listsubset, **kwargs)
    def full_volume(self):
        return self.data[0].full_volume()
    def site_shape(self):
        return self.data[0].site_shape()
    def get_site(self, *xs):
        ss = SubSet.from_coord(xs)
        if ss not in self.subset:
            raise ValueError(f'coord {xs} not in subset {self.subset}')
        ls = [lat for lat in self.data if ss in lat.subset]
        if len(ls)!=1:
            raise ValueError(f'coord {xs} found in {len(ls)} sub-lattice')
        return ls[0].get_site(*xs)
    def batch_size(self):
        bd = self.batch_dim
        if bd<0:
            return 0
        else:
            return self.data[0].batch_size()
    def get_batch(self, b):
        if self.batch_dim<0:
            return self
        else:
            return self.wrap([lat.get_batch(b) for lat in self.data])
    def batch_update(self, cond, other):
        return self.wrap([lat.batch_update(cond, o) for lat,o in zip(self.data,other.data)])
    def if_compatible(self, other, f, e=None, keep_unmatched_subset=False):
        """
        If compatible, run f on data, else run e on our data.
        Result drops unmatched subset, unless keep_unmatched_subset is True.
        Always raise ValueError if subsets do not overlap.
        """
        if self.is_compatible(other):
            if self.subset==other.subset:
                return self.wrap([f(a,b) for a,b in zip(self.data,other.data)])
            elif self.subset in other.subset:
                res = self.wrap([f(a,b) for a,b in zip(self.data,other.get_subset(self.subset).unwrap())])
                if keep_unmatched_subset:
                    res = combine_subsets(other.get_subset(other.subset-self.subset), res)
                return res
            elif other.subset in self.subset:
                res = other.wrap([f(a,b) for a,b in zip(self.get_subset(other.subset).unwrap(),other.data)])
                if keep_unmatched_subset:
                    res = combine_subsets(self.get_subset(self.subset-other.subset), res)
                return res
            else:
                ss = self.subset.intersection(other.subset)
                if not ss.is_empty():
                    res = self.wrap([f(a,b) for a,b in zip(self.get_subset(ss).unwrap(),other.get_subset(ss).unwrap())], subset=ss)
                    if keep_unmatched_subset:
                        res = combine_subsets(self.get_subset(self.subset-ss), other.get_subset(other.subset-ss), res)
                    return res
                else:
                    raise ValueError(f'subsets have no intersection {self} vs {other}')
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            return self.wrap([e(a) for a in self.data])
    def get_subset(self, subset):
        return self.wrap([l for l in self.unwrap() if l.subset in subset], subset=subset)
    def __str__(self):
        return f'{self.__class__.__name__}{(self.nd,self.batch_dim,self.subset,self.data)}'
    def __repr__(self):
        return f'{self.__class__.__name__}#{hex(id(self))}{(self.nd,self.batch_dim,self.subset,self.data)}'
    def __neg__(self):
        return self.wrap([-l for l in self.data])
    def __pos__(self):
        return self.wrap([+l for l in self.data])

class LatticeHypercubeParts(LatticeList):
    """
    Same as Lattice with the lattice data represented as 2^nd tensors, each from one corner of the 2^nd hypercube.
    """
    def combine_hypercube(self):
        if self.subset!=SubSetAll:
            raise ValueError(f'subset {self.subset} is not SubSetAll')
        batch_dim = self.batch_dim
        nd = self.nd
        shape = self.data[0].unwrap().shape
        if batch_dim==0:
            batch_size = shape[0]
            lat_shape = tuple([2*d for d in shape[1:1+nd]])
            site_shape = shape[1+nd:]
            full_shape = shape[:1]+lat_shape+site_shape
        else:
            batch_size = 0
            lat_shape = tuple([2*d for d in shape[:nd]])
            site_shape = shape[nd:]
            full_shape = lat_shape+site_shape
        part = hypercube_mask(lat_shape,batch_size)
        cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 1<<nd)
        flat = [tf.reshape(p.unwrap(), (-1,)+tuple(site_shape)) for p in self.data]
        return Lattice(
            tf.reshape(tf.dynamic_stitch(cond, flat), full_shape),
            nd = nd,
            batch_dim = batch_dim,
            subset = self.subset)

def from_hypercube_parts(parts, **kwargs):
    """pass kwargs to LatticeHypercubeParts"""
    kwargs = {'subset':SubSetAll, **kwargs}
    subset = kwargs['subset']
    ix = subset.to_corner_indices()
    if len(ix)!=len(parts):
        raise ValueError(f'subset {subset} needs {len(ix)} parts but got only {len(parts)}')
    pargs = lambda i: {**kwargs, 'subset':SubSet.from_corner_index(i)}
    return LatticeHypercubeParts([Lattice(p, **pargs(i)) for i,p in zip(ix,parts)], **kwargs)

def combine_subsets(*lats):
    subset = SubSet()
    ps = []
    for lat in lats:
        if not isinstance(lat, LatticeHypercubeParts):
            raise ValueError(f'lattice {lat} is not a LatticeHypercubeParts')
        if not lats[0].is_compatible(lat):
            raise ValueError(f'lattice not compatible {lats[0]} and {lat}')
        subset = subset.union(lat.subset)
        ps += lat.unwrap()
    return lats[0].wrap(ps, subset=subset)

def hypercube_mask(dim, batch_size=0):
    """
    Output: tensor with elements in range(1<<nd), each label a corner of the hypercube,
            shape dim if batch_size==0 else (batch_size,)+dim
    Input:
        dim: the lattice shape for computing hypercube corners
        batch_shape: shape of the batch
    """
    nd = len(dim)
    for d in dim:
        if d%2 != 0:
            raise ValueError(f'dim is odd: {dim}')
    cube = tf.reshape(tf.range(1<<nd), (2,)*nd)
    mask = tf.tile(cube, [x//2 for x in dim])
    if batch_size>0:
        mask = tf.expand_dims(mask, 0)
        mask = tf.tile(mask, (batch_size,)+(1,)*nd)
    return mask

def evenodd_mask(dim, batch_size=0):
    """
    Output: tensor with elements 0 (even) or 1 (odd),
            shape dim if batch_size==0 else (batch_size,)+dim
    Input:
        dim: the lattice shape for computing even and odd
        batch_shape: shape of the batch
    """
    nd = len(dim)
    if nd>4:
        raise ValueError(f'unimplemented for dim {dim}')
    for d in dim:
        if d%2 != 0:
            raise ValueError(f'dim is odd: {dim}')
    eo = [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]
    cube = tf.reshape(eo[:1<<nd], (2,)*nd)
    mask = tf.tile(cube, [x//2 for x in dim])
    if batch_size>0:
        mask = tf.expand_dims(mask, 0)
        mask = tf.tile(mask, (batch_size,)+(1,)*nd)
    return mask

if __name__=='__main__':
    import group as g
    import sys

    lat = (4,4,4,4)
    gauge = tf.zeros(lat, dtype=tf.int32)
    gauge += tf.reshape(tf.range(4, dtype=tf.int32), (1,1,1,lat[3]))
    gauge += tf.reshape(tf.range(4, dtype=tf.int32)*10, (1,1,lat[3],1))
    gauge += tf.reshape(tf.range(4, dtype=tf.int32)*100, (1,lat[3],1,1))
    gauge += tf.reshape(tf.range(4, dtype=tf.int32)*1000, (lat[3],1,1,1))

    tf.print('Full gauge:')
    tf.print(gauge)

    gcube = Lattice(gauge).hypercube_partition()
    for i in range(len(gcube.unwrap())):
        tf.print('cube',i)
        tf.print(gcube.unwrap()[i])

    gfull = gcube.combine_hypercube()
    tf.print('recombined')
    tf.print(gfull.unwrap())

    tf.print('diff',g.norm2(gfull.unwrap()-gauge, range(len(gauge.shape))))
