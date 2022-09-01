import math
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
            raise ValueError(f'subset ({subset.__class__}) is not a SubSet')
        return subset.s == subset.s&self.s
    def from_corner_index(i):
        return SubSet(1<<i)
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

def zeros(lat, **kwargs):
    """Pass kwargs to Lattice.zeros"""
    if isinstance(lat, Lattice):
        return lat.zeros(**kwargs)
    elif len(kwargs)>0:
        raise ValueError(f'unsupported kwargs {kwargs}')
    elif isinstance(lat, list):
        return [zeros(x) for x in lat]
    elif isinstance(lat, tuple):
        return tuple([zeros(x) for x in lat])
    else:
        return tf.zeros(lat.shape, dtype=lat.dtype)

def typecast(lat, x):
    """makes x the same dtype as lat"""
    if isinstance(lat, Lattice):
        return lat.typecast(x)
    elif isinstance(lat, (tuple, list)):
        return typecast(lat[0], x)
    else:
        return tf.cast(x, dtype=lat.dtype)

def trace(lat):
    """
    Output: trace of the site over lattice
    Input:
        lattice: assuming matrix valued sites, and batch_dim never in the matrix
    """
    if isinstance(lat, Lattice):
        return lat.trace()
    elif isinstance(lat, list):
        return [trace(x) for x in lat]
    elif isinstance(lat, tuple):
        return tuple([trace(x) for x in lat])
    else:
        return tf.linalg.trace(lat)

def reduce_sum(lat, scope='lattice', exclude=None):
    if isinstance(lat, Lattice):
        return lat.reduce_sum(scope=scope, exclude=exclude)
    elif isinstance(lat, (tuple, list)):
        return sum([reduce_sum(x, scope=scope, exclude=exclude) for x in lat])
    else:
        if exclude is None:
            return tf.math.reduce_sum(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tf.math.reduce_sum(lat, axis=axis)

def reduce_mean(lat, scope='lattice', exclude=None):
    if isinstance(lat, Lattice):
        return lat.reduce_mean(scope=scope, exclude=exclude)
    elif isinstance(lat, (tuple, list)):
        return sum([reduce_mean(x, scope=scope, exclude=exclude) for x in lat])/len(lat)
    else:
        if exclude is None:
            return tf.math.reduce_mean(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tf.math.reduce_mean(lat, axis=axis)

def reduce_max(lat, scope='lattice', exclude=None):
    if isinstance(lat, Lattice):
        return lat.reduce_max(scope=scope, exclude=exclude)
    elif isinstance(lat, (tuple, list)):
        return max([reduce_max(x, scope=scope, exclude=exclude) for x in lat])
    else:
        if exclude is None:
            return tf.math.reduce_max(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tf.math.reduce_max(lat, axis=axis)

def norm2(lat, scope='lattice', exclude=None):
    if isinstance(lat, Lattice):
        return lat.norm2(scope=scope, exclude=exclude)
    elif isinstance(lat, (tuple, list)):
        return sum([norm2(x, scope=scope, exclude=exclude) for x in lat])
    else:
        if lat.dtype==tf.complex128 or lat.dtype==tf.complex64:
            lat = tf.abs(lat)
        lat = tf.math.square(lat)
        if exclude is None:
            return tf.math.reduce_sum(lat)
        else:
            axis = [i for i in range(len(lat.shape)) if i not in exclude]
            return tf.math.reduce_sum(lat, axis=axis)

def redot(x, y, scope='lattice', exclude=None):
    if isinstance(x, Lattice):
        return x.redot(y, scope=scope, exclude=exclude)
    elif isinstance(y, Lattice):
        return y.rredot(x, scope=scope, exclude=exclude)
    elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return sum([redot(a,b, scope=scope, exclude=exclude) for a,b in zip(x,y)])
    elif isinstance(x, (tuple, list)):
        return sum([redot(a,y, scope=scope, exclude=exclude) for a in x])
    elif isinstance(y, (tuple, list)):
        return sum([redot(x,b, scope=scope, exclude=exclude) for b in y])
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
    def size(self):
        return tf.size(self.data)
    def site_shape(self):
        shape = self.data.shape
        nd = self.nd
        if self.batch_dim<0:
            return shape[nd:]
        elif self.batch_dim<=nd:
            return shape[nd+1:]
        else:
            raise ValueError(f'got shape {shape} with nd {nd} and batch_dim {self.batch_dim}')
    def if_compatible(self, other, f, e=None):
        if self.is_compatible(other):
            return self.wrap(f(self.data,other.data))
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
        axis = self.nd-direction-1
        if self.batch_dim==0:
            axis = axis+1
        if self.subset in subset:
            shiftedsubset = self.subset.shift(direction, length)
            return self.wrap(shift(self.unwrap(), direction, length, subset=self.subset, axis=axis), subset=shiftedsubset)
        else:
            raise ValueError(f'subset mismatch in shift: {self.subset} is not in the requested {subset}')
    def zeros(self, **kwargs):
        "kwargs pass to self.wrap"
        return self.wrap(zeros(self.unwrap()), **kwargs)
    def typecast(self, x):
        return typecast(self.unwrap(), x)
    def trace(self):
        return self.wrap(trace(self.unwrap()))
    def reduce(self, functor, scope='lattice', exclude=None):
        if scope=='lattice':
            if exclude is None:
                exclude = (self.batch_dim,)
            return functor(self.unwrap(), scope=scope, exclude=exclude)
        elif scope=='site':
            if exclude is None:
                exclude = tuple(range(self.nd))
                if self.batch_dim in exclude:
                    exclude += (self.nd+1,)
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
                    exclude += (self.nd+1,)
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
                    exclude += (self.nd+1,)
                else:
                    exclude += (self.batch_dim,)
            return self.wrap(redot(other, self.unwrap(), scope=scope, exclude=exclude))
        else:
            raise ValueError(f'reduction scope should be "lattice" or "site", but got {scope}')
    def matmul(self, mat, adjoint_l=False, adjoint_r=False):
        if self.is_compatible(mat):
            if self.subset==mat.subset:
                return self.wrap(matmul(self.unwrap(), mat.unwrap(), adjoint_l=adjoint_l, adjoint_r=adjoint_r))
            elif (s:=self.subset) in mat.subset:
                return self.wrap(matmul(self.unwrap(), mat.get_subset(s).unwrap(), adjoint_l=adjoint_l, adjoint_r=adjoint_r))
            elif (s:=mat.subset) in self.subset:
                return mat.wrap(matmul(self.get_subset(s).unwrap(), mat.unwrap(), adjoint_l=adjoint_l, adjoint_r=adjoint_r))
            elif not (s:=self.subset.intersection(mat.subset)).is_empty():
                return self.wrap(matmul(self.get_subset(s).unwrap(), mat.get_subset(s).unwrap(), adjoint_l=adjoint_l, adjoint_r=adjoint_r), subset=s)
            else:
                raise ValueError(f'subsets have no intersection {self.subset} and {mat.subset}')
        else:
            raise ValueError(f'incompatible lattice {self.__class__} and {mat.__class__}')
    def matvec(self, vec, adjoint=False):
        if self.is_compatible(vec):
            if self.subset==vec.subset:
                return self.wrap(matvec(self.unwrap(), vec.unwrap(), adjoint=adjoint))
            elif self.subset in vec.subset:
                return self.wrap(matvec(self.unwrap(), vec.get_subset(self.subset).unwrap(), adjoint=adjoint))
            elif vec.subset in self.subset:
                return vec.wrap(matvec(self.get_subset(vec.subset).unwrap(), vec.unwrap(), adjoint=adjoint))
            elif not self.subset.intersection(vec.subset).is_empty():
                s = self.subset.intersection(vec.subset)
                return self.wrap(matvec(self.get_subset(s).unwrap(), vec.get_subset(s).unwrap(), adjoint=adjoint), subset=s)
            else:
                raise ValueError(f'subsets have no intersection {self.subset} and {vec.subset}')
        else:
            raise ValueError(f'incompatible lattice {self.__class__} and {vec.__class__}')

    def is_compatible(self, other):
        return isinstance(other, self.__class__) and self.__class__==other.__class__ and self.nd==other.nd and self.batch_dim==other.batch_dim

    # builtin generics

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
        return self.wrap(-self.data)
    def __pos__(self):
        return self.wrap(+self.data)

class LatticeHypercubeParts(Lattice):
    """
    Same as Lattice with the lattice data represented as 2^nd tensors, each from one corner of the 2^nd hypercube.
    """
    def __init__(self, hypercube_parts, **kwargs):
        subset = SubSet()
        for lat in hypercube_parts:
            if not isinstance(lat, Lattice):
                raise ValueError(f'lat ({lat.__class__}) is not a Lattice')
            subset = subset.union(lat.subset)
        super(LatticeHypercubeParts, self).__init__(sorted(hypercube_parts,key=lambda x:x.subset), **kwargs)
        if self.subset!=subset:
            raise ValueError(f'initialized subset {self.subset} and subset from parts {subset} mismatch')
    def size(self):
        return sum([l.size() for l in self.data])
    def site_shape(self):
        return self.data[0].site_shape()
    def if_compatible(self, other, f, e=None):
        if self.is_compatible(other):
            return self.wrap([f(a,b) for a,b in zip(self.data,other.data)])
        elif e is None:
            raise ValueError('operation only possible with Lattice object')
        else:
            return self.wrap([e(a) for a in self.data])
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
    def get_subset(self, subset):
        return self.wrap([l for l in self.unwrap() if l.subset in subset], subset=subset)

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
    Output: tensor with elements of 0 (even) and 1 (odd), shape dim if batch_size==0 else (batch_size,)+dim
    Input:
        dim: the lattice shape for computing even and odd
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
