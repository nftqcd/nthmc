import math
import tensorflow as tf
import parts

def hypercube_mask(dim, batch_shape=()):
    """
    Output: tensor with elements of 0 (even) and 1 (odd), shape batch_shape+dim
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
    nb = len(batch_shape)
    if nb>0:
        mask = tf.reshape(mask, (1,)*nb+mask.shape)
        mask = tf.tile(mask, batch_shape+(1,)*len(dim))
    return mask

def hypercube_partition(lattice, nd=4, batch_dims=0):
    """
    Output: 2^nd lattice, eoch shape: batch_shape T//2 .. X//2 Site ...
    Input:
        lattice: tensor, shape: batch_shape Dims... Site ...
            Dims from T to X, with X direction fastest
            Assuming all dims are even numbers.
        nd: length of the dimension of the lattice
        batch_dims: number of dimensions in the lattice.shape belong to batches
    """
    batch_shape = lattice.shape[:batch_dims]
    lat_shape = lattice.shape[batch_dims:batch_dims+nd]
    mask = hypercube_mask(lat_shape,batch_shape)
    latparts = tf.dynamic_partition(lattice, mask, 1<<nd)
    return parts.HypercubeParts(
        [tf.reshape(p, batch_shape+tuple([s//2 for s in lat_shape])+lattice.shape[batch_dims+nd:])
            for p in latparts],
        subset='all')

def combine_hypercube(lattice, nd=4, batch_dims=0):
    """
    Output: lattice, shape batch_shape T .. X Site ..., combined from the input even-odd
    Input:
        lattice: a list of 2^nd tensor, shape: batch_shape T .. X//2 Site ...
        nd: length of the dimension of the lattice
        batch_dims: number of dimensions in the lattice.shape belong to batches
    """
    batch_shape = lattice[0].shape[:batch_dims]
    lat_shape = tuple([2*d for d in lattice[0].shape[batch_dims:batch_dims+nd]])
    site_shape = lattice[0].shape[batch_dims+nd:]
    part = hypercube_mask(lat_shape,batch_shape)
    cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 1<<nd)
    flat = [tf.reshape(p, (-1,)+tuple(site_shape)) for p in lattice]
    return tf.reshape(tf.dynamic_stitch(cond, flat), batch_shape+lat_shape+site_shape)

def size(x):
    if isinstance(x, parts.HypercubeParts):
        return sum([size(l) for l in x])
    else:
        return tf.size(x)

# NOTE: the following only works for nd<=4
def is_even_index(x):
    return 0 == ((x&0b1) ^ ((x&0b10)>>1) ^ ((x&0b100)>>2) ^ ((x&0b1000)>>3))
even_indices = [0b0, 0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100, 0b1111]
odd_indices = [0b1, 0b10, 0b100, 0b111, 0b1000, 0b1011, 0b1101, 0b1110]

def get_even(lattice):
    n = len(lattice)
    return parts.HypercubeParts([lattice[i] for i in even_indices if i<n], subset='even')

def get_odd(lattice):
    n = len(lattice)
    return parts.HypercubeParts([lattice[i] for i in odd_indices if i<n], subset='odd')

def get_subset(lattice, subset):
    if lattice.subset=='all':
        if subset=='even':
            return get_even(lattice)
        elif subset=='odd':
            return get_odd(lattice)
        else:
            raise ValueError(f'unsupported subset {subset}')
    else:
        raise ValueError(f'unsupported lattice with subset {lattice.subset}')

def mix_evenodd(latE, latO):
    if not(isinstance(latE, parts.HypercubeParts) and isinstance(latO, parts.HypercubeParts)):
        raise ValueError('latE and latO must be HypercubeParts')
    if len(latE)!=len(latO):
        raise ValueError(f'even odd partitions have different sizes: even {len(latE)} odd {len(latO)}')
    return parts.HypercubeParts(
        [latE[even_indices.index(i)] if is_even_index(i) else latO[odd_indices.index(i)]
            for i in range(2*len(latE))],
        subset='all')

def zeros_like(v):
    if isinstance(v, parts.HypercubeParts):
        return parts.HypercubeParts([zeros_like(x) for x in v], subset=v.subset)
    else:
        return tf.zeros(v.shape, dtype=v.dtype)

def typecast(x, target):
    if isinstance(target, parts.HypercubeParts):
        return typecast(x, target[0])
    else:
        return tf.cast(x, dtype=target.dtype)

def mul(l, r, nd, batch_dims=0, adjoint_l=False, adjoint_r=False):
    """
    Output: lattice of site-wide matrix-matrix, matrix-vector, or vector-matrix multiplications
    Input:
        l,r: left and right lattice fields
        nd: length of the dimension of the lattice
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
        adjoint_l,adjoint_r: take the adjoint of l/r before multiplication if it is a matrix field
    """
    if isinstance(l, parts.HypercubeParts) and isinstance(r, parts.HypercubeParts):
        if l.subset==r.subset and len(l)==len(r):
            return parts.HypercubeParts(
                [mul(a, b, nd, batch_dims=batch_dims, adjoint_l=adjoint_l, adjoint_r=adjoint_r)
                    for a,b in zip(l,r)],
                subset=l.subset)
        elif l.subset=='all':
            return mul(get_subset(l, r.subset), r, nd, batch_dims=batch_dims, adjoint_l=adjoint_l, adjoint_r=adjoint_r)
        elif r.subset=='all':
            return mul(l, get_subset(r, l.subset), nd, batch_dims=batch_dims, adjoint_l=adjoint_l, adjoint_r=adjoint_r)
        else:
            raise ValueError(f'unsupported subset pair {l.subset} and {r.subset}')
    elif isinstance(l, parts.HypercubeParts) or isinstance(r, parts.HypercubeParts):
        raise ValueError(f'incompatible l and r')
    nsl = len(l.shape)-nd-batch_dims
    nsr = len(r.shape)-nd-batch_dims
    if nsl<1 or nsl>2:
        raise ValueError(f'unrecognized l.shape {l.shape}')
    elif nsr<1 or nsr>2:
        raise ValueError(f'unrecognized r.shape {r.shape}')
    elif nsl==1 and nsr==1:
        raise ValueError(f'cannot group multiply two shapes {l.shape} {r.shape}')
    elif nsl==2 and nsr==2:
        return tf.linalg.matmul(l, r, adjoint_a=adjoint_l, adjoint_b=adjoint_r)
    elif nsl==2 and nsr==1:
        if adjoint_r:
            raise ValueError(f'cannot compute adjoint of tensor with shape {r.shape}')
        return tf.linalg.matvec(l, r, adjoint_a=adjoint_l)
    else:
        if adjoint_l:
            raise ValueError(f'cannot compute adjoint of tensor with shape {l.shape}')
        return tf.linalg.matvec(r, l, transpose_a=not adjoint_r)

def tr(lattice):
    """
    Output: trace of the lattice
    Input:
        lattice: assuming matrix valued sites
    """
    if isinstance(lattice, parts.HypercubeParts):
        return parts.HypercubeParts([tr(l) for l in lattice], subset=lattice.subset)
    else:
        return tf.linalg.trace(lattice)

def shift(lattice, direction, length, nd=4, batch_dims=0):
    """
    Output: circular shifted lattice along direction with length
    Input:
        lattice: tensor, shape: [batch ...] Dims... Site ...
            Dims from T to X, with X direction fastest
        direction: 0,1,2,... corresponding to x,y,z,...
        length: +/- denotes forward/backward direction corresponding to tf.roll with -/+ shifts
        nd: length of the dimension of the lattice
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
    """
    if direction<0:
        raise ValueError(f'direction less than 0: {direction}')
    axis = batch_dims+nd-direction-1
    if axis<0:
        raise ValueError(f'direction too large: {direction} with nd={nd}')
    if length==0:
        return lattice
    else:
        if isinstance(lattice, parts.HypercubeParts):
            shiftedsubset = lattice.subset
            dbit = 1<<direction
            def doshift(x, i):
                # direction 0, length 1
                # 0101 -> 1010
                # 2323    3232
                if (i&dbit)==0:
                    return tf.roll(x, shift=-length//2, axis=axis)
                else:
                    return tf.roll(x, shift=-(length-1)//2, axis=axis)
            if lattice.subset=='all':
                if len(lattice)!=(1<<nd):
                    raise ValueError(f'for {nd} dimension we expect {1<<nd} partitions, but got {len(lattice)}')
                v = [doshift(lattice[i], i) for i in range(len(lattice))]
                if length%2==1:
                    v = [v[i^dbit] for i in range(len(lattice))]
            elif lattice.subset=='even':
                if len(lattice)!=(1<<(nd-1)):
                    raise ValueError(f'for {nd} dimension we expect {1<<(nd-1)} partitions for even subset, but got {len(lattice)}')
                v = [doshift(lattice[i], even_indices[i]) for i in range(len(lattice))]
                if length%2==1:
                    v = [v[even_indices.index(i^dbit)] for i in odd_indices]
                    shiftedsubset = 'odd'
            elif lattice.subset=='odd':
                if len(lattice)!=(1<<(nd-1)):
                    raise ValueError(f'for {nd} dimension we expect {1<<(nd-1)} partitions for odd subset, but got {len(lattice)}')
                v = [doshift(lattice[i], odd_indices[i]) for i in range(len(lattice))]
                if length%2==1:
                    v = [v[odd_indices.index(i^dbit)] for i in even_indices]
                    shiftedsubset = 'even'
            else:
                raise ValueError(f'unsupported subset: {subset}')
            v = parts.HypercubeParts(v, subset=shiftedsubset)
        else:
            v = tf.roll(lattice, shift=-length, axis=axis)
        return v

def transport(lattice, gauge, direction, length, nd=4, batch_dims=0):
    """
    Output: circular transported lattice along direction with length
    Input:
        lattice: tensor, shape: [batch ...] Dims... Site ...
            Dims from T to X, with X direction fastest
        gauge: the gauge field, shape: dim [batch ...] Dims... Site ...
        direction: 0,1,2,... corresponding to x,y,z,...
        length: +/- denotes forward/backward direction corresponding to tf.roll with -/+ shifts
        nd: length of the dimension of the lattice
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
    """
    v = lattice
    if isinstance(gauge, parts.HypercubeParts) and isinstance(lattice, parts.HypercubeParts):
        U = parts.HypercubeParts([u[direction] for u in gauge], subset=gauge.subset)
    elif isinstance(gauge, parts.HypercubeParts) or isinstance(lattice, parts.HypercubeParts):
        raise ValueError('incompatible lattice and gauge')
    else:
        U = gauge[direction]
    if length>0:
        for i in range(length):
            v = shift(v, direction, 1, nd, batch_dims=batch_dims)
            v = mul(U, v, nd, batch_dims=batch_dims)
    elif length<0:
        for i in range(-length):
            v = mul(U, v, nd, batch_dims=batch_dims, adjoint_l=True)
            v = shift(v, direction, -1, nd, batch_dims=batch_dims)
    return v

def plaqField(gauge, nd, batch_dims=0):
    if isinstance(gauge, parts.HypercubeParts):
        U = [parts.HypercubeParts([u[mu] for u in gauge], subset=gauge.subset) for mu in range(nd)]
    else:
        U = [gauge[mu] for mu in range(nd)]
    ps = []
    for mu in range(1,nd):
        for nu in range(0,mu):
            Umunu = transport(U[nu], gauge, mu, 1, nd=nd, batch_dims=batch_dims)
            Unumu = transport(U[mu], gauge, nu, 1, nd=nd, batch_dims=batch_dims)
            ps.append(mul(Umunu, Unumu, nd, batch_dims, adjoint_r=True))
    return ps

def plaquette(gauge, nd, batch_dims=0):
    pf = [tr(p) for p in plaqField(gauge, nd, batch_dims)]
    if isinstance(pf[0], parts.HypercubeParts):
        nc = gauge[0].shape[-1]
        ps = [sum([tf.reduce_mean(p, axis=range(batch_dims,len(p.shape)))/nc for p in P])/len(P) for P in pf]
    else:
        nc = gauge.shape[-1]
        ps = [tf.reduce_mean(P, axis=tuple(range(batch_dims,len(P.shape))))/nc for P in pf]
    return ps

def setBC(gauge, batch_dims=0):
    """
    Output: new gauge with antiperiodic BC in T dir for fermions.
    Input: gauge, shape: dim [batch] T Z ... X C C
        dim: [0,1,2,3] ~ [X,Y,Z,T]
    Only works for batch_dims == 0 or 1
    """
    if isinstance(gauge, parts.HypercubeParts):
        if gauge.subset!='all':
            raise ValueError(f'unsupported subset {gauge.subset}')
        return parts.HypercubeParts(
            [gauge[i] if i&8==0 else setBC(gauge[i], batch_dims) for i in range(len(gauge))],
            subset=gauge.subset)
    shape = gauge.shape
    td = gauge.shape[0]-1
    if batch_dims==0:
        tb = gauge.shape[1]-1
        tbgauge = (-1.0) * gauge[td,tb]
        ix = [[td,tb]]
        g = tf.tensor_scatter_nd_update(gauge, ix, tf.expand_dims(tbgauge,0))
    elif batch_dims==1:
        bn = gauge.shape[1]
        tb = gauge.shape[2]-1
        tbgauge = (-1.0) * gauge[td,:,tb]
        ix = [[td,i,tb] for i in range(bn)]
        g = tf.tensor_scatter_nd_update(gauge, ix, tbgauge)
    else:
        raise ValueError(f'unsupported batch_dims: {batch_dims}')
    return g

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

    gcube = hypercube_partition(gauge)
    for i in range(len(gcube)):
        tf.print('cube',i)
        tf.print(gcube[i])

    gfull = combine_hypercube(gcube)
    tf.print('recombined')
    tf.print(gfull)

    tf.print('diff',g.norm2(gfull-gauge, range(len(gauge.shape))))
