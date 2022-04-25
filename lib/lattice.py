import math
import tensorflow as tf

def evenodd_mask(dim, batch_shape=()):
    """
    Output: tensor with elements of 0 (even) and 1 (odd), shape batch_shape+dim
    Input:
        dim: the lattice shape for computing even and odd
        batch_shape: shape of the batch
    """
    nd = len(dim)
    if nd==1:
        eo = tf.constant([0,1])
    elif nd==2:
        eo = tf.constant([[0,1],[1,0]])
    elif nd==3:
        eo = tf.constant([[[0,1],[1,0]],[[1,0],[0,1]]])
    elif nd==4:
        eo = tf.constant([[[[0,1],[1,0]],[[1,0],[0,1]]],[[[1,0],[0,1]],[[0,1],[1,0]]]])
    else:
        raise ValueError(f'unsupported dim = {dim}')
    mask = tf.slice(tf.tile(eo, [(x+1)//2 for x in dim]), tf.zeros(nd, dtype=tf.int32), dim)    # takes care of cases where dim[i]%2!=0
    nb = len(batch_shape)
    if nb>0:
        mask = tf.reshape(mask, (1,)*nb+mask.shape)
        mask = tf.tile(mask, batch_shape+(1,)*len(dim))
    return mask

def evenodd_partition(lattice, nd=4, batch_dims=0):
    """
    Output: even-odd partitioned lattice, shape: batch_shape EO T .. X//2 Site ...
    Input:
        lattice: tensor, shape: batch_shape Dims... Site ...
            Dims from T to X, with X direction fastest
            Assuming all dims are even numbers.
        nd: length of the dimension of the lattice
        batch_dims: number of dimensions in the lattice.shape belong to batches
    """
    batch_shape = lattice.shape[:batch_dims]
    lat_shape = lattice.shape[batch_dims:batch_dims+nd]
    part = evenodd_mask(lat_shape,batch_shape)
    eo = tf.dynamic_partition(lattice, part, 2)
    return tf.stack(
        [tf.reshape(p, batch_shape+lat_shape[:-1]+(lat_shape[-1]//2,)+lattice.shape[batch_dims+nd:])
            for p in eo],
        batch_dims)

def combine_evenodd(lattice, nd=4, batch_dims=0):
    """
    Output: lattice, shape batch_shape T .. X Site ..., combined from the input even-odd
    Input:
        lattice: tensor, shape: batch_shape EO T .. X//2 Site ...
        nd: length of the dimension of the lattice
        batch_dims: number of dimensions in the lattice.shape belong to batches
    """
    batch_shape = lattice.shape[:batch_dims]
    lat_shape = lattice.shape[batch_dims+1:batch_dims+nd] + (2*lattice.shape[batch_dims+nd],)
    site_shape = lattice.shape[batch_dims+1+nd:]
    part = evenodd_mask(lat_shape,batch_shape)
    cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 2)
    eo = tf.transpose(lattice, (batch_dims,)+tuple(range(batch_dims))+tuple(range(batch_dims+1,len(lattice.shape))))
    eoflat = tf.reshape(eo, (2,-1)+tuple(site_shape))
    return tf.reshape(tf.dynamic_stitch(cond, eoflat), batch_shape+lat_shape+site_shape)

def swap_evenodd(lattice, batch_dims=0):
    """
    Output: even-odd partitioned lattice, shape [batch ...] EO T .. X//2 Site ...
        where even and odd swapped place from input
    Input:
        lattice: tensor, shape: [batch ...] EO Dims... Site ...
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
    """
    # We are pushing the limit of Tensorflow
    # Direct indexing would result in
    #     tensorflow.python.framework.errors_impl.UnimplementedError: Unhandled input dimensions 9 [Op:StridedSlice] name: strided_slice/
    # Using reverse would result in
    #     tensorflow.python.framework.errors_impl.UnimplementedError: reverse is not implemented for tensors of rank > 8. [Op:ReverseV2]
    if tf.rank(lattice) > 8:
        r = tf.reshape(lattice, tuple(lattice.shape[:batch_dims])+(2,-1))
        r = tf.reverse(r, [batch_dims])
        r = tf.reshape(r, lattice.shape)
    else:
        r = tf.reverse(lattice, [batch_dims])
    return r

def mul(l, r, nd, isEO, batch_dims=0, adjoint_l=False, adjoint_r=False):
    """
    Output: lattice of site-wide matrix-matrix, matrix-vector, or vector-matrix multiplications
    Input:
        l,r: left and right lattice fields
        nd: length of the dimension of the lattice
        isEO: whether the first dimension of lattice is EO
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
        adjoint_l,adjoint_r: take the adjoint of l/r before multiplication if it is a matrix field
    """
    neo = 1 if isEO else 0
    nsl = len(l.shape)-nd-neo-batch_dims
    nsr = len(r.shape)-nd-neo-batch_dims
    if nsl<1 or nsl>2:
        raise ValueError(f'unrecognized l.shape {l.shape}')
    elif nsr<1 or nsr>2:
        raise ValueError(f'unrecognized r.shape {r.shape}')
    elif nsr==1 and nsr==1:
        raise ValueError(f'cannot group multiply two shapes {l.shape} {r.shape}')
    elif nsr==2 and nsr==2:
        return tf.linalg.matmul(l, r, adjoint_a=adjoint_l, adjoint_b=adjoint_r)
    elif nsr==2 and nsr==1:
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
    return tf.linalg.trace(lattice)

def shift(lattice, direction, length, nd=4, isEO=True, batch_dims=0):
    """
    Output: circular shifted lattice along direction with length
    Input:
        lattice: tensor, shape: [batch ...] [EO] Dims... Site ...
            Dims from T to X, with X direction fastest
        direction: 0,1,2,... corresponding to x,y,z,...
        length: +/- denotes forward/backward direction corresponding to tf.roll with -/+ shifts
        nd: length of the dimension of the lattice
        isEO: whether the first dimension of lattice is EO
        batch_dims: the first batch_dims dims in lattice.shape belon to batch
    """
    if direction<0:
        raise ValueError(f'direction less than 0: {direction}')
    axis = batch_dims+nd-direction
    if axis<0:
        raise ValueError(f'direction too large: {direction} with nd={nd}')
    if length==0:
        return lattice
    else:
        if not isEO:
            axis = axis-1
        if isEO and direction==0:
            batch_shape = lattice.shape[:batch_dims]
            latNoX = lattice.shape[batch_dims+1:batch_dims+nd]
            latX = lattice.shape[batch_dims+nd]
            site = lattice.shape[batch_dims+1+nd:]
            # to avoid Tensorflow indexing error
            # tensorflow.python.framework.errors_impl.UnimplementedError: Unhandled input dimensions 9 [Op:StridedSlice] name: strided_slice/
            latticeR = tf.reshape(lattice, tuple(batch_shape)+(2,-1))
            if batch_dims==0:
                part = evenodd_mask(latNoX)
                elat = latticeR[0]
                olat = latticeR[1]
            elif batch_dims==1:
                part = evenodd_mask(latNoX,lattice.shape[:1])
                elat = latticeR[:,0]
                olat = latticeR[:,1]
            elif batch_dims==2:
                part = evenodd_mask(latNoX,lattice.shape[:2])
                elat = latticeR[:,:,0]
                olat = latticeR[:,:,1]
            else:
                raise ValueError(f'unsupported batch_dims {batch_dims}')
            elat = tf.reshape(elat, batch_shape+lattice.shape[batch_dims+1:])
            olat = tf.reshape(olat, batch_shape+lattice.shape[batch_dims+1:])
            cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 2)
            # even subset splitted
            elatEO = [tf.reshape(x, (-1,latX)+tuple(site)) for x in tf.dynamic_partition(elat, part, 2)]
            elatEO = [tf.roll(elatEO[0], shift=-length//2, axis=1),
                      tf.roll(elatEO[1], shift=-(length-1)//2, axis=1)]
            elat = tf.reshape(tf.dynamic_stitch(cond, elatEO), batch_shape+lattice.shape[batch_dims+1:])
            # odd subset splitted
            olatEO = [tf.reshape(x, (-1,latX)+tuple(site)) for x in tf.dynamic_partition(olat, part, 2)]
            olatEO = [tf.roll(olatEO[0], shift=-(length-1)//2, axis=1),
                      tf.roll(olatEO[1], shift=-length//2, axis=1)]
            olat = tf.reshape(tf.dynamic_stitch(cond, olatEO), batch_shape+lattice.shape[batch_dims+1:])
            if length%2==1:
                v = tf.stack([olat,elat], batch_dims)
            else:
                v = tf.stack([elat,olat], batch_dims)
        else:
            v = tf.roll(lattice, shift=-length, axis=axis)
            if isEO and length%2==1:
                v = swap_evenodd(v, batch_dims)
        return v

def transport(lattice, gauge, direction, length, nd, isEO, batch_dims=0):
    """
    Output: circular shifted lattice along direction with length
    Input:
        lattice: tensor, shape: [batch ...] Dims... Site ...
            Dims from T to X, with X direction fastest
        direction: 0,1,2,... corresponding to x,y,z,...
        length: +/- denotes forward/backward direction corresponding to tf.roll with -/+ shifts
        nd: length of the dimension of the lattice
        isEO: whether the first dimension of lattice is EO
    """
    if length==0:
        return lattice
    else:
        v = lattice
        if batch_dims==0:
            U = gauge[direction]
        else:
            U = tf.transpose(gauge, [batch_dims]+list(range(batch_dims)))[direction]
        if length>0:
            for i in range(length):
                v = shift(v, direction, 1, nd, isEO, batch_dims)
                v = mul(U, v, nd, isEO, batch_dims)
        else:
            for i in range(-length):
                v = mul(U, v, nd, isEO, batch_dims, adjoint_l=True)
                v = shift(v, direction, -1, nd, isEO, batch_dims)
        return v

## JXY WAS HERE
## transpose twice one in plaqField, one in transport
## average plaquette is wrong for spatial plaquette

def plaqField(gauge, nd, isEO, batch_dims=0):
    ps = []
    if batch_dims==0:
        U = gauge
    else:
        U = tf.transpose(gauge, [batch_dims]+list(range(batch_dims)))
    for mu in range(1,nd):
        for nu in range(0,mu):
            Umunu = transport(U[nu], gauge, mu, 1, nd, isEO, batch_dims)
            Unumu = transport(U[mu], gauge, nu, 1, nd, isEO, batch_dims)
            ps.append(mul(Umunu, Unumu, nd, isEO, batch_dims, adjoint_r=True))
    return ps

def plaquette(gauge, nd, isEO, batch_dims=0):
    nc = gauge.shape[-1]
    ps = [tf.reduce_mean(tr(P), axis=range(batch_dims,len(P.shape)-2))/nc for P in plaqField(gauge, nd, isEO, batch_dims)]
    return ps

if __name__=='__main__':
    import sys
    import fieldio

    gconf,lat = fieldio.readLattice(sys.argv[1])
    nd = len(lat)
    print(gconf.shape)
    print(lat)

    print('Plaq:', plaquette(gconf, nd, False))

    gaugeEO = tf.stack([evenodd_partition(gconf[mu], nd) for mu in range(nd)], 0)
    print(gaugeEO.shape)
    for y in range(lat[1]//2):
        for x in range(lat[0]//2):
            if y%2==0:
                print(f'[E,{x},{y},0,0],0,[0,0] {gaugeEO[0,0,0,0,y,x,0,0].numpy()}')
                print(f'[O,{x},{y},0,0],0,[0,0] {gaugeEO[0,1,0,0,y,x,0,0].numpy()}')
            else:
                print(f'[O,{x},{y},0,0],0,[0,0] {gaugeEO[0,1,0,0,y,x,0,0].numpy()}')
                print(f'[E,{x},{y},0,0],0,[0,0] {gaugeEO[0,0,0,0,y,x,0,0].numpy()}')

    print('Plaq:', plaquette(gaugeEO, nd, True))

    gaugeS = tf.stack([combine_evenodd(gaugeEO[mu], nd) for mu in range(nd)], 0)
    print(gaugeS.shape)
    for y in range(lat[1]):
        for x in range(lat[0]):
            print(f'[{x},{y},0,0],0,[0,0] {gaugeS[0,0,0,y,x,0,0].numpy()}')

    print('Plaq:', plaquette(gaugeS, nd, False))
