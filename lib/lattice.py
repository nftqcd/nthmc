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

def get_even(lattice, batch_dims=0):
    if batch_dims==0:
        return lattice[0]
    elif batch_dims==1:
        return lattice[:,0]
    elif batch_dims==2:
        if tf.rank(lattice)>8:
            return tf.reshape(tf.reshape(lattice, tuple(lattice.shape[:3])+(-1,))[:,:,0], lattice.shape[:2]+lattice.shape[3:])
        else:
            return lattice[:,:,0]
    else:
        raise ValueError(f'unsupported batch_dims {batch_dims}')

def get_odd(lattice, batch_dims=0):
    if batch_dims==0:
        return lattice[1]
    elif batch_dims==1:
        return lattice[:,1]
    elif batch_dims==2:
        if tf.rank(lattice)>8:
            return tf.reshape(tf.reshape(lattice, tuple(lattice.shape[:3])+(-1,))[:,:,1], lattice.shape[:2]+lattice.shape[3:])
        else:
            return lattice[:,:,1]
    else:
        raise ValueError(f'unsupported batch_dims {batch_dims}')

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

def shift(lattice, direction, length, nd=4, isEO=True, subset='all', batch_dims=0, cond=None, part=None):
    """
    Output: circular shifted lattice along direction with length
    Input:
        lattice: tensor, shape: [batch ...] [EO] Dims... Site ...
            Dims from T to X, with X direction fastest
        direction: 0,1,2,... corresponding to x,y,z,...
        length: +/- denotes forward/backward direction corresponding to tf.roll with -/+ shifts
        nd: length of the dimension of the lattice
        isEO: whether the first dimension of lattice is EO
        subset: {'all','even','odd'}, the given lattice only contains the subset
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
        if isEO:
            if subset!='all':
                raise ValueError(f'incompatible subset {subset} with EO partitioned data')
        else:
            axis = axis-1
        if isEO and direction==0:
            batch_shape = lattice.shape[:batch_dims]
            latNoX = lattice.shape[batch_dims+1:batch_dims+nd]
            latX = lattice.shape[batch_dims+nd]
            site = lattice.shape[batch_dims+1+nd:]
            if batch_dims==0:
                part = evenodd_mask(latNoX)
            elif batch_dims==1:
                part = evenodd_mask(latNoX,lattice.shape[:1])
            elif batch_dims==2:
                part = evenodd_mask(latNoX,lattice.shape[:2])
            else:
                raise ValueError(f'unsupported batch_dims {batch_dims}')
            cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 2)
            elat = get_even(lattice, batch_dims)
            olat = get_odd(lattice, batch_dims)
            elat = shift(elat, direction, length, nd=nd, isEO=False, subset='even', batch_dims=batch_dims, cond=cond, part=part)
            olat = shift(olat, direction, length, nd=nd, isEO=False, subset='odd', batch_dims=batch_dims, cond=cond, part=part)
            if length%2==1:
                v = tf.stack([olat,elat], batch_dims)
            else:
                v = tf.stack([elat,olat], batch_dims)
        elif (subset=='even' or subset=='odd') and direction==0:
            latNoX = lattice.shape[batch_dims:batch_dims+nd-1]
            if part is None:
                if batch_dims==0:
                    part = evenodd_mask(latNoX)
                elif batch_dims==1:
                    part = evenodd_mask(latNoX,lattice.shape[:1])
                elif batch_dims==2:
                    part = evenodd_mask(latNoX,lattice.shape[:2])
                else:
                    raise ValueError(f'unsupported batch_dims {batch_dims}')
            if cond is None:
                cond = tf.dynamic_partition(tf.range(math.prod(part.shape)), tf.reshape(part, [-1]), 2)
            flat_x_site = (-1,) + tuple(lattice.shape[batch_dims+nd-1:])
            latEO = [tf.reshape(x, flat_x_site) for x in tf.dynamic_partition(lattice, part, 2)]
            if subset=='even':
                latEO = [tf.roll(latEO[0], shift=-length//2, axis=1),
                         tf.roll(latEO[1], shift=-(length-1)//2, axis=1)]
            else:
                latEO = [tf.roll(latEO[0], shift=-(length-1)//2, axis=1),
                         tf.roll(latEO[1], shift=-length//2, axis=1)]
            v = tf.reshape(tf.dynamic_stitch(cond, latEO), lattice.shape)
        else:
            v = tf.roll(lattice, shift=-length, axis=axis)
            if isEO and length%2==1:
                v = swap_evenodd(v, batch_dims)
        return v

def transport(lattice, gauge, direction, length, nd=4, isEO=True, subset='all', batch_dims=0):
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
        if (subset=='even' or subset=='odd') and not isEO:
            raise ValueError(f'incompatible subset {subset} without EO partitioned field.')
        v = lattice
        if batch_dims==0:
            U = gauge[direction]
        else:
            if isEO:
                U = tf.transpose(gauge, [batch_dims,batch_dims+1]+list(range(batch_dims)))[direction]
            else:
                U = tf.transpose(gauge, [batch_dims]+list(range(batch_dims)))[direction]
        if length>0:
            for i in range(length):
                v = shift(v, direction, 1, nd, isEO=isEO, subset=subset, batch_dims=batch_dims)
                if subset=='even':
                    subset = 'odd'
                    v = mul(U[1], v, nd, isEO, batch_dims)
                elif subset=='odd':
                    subset = 'even'
                    v = mul(U[0], v, nd, isEO, batch_dims)
                else:
                    v = mul(U, v, nd, isEO, batch_dims)
        else:
            for i in range(-length):
                if subset=='even':
                    v = mul(U[0], v, nd, isEO, batch_dims, adjoint_l=True)
                    subset = 'odd'
                elif subset=='odd':
                    v = mul(U[1], v, nd, isEO, batch_dims, adjoint_l=True)
                    subset = 'even'
                else:
                    v = mul(U, v, nd, isEO, batch_dims, adjoint_l=True)
                v = shift(v, direction, -1, nd, isEO=isEO, subset=subset, batch_dims=batch_dims)
        return v

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

def setBC(gauge, isEO=True, batch_dims=0):
    """
    Output: new gauge with antiperiodic BC in T dir for fermions.
    Input: gauge, shape: [batch] dim [EO] T Z ... X C C
        dim: [0,1,2,3] ~ [X,Y,Z,T]
    Only works for batch_dims == 0 or 1
    """
    shape = gauge.shape
    if len(shape)>8:
        gauge = tf.reshape(gauge, tuple(shape[:7])+(-1,))
    if batch_dims==0:
        td = gauge.shape[0]-1
        if isEO:
            tb = gauge.shape[2]-1
            tbgauge = (-1.0) * gauge[td,:,tb]
            ix = [[td,0,tb],[td,1,tb]]
            g = tf.tensor_scatter_nd_update(gauge, ix, tbgauge)
        else:
            tb = gauge.shape[1]-1
            tbgauge = (-1.0) * gauge[td,tb]
            ix = [[td,tb]]
            g = tf.tensor_scatter_nd_update(gauge, ix, tf.expand_dims(tbgauge,0))
    elif batch_dims==1:
        bn = gauge.shape[0]
        td = gauge.shape[1]-1
        if isEO:
            tb = gauge.shape[3]-1
            tbgauge = (-1.0) * gauge[:,td,:,tb]
            ix = [[[i,td,0,tb],[i,td,1,tb]] for i in range(bn)]
            g = tf.tensor_scatter_nd_update(gauge, ix, tbgauge)
        else:
            tb = gauge.shape[2]-1
            tbgauge = (-1.0) * gauge[:,td,tb]
            ix = [[i,td,tb] for i in range(bn)]
            g = tf.tensor_scatter_nd_update(gauge, ix, tbgauge)
    if len(shape)>8:
        g = tf.reshape(g, shape)
    return g

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
