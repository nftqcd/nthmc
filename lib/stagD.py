## Staggered Dslash in 4D

import tensorflow as tf
import lattice as l

def phase(gauge, isEO=True, batch_dims=0):
    """
    Input: gauge, shape: [batch] dim [EO] T Z Y X ... site
    phase(mu,N) = (-)^(sum_i N_i | i<mu) with N = {t,x,y,z}
    """
    if isEO:
        # take the shortcut, combine EO first
        g = l.combine_evenodd(gauge, batch_dims=batch_dims+1)    # +1 for gauge dim
        g = phase(g, isEO=False, batch_dims=batch_dims)
        pg = l.evenodd_partition(g, batch_dims=batch_dims+1)    # +1 for gauge dim
    else:
        shape = gauge.shape
        dims = shape[batch_dims+1:batch_dims+5]    # 4D without EO

        # avoid indexing into rank>8
        gauge = tf.reshape(gauge, tuple(shape[:-2])+(-1,))
        ld = batch_dims * (1,)    # excluding the 'dim' axis

        px = tf.reshape(tf.tile(tf.constant([1,-1],dtype=tf.complex128), [dims[0]//2]), ld+(dims[0],1,1,1,1))
        py = tf.reshape(tf.tile(tf.constant([[1,-1],[-1,1]],dtype=tf.complex128), [dims[0]//2,dims[3]//2]), ld+(dims[0],1,1,dims[3],1))
        pz = tf.reshape(tf.tile(tf.constant([[[1,-1],[-1,1]],[[-1,1],[1,-1]]],dtype=tf.complex128), [dims[0]//2,dims[2]//2,dims[3]//2]), ld+(dims[0],1,dims[2],dims[3],1))
        if batch_dims==0:
            gx = gauge[0]
            gy = gauge[1]
            gz = gauge[2]
            gt = gauge[3]
        elif batch_dims==1:
            gx = gauge[:,0]
            gy = gauge[:,1]
            gz = gauge[:,2]
            gt = gauge[:,3]
        else:
            raise ValueError(f'unsupported batch_dims {batch_dims}')
        pg = tf.reshape(tf.stack([px*gx, py*gy, pz*gz, gt], axis=batch_dims), shape)
    return pg

def D2(gauge, x, subset='all', batch_dims=0):
    r = 0
    for mu in range(4):
        f = l.transport(x, gauge, mu, 1, 4, subset=subset, isEO=True, batch_dims=batch_dims)
        b = l.transport(x, gauge, mu, -1, 4, subset=subset, isEO=True, batch_dims=batch_dims)
        r += f-b
    return r

def D(gauge, x, m, subset='all', batch_dims=0):
    return m*x + 0.5*D2(gauge, x, subset=subset, batch_dims=batch_dims)

def Ddag(gauge, x, m, subset='all', batch_dims=0):
    return m*x - 0.5*D2(gauge, x, subset=subset, batch_dims=batch_dims)

def D2ee(gauge, x, m2, batch_dims=0):
    return 4.0*m2*x - D2(gauge, D2(gauge, x, subset='even', batch_dims=batch_dims), subset='odd', batch_dims=batch_dims)

def eoReconstruct(gauge, r, b, m, batch_dims=0):
    """
    Output: full field
    Input:
        r: even
        b: odd
    """
    ro = (b - 0.5*D2(gauge, r, subset='even', batch_dims=batch_dims)) / m
    return tf.stack([r,ro], batch_dims)

if __name__=='__main__':
    import sys
    import fieldio
    import group

    gconf,lat = fieldio.readLattice(sys.argv[1])
    nd = len(lat)
    print(gconf.shape)
    print(lat)

    print('Plaq:', l.plaquette(gconf, nd, False))

    gaugeEO = tf.stack([l.evenodd_partition(gconf[mu], nd) for mu in range(nd)], 0)
    gaugeEO = l.setBC(gaugeEO, isEO=True, batch_dims=0)
    gaugeEO = phase(gaugeEO, isEO=True, batch_dims=0)
    print(gaugeEO.shape)

    v1 = tf.zeros(lat[::-1]+[3], dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    print(f'v {v1.shape} {group.norm2(v1, range(tf.rank(v1)))}')
    v1 = l.evenodd_partition(v1)

    m = 0.1
    m2 = m*m

    print('D:')
    v2 = v1
    for i in range(3):
        v2 = D(gaugeEO, v2, m)
        print(f'Dv2 {v2.shape} {group.norm2(v2, range(tf.rank(v2)))}')
    print('Ddag:')
    v2 = v1
    for i in range(3):
        v2 = Ddag(gaugeEO, v2, m)
        print(f'Dv2 {v2.shape} {group.norm2(v2, range(tf.rank(v2)))}')

    print('D2ee:')
    v1e = l.get_even(v1)
    print(f've {v1e.shape} {group.norm2(v1e, range(tf.rank(v1e)))}')
    v2e = v1e
    for i in range(3):
        v2e = D2ee(gaugeEO, v2e, m2)
        print(f'D2ee {v2e.shape} {group.norm2(v2e, range(tf.rank(v2e)))}')

    def time(s,f,n=1,niter=1):
        for i in range(n):
            t0 = tf.timestamp()
            for iter in range(niter):
                ret = f()
            dt = tf.timestamp()-t0
            if n>1:
                if niter>1:
                    print(f'{s} run {i}: Time {dt} sec / {niter} = {dt/niter} sec', flush=True)
                else:
                    print(f'{s} run {i}: Time {dt} sec', flush=True)
            else:
                if niter>1:
                    print(f'{s}: Time {dt} sec / {niter} = {dt/niter} sec', flush=True)
                else:
                    print(f'{s}: Time {dt} sec', flush=True)
        return ret

    niter = 100
    v2e = v1e
    t0 = tf.timestamp()
    for i in range(niter):
        v2e = D2ee(gaugeEO, v2e, m2)
    dt = tf.timestamp()-t0
    print(f'time {dt} / {niter} = {dt/niter}')
    print(f'D2ee {v2e.shape} {group.norm2(v2e, range(tf.rank(v2e)))}')

    f_fun = tf.function(D2ee)
    f_jit = tf.function(D2ee,jit_compile=True)

    D2ee_fun = time('fun get concrete',lambda:f_fun.get_concrete_function(gaugeEO, v2e, m2))
    D2ee_jit = time('jit get concrete',lambda:f_jit.get_concrete_function(gaugeEO, v2e, m2))

    def run_fun(f):
        global v2e
        v2e = f(gaugeEO,v2e,m2)
        return v2e

    print('Benchmark eager')
    v2e = v1e
    v2e = time('D2ee eager',lambda:run_fun(D2ee),n=3)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')

    v2e = v1e
    v2e = time('D2ee eager',lambda:run_fun(D2ee),n=3,niter=100)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')

    print('Benchmark function')
    v2e = v1e
    v2e = time('D2ee fun',lambda:run_fun(D2ee_fun),n=3)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')

    v2e = v1e
    v2e = time('D2ee fun',lambda:run_fun(D2ee_fun),n=3,niter=100)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')

    print('Benchmark jit_compile')
    v2e = v1e
    v2e = time('D2ee jit',lambda:run_fun(D2ee_jit),n=3)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')

    v2e = v1e
    v2e = time('D2ee jit',lambda:run_fun(D2ee_jit),n=3,niter=100)
    print(f'{group.norm2(v2e, range(tf.rank(v2e)))}')
