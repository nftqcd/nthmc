## Staggered Dslash in 4D

import tensorflow as tf
import lattice as l
import parts

def phase(gauge, batch_dims=0):
    """
    Input: gauge, shape: dim [batch] T Z Y X ... site
    phase(mu,N) = (-)^(sum_i N_i | i<mu) with N = {t,x,y,z}
    """
    if isinstance(gauge, parts.HypercubeParts):
        sh = (4,)+(len(gauge[0].shape)-1)*(1,)
        def ph(i):
            # index bits: TZYX
            px = 1 if i&8==0 else -1
            py = 1 if ((i&8)>>3)^(i&1)==0 else -1
            pz = 1 if ((i&8)>>3)^(i&1)^((i&2)>>1)==0 else -1
            return tf.reshape(tf.constant([px,py,pz,1], dtype=gauge[0].dtype), sh)
        pg = parts.HypercubeParts([ph(i)*gauge[i] for i in range(len(gauge))], subset=gauge.subset)
    else:
        shape = gauge.shape
        dims = shape[batch_dims+1:batch_dims+5]    # 4D

        ld = batch_dims * (1,)    # excluding the 'dim' axis
        site = (len(shape)-batch_dims-5) * (1,)

        px = tf.reshape(tf.tile(tf.constant([1,-1],dtype=tf.complex128), [dims[0]//2]), ld+(dims[0],1,1,1)+site)
        py = tf.reshape(tf.tile(tf.constant([[1,-1],[-1,1]],dtype=tf.complex128), [dims[0]//2,dims[3]//2]), ld+(dims[0],1,1,dims[3])+site)
        pz = tf.reshape(tf.tile(tf.constant([[[1,-1],[-1,1]],[[-1,1],[1,-1]]],dtype=tf.complex128), [dims[0]//2,dims[2]//2,dims[3]//2]), ld+(dims[0],1,dims[2],dims[3])+site)
        gx = gauge[0]
        gy = gauge[1]
        gz = gauge[2]
        gt = gauge[3]
        pg = tf.stack([px*gx, py*gy, pz*gz, gt], axis=0)
    return pg

def D2(gauge, x, batch_dims=0):
    if isinstance(gauge, parts.HypercubeParts):
        if x.subset=='all':
            ss = 'all'
        elif x.subset=='even':
            ss = 'odd'
        elif x.subset=='odd':
            ss = 'even'
        else:
            raise ValueError(f'unsupported subset {subset}')
        r = parts.HypercubeParts(len(x)*[0], subset=ss)
    else:
        r = 0
    for mu in range(4):
        f = l.transport(x, gauge, mu, 1, 4, batch_dims=batch_dims)
        b = l.transport(x, gauge, mu, -1, 4, batch_dims=batch_dims)
        r += f-b
    return r

def D(gauge, x, m, batch_dims=0, s=0.5):
    D2x = D2(gauge, x, batch_dims=batch_dims)
    return m*x + s*D2x

def Ddag(gauge, x, m, batch_dims=0):
    return D(gauge, x, m, batch_dims=batch_dims, s=-0.5)

def D2ee(gauge, x, m2, batch_dims=0):
    Dx = D2(gauge, x, batch_dims=batch_dims)
    DDx = D2(gauge, Dx, batch_dims=batch_dims)
    return 4.0*m2*x - DDx

def eoReconstruct(gauge, r, b, m, batch_dims=0):
    """
    Output: full field
    Input:
        r: even
        b: odd
    """
    D2r = D2(gauge, r, batch_dims=batch_dims)
    ro = (1.0/m) * (b - 0.5*D2r)
    return l.mix_evenodd(r, ro)

if __name__=='__main__':
    import sys
    import fieldio
    import group

    gconf,lat = fieldio.readLattice(sys.argv[1])
    nd = len(lat)
    print(gconf.shape)
    print(lat)

    print('Plaq:')
    for p in l.plaquette(gconf, nd):
        print(p)

    gaugeEO = l.hypercube_partition(gconf, nd, batch_dims=1)    # Count dim axis as batch
    print(gaugeEO[0].shape)
    print('Plaq from EO:')
    for p in l.plaquette(gaugeEO, nd):
        print(p)

    gaugeEO = l.setBC(gaugeEO, batch_dims=0)
    gaugeEO = phase(gaugeEO, batch_dims=0)

    def norm2(v):
        if isinstance(v, parts.HypercubeParts):
            return group.norm2(v, range(tf.rank(v[0])), allreduce=True)
        else:
            return group.norm2(v, range(tf.rank(v)))

    v1 = tf.zeros(lat[::-1]+[3], dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,2,0]], [0.5])
    print(f'v {norm2(v1)}')
    v1 = l.hypercube_partition(v1)

    m = 0.1
    m2 = tf.constant(m*m, dtype=tf.complex128)

    print('D:')
    v2 = v1
    for i in range(3):
        v2 = D(gaugeEO, v2, m)
        print(f'Dv2 {norm2(v2)}')
    print('Ddag:')
    v2 = v1
    for i in range(3):
        v2 = Ddag(gaugeEO, v2, m)
        print(f'Dv2 {norm2(v2)}')

    print('D2ee:')
    v1e = l.get_even(v1)
    print(f've {norm2(v1e)}')
    v2e = v1e
    for i in range(3):
        v2e = D2ee(gaugeEO, v2e, m2)
        print(f'D2ee {norm2(v2e)}')

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
    print(f'D2ee {norm2(v2e)}')

    # NOTE: tf.function hates custom objects.
    def D2ee_eager(gt, vt, m):
        g = parts.HypercubeParts(gt,subset='all')
        v = parts.HypercubeParts(vt,subset='even')
        y = D2ee(g, v, m)
        return y.parts
    f_fun = tf.function(D2ee_eager)
    f_jit = tf.function(D2ee_eager,jit_compile=True)

    D2ee_fun = time('fun get concrete',lambda:f_fun.get_concrete_function(gaugeEO.parts, v2e.parts, m2))
    D2ee_jit = time('jit get concrete',lambda:f_jit.get_concrete_function(gaugeEO.parts, v2e.parts, m2))

    def run_fun(f):
        global v2e
        v2e = parts.HypercubeParts(f(gaugeEO.parts,v2e.parts,m2),subset='even')
        return v2e

    print('Benchmark eager')
    v2e = v1e
    v2e = time('D2ee eager',lambda:run_fun(D2ee_eager),n=3)
    print(f'{norm2(v2e)}')

    v2e = v1e
    v2e = time('D2ee eager',lambda:run_fun(D2ee_eager),n=3,niter=100)
    print(f'{norm2(v2e)}')

    print('Benchmark function')
    v2e = v1e
    v2e = time('D2ee fun',lambda:run_fun(D2ee_fun),n=3)
    print(f'{norm2(v2e)}')

    v2e = v1e
    v2e = time('D2ee fun',lambda:run_fun(D2ee_fun),n=3,niter=100)
    print(f'{norm2(v2e)}')

    print('Benchmark jit_compile')
    v2e = v1e
    v2e = time('D2ee jit',lambda:run_fun(D2ee_jit),n=3)
    print(f'{norm2(v2e)}')

    v2e = v1e
    v2e = time('D2ee jit',lambda:run_fun(D2ee_jit),n=3,niter=100)
    print(f'{norm2(v2e)}')
