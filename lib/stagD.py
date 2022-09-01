## Staggered Dslash in 4D

import tensorflow as tf
from lattice import SubSetAll, SubSetEven, SubSetOdd, Lattice, combine_subsets
from gauge import Gauge

def phase(gauge):
    """
    Input: Gauge
    phase(mu,N) = (-)^(sum_i N_i | i<mu) with N = {t,x,y,z}
    """
    if isinstance(gauge, Gauge) and len(gauge)==4:
        return Gauge([t.wrap(phase_dir(i, t.unwrap())) for i,t in enumerate(gauge)])
    else:
        raise ValueError(f'phase only works with 4D gauge, but got {gauge}')

def phase_dir(dir, lattice, batch_dim=None):
    if dir==3:  # T dir
        return lattice
    if isinstance(lattice, Lattice):
        corners = lattice.subset.to_corner_indices()
        if len(corners)>1:
            pl = phase_dir(dir, lattice.unwrap(), lattice.batch_dim)
            return lattice.wrap(pl)
        else:
            # index bits: TZYX
            i = corners[0]
            if (dir==0 and i&8==0) or (dir==1 and ((i&8)>>3)^(i&1)==0) or (dir==2 and ((i&8)>>3)^(i&1)^((i&2)>>1)==0):
                return lattice
            else:
                return -lattice
    elif isinstance(lattice, list):
        return [phase_dir(dir, l, batch_dim=batch_dim) for l in lattice]
    elif isinstance(lattice, tuple):
        return tuple([phase_dir(dir, l, batch_dim=batch_dim) for l in lattice])
    else:
        if batch_dim is None:
            raise ValueError(f'unknown batch_dim for dir {dir} with lattice {lattice}')
        shape = lattice.shape
        if batch_dim==0:
            bd = (1,)
            dims = shape[1:5]    # 4D
            site = (len(shape)-5)*(1,)
        else:
            bd = ()
            dims = shape[0:4]
            site = (len(shape)-4)*(1,)
        dt = lattice.dtype
        if dir==0:
            p = tf.reshape(tf.tile(tf.constant([1,-1],dtype=dt), [dims[0]//2]), bd+(dims[0],1,1,1)+site)
        elif dir==1:
            p = tf.reshape(tf.tile(tf.constant([[1,-1],[-1,1]],dtype=dt), [dims[0]//2,dims[3]//2]), bd+(dims[0],1,1,dims[3])+site)
        else:  # dir==2
            p = tf.reshape(tf.tile(tf.constant([[[1,-1],[-1,1]],[[-1,1],[1,-1]]],dtype=dt), [dims[0]//2,dims[2]//2,dims[3]//2]), bd+(dims[0],1,dims[2],dims[3])+site)
        return p*lattice

def D2(gauge, x):
    if x.subset==SubSetAll:
        ss = SubSetAll
    elif x.subset==SubSetEven:
        ss = SubSetOdd
    elif x.subset==SubSetOdd:
        ss = SubSetEven
    else:
        raise ValueError(f'unsupported subset {x.subset}')
    r = 0
    for mu in range(4):
        f = gauge[mu](x)
        b = gauge[mu].adjoint()(x)
        r += f-b
    return r

def D(gauge, x, m, s=0.5):
    D2x = D2(gauge, x)
    return m*x + s*D2x

def Ddag(gauge, x, m):
    return D(gauge, x, m, s=-0.5)

def D2ee(gauge, x, m2):
    Dx = D2(gauge, x)
    DDx = D2(gauge, Dx)
    return 4.0*m2*x - DDx

def eoReconstruct(gauge, r, b, m):
    """
    Output: full field
    Input:
        r: even
        b: odd
    """
    D2r = D2(gauge, r)
    ro = (1.0/m) * (b - 0.5*D2r)
    return combine_subsets(r, ro)

if __name__=='__main__':
    import sys
    import gauge
    from lattice import norm2, Lattice, SubSetEven

    gconf = gauge.readGauge(sys.argv[1])
    nd = len(gconf)
    print('Plaq:')
    for p in gauge.plaquette(gconf):
        print(p)

    gaugeEO = gconf.hypercube_partition()
    print(gaugeEO[0].lattice[0].unwrap().shape)
    print('Plaq from EO:')
    for p in gauge.plaquette(gaugeEO):
        print(p)

    gaugeEO = gauge.setBC(gaugeEO)
    gaugeEO = phase(gaugeEO)

    v1 = tf.zeros(gconf[0].lattice.unwrap().shape[:nd]+(3,), dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,2,0]], [0.5])
    print(f'v {norm2(v1)}')
    v1 = Lattice(v1).hypercube_partition()
    print(f'v {norm2(v1)}')

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
    v1e = v1.get_subset(SubSetEven)
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
        g = gaugeEO.from_tensors(gt)
        v = v2e.from_tensors(vt)
        y = D2ee(g, v, m)
        return y.to_tensors()

    print(f've {norm2(v1e)}')
    v2e = v1e
    for i in range(3):
        v2e = v2e.from_tensors(D2ee_eager(gaugeEO.to_tensors(), v2e.to_tensors(), m2))
        print(f'D2ee_eager {norm2(v2e)}')

    f_fun = tf.function(D2ee_eager)
    f_jit = tf.function(D2ee_eager,jit_compile=True)

    D2ee_fun = time('fun get concrete',lambda:f_fun.get_concrete_function(gaugeEO.to_tensors(), v2e.to_tensors(), m2))
    D2ee_jit = time('jit get concrete',lambda:f_jit.get_concrete_function(gaugeEO.to_tensors(), v2e.to_tensors(), m2))

    def run_fun(f):
        global v2e
        v2e = v2e.from_tensors(f(gaugeEO.to_tensors(), v2e.to_tensors(), m2))
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
