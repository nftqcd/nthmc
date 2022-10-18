import tensorflow as tf
from . import gauge as g
from . import stagD as s
from .lattice import SubSetEven, SubSetOdd, Lattice, from_hypercube_parts, norm2, redot

def cg_iter(A,x,p,r,r2,r2o):
    beta = p.typecast(r2/r2o)
    # print(f'beta: {beta}')
    p = r + beta*p
    Ap = A(p)
    pAp = redot(p, Ap)
    alpha = p.typecast(r2/pAp)
    x += alpha*p
    r -= alpha*Ap
    return x,p,r,norm2(r),r2

def cg(A, x, b, r2req, maxits):
    b2 = norm2(b)
    # print(f'input norm2: {b2}')
    Ap = A(x)
    r = b - Ap
    p = r.zeros()
    r2 = norm2(r)
    # print(f'r2: {r2}')
    r2stop = r2req * b2
    r2o = tf.constant(1.0, dtype=tf.float64)
    itn = tf.constant(0, dtype=tf.int64)
    # print(f'CG {itn}  {r2/b2}')
    if isinstance(x, Lattice):
        # NOTE: tf.function cannot handle wrapper class objects in python while loop.
        def cond(itn,xp,pp,rp,r2,r2o):
            return itn<maxits and r2>r2stop
        def body(itn,xp,pp,rp,r2,r2o):
            itn += 1
            x_ = x.from_tensors(xp)
            p_ = p.from_tensors(pp)
            r_ = r.from_tensors(rp)
            x_,p_,r_,r2,r2o = cg_iter(A,x_,p_,r_,r2,r2o)
            return itn,x_.to_tensors(),p_.to_tensors(),r_.to_tensors(),r2,r2o
        itn,xp,pp,rp,r2,r2o = tf.while_loop(cond, body, [itn,x.to_tensors(),p.to_tensors(),r.to_tensors(),r2,r2o])
        x = x.from_tensors(xp)
    else:
        def cond(itn,x,p,r,r2,r2o):
            return itn<maxits and r2>r2stop
        def body(itn,x,p,r,r2,r2o):
            itn += 1
            x,p,r,r2,r2o = cg_iter(A,x,p,r,r2,r2o)
            return itn,x,p,r,r2,r2o
        itn,x,p,r,r2,r2o = tf.while_loop(cond, body, [itn,x,p,r,r2,r2o])
    return x,itn

@tf.function(jit_compile=True)
def TFsolveEE(gaugeT, rT, xT, m, r2req, maxits):
    gauge = g.from_hypercube_parts(gaugeT)
    r = from_hypercube_parts(rT, subset=SubSetEven)
    x = from_hypercube_parts(xT, subset=SubSetEven)
    m2 = m*m
    def op(v):
        return s.D2ee(gauge, v, m2)
    result,iter = cg(op, r, x, r2req, maxits)
    flops = (4*4*72+60)*x.full_volume()/2*tf.cast(iter,dtype=tf.float64)
    return result.to_tensors(),iter,flops

def solveEE(gauge, r, x, m, r2req, maxits):
    rT,iter,flops = TFsolveEE(gauge.to_tensors(), r.to_tensors(), x.to_tensors(), tf.constant(m, dtype=tf.complex128), r2req, tf.constant(maxits,dtype=tf.int64))
    r = r.from_tensors(rT)
    return r,iter,flops

def solve(gauge, x, b, m, r2req, maxits):
    t0 = tf.timestamp()
    c = b - s.D(gauge, x, m)
    b2 = norm2(b)
    r2 = norm2(c)
    r2stop = r2req * b2
    # print(f'stagSolve b2: {b2}  r2: {r2/b2}  r2stop: {r2stop}')
    d = s.Ddag(gauge, c, m)
    de = d.get_subset(SubSetEven)
    d2e = norm2(de)
    y = de.zeros()
    y,iter,flops = solveEE(gauge, y, de, m, r2req*b2*m*m/d2e, maxits)
    x += s.eoReconstruct(gauge, 4.0*y, c.get_subset(SubSetOdd), m)
    c = b - s.D(gauge, x, m)
    r2 = norm2(c)
    dt = tf.timestamp()-t0
    flops += (4*4*72+24)*x.full_volume()/2
    return x,iter,dt,flops

if __name__=='__main__':
    import sys, os
    import gauge
    from group import SU3

    if os.path.exists(sys.argv[1]):
        gconf = gauge.readGauge(sys.argv[1])
        nd = len(gconf)
        lat = gconf[0].lattice.unwrap().shape[:nd]
    else:
        nd = 4
        lat = tuple([int(sys.argv[i]) for i in range(4,0,-1)])
        gconf = gauge.from_tensor(SU3.random((4,)+lat+(3,3), tf.random.Generator.from_seed(7654321)))
    print(lat)

    print('Plaq:')
    for p in gauge.plaquette(gconf):
        print(p)

    if gconf[0].lattice.unwrap().dtype==tf.complex64:
        gconf = gconf.from_tensors(tf.cast(gconf.to_tensors(), dtype=tf.complex128))
        print(f'checkSU: {gconf.checkSU()}')
        gconf = gconf.projectSU()
        print(f'checkSU: {gconf.checkSU()}')
        print('Plaq:')
        for p in gauge.plaquette(gconf):
            print(p)

    gaugeEO = gconf.hypercube_partition()
    gaugeEO = gauge.setBC(gaugeEO)
    gaugeEO = s.phase(gaugeEO)

    v1 = tf.zeros(lat+(3,), dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,2,0]], [0.5])
    print(f'v {norm2(v1)}')
    v1 = Lattice(v1).hypercube_partition()
    print(f'v {norm2(v1)}')

    m = 0.1
    m2 = tf.constant(m*m, dtype=tf.complex128)

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.02, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.01, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = v1.zeros()
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.001, 1e-20, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')
