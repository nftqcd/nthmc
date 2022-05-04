import tensorflow as tf
import parts
import group as g
import lattice as l
import stagD as s

def norm2(v):
    return g.norm2(v, allreduce=True, exclude=[])

def cg_iter(A,x,p,r,r2,r2o):
    beta = l.typecast(r2/r2o,target=p)
    # print(f'beta: {beta}')
    p = r + beta*p
    Ap = A(p)
    pAp = g.redot(p, Ap)
    alpha = l.typecast(r2/pAp,target=p)
    x += alpha*p
    r -= alpha*Ap
    return x,p,r,norm2(r),r2

def cg(A, x, b, r2req, maxits):
    b2 = norm2(b)
    # print(f'input norm2: {b2}')
    Ap = A(x)
    r = b - Ap
    p = l.zeros_like(r)
    r2 = norm2(r)
    # print(f'r2: {r2}')
    r2stop = r2req * b2
    r2o = tf.constant(1.0, dtype=tf.float64)
    itn = tf.constant(0, dtype=tf.int64)
    # print(f'CG {itn}  {r2/b2}')
    if isinstance(x, parts.HypercubeParts):
        # NOTE: tf.function cannot handle wrapper class objects in python while loop.
        def cond(itn,xp,pp,rp,r2,r2o):
            return itn<maxits and r2>r2stop
        ss = x.subset
        def body(itn,xp,pp,rp,r2,r2o):
            itn += 1
            x = parts.HypercubeParts(xp, subset=ss)
            p = parts.HypercubeParts(pp, subset=ss)
            r = parts.HypercubeParts(rp, subset=ss)
            x,p,r,r2,r2o = cg_iter(A,x,p,r,r2,r2o)
            return itn,x.parts,p.parts,r.parts,r2,r2o
        itn,xp,pp,rp,r2,r2o = tf.while_loop(cond, body, [itn,x.parts,p.parts,r.parts,r2,r2o])
        x = parts.HypercubeParts(xp, subset=ss)
    else:
        def cond(itn,x,p,r,r2,r2o):
            return itn<maxits and r2>r2stop
        ss = x.subset
        def body(itn,x,p,r,r2,r2o):
            itn += 1
            x,p,r,r2,r2o = cg_iter(A,x,p,r,r2,r2o)
            return itn,x,p,r,r2,r2o
        itn,x,p,r,r2,r2o = tf.while_loop(cond, body, [itn,x,p,r,r2,r2o])
    return x,itn

@tf.function(jit_compile=True)
def TFsolveEE(gaugeT, rT, xT, m, r2req, maxits):
    gauge = parts.HypercubeParts(gaugeT, subset='all')
    r = parts.HypercubeParts(rT, subset='even')
    x = parts.HypercubeParts(xT, subset='even')
    m2 = m*m
    def op(v):
        return s.D2ee(gauge, v, m2)
    result,iter = cg(op, r, x, r2req, maxits)
    flops = (4*4*72+60)*tf.cast(l.size(x),dtype=tf.float64)*tf.cast(iter,dtype=tf.float64)
    return result.parts,iter,flops

def solveEE(gauge, r, x, m, r2req, maxits):
    rT,iter,flops = TFsolveEE(gauge.parts, r.parts, x.parts, tf.constant(m, dtype=tf.complex128), r2req, tf.constant(maxits,dtype=tf.int64))
    r = parts.HypercubeParts(rT, subset='even')
    return r,iter,flops

def solve(gauge, x, b, m, r2req, maxits):
    t0 = tf.timestamp()
    c = b - s.D(gauge, x, m)
    b2 = norm2(b)
    r2 = norm2(c)
    r2stop = r2req * b2
    # print(f'stagSolve b2: {b2}  r2: {r2/b2}  r2stop: {r2stop}')
    d = s.Ddag(gauge, c, m)
    de = l.get_even(d)
    d2e = norm2(de)
    y = l.zeros_like(de)
    y,iter,flops = solveEE(gauge, y, de, m, r2req*b2*m*m/d2e, maxits)
    x += s.eoReconstruct(gauge, 4.0*y, l.get_odd(c), m)
    c = b - s.D(gauge, x, m)
    r2 = norm2(c)
    dt = tf.timestamp()-t0
    flops += (4*4*72+24)*l.size(x)/2
    return x,iter,dt,flops

if __name__=='__main__':
    import sys, os
    import fieldio

    if os.path.exists(sys.argv[1]):
        gconf,lat = fieldio.readLattice(sys.argv[1])
    else:
        lat = [int(sys.argv[i]) for i in range(1,5)]
        gconf = g.SU3.random([4]+lat[::-1]+[3,3], tf.random.Generator.from_seed(7654321))
    nd = len(lat)
    print(gconf.shape)
    print(lat)

    print('Plaq:', l.plaquette(gconf, nd, False))

    gaugeEO = l.hypercube_partition(gconf, nd, batch_dims=1)    # Count dim axis as batch
    gaugeEO = l.setBC(gaugeEO, batch_dims=0)
    gaugeEO = s.phase(gaugeEO, batch_dims=0)

    v1 = tf.zeros(lat[::-1]+[3], dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,2,0]], [0.5])
    print(f'v {v1.shape} {norm2(v1)}')
    v1 = l.hypercube_partition(v1)

    m = 0.1
    m2 = tf.constant(m*m, dtype=tf.complex128)

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.02, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.01, 1e-12, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = l.zeros_like(v1)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.001, 1e-20, 10000)
    print(f'solve {iter}  x {norm2(x)}  time {dt} sec  Gf/s {1e-9*flops/dt}')
