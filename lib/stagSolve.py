import tensorflow as tf
import group as g
import lattice as l
import stagD as s

def cg_iter(A,x,p,r,r2,r2o):
    dty = r.dtype
    beta = tf.cast(r2/r2o,dtype=dty)
    # print(f'beta: {beta}')
    p = r + beta*p
    Ap = A(p)
    pAp = g.redot(p, Ap, range(tf.rank(p)))
    alpha = tf.cast(r2/pAp,dtype=dty)
    x += alpha*p
    r -= alpha*Ap
    return x,p,r,g.norm2(r, range(tf.rank(r))),r2

def cg(A, x, b, r2req, maxits):
    dty = b.dtype
    b2 = g.norm2(b, range(tf.rank(b)))
    # print(f'input norm2: {b2}')
    Ap = A(x)
    r = b - Ap
    p = tf.zeros(r.shape, dtype=dty)
    r2 = g.norm2(r, range(tf.rank(r)))
    # print(f'r2: {r2}')
    r2stop = r2req * b2
    r2o = tf.constant(1.0, dtype=tf.float64)
    itn = tf.constant(0, dtype=tf.int64)
    # print(f'CG {itn}  {r2/b2}')
    while itn<maxits and r2>r2stop:
        itn += 1
        x,p,r,r2,r2o = cg_iter(A,x,p,r,r2,r2o)
        # print(f'CG {itn}  {r2/b2}')
    return x,itn

@tf.function
def solveEE(gauge, r, x, m, r2req, maxits):
    m2 = m*m
    def op(v):
        return s.D2ee(gauge, v, m2)
    r,iter = cg(op, r, x, r2req, maxits)
    flops = (4*4*72+60)*tf.cast(tf.size(x),dtype=tf.float64)*tf.cast(iter,dtype=tf.float64)
    return r,iter,flops

def solve(gauge, x, b, m, r2req, maxits):
    t0 = tf.timestamp()
    c = b - s.D(gauge, x, m)
    b2 = g.norm2(b, range(tf.rank(b)))
    r2 = g.norm2(c, range(tf.rank(c)))
    r2stop = r2req * b2
    print(f'stagSolve b2: {b2}  r2: {r2/b2}  r2stop: {r2stop}')
    d = s.Ddag(gauge, c, m)
    de = l.get_even(d)
    d2e = g.norm2(de, range(tf.rank(de)))
    y = tf.zeros(de.shape, dtype=de.dtype)
    y,iter,flops = solveEE(gauge, y, de, tf.constant(m, dtype=tf.complex128), r2req*b2*m*m/d2e, tf.constant(maxits,dtype=tf.int64))
    x += s.eoReconstruct(gauge, 4.0*y, l.get_odd(c), m)
    c = b - s.D(gauge, x, m)
    r2 = g.norm2(c, range(tf.rank(c)))
    dt = tf.timestamp()-t0
    flops += (4*4*72+24)*tf.size(x)/2
    return x,iter,dt,flops

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
    gaugeEO = s.phase(gaugeEO, isEO=True, batch_dims=0)
    print(gaugeEO.shape)

    v1 = tf.zeros(lat[::-1]+[3], dtype=tf.complex128)
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,0,0]], [1])
    v1 = tf.tensor_scatter_nd_update(v1, [[0,0,0,2,0]], [0.5])
    print(f'v {v1.shape} {group.norm2(v1, range(tf.rank(v1)))}')
    v1 = l.evenodd_partition(v1)

    m = 0.1
    m2 = m*m

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-8, 100)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.1, 1e-12, 10000)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.02, 1e-12, 10000)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.01, 1e-12, 10000)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')

    x = tf.zeros(v1.shape, dtype=v1.dtype)
    x,iter,dt,flops = solve(gaugeEO, x, v1, 0.001, 1e-20, 10000)
    print(f'solve {iter}  x {g.norm2(x, range(tf.rank(x)))}  time {dt} sec  Gf/s {1e-9*flops/dt}')
