import tensorflow as tf
import math

class Group:
    """
    Gauge group represented as matrices
    in the last two dimension in tensors.
    """
    def mul(l,r,adjoint_l=False,adjoint_r=False):
        return tf.linalg.matmul(l,r,adjoint_a=adjoint_l,adjoint_b=adjoint_r)
    def adjoint(x):
        return tf.linalg.adjoint(x)
    def trace(x):
        return tf.linalg.trace(x)

class U1Phase(Group):
    def mul(l,r,adjoint_l=False,adjoint_r=False):
        if adjoint_l and adjoint_r:
            return -l-r
        elif adjoint_l:
            return -l+r
        elif adjoint_r:
            return l-r
        else:
            return l+r
    def adjoint(x):
        return -x
    def trace(x):
        return tf.cos(x)
    def diffTrace(x):
        return -tf.sin(x)
    def diff2Trace(x):
        return -tf.cos(x)
    def compatProj(x):
        return tf.math.floormod(x+math.pi, 2*math.pi)-math.pi
    def random(shape, rng):
        return rng.uniform(shape, -math.pi, math.pi, dtype=tf.float64)
    def randomMom(shape, rng):
        return rng.normal(shape, dtype=tf.float64)
    def momEnergy(p):
        return 0.5*tf.reduce_sum(tf.reshape(p, [p.shape[0], -1])**2, axis=1)

class SU2(Group):
    pass

class SU3(Group):
    dtype=tf.complex128
    size=[3,3]
    def mul(l,r,adjoint_l=False,adjoint_r=False):
        return tf.linalg.matmul(l,r,adjoint_a=adjoint_l,adjoint_b=adjoint_r)
    def adjoint(x):
        return tf.linalg.adjoint(x)
    def trace(x):
        return tf.linalg.trace(x)
    def diffTrace(x):
        tf.print('TODO')
    def diff2Trace(x):
        tf.print('TODO')
    def exp(m):
        return exp(m)
    def projectTAH(x):
        return projectTAH(x)
    def random(shape, rng):
        r = rng.normal(shape, dtype=tf.float64)
        i = rng.normal(shape, dtype=tf.float64)
        return projectSU(tf.dtypes.complex(r,i))
    def randomMom(shape, rng):
        return randTAH3(shape[:-2], rng)
    def momEnergy(p):
        p2 = norm2(p) - 8.0
        return 0.5*tf.math.reduce_sum(tf.reshape(p2, [p.shape[0], -1]), axis=1)

def unit(shape):
    return tf.eye(*shape[-2:], batch_shape=shape[:-2], dtype=tf.complex128)

def eyeOf(m):
    return tf.eye(*m.shape[-2:], batch_shape=[1]*(len(m.shape)-2), dtype=m.dtype)

def norm2(m, axis=[-2,-1], exclude=None):
    """
    Axis is ignored if exclude is not None.
    No reduction if axis is empty.
    """
    if m.dtype==tf.complex128 or m.dtype==tf.complex64:
        m = tf.abs(m)
    n = tf.math.square(m)
    if exclude is None:
        if len(axis)==0:
            return n
        else:
            return tf.math.reduce_sum(n, axis=axis)
    else:
        return tf.math.reduce_sum(n, axis=[i for i in range(len(n.shape)) if i not in exclude])

def redot(x,y):
    n = tf.math.real(tf.math.conj(x)*y)
    return tf.math.reduce_sum(n, axis=range(tf.rank(n)))

# Converted from qex/src/maths/matrixFunctions.nim
# Last two dims in a tensor contain matrices.
# WARNING: below only works for SU3 for now

def randTAH3(shape, s):
    s2 = 0.70710678118654752440    # sqrt(1/2)
    s3 = 0.57735026918962576450    # sqrt(1/3)
    r3 = s2 * s.normal(shape, dtype=tf.float64)
    r8 = s2 * s3 * s.normal(shape, dtype=tf.float64)
    m00 = tf.dtypes.complex(tf.cast(0.0,tf.float64), r8+r3)
    m11 = tf.dtypes.complex(tf.cast(0.0,tf.float64), r8-r3)
    m22 = tf.dtypes.complex(tf.cast(0.0,tf.float64), -2*r8)
    r01 = s2 * s.normal(shape, dtype=tf.float64)
    r02 = s2 * s.normal(shape, dtype=tf.float64)
    r12 = s2 * s.normal(shape, dtype=tf.float64)
    i01 = s2 * s.normal(shape, dtype=tf.float64)
    i02 = s2 * s.normal(shape, dtype=tf.float64)
    i12 = s2 * s.normal(shape, dtype=tf.float64)
    m01 = tf.dtypes.complex( r01, i01)
    m10 = tf.dtypes.complex(-r01, i01)
    m02 = tf.dtypes.complex( r02, i02)
    m20 = tf.dtypes.complex(-r02, i02)
    m12 = tf.dtypes.complex( r12, i12)
    m21 = tf.dtypes.complex(-r12, i12)
    return tf.stack([
        tf.stack([m00,m10,m20], axis=-1),
        tf.stack([m01,m11,m21], axis=-1),
        tf.stack([m02,m12,m22], axis=-1),
    ], axis=-1)

def eigs3(tr,p2,det):
    tr3 = (1.0/3.0)*tr
    p23 = (1.0/3.0)*p2
    tr32 = tr3*tr3
    q = tf.math.abs(0.5*(p23-tr32))
    r = 0.25*tr3*(5*tr32-p2) - 0.5*det
    sq = tf.math.sqrt(q)
    sq3 = q*sq
    isq3 = 1.0/sq3
    maxv = tf.constant( 3e38, shape=isq3.shape, dtype=isq3.dtype)
    minv = tf.constant(-3e38, shape=isq3.shape, dtype=isq3.dtype)
    isq3c = tf.math.minimum(maxv, tf.math.maximum(minv,isq3))
    rsq3c = r * isq3c
    maxv = tf.constant( 1, shape=isq3.shape, dtype=isq3.dtype)
    minv = tf.constant(-1, shape=isq3.shape, dtype=isq3.dtype)
    rsq3 = tf.math.minimum(maxv, tf.math.maximum(minv,rsq3c))
    t = (1.0/3.0)*tf.math.acos(rsq3)
    st = tf.math.sin(t)
    ct = tf.math.cos(t)
    sqc = sq*ct
    sqs = 1.73205080756887729352*sq*st  # sqrt(3)
    ll = tr3 + sqc
    e0 = tr3 - 2*sqc
    e1 = ll + sqs
    e2 = ll - sqs
    return e0,e1,e2

def rsqrtPHM3f(tr,p2,det):
    l0,l1,l2 = eigs3(tr,p2,det)
    sl0 = tf.math.sqrt(tf.math.abs(l0))
    sl1 = tf.math.sqrt(tf.math.abs(l1))
    sl2 = tf.math.sqrt(tf.math.abs(l2))
    u = sl0 + sl1 + sl2
    w = sl0 * sl1 * sl2
    d = w*(sl0+sl1)*(sl0+sl2)*(sl1+sl2)
    di = 1.0/d
    c0 = (w*u*u+l0*sl0*(l1+l2)+l1*sl1*(l0+l2)+l2*sl2*(l0+l1))*di
    c1 = -(tr*u+w)*di
    c2 = u*di
    return c0,c1,c2

def rsqrtPHM3(x):
    tr = tf.math.real(tf.linalg.trace(x))
    x2 = tf.linalg.matmul(x,x)
    p2 = tf.math.real(tf.linalg.trace(x2))
    det = tf.math.real(tf.linalg.det(x))
    c0,c1,c2 = rsqrtPHM3f(tr, p2, det)
    return tf.cast(tf.reshape(c0,c0.shape+[1,1])*tf.eye(3,batch_shape=[1]*len(c0.shape),dtype=c0.dtype),x.dtype) + tf.reshape(tf.cast(c1,x.dtype),c1.shape+[1,1])*x + tf.reshape(tf.cast(c2,x.dtype),c2.shape+[1,1])*x2

def projectU(x):
    "x (x'x)^{-1/2}"
    nc = x.shape[-1]
    t = tf.linalg.matmul(x,x,adjoint_a=True)
    t2 = rsqrtPHM3(t)
    return tf.linalg.matmul(x, t2)

def projectSU(x):
    nc = x.shape[-1]
    m = projectU(x)
    d = tf.linalg.det(m)    # after projectU: 1=|d
    p = (1.0/(-nc)) * tf.math.atan2(tf.math.imag(d), tf.math.real(d))
    return tf.reshape(tf.dtypes.complex(tf.math.cos(p), tf.math.sin(p)),p.shape+[1,1]) * m

def projectTAH(x):
    """
    returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
    R = - T^a tr[T^a (X - X†)]
      = T^a ∂_a (- tr[X + X†])
    """
    nc = x.shape[-1]
    r = 0.5*(x - tf.linalg.adjoint(x))
    d = tf.linalg.trace(r) / nc
    r -= tf.reshape(d,d.shape+[1,1])*eyeOf(x)
    return r

def checkU(x):
    ## Returns the average and maximum of the sum of deviations of x^dag x.
    nc = x.shape[-1]
    d = norm2(tf.linalg.matmul(x,x,adjoint_a=True) - eyeOf(x))
    a = tf.math.reduce_mean(d, axis=range(1,len(d.shape)))
    b = tf.math.reduce_max(d, axis=range(1,len(d.shape)))
    c = 2*(nc*nc+1)
    return tf.math.sqrt(a/c),tf.math.sqrt(b/c)

def checkSU(x):
    ## Returns the average and maximum of the sum of deviations of x^dag x and det(x) from unitarity.
    nc = x.shape[-1]
    d = norm2(tf.linalg.matmul(x,x,adjoint_a=True) - eyeOf(x))
    d += norm2(-1 + tf.linalg.det(x), axis=[])
    a = tf.math.reduce_mean(d, axis=range(1,len(d.shape)))
    b = tf.math.reduce_max(d, axis=range(1,len(d.shape)))
    c = 2*(nc*nc+1)
    return tf.math.sqrt(a/c),tf.math.sqrt(b/c)

def su3vec(x):
    """
    Only for x in 3x3 anti-Hermitian.  Return 8 real numbers, X^a T^a = X - 1/3 tr(X).
    Convention: tr{T^a T^a} = -1/2
    X^a = - 2 tr[T^a X]
    """
    s3 = 0.57735026918962576451    # sqrt(1/3)
    # r3 = 1.7320508075688772935    # sqrt(3)
    c = -2
    return tf.stack([
        c*tf.math.imag(x[...,0,1]), c*tf.math.real(x[...,0,1]),
        tf.math.imag(x[...,1,1])-tf.math.imag(x[...,0,0]),
        c*tf.math.imag(x[...,0,2]), c*tf.math.real(x[...,0,2]),
        c*tf.math.imag(x[...,1,2]), c*tf.math.real(x[...,1,2]),
        s3*(2*tf.math.imag(x[...,2,2])-tf.math.imag(x[...,1,1])-tf.math.imag(x[...,0,0]))
        # r3*tf.math.imag(x[...,2,2])
    ], axis=-1)

def su3fromvec_direct(v):
    """
    X = X^a T^a
    tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) δ^ab = -1/2 X^b
    X^a = -2 X_ij T^a_ji
    """
    s3 = 0.57735026918962576451    # sqrt(1/3)
    c = -0.5
    zero = tf.zeros(v[...,0].shape, dtype=v[...,0].dtype)
    x01 = c*tf.dtypes.complex(v[...,1], v[...,0])
    x02 = c*tf.dtypes.complex(v[...,4], v[...,3])
    x12 = c*tf.dtypes.complex(v[...,6], v[...,5])
    x2i = s3*v[...,7]
    x0i = c*(x2i+v[...,2])
    x1i = c*(x2i-v[...,2])
    return tf.stack([
        tf.stack([tf.dtypes.complex(zero,x0i), -tf.math.conj(x01), -tf.math.conj(x02)], axis=-1),
        tf.stack([x01,                tf.dtypes.complex(zero,x1i), -tf.math.conj(x12)], axis=-1),
        tf.stack([x02,                x12,                tf.dtypes.complex(zero,x2i)], axis=-1),
    ], axis=-1)

def su3fromvec_mat(v):
    """
    X = X^a T^a
    tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) δ^ab = -1/2 X^b
    X^a = -2 X_ij T^a_ji
    This implements a matvec converter.
    real:            0          c*v1     c*v4    # signs are different from above tf.stack
                 -c*v1             0     c*v6    # because tf.stack collumn order
                 -c*v4         -c*v6        0
    imag: c*(s3*v7+v2)          c*v0     c*v3
                  c*v0  c*(s3*v7-v2)     c*v5
                  c*v3          c*v5    s3*v7
    """
    s3 = 0.57735026918962576451    # sqrt(1/3)
    c = -0.5
    dt = v.dtype
    shmat = v.shape[:-1]+(3,3)
    shconv = (1,)*(len(v.shape)-1)+(9,8)
    # 9x8 matrix for real and imag
    convR = tf.reshape(tf.constant([
        0,0,0,0,0,0,0,0,
        0, c,0,0,0,0,0,0,
        0,0,0,0, c,0,0,0,
        0,-c,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0, c,0,
        0,0,0,0,-c,0,0,0,
        0,0,0,0,0,0,-c,0,
        0,0,0,0,0,0,0,0], dtype=dt), shconv)
    convI = tf.reshape(tf.constant([
        0,0, c,0,0,0,0,c*s3,
        c,0,0,0,0,0,0,0,
        0,0,0,c,0,0,0,0,
        c,0,0,0,0,0,0,0,
        0,0,-c,0,0,0,0,c*s3,
        0,0,0,0,0,c,0,0,
        0,0,0,c,0,0,0,0,
        0,0,0,0,0,c,0,0,
        0,0,0,0,0,0,0,s3], dtype=dt), shconv)
    return tf.reshape(tf.dtypes.complex(tf.linalg.matvec(convR, v), tf.linalg.matvec(convI, v)), shmat)

su3fromvec = su3fromvec_mat

def su3fabc(v):
    """
    returns f^{abc} v[...,c]
    [T^a, T^b] = f^abc T^c
    """
    f012 = 1.0
    f036 = 0.5
    f045 = -0.5
    f135 = 0.5
    f146 = 0.5
    f234 = 0.5
    f256 = -0.5
    f347 = 0.86602540378443864676    # sqrt(3/4)
    f567 = 0.86602540378443864676
    a01 =   f012 *v[...,2]
    a02 = (-f012)*v[...,1]
    a03 =   f036 *v[...,6]
    a04 =   f045 *v[...,5]
    a05 = (-f045)*v[...,4]
    a06 = (-f036)*v[...,3]
    a12 =   f012 *v[...,0]
    a13 =   f135 *v[...,5]
    a14 =   f146 *v[...,6]
    a15 = (-f135)*v[...,3]
    a16 = (-f146)*v[...,4]
    a23 =   f234 *v[...,4]
    a24 = (-f234)*v[...,3]
    a25 =   f256 *v[...,6]
    a26 = (-f256)*v[...,5]
    a34 =   f347 *v[...,7] + f234 *v[...,2]
    a35 =   f135 *v[...,1]
    a36 =   f036 *v[...,0]
    a37 = (-f347)*v[...,4]
    a45 =   f045 *v[...,0]
    a46 =   f146 *v[...,1]
    a47 =   f347 *v[...,3]
    a56 =   f567 *v[...,7] + f256 *v[...,2]
    a57 = (-f567)*v[...,6]
    a67 =   f567 *v[...,5]
    zero = tf.zeros(v[...,0].shape, dtype=v[...,0].dtype)
    return tf.stack([
        tf.stack([zero,-a01,-a02,-a03,-a04,-a05,-a06,zero], axis=-1),
        tf.stack([ a01,zero,-a12,-a13,-a14,-a15,-a16,zero], axis=-1),
        tf.stack([ a02, a12,zero,-a23,-a24,-a25,-a26,zero], axis=-1),
        tf.stack([ a03, a13, a23,zero,-a34,-a35,-a36,-a37], axis=-1),
        tf.stack([ a04, a14, a24, a34,zero,-a45,-a46,-a47], axis=-1),
        tf.stack([ a05, a15, a25, a35, a45,zero,-a56,-a57], axis=-1),
        tf.stack([ a06, a16, a26, a36, a46, a56,zero,-a67], axis=-1),
        tf.stack([zero,zero,zero, a37, a47, a57, a67,zero], axis=-1),
    ], axis=-1)

def su3dabc(v):
    """
    returns d^abc v[...,c]
    {T^a,T^b} = -1/3δ^ab + i d^abc T^c
    """
    # NOTE: negative sign of what's on wikipedia
    d007 = -0.57735026918962576451    # -sqrt(1/3)
    d035 = -0.5
    d046 = -0.5
    d117 = -0.57735026918962576451
    d136 = 0.5
    d145 = -0.5
    d227 = -0.57735026918962576451
    d233 = -0.5
    d244 = -0.5
    d255 = 0.5
    d266 = 0.5
    d337 = 0.28867513459481288225    # sqrt(1/3)/2
    d447 = 0.28867513459481288225
    d557 = 0.28867513459481288225
    d667 = 0.28867513459481288225
    d777 = 0.57735026918962576451
    a00 = d007*v[...,7]
    a03 = d035*v[...,5]
    a04 = d046*v[...,6]
    a05 = d035*v[...,3]
    a06 = d046*v[...,4]
    a07 = d007*v[...,0]
    a11 = d117*v[...,7]
    a13 = d136*v[...,6]
    a14 = d145*v[...,5]
    a15 = d145*v[...,4]
    a16 = d136*v[...,3]
    a17 = d117*v[...,1]
    a22 = d227*v[...,7]
    a23 = d233*v[...,3]
    a24 = d244*v[...,4]
    a25 = d255*v[...,5]
    a26 = d266*v[...,6]
    a27 = d227*v[...,2]
    a33 = d337*v[...,7]+d233*v[...,2]
    a35 = d035*v[...,0]
    a36 = d136*v[...,1]
    a37 = d337*v[...,3]
    a44 = d447*v[...,7]+d244*v[...,2]
    a45 = d145*v[...,1]
    a46 = d046*v[...,0]
    a47 = d447*v[...,4]
    a55 = d557*v[...,7]+d255*v[...,2]
    a57 = d557*v[...,5]
    a66 = d667*v[...,7]+d266*v[...,2]
    a67 = d667*v[...,6]
    a77 = d777*v[...,7]
    zero = tf.zeros(v[...,0].shape, dtype=v[...,0].dtype)
    return tf.stack([
        tf.stack([ a00,zero,zero, a03, a04, a05, a06, a07], axis=-1),
        tf.stack([zero, a11,zero, a13, a14, a15, a16, a17], axis=-1),
        tf.stack([zero,zero, a22, a23, a24, a25, a26, a27], axis=-1),
        tf.stack([ a03, a13, a23, a33,zero, a35, a36, a37], axis=-1),
        tf.stack([ a04, a14, a24,zero, a44, a45, a46, a47], axis=-1),
        tf.stack([ a05, a15, a25, a35, a45, a55,zero, a57], axis=-1),
        tf.stack([ a06, a16, a26, a36, a46,zero, a66, a67], axis=-1),
        tf.stack([ a07, a17, a27, a37, a47, a57, a67, a77], axis=-1),
    ], axis=-1)

def SU3Ad(x):
    """
    X T^c X† = AdX T^c = T^b AdX^bc
    Input x must be in SU(3) group.
    AdX^bc = - 2 tr[T^b X T^c X†] = - 2 tr[T^c X† T^b X]
    """
    y = tf.expand_dims(x, -3)
    return su3vec(tf.linalg.matmul(y, tf.linalg.matmul(su3gen(), y), adjoint_a=True))

def su3ad(x):
    """
    adX^{ab} = - f^{abc} X^c = f^{abc} 2 tr(X T^c) = 2 tr(X [T^a, T^b])
    Input x must be in su(3) algebra.
    """
    return su3fabc(-su3vec(x))

def su3adapply(adx, y):
    """
    adX(Y) = [X, Y]
    adX(T^b) = T^a adX^{ab} = - T^a f^{abc} X^c = X^c f^{cba} T^a = X^c [T^c, T^b] = [X, T^b]
    adX(Y) = T^a adX^{ab} Y^b = T^a adX^{ab} (-2) tr{T^b Y}
    """
    return su3fromvec(tf.linalg.matvec(adx, su3vec(y)))

def gellMann():
    s3 = 0.57735026918962576451    # sqrt(1/3)
    zero3 = tf.zeros([3,3], dtype=tf.float64)
    return tf.stack([
        tf.dtypes.complex(tf.reshape(tf.constant([0,1,0,1,0,0,0,0,0],dtype=tf.float64),[3,3]), zero3),
        tf.dtypes.complex(zero3, tf.reshape(tf.constant([0,-1,0,1,0,0,0,0,0],dtype=tf.float64),[3,3])),
        tf.dtypes.complex(tf.reshape(tf.constant([1,0,0,0,-1,0,0,0,0],dtype=tf.float64),[3,3]), zero3),
        tf.dtypes.complex(tf.reshape(tf.constant([0,0,1,0,0,0,1,0,0],dtype=tf.float64),[3,3]), zero3),
        tf.dtypes.complex(zero3, tf.reshape(tf.constant([0,0,-1,0,0,0,1,0,0],dtype=tf.float64),[3,3])),
        tf.dtypes.complex(tf.reshape(tf.constant([0,0,0,0,0,1,0,1,0],dtype=tf.float64),[3,3]), zero3),
        tf.dtypes.complex(zero3, tf.reshape(tf.constant([0,0,0,0,0,-1,0,1,0],dtype=tf.float64),[3,3])),
        s3*tf.dtypes.complex(tf.reshape(tf.constant([1,0,0,0,1,0,0,0,-2],dtype=tf.float64),[3,3]), zero3)])

_su3gen_private_global_cache_ = None
def su3gen():
    """
    T[a,i,j] = T^a_ij
    Traceless Anti-Hermitian basis.  tr{T^a T^a} = -1/2
    """
    global _su3gen_private_global_cache_
    if _su3gen_private_global_cache_ is None:
        _su3gen_private_global_cache_ = tf.dtypes.complex(tf.constant(0,dtype=tf.float64),tf.constant(-0.5,dtype=tf.float64)) * gellMann()
    return _su3gen_private_global_cache_

def exp(m, order=12):
    eye = eyeOf(m)
    x = eye + 1.0/order * m
    for i in tf.range(order-1, 0, -1):
        x = eye + 1.0/tf.cast(i,m.dtype)*tf.linalg.matmul(m,x)
    return x

def diffprojectTAH(m, p = None):
    """
    returns ∂_c p^a = ∂_c projectTAH(m)^a = - tr[T^a (T^c M + M† T^c)]
    P^a = -2 tr[T^a {- T^d tr[T^d (M - M†)]}]
        = - tr[T^a (M - M†)]
        = - ∂_a tr[M + M†]
    ∂_c P^a = - tr[T^a (T^c M + M† T^c)]
            = - 1/2 tr[{T^a,T^c} (M+M†) + [T^a,T^c] (M-M†)]
            = - 1/2 tr[d^acb T^b i (M+M†) - 1/3 δ^ac (M+M†) + f^acb T^b (M-M†)]
            = - 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) - f^acb F^b }
            = - 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) + adF^ac }
    Note:
        T^a T^b = 1/2 {(f^abc + i d^abc) T^c - 1/3 δ^ab}
    """
    if p is None:
        p = projectTAH(m)
    mhalfadP = su3ad((-0.5) * p)
    Ms = m+tf.linalg.adjoint(m)
    trMs = tf.math.real(tf.linalg.trace(Ms))/6.0
    I = tf.dtypes.complex(tf.constant(0,dtype=tf.float64), tf.constant(1,dtype=tf.float64))
    return su3dabc(0.25*su3vec(I*Ms)) + tf.reshape(trMs,trMs.shape+[1,1])*eyeOf(mhalfadP) + mhalfadP

def diffprojectTAHCross(m, x = None, Adx = None, p = None):
    """
    returns R^ac = ∇_c p^a = ∇_c projectTAH(X Y)^a = - ∇_c ∂_a tr[X Y + Y† X†], where M = X Y
    The derivatives ∂ is on X and ∇ is on Y.
    ∇_c P^a = - 2 ReTr[T^a X T^c Y]
            = - tr[T^a (X T^c X† X Y + Y† X† X T^c X†)]
            = - tr[T^a (T^b M + M† T^b)] AdX^bc
    """
    if Adx is None:
        if x is None:
            raise ValueError(f'diffprojectTAHCross must either provide x or Adx.')
        Adx = SU3Ad(x)
    return tf.linalg.matmul(diffprojectTAH(m, p), Adx)

def diffexp(adX, order=13):
    """
    return J(X) = (1-exp{-adX})/adX = Σ_{k=0}^\infty 1/(k+1)! (-adX)^k  upto k=order
    [exp{-X(t)} d/dt exp{X(t)}]_ij = [J(X) d/dt X(t)]_ij = T^a_ij J(X)^ab (-2) T^b_kl [d/dt X(t)]_lk
    J(X) = 1 + 1/2 (-adX) (1 + 1/3 (-adX) (1 + 1/4 (-adX) (1 + ...)))
    J(x) ∂_t x
        = T^a J(x)^ab (-2) tr[T^b ∂_t x]
        = exp(-x) ∂_t exp(x)
    J(s x) ∂_t x = exp(-s x) ∂_t exp(s x)
    ∂_s J(s x) ∂_t x
        = - exp(-s x) x ∂_t exp(s x) + exp(-s x) ∂_t x exp(s x)
        = - exp(-s x) x ∂_t exp(s x) + exp(-s x) [∂_t x] exp(s x) + exp(-s x) x ∂_t exp(s x)
        = exp(-s x) [∂_t x] exp(s x)
        = exp(-s adx) ∂_t x
        = Σ_k 1/k! (-1)^k s^k (adx)^k ∂_t x
    J(0) = 0
    J(x) ∂_t x
        = ∫_0^1 ds Σ_{k=0} 1/k! (-1)^k s^k (adx)^k ∂_t x
        = Σ_{k=0} 1/(k+1)! (-1)^k (adx)^k ∂_t x
    """
    m = -adX
    eye = eyeOf(m)
    x = eye + 1.0/(order+1.0) * m
    for i in tf.range(order, 1, -1):
        x = eye + 1.0/tf.cast(i,m.dtype)*tf.linalg.matmul(m,x)
    return x

def SU3GradientTF(f, x):
    """
    Compute gradient using TensorFlow GradientTape.
    f(x) must be a real scalar value.
    Returns (f(x),D), where D = T^a D^a = T^a ∂_a f(x)
    Use real vector derivatives.
    D^a = ∂_a f(x)
        = ∂_t f(exp(t T^a) x) |_t=0
    """
    v = tf.zeros(8, dtype=tf.float64)
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(v)
        r = f(tf.linalg.matmul(exp(su3fromvec(v)),x))
    d = t.gradient(r, v)
    return r,d

def SU3GradientTFMat(f, x):
    """
    Compute gradient using TensorFlow GradientTape.
    f(x) must be a real scalar value.
    Returns (f(x),D), where D = T^a D^a = T^a ∂_a f(x)
    Use Matrix derivatives.
    D^a = ∂_a f(x)
        = [∂_a x_ij] [d/dx_ij f(x)]
        = [T^a_ik x_kj] [d/dx_ij f(x)]
        = [T^a_ik x†_kj] [d/dx†_ij f(x)]
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)† = 2 [∂_z Re(f)]† = 2 ∂_z† Re(f)
    """
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(x)
        r = f(x)
    d = 0.5*projectTAH(tf.linalg.matmul(t.gradient(r, x), x, adjoint_b=True))
    return r,d

def SU3JacobianTF(f, x, is_SU3=True):
    """
    Compute Jacobian using TensorFlow GradientTape with real vector derivatives.
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)†
    In order to have the proper gradient info, we always project the result to su(3).
    If is_SU3 is True, we multiply the result by its adjoint before projecting.
    Otherwise we assume the result is su3 and project it directly.
    The input x must be in SU(3).
    Returns f(x) and its Jacobian in ad.
    [d/dSU(3)] SU(3)
        T^c_km X_ml (-2) (∂_X_kl F(X)_in) F(X)†_nj T^a_ji
          = T^c_km X_ml (-2) F'(X)_{kl,in} F(X)†_nj T^a_ji
    [d/dSU(3)] su(3)
        (T^c X)_kl (∂_X_kl F(X)^a)
        T^c_km X_ml (∂_X_kl F(X)^a)
          = T^c_km X_ml F'(X)^a_{kl}
    """
    v = tf.zeros(8, dtype=tf.float64)
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
        t.watch(v)
        Z = f(tf.linalg.matmul(exp(su3fromvec(v)),x))
        if is_SU3:
            z = tf.linalg.matmul(Z,tf.stop_gradient(Z),adjoint_b=True)
        else:
            z = Z
        z = su3vec(z)
    tj = t.jacobian(z,v,experimental_use_pfor=False)
    return Z,tj

def SU3JacobianTFMat(f, x, is_SU3=True):
    """
    Compute Jacobian using TensorFlow GradientTape with matrix derivatives.
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)†
    In order to have the proper gradient info, we always project the result to su(3).
    If is_SU3 is True, we multiply the result by its adjoint before projecting.
    Otherwise we assume the result is su3 and project it directly.
    The input x must be in SU(3).
    Returns f(x) and its Jacobian in ad.
    [d/dSU(3)] SU(3)
        T^c_km X_ml (-2) (∂_X_kl F(X)_in) F(X)†_nj T^a_ji
          = T^c_km X_ml (-2) F'(X)_{kl,in} F(X)†_nj T^a_ji
    [d/dSU(3)] su(3)
        (T^c X)_kl (∂_X_kl F(X)^a)
        T^c_km X_ml (∂_X_kl F(X)^a)
          = T^c_km X_ml F'(X)^a_{kl}
    """
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
        t.watch(x)
        Z = f(x)
        if is_SU3:
            z = tf.linalg.matmul(Z,tf.stop_gradient(Z),adjoint_b=True)
        else:
            z = Z
        z = tf.cast(su3vec(z), tf.complex128)
    jzx = t.jacobian(z,x,experimental_use_pfor=False)
    tj = tf.math.real(tf.einsum('aik,kj,bij->ba', su3gen(), x, tf.math.conj(jzx)))
    return Z,tj
