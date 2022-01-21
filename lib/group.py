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
        eye = tf.eye(3,batch_shape=[1]*(len(m.shape)-2),dtype=m.dtype)
        x = eye + 1.0/12.0 * m
        for i in tf.range(11, 0, -1):
            x = eye + 1.0/tf.cast(i,m.dtype)*tf.linalg.matmul(m,x)
        return x
    def projectTAH(x):
        r = 0.5*(x - tf.linalg.adjoint(x))
        d = tf.linalg.trace(r) / 3.0
        r -= tf.reshape(d,d.shape+[1,1])*tf.eye(3,batch_shape=[1]*len(d.shape),dtype=d.dtype)
        return r
    def random(shape, rng):
        r = rng.normal(shape, dtype=tf.float64)
        i = rng.normal(shape, dtype=tf.float64)
        return projectSU(tf.dtypes.complex(r,i))
    def randomMom(shape, rng):
        return randTAH3(shape[:-2], rng)
    def momEnergy(p):
        p2 = tf.math.real(tf.norm(p, ord='fro', axis=[-2,-1]))**2 - 8.0
        return 0.5*tf.math.reduce_sum(tf.reshape(p2, [p.shape[0], -1]), axis=1)

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
    nc = 3
    t = tf.linalg.matmul(x,x,adjoint_a=True)
    t2 = rsqrtPHM3(t)
    return tf.linalg.matmul(x, t2)

def projectSU(x):
    nc = 3
    m = projectU(x)
    d = tf.linalg.det(m)    # after projectU: 1=|d
    p = (1.0/(-nc)) * tf.math.atan2(tf.math.imag(d), tf.math.real(d))
    return tf.reshape(tf.dtypes.complex(tf.math.cos(p), tf.math.sin(p)),p.shape+[1,1]) * m
