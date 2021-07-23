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
    pass
