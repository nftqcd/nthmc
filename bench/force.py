import tensorflow as tf
import os, sys
from benchutil import bench, time, mem
sys.path.append("../lib")
from transform import Ident
from gauge import readGauge, random
from action import C1DBW2, SU3d4, Dynamics, TransformedActionMatrixBase, TransformedActionVectorFromMatrixBase
from nthmc import setup, Conf

conf = Conf(nbatch=1, nepoch=2, nstepEpoch=8, trajLength=0.2, stepPerTraj=2)
setup(conf)

act = SU3d4(beta=0.7796, c1=C1DBW2)
rng = tf.random.Generator.from_seed(conf.seed)

if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
    gconf = readGauge(sys.argv[1])
elif len(sys.argv)>=5:
    lat = [int(x) for x in sys.argv[1:5]]
    if any([x<2 or x%2==1 for x in lat]):
        raise ValueError(f'received incorrect lattice size {sys.argv[1:]}')
    nb = 0 if len(sys.argv)==5 else int(sys.argv[5])
    gconf = random(rng, lat, nbatch=nb)
    tf.print('### lat',lat,'nbatch',nb)
else:
    gconf = random(rng, [8,8,8,16], nbatch=4)

gconfP = gconf.hypercube_partition()

def run(x, act):
    if isinstance(act, TransformedActionMatrixBase):
        p = x.randomTangent(rng)
    else:
        p = x.randomTangentVector(rng)
    v,_,_ = act(x)
    tf.print('V',v)
    def fun_(x_):
        x_ = x.from_tensors(x_)
        f,_,_ = act.gradient(x_)
        return f.to_tensors()
    def fun(fn,x):
        f_ = fn(x.to_tensors())
        return p.from_tensors(f_)
    mem('before')

    f,_ = bench('egr',lambda:fun(fun_,x))
    tf.print('f2',f.norm2())

    ffun_,_ = time('tf.function',lambda:tf.function(fun_))
    ffun,_ = time('fun concrete',lambda:ffun_.get_concrete_function(x.to_tensors()))
    mem('procfun')

    f,_ = bench('fun',lambda:fun(ffun,x))
    tf.print('f2',f.norm2())

    fjit_,_ = time('tf.func:jit',lambda:tf.function(fun_,jit_compile=True))
    fjit,_ = time('jit concrete',lambda:fjit_.get_concrete_function(x.to_tensors()))
    mem('procjit')

    f,_ = bench('jit',lambda:fun(fjit,x))
    tf.print('f2',f.norm2())

tf.print('1. Mat Mom')
run(gconf, TransformedActionMatrixBase(transform=Ident(), action=act))
tf.print('2. Vec Mom from Mat')
run(gconf, TransformedActionVectorFromMatrixBase(transform=Ident(), action=act))
tf.print('3. Part Mat Mom')
run(gconfP, TransformedActionMatrixBase(transform=Ident(), action=act))
tf.print('4. Part Vec Mom from Mat')
run(gconfP, TransformedActionVectorFromMatrixBase(transform=Ident(), action=act))
