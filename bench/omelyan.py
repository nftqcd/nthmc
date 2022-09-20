import tensorflow as tf
import os, sys
from benchutil import bench, time
sys.path.append("../lib")
from transform import Ident
from gauge import readGauge, random
from action import C1DBW2, SU3d4, Dynamics, TransformedActionMatrixBase, TransformedActionVectorBase, TransformedActionVectorFromMatrixBase
from evolve import Omelyan2MN
from nthmc import setup, Conf

conf = Conf(nbatch=1, nepoch=2, nstepEpoch=8, trajLength=0.2, stepPerTraj=4)
setup(conf)

act = SU3d4(beta=0.7796, c1=C1DBW2)
rng = tf.random.Generator.from_seed(9876543211)

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
    gconf = random(rng, [8,8,8,16], nbatch=32)

gconfP = gconf.hypercube_partition()

def mem(s=''):
    if tf.config.list_physical_devices('GPU'):
        s = 'mem' if s=='' else s+' mem'
        tf.print(s,tf.config.experimental.get_memory_info('GPU:0'))
        tf.config.experimental.reset_memory_stats('GPU:0')

def run(x0, act):
    if isinstance(act, (TransformedActionVectorBase,TransformedActionVectorFromMatrixBase)):
        p0 = x0.randomTangentVector(rng)
    else:
        p0 = x0.randomTangent(rng)
    x,p = x0,p0
    md = Omelyan2MN(conf, Dynamics(act))
    v0,_,_ = md.dynamics.V(x)
    t0 = md.dynamics.T(p)
    tf.print('H0',v0+t0,v0,t0)
    def mdfun_(x_,p_):
        x = x0.from_tensors(x_)
        p = p0.from_tensors(p_)
        xn,pn,_,_,_,_ = md(x,p)
        return xn.to_tensors(),pn.to_tensors()
    def mdfun(f,x,p):
        x_,p_ = f(x.to_tensors(),p.to_tensors())
        return x.from_tensors(x_),p.from_tensors(p_)
    ffun_,_ = time('tf.function',lambda:tf.function(mdfun_))
    fjit_,_ = time('tf.func:jit',lambda:tf.function(mdfun_,jit_compile=True))
    ffun,_ = time('fun concrete',lambda:ffun_.get_concrete_function(x.to_tensors(),p.to_tensors()))
    fjit,_ = time('jit concrete',lambda:fjit_.get_concrete_function(x.to_tensors(),p.to_tensors()))
    mem('before')

    x,p = x0,p0
    (x,p),_ = bench('egr',lambda:mdfun(mdfun_,x,p))
    v1,_,_ = md.dynamics.V(x)
    t1 = md.dynamics.T(p)
    tf.print('H1',v1+t1,v1,t1)
    tf.print('dH',v1+t1-v0-t0)
    mem('egr')

    x,p = x0,p0
    (x,p),_ = bench('fun',lambda:mdfun(ffun,x,p))
    v1,_,_ = md.dynamics.V(x)
    t1 = md.dynamics.T(p)
    tf.print('H1',v1+t1,v1,t1)
    tf.print('dH',v1+t1-v0-t0)
    mem('fun')

    x,p = x0,p0
    (x,p),_ = bench('jit',lambda:mdfun(fjit,x,p))
    v1,_,_ = md.dynamics.V(x)
    t1 = md.dynamics.T(p)
    tf.print('H1',v1+t1,v1,t1)
    tf.print('dH',v1+t1-v0-t0)
    mem('jit')

tf.print('1. Mat Mom')
run(gconf, TransformedActionMatrixBase(transform=Ident(), action=act))
tf.print('2. Vec Mom')
run(gconf, TransformedActionVectorBase(transform=Ident(), action=act))
tf.print('3. Vec Mom from Mat')
run(gconf, TransformedActionVectorFromMatrixBase(transform=Ident(), action=act))
tf.print('4. Part Mat Mom')
run(gconfP, TransformedActionMatrixBase(transform=Ident(), action=act))
tf.print('5. Part Vec Mom')
run(gconfP, TransformedActionVectorBase(transform=Ident(), action=act))
tf.print('6. Part Vec Mom from Mat')
run(gconfP, TransformedActionVectorFromMatrixBase(transform=Ident(), action=act))
