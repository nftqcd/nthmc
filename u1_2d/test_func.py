import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, evolve
import timeit
import sys
sys.path.append('../lib')
import field

trajLength = 4
nstep = 10

# conf = nthmc.Conf(nbatch=64, nepoch=1, nstepEpoch=512, nstepMixing=64, nstepPostTrain=1, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=16, nthrIop=2, xlaCluster=True)
conf = nthmc.Conf(nbatch=64, nepoch=1, nstepEpoch=512, nstepMixing=64, nstepPostTrain=1, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=16, nthrIop=2, xlaCluster=False)
nthmc.setup(conf)
op0 = (((1,2,-1,-2), (1,-2,-1,2)),
       ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
# requires different coefficient bounds:
# (1,2,-1,-2,1,-2,-1,2)
# (1,2,-1,-2,1,2,-1,-2)
# (1,-2,-1,2,1,-2,-1,2)
op1 = (((2,-1,-2,1), (2,1,-2,-1)),
       ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
fixedP = (1,2,-1,-2)
fixedR0 = (2,2,1,-2,-2,-1)
fixedR1 = (1,1,2,-1,-1,-2)
convP0 = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convP1 = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convR = lambda pad: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
), pad)
conv = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
transform = lambda: ftr.TransformChain([
    # ftr.GenericStoutSmear(((0,0),(2,2)), (((1,2,-1,-2), (1,-2,-1,2)),), [(fixedP, convP0())], lambda x:x),
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
])
# ftr.checkDep(transform())
rng = tf.random.Generator.from_seed(conf.seed)
mcmc = nthmc.Metropolis(conf, nthmc.LeapFrog(conf, nthmc.U1d2(beta=7.0, beta0=7.0, size=(64,64), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])))
# mcmc = nthmc.Metropolis(conf, nthmc.LeapFrog(conf, nthmc.U1d2(beta=7.0, beta0=7.0, size=(64,64), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])))

x = mcmc.generate.action.random()
p = mcmc.generate.action.randomMom()
r = mcmc.generate.action.rng.uniform([x.shape[0]], dtype=tf.float64)

def my_func(x,p,r):
    print('* tracing',x,p,r)
    return mcmc(x,p,r)

f0 = tf.function(my_func).get_concrete_function(x,p,r)
f1 = tf.function(my_func,jit_compile=True).get_concrete_function(x,p,r)

print(tf.function(my_func,jit_compile=True).experimental_get_compiler_ir(x,p,r)(stage='hlo'))

print('* first run')
print('tf.function:', timeit.timeit(lambda: f0(x,p,r), number=1))
print('tf.function(jit_compile=True):', timeit.timeit(lambda: f1(x,p,r), number=1))
print('* benchmark')
print('tf.function:', timeit.timeit(lambda: f0(x,p,r), number=10))
print('tf.function(jit_compile=True):', timeit.timeit(lambda: f1(x,p,r), number=10))
