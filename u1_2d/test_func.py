import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, evolve
import sys
sys.path.append('../lib')
import field

trajLength = 4
nstep = 10

conf = nthmc.Conf(nbatch=3, nepoch=1, nstepEpoch=512, nstepMixing=64, nstepPostTrain=1, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=16, nthrIop=2, xlaCluster=False, softPlace=True)
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
hmc = nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, nthmc.U1d2(beta=7.0, beta0=7.0, size=(512,512), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])))
thmc = nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, nthmc.U1d2(beta=7.0, beta0=7.0, size=(512,512), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])))

x = thmc.generate.action.random()
p = thmc.generate.action.randomMom()
r = thmc.generate.action.rng.uniform([x.shape[0]], dtype=tf.float64)

def pmem():
    if tf.config.list_physical_devices('GPU'):
        print(tf.config.experimental.get_memory_info('GPU:0'), flush=True)
        tf.config.experimental.reset_memory_stats('GPU:0')

def pint():
    print(f'* hmc dt {hmc.generate.dt} steps/traj {hmc.generate.stepPerTraj}')
    print(f'* thmc dt {thmc.generate.dt} steps/traj {thmc.generate.stepPerTraj}')

def time(s,f,n=1):
    for i in range(n):
        t0 = tf.timestamp()
        ret = f(x,p,r)
        if n>1:
            print(f'{s} run {i} took {tf.timestamp()-t0} sec.', flush=True)
        else:
            print(f'{s} took {tf.timestamp()-t0} sec.', flush=True)
    return ret

def my_func(mc,x,p,r):
    print(f'tracing {mc} {x} {p} {r}', flush=True)
    return mc(x,p,r)

pint()

time('hmc',hmc,4)
pmem()
time('thmc',thmc,4)
pmem()

f_fun = tf.function(my_func)
f_jit = tf.function(my_func,jit_compile=True)

hmc_fun = time('hmc_fun get concrete',lambda x,p,r:f_fun.get_concrete_function(hmc,x,p,r))
pmem()
hmc_jit = time('hmc_jit get concrete',lambda x,p,r:f_jit.get_concrete_function(hmc,x,p,r))
pmem()

thmc_fun = time('thmc_fun get concrete',lambda x,p,r:f_fun.get_concrete_function(thmc,x,p,r))
pmem()
thmc_jit = time('thmc_jit get concrete',lambda x,p,r:f_jit.get_concrete_function(thmc,x,p,r))
pmem()

# print(f_jit.experimental_get_compiler_ir(hmc,x,p,r)(stage='hlo'), flush=True)
# print(f_jit.experimental_get_compiler_ir(thmc,x,p,r)(stage='hlo'), flush=True)

# Note: concrete functions do not accept python objects.

time('hmc_fun',hmc_fun,4)
pmem()
time('hmc_jit',hmc_jit,4)
pmem()

time('thmc_fun',thmc_fun,4)
pmem()
time('thmc_jit',thmc_jit,4)
pmem()

hmc.generate.dt.assign(0.2)
hmc.generate.stepPerTraj.assign(20)

thmc.generate.dt.assign(0.2)
thmc.generate.stepPerTraj.assign(20)

pint()

time('hmc',hmc,4)
pmem()
time('thmc',thmc,4)
pmem()

time('hmc_fun',hmc_fun,4)
pmem()
time('hmc_jit',hmc_jit,4)
pmem()

time('thmc_fun',thmc_fun,4)
pmem()
time('thmc_jit',thmc_jit,4)
pmem()
