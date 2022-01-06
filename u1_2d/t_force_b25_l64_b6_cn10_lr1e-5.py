import tensorflow as tf
import tensorflow.keras as tk
import numpy
import nthmc, ftr, forcetrain

conf = nthmc.Conf(nbatch=128, nepoch=1, nstepEpoch=1024, nstepMixing=128, nstepPostTrain=1024, initDt=0.2, stepPerTraj=10, nthr=20, nthrIop=2, seed=1234567)
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
    tk.layers.Conv2D(4, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convP1 = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(4, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convR = lambda pad: ftr.PeriodicConv((
    tk.layers.Conv2D(4, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
), pad)
conv = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(4, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
transform = lambda: ftr.TransformChain([
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
])
ftr.checkDep(transform())
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=6.0, beta0=6.0, size=(64,64), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])
loss = lambda action: forcetrain.LossFun(action, betaMap=2.5, cNorm2=1.0, cNorm4=1.0, cNorm6=1.0, cNorm8=1.0, cNorm10=1.0, cNormInf=1.0)
opt = tk.optimizers.Adam(learning_rate=0.00001)
mcmc = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))
def loadConf(n):
    # n from 0 to 1023
    ab = 'abcdefgh'[n%8]
    k = n//8
    i = k//8
    j = k%8
    return tf.constant(numpy.load(f'configs/s_hmc_l64_b6_{ab}/conf.b6.0.n1024.l64_64.{8192*(i+1):05d}.npy', allow_pickle=False, fix_imports=False)[128*j:128*(j+1)], dtype=tf.float64)
forcetrain.trainWithConfs(conf, loadConf, actionFun, loss, opt, loadWeights='weights/t_force_b25_l64_b5', saveWeights='weights/t_force_b25_l64_b6_cn10_lr1e-5')
