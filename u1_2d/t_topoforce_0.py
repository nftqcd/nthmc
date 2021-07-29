import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, hmctrain
import sys
sys.path.append('../lib')
import field

conf = nthmc.Conf(nbatch=64, nepoch=5, nstepEpoch=1024, nstepMixing=128, initDt=0.1, stepPerTraj=5)
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
loss = lambda action: nthmc.LossFun(action, cCosDiff=0.0, cTopoDiff=1.0, cForce2=0.2, dHmin=0.5, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=0.001)
rng = tf.random.Generator.from_seed(conf.seed)
mcmc = lambda: nthmc.Metropolis(conf, nthmc.LeapFrog(conf, nthmc.U1d2(beta=7.0, beta0=2.0, size=(16,16), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])))
hmctrain.run(conf, mcmc, loss, opt)
