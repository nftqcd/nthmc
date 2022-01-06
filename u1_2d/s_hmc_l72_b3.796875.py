import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 10
fac = (72./64.)**2

conf = nthmc.Conf(nbatch=128, nepoch=1, nstepEpoch=16*16384, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1, seed=7*11*13*17*19)
nthmc.setup(conf)
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=3.*fac, beta0=0.0, size=(72,72), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

forcetrain.runInfer(conf, actionFun, mcmcFun, saveFile='configs/s_hmc_l72_b3.796875/conf', saveFreq=16*16)
