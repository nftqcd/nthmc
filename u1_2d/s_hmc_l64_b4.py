import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 17

conf = nthmc.Conf(nbatch=128, nepoch=1, nstepEpoch=32*16384, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1, seed=7*11*13*23)
nthmc.setup(conf)
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=4.0, beta0=0.0, size=(64,64), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

forcetrain.runInfer(conf, actionFun, mcmcFun, saveFile='configs/s_hmc_l64_b4/conf', saveFreq=32*16)
