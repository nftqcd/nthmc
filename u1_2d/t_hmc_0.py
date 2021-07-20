import tensorflow.keras as tk
import nthmc, ftr, hmctrain
import sys
sys.path.append('../lib')
import field

conf = nthmc.Conf(nbatch=2048, nepoch=20, nstepEpoch=128, nstepMixing=64, nstepPostTrain=1024, initDt=0.1, stepPerTraj=40)
nthmc.setup(conf)
loss = lambda action: nthmc.LossFun(action, cCosDiff=0.0, cTopoDiff=1.0, cForce2=0.0, dHmin=0.0, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=0.001)
mcmc = lambda: nthmc.Metropolis(conf, nthmc.LeapFrog(conf, nthmc.U1d2(beta=7.0, beta0=2.0, size=(16,16), transform=ftr.Ident())))
hmctrain.run(conf, mcmc, loss, opt)
