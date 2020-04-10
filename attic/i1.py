import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=32, nepoch=1, nstepEpoch=32*1024, initDt=0.4, refreshOpt=False, checkReverse=False)
nthmc.setup(conf)
beta=1.9375
action = nthmc.OneD(beta=beta, transforms=[nthmc.Ident()])
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.37381486643052653,
   beta]))
nthmc.runInfer(conf, action, loss, weights, x0)
