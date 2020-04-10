import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=32, nepoch=1, nstepEpoch=1024, initDt=0.4, refreshOpt=False, checkReverse=False)
nthmc.setup(conf)
beta=1.9375

action = nthmc.OneD(beta=beta, transform=nthmc.Ident())
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.37381486643052653,
   beta]))
x1 = nthmc.runInfer(conf, action, loss, weights, x0)

conf = nthmc.Conf(nbatch=32, nepoch=1, nstepEpoch=1024, initDt=0.4, refreshOpt=False, checkReverse=True)
action = nthmc.OneD(beta=beta, transform=nthmc.TransformChain([
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even',distance=2), nthmc.OneDNeighbor(mask='odd',distance=2),
  nthmc.OneDNeighbor(mask='even',distance=4), nthmc.OneDNeighbor(mask='odd',distance=4),
  nthmc.OneDNeighbor(mask='even',distance=8), nthmc.OneDNeighbor(mask='odd',distance=8),
  nthmc.OneDNeighbor(mask='even',distance=16), nthmc.OneDNeighbor(mask='odd',distance=16),
  nthmc.OneDNeighbor(mask='even',distance=32), nthmc.OneDNeighbor(mask='odd',distance=32),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even',distance=2), nthmc.OneDNeighbor(mask='odd',distance=2),
  nthmc.OneDNeighbor(mask='even',distance=4), nthmc.OneDNeighbor(mask='odd',distance=4),
  nthmc.OneDNeighbor(mask='even',distance=8), nthmc.OneDNeighbor(mask='odd',distance=8),
  nthmc.OneDNeighbor(mask='even',distance=16), nthmc.OneDNeighbor(mask='odd',distance=16),
  nthmc.OneDNeighbor(mask='even',distance=32), nthmc.OneDNeighbor(mask='odd',distance=32),
]))
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.11497602727990924,
   -0.43679630202335662,
   -0.43531393550851105,
   0.65179777154399243,
   0.70347772415076748,
   0.85116289572552506,
   0.79473886622749479,
   0.10736445223548315,
   0.11543297379410809,
   -0.013048946360112276,
   -0.053321622254097173,
   -0.0056573417511179983,
   0.0096381787628389334,
   -0.48364042980839506,
   -0.4403052818739091,
   0.61204711948093538,
   0.58896605234125432,
   0.92761507356535189,
   0.80001609145137886,
   0.14445059557920209,
   0.14522471260758876,
   -0.01612720057105629,
   -0.018861062984923068,
   0.018166008562767483,
   0.0039632680885227428,
   beta]))
nthmc.runInfer(conf, action, loss, weights, x1)
