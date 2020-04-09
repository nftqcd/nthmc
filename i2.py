import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=32, nepoch=1, nstepEpoch=1024, initDt=0.4, refreshOpt=False, checkReverse=False)
nthmc.setup(conf)
beta=1.9375

action = nthmc.OneD(beta=beta, transforms=[nthmc.Ident()])
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.37381486643052653,
   beta]))
x1 = nthmc.runInfer(conf, action, loss, weights, x0)

conf = nthmc.Conf(nbatch=32, nepoch=1, nstepEpoch=1024, initDt=0.4, refreshOpt=False, checkReverse=True)
action = nthmc.OneD(beta=beta, transforms=[
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
])
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.0939129063675943,
   -1.0069582001526891,
   -0.98578897150452327,
   0.58601096871733971,
   0.63773743108828174,
   0.67372626670660918,
   0.604494550139737,
   -0.29201241032522657,
   -0.34871955246881359,
   -0.27641210577142583,
   -0.32063167414646221,
   -0.1943108665025394,
   -0.26210792555982915,
   -1.0635526243856992,
   -1.0302940098964493,
   0.90709268937518528,
   0.82812192327363887,
   0.63831143496833609,
   0.47218304196084754,
   0.010947298119975208,
   0.015418726038448363,
   -0.098937911431820569,
   -0.093123389412637564,
   -0.13405518193588081,
   -0.15697499994193129,
   beta]))
nthmc.runInfer(conf, action, loss, weights, x1)
