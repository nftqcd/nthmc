import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=128, nepoch=1, nstepEpoch=1024, initDt=0.4, refreshOpt=False, checkReverse=False)
nthmc.setup(conf)
beta=3.5

action = nthmc.OneD(beta=beta, transform=nthmc.Ident())
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.268831031592305,
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
  [0.0035775747302057011,
   1.084299918472821,
   1.1517185401340932,
   1.7866908237787602,
   1.84446634992922,
   1.5897527922274048,
   1.5090234261172555,
   0.40009798582726053,
   0.36719130339695016,
   -0.1045537852602493,
   -0.1291604500546624,
   -0.006726216684042362,
   -0.0083663085334568128,
   0.82851521495902436,
   0.7947825950114511,
   1.7545063582621507,
   1.6907270907398568,
   1.78371238253251,
   1.5816884157341682,
   0.6021640983433787,
   0.5701408394016011,
   -0.0013953386622795566,
   0.010986364163892717,
   2.8098807327356654e-05,
   -0.010142639658214804,
   beta]))
nthmc.runInfer(conf, action, loss, weights, x1)
