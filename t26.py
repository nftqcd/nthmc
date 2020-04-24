import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=2048, nepoch=4, nstepEpoch=2048, nstepMixing=128, initDt=0.5, trainDt=False, refreshOpt=False, nthr=4)
nthmc.setup(conf)
#action = OneD(transforms=[Ident()])
action = nthmc.OneD(beta=3.5, transform=nthmc.TransformChain([
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even',distance=2), nthmc.OneDNeighbor(mask='odd',distance=2),
  nthmc.OneDNeighbor(mask='even',distance=4), nthmc.OneDNeighbor(mask='odd',distance=4),
  nthmc.OneDNeighbor(mask='even',distance=8), nthmc.OneDNeighbor(mask='odd',distance=8),
  nthmc.OneDNeighbor(mask='even',distance=16), nthmc.OneDNeighbor(mask='odd',distance=16),
  nthmc.OneDNeighbor(mask='even',distance=32), nthmc.OneDNeighbor(mask='odd',distance=32),
  nthmc.OneDNeighbor(mask='even',order=2), nthmc.OneDNeighbor(mask='odd',order=2),
  nthmc.OneDNeighbor(mask='even',order=2,distance=2), nthmc.OneDNeighbor(mask='odd',order=2,distance=2),
  nthmc.OneDNeighbor(mask='even',order=2,distance=4), nthmc.OneDNeighbor(mask='odd',order=2,distance=4),
  nthmc.OneDNeighbor(mask='even',order=2,distance=8), nthmc.OneDNeighbor(mask='odd',order=2,distance=8),
  nthmc.OneDNeighbor(mask='even',order=2,distance=16), nthmc.OneDNeighbor(mask='odd',order=2,distance=16),
  nthmc.OneDNeighbor(mask='even',order=2,distance=32), nthmc.OneDNeighbor(mask='odd',order=2,distance=32),
  nthmc.OneDNeighbor(mask='even',order=3), nthmc.OneDNeighbor(mask='odd',order=3),
  nthmc.OneDNeighbor(mask='even',order=3,distance=2), nthmc.OneDNeighbor(mask='odd',order=3,distance=2),
  nthmc.OneDNeighbor(mask='even',order=3,distance=4), nthmc.OneDNeighbor(mask='odd',order=3,distance=4),
  nthmc.OneDNeighbor(mask='even',order=3,distance=8), nthmc.OneDNeighbor(mask='odd',order=3,distance=8),
  nthmc.OneDNeighbor(mask='even',order=3,distance=16), nthmc.OneDNeighbor(mask='odd',order=3,distance=16),
  nthmc.OneDNeighbor(mask='even',order=3,distance=32), nthmc.OneDNeighbor(mask='odd',order=3,distance=32),
  nthmc.OneDNeighbor(mask='even',order=4), nthmc.OneDNeighbor(mask='odd',order=4),
  nthmc.OneDNeighbor(mask='even',order=4,distance=2), nthmc.OneDNeighbor(mask='odd',order=4,distance=2),
  nthmc.OneDNeighbor(mask='even',order=4,distance=4), nthmc.OneDNeighbor(mask='odd',order=4,distance=4),
  nthmc.OneDNeighbor(mask='even',order=4,distance=8), nthmc.OneDNeighbor(mask='odd',order=4,distance=8),
  nthmc.OneDNeighbor(mask='even',order=4,distance=16), nthmc.OneDNeighbor(mask='odd',order=4,distance=16),
  nthmc.OneDNeighbor(mask='even',order=4,distance=32), nthmc.OneDNeighbor(mask='odd',order=4,distance=32),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even',distance=2), nthmc.OneDNeighbor(mask='odd',distance=2),
  nthmc.OneDNeighbor(mask='even',distance=4), nthmc.OneDNeighbor(mask='odd',distance=4),
  nthmc.OneDNeighbor(mask='even',distance=8), nthmc.OneDNeighbor(mask='odd',distance=8),
  nthmc.OneDNeighbor(mask='even',distance=16), nthmc.OneDNeighbor(mask='odd',distance=16),
  nthmc.OneDNeighbor(mask='even',distance=32), nthmc.OneDNeighbor(mask='odd',distance=32),
  nthmc.OneDNeighbor(mask='even',order=2), nthmc.OneDNeighbor(mask='odd',order=2),
  nthmc.OneDNeighbor(mask='even',order=2,distance=2), nthmc.OneDNeighbor(mask='odd',order=2,distance=2),
  nthmc.OneDNeighbor(mask='even',order=2,distance=4), nthmc.OneDNeighbor(mask='odd',order=2,distance=4),
  nthmc.OneDNeighbor(mask='even',order=2,distance=8), nthmc.OneDNeighbor(mask='odd',order=2,distance=8),
  nthmc.OneDNeighbor(mask='even',order=2,distance=16), nthmc.OneDNeighbor(mask='odd',order=2,distance=16),
  nthmc.OneDNeighbor(mask='even',order=2,distance=32), nthmc.OneDNeighbor(mask='odd',order=2,distance=32),
  nthmc.OneDNeighbor(mask='even',order=3), nthmc.OneDNeighbor(mask='odd',order=3),
  nthmc.OneDNeighbor(mask='even',order=3,distance=2), nthmc.OneDNeighbor(mask='odd',order=3,distance=2),
  nthmc.OneDNeighbor(mask='even',order=3,distance=4), nthmc.OneDNeighbor(mask='odd',order=3,distance=4),
  nthmc.OneDNeighbor(mask='even',order=3,distance=8), nthmc.OneDNeighbor(mask='odd',order=3,distance=8),
  nthmc.OneDNeighbor(mask='even',order=3,distance=16), nthmc.OneDNeighbor(mask='odd',order=3,distance=16),
  nthmc.OneDNeighbor(mask='even',order=3,distance=32), nthmc.OneDNeighbor(mask='odd',order=3,distance=32),
  nthmc.OneDNeighbor(mask='even',order=4), nthmc.OneDNeighbor(mask='odd',order=4),
  nthmc.OneDNeighbor(mask='even',order=4,distance=2), nthmc.OneDNeighbor(mask='odd',order=4,distance=2),
  nthmc.OneDNeighbor(mask='even',order=4,distance=4), nthmc.OneDNeighbor(mask='odd',order=4,distance=4),
  nthmc.OneDNeighbor(mask='even',order=4,distance=8), nthmc.OneDNeighbor(mask='odd',order=4,distance=8),
  nthmc.OneDNeighbor(mask='even',order=4,distance=16), nthmc.OneDNeighbor(mask='odd',order=4,distance=16),
  nthmc.OneDNeighbor(mask='even',order=4,distance=32), nthmc.OneDNeighbor(mask='odd',order=4,distance=32),
]))
loss = nthmc.LossFun(action, cCosDiff=0.1, cTopoDiff=1.0, dHmin=0.0, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=0.0001)
x0 = action.initState(conf.nbatch)
nthmc.run(conf, action, loss, opt, x0)
