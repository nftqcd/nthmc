import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=2048, nepoch=32, nstepEpoch=1024, initDt=0.4, refreshOpt=False, nthr=8)
nthmc.setup(conf)
#action = OneD(transforms=[Ident()])
action = nthmc.OneD(beta=6, transform=nthmc.TransformChain([
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
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=100.0, dHmin=0.5, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=0.001)
x0 = action.initState(conf.nbatch)
nthmc.run(conf, action, loss, opt, x0)
