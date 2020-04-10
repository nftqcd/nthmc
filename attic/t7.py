import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=1024, nepoch=16, nstepEpoch=2048, initDt=0.4, refreshOpt=False)
nthmc.setup(conf)
#action = OneD(transforms=[Ident()])
action = nthmc.OneD(beta=6, transforms=[
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd'),
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd')])
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=10.0, dHmin=0.5, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=0.001)
x0 = action.initState(conf.nbatch)
nthmc.run(conf, action, loss, opt, x0)
