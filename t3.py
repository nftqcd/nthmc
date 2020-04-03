import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=256, nepoch=16, nstepEpoch=4096, initDt=0.4, refreshOpt=False)
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
  nthmc.OneDNeighbor(mask='even'), nthmc.OneDNeighbor(mask='odd')])
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.5, topoFourierN=9)
opt = tk.optimizers.Adam(learning_rate=0.001)
x0 = action.initState(conf.nbatch)
nthmc.run(conf, action, loss, opt, x0)
