import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=2048, nepoch=1, nstepEpoch=2048, nstepMixing=64, initDt=0.4, refreshOpt=False, nthr=8)
beta=3.5
nthmc.setup(conf)
action = nthmc.OneD(beta=beta, transform=nthmc.TransformChain([
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
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.5, topoFourierN=1)
opt = tk.optimizers.Adam(learning_rate=1E-5)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64), [
 0.426161809940765,
 -0.320109120400013,
 -0.3209002024382495,
 -0.03118271698489185,
 -0.036169773339796464,
 0.055714318919392686,
 0.057602389890724234,
 0.029411886986087127,
 0.02048733243498738,
 0.0009483945522790476,
 -0.003336858749749962,
 0.004283181019440162,
 0.0055589091837478805,
 0.1523380013134244,
 0.15163036003180105,
 0.017450942775123303,
 0.01366963403033924,
 -0.015362176729137129,
 -0.023842410298148348,
 -0.007731245793489482,
 -0.0013628219442876222,
 0.0011295376199805572,
 -0.0009141005452412725,
 -0.0005934186447350823,
 0.0025111964348351304,
 -0.016444424617664447,
 -0.015570829270105238,
 0.0019647033660882846,
 0.005939361346840814,
 0.006460016703292643,
 0.004736273804986227,
 0.0022333630983046664,
 -0.0011657888127998832,
 0.00019669260733786145,
 -0.0030779286401902473,
 0.002774947111944009,
 -9.643393833526736e-05,
 0.0083785133367789,
 0.0053008391565818914,
 -0.0014080778872983919,
 -0.0024396905236594682,
 -0.0015531026667714104,
 -0.0015796761344081557,
 -0.001253733487886692,
 -0.0015042727436904697,
 0.0011413533343287735,
 0.0009722780451509098,
 -0.00046677598847423714,
 0.0006355633832931227,
 -0.32071868062103076,
 -0.3214818015929604,
 -0.00986116406882059,
 -0.017335584106134748,
 0.06802936969063668,
 0.06691802024265854,
 0.030819349510999603,
 0.023206203501044503,
 0.0017779135561217525,
 -0.0034133032476216588,
 0.002189343578032792,
 0.00656004530207795,
 0.11256550758203428,
 0.11055222402865708,
 0.049446153758141626,
 0.04565898588776925,
 -0.01758171549794033,
 -0.026933901536123416,
 -0.011986081801134148,
 -0.0048059039456269485,
 0.0017878663762805563,
 -0.0025517310832571327,
 0.00019610673621250042,
 0.003797903258295098,
 -0.04866943996936729,
 -0.04588564019763426,
 -0.030946502446712494,
 -0.02598814368018486,
 0.005873979914149713,
 0.004419541888295364,
 0.0029309881330323194,
 -0.004230773448561739,
 -0.000379102785780568,
 -0.0004200660801947094,
 -0.000890702512832992,
 -0.0015533078274466545,
 0.018431797429963044,
 0.01296582266989706,
 0.008373080763779048,
 0.0071470949531473186,
 -0.0006280677552497352,
 0.0008691134144185065,
 -0.00011310686430592162,
 0.001019738436482968,
 -0.0004266479170588166,
 -0.0006059400331239689,
 8.359503352565366e-05,
 -0.0007053316682491896,
 beta]))
nthmc.run(conf, action, loss, opt, x0, weights=weights)
