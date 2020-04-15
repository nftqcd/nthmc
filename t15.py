import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=2048, nepoch=1, nstepEpoch=256, nstepMixing=64, initDt=0.4, refreshOpt=False, nthr=8)
beta=1.625
nthmc.setup(conf)

action = nthmc.OneD(beta=beta, transform=nthmc.Ident())
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
  [0.3737090001953306,
   beta]))
tf.print('-------- pretrain mixing --------')
x0 = nthmc.runInfer(conf, action, loss, weights, x0, detail=False)
tf.print('-------- done pretrain mixing --------')

conf.nstepEpoch=2048
conf.nstepMixing=0
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
opt = tk.optimizers.Adam(learning_rate=5E-5)

weights=list(map(lambda x:tf.constant(x,dtype=tf.float64), [
 0.39928005894476953,
 -0.16646589446724119,
 -0.165116196190377,
 0.030407332523959697,
 0.030213236259768468,
 0.079470890222058513,
 0.0761346381697804,
 0.029619192505227931,
 0.030915611020612837,
 0.00403555847393147,
 0.00407719851568374,
 -0.00060822007493423636,
 0.0037353011339751178,
 0.069686089040409807,
 0.070473588467025811,
 0.033146255849164606,
 0.033379928079238383,
 -0.0029161974044230022,
 -0.0017224631344893938,
 -0.00069061113081232792,
 -0.0016410929512909317,
 0.0016876364859234507,
 -0.000733623769599814,
 0.0014529279510181758,
 -0.00091449778170147266,
 -0.019901824910881289,
 -0.017959584894213086,
 -0.0059090578292857058,
 -0.0054266495233532761,
 0.0013726690186972,
 0.00021210992451173647,
 -0.0001498695177544983,
 0.00064305655082401761,
 0.0010931278372980787,
 0.00037689345534901728,
 -0.0014984995098818561,
 -0.00040476075088637781,
 0.0046935831026250876,
 0.0032850096553108288,
 -0.00054541015203022974,
 -0.0014208086412517168,
 -0.0002359329393992865,
 -0.00035542688976354463,
 -1.2157678571547889e-05,
 0.00015490831515802204,
 -0.00076950136336040114,
 -0.00031333861450947426,
 5.097857409197952e-05,
 -0.00012148501847680332,
 -0.16518081785315231,
 -0.16337905450177662,
 0.035184121942295171,
 0.034570717385232527,
 0.080465773703933,
 0.0774896127221109,
 0.02912121009107339,
 0.030940522095703058,
 0.0043964429072142538,
 0.0040451007928214251,
 -0.00080468042839712994,
 0.0035457375499732395,
 0.06101007963274057,
 0.061368775130318916,
 0.042444107322532766,
 0.0429949487047859,
 -0.0027232705295604813,
 -0.0012932981224013512,
 -0.000984564284924616,
 -0.0024456764643747803,
 0.0015834011617584004,
 -0.00090531730999972814,
 0.0017613431423082497,
 -0.0012386881834937134,
 -0.023626271538814435,
 -0.021598075508490612,
 -0.012897707141515927,
 -0.012881432717533042,
 0.0014793362615386902,
 9.2105145307772054e-06,
 -0.00020941704974683913,
 0.00023779728215206694,
 0.0014388740734254534,
 0.00038662450216112368,
 -0.0012415944776245824,
 -5.7876896633756865e-05,
 0.00847176568981238,
 0.00680656254828831,
 0.0038699954560532414,
 0.002672203307567224,
 -0.00032310477908741877,
 -0.00027817807890187128,
 2.9749369975343604e-07,
 0.00056912541337158064,
 -0.00016832076473673023,
 -6.8163634028702889e-05,
 0.00038894121879160768,
 0.00021929053651325786,
 beta]))
nthmc.run(conf, action, loss, opt, x0, weights=weights, requireInv=True)
