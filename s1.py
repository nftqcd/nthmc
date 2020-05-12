import tensorflow as tf
import tensorflow.keras as tk
import nthmc

conf = nthmc.Conf(nbatch=1, nepoch=1, nstepEpoch=1024, nstepMixing=64, stepPerTraj = 10,
  initDt=0.4, refreshOpt=False, checkReverse=False, nthr=4)
nthmc.setup(conf)
beta=6.0

action = nthmc.OneD(beta=beta, transform=nthmc.Ident())
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.0, topoFourierN=1)
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
 [0.20047195332489751,
  beta]))
nthmc.showTransform(conf, action, loss, weights)

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
loss = nthmc.LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.0, topoFourierN=1)

beta=6.0
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
 # 02f:"cy$@c:r!awk -v beta=6.0 '/^beta: /{b=$2} p>0{w=w "\n" $0} b==beta&&/^weights: /{p=1;w=$0} p==1&&/]$/{p=0} END{print w}' t28.log
 [0.31095081795527452,
 -0.31990243517223638,
 -0.32109470828449777,
 0.00061345103106714928,
 -0.0014288837697507305,
 0.044385788985282018,
 0.04739037868609252,
 0.030358291179336333,
 0.021905379270682833,
 0.0038329889950628466,
 0.00041170927229429493,
 0.0054020647322860554,
 0.0022722036434059895,
 0.17275753175664138,
 0.17369996388805223,
 0.031055318185354733,
 0.026101611944405891,
 -0.023682589223374958,
 -0.029082838949610015,
 -0.01155545173109941,
 -0.0049888467491186949,
 0.0012223783891987397,
 -0.0027864989912063025,
 -0.00031668966420504555,
 -0.0025950147861293804,
 -0.0191200076393723,
 -0.018360283547626928,
 -0.0008892057675957514,
 0.0011183421081395824,
 0.00908718584563754,
 0.0079555636549820275,
 0.0027707297184971411,
 -9.0923509933336954e-05,
 0.00022561579362373612,
 -0.0016363095621011938,
 0.0023224255253234869,
 -0.0015588467362802466,
 0.011192337723435811,
 0.01001928900967989,
 -0.0020209909440646904,
 -0.0038184474653411814,
 -0.0012084607345820985,
 -0.0026907165336564403,
 0.0011421252742597051,
 -0.00080824431530238149,
 0.001791035598967798,
 0.0014856840742477648,
 0.00013125540495647592,
 0.00089364934054835064,
 -0.32118506455984025,
 -0.32278639780010793,
 0.02201193164572543,
 0.01847830215045701,
 0.057868016994201604,
 0.057961777416760368,
 0.031801502531235981,
 0.025875727600236865,
 0.0048232000890446724,
 0.000350707193304461,
 0.0031772671473503187,
 0.003938131166258135,
 0.12954009355818141,
 0.12873408258744853,
 0.064697650862689859,
 0.059275173056005478,
 -0.027113534565581736,
 -0.035027107076422641,
 -0.016766235243913494,
 -0.0090755405725385,
 0.0015868604632436776,
 -0.0055585159089984668,
 0.00080978873896573927,
 -0.0015156117022210669,
 -0.055199474854345822,
 -0.053896634221432628,
 -0.0363622130823767,
 -0.03260865283701795,
 0.0090064171950468962,
 0.0061976634512000009,
 0.0030965148747710762,
 -0.0042718714973962119,
 -0.0002045477947971253,
 0.0016792656339733915,
 0.0010079415242270158,
 -0.000665126246783663,
 0.0194533161508012,
 0.014387919320054194,
 0.0085723711227567688,
 0.0082458698714821359,
 -0.0033886220204170114,
 -0.0023383752668644617,
 0.0026611607382027104,
 0.00040529789179773678,
 0.00051303784018107092,
 -0.00017297230444901615,
 0.0019364894410578641,
 -0.0020016468715906323,
 beta]))
tf.print('beta: ',beta)
nthmc.showTransform(conf, action, loss, weights)
