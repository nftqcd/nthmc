import tensorflow as tf
import tensorflow.keras.layers as tl
import tensorflow.keras as tk
if __name__=='__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.normpath(path.join(path.dirname(__file__),'../..')))
    __package__ = 'nthmc.su3_4d'
from . import dataset
from ..lib import action, gauge, transform, nthmc, forcetrain

# a/r₀, Eq (4.11), S. Necco, Nucl. Phys. B683, 137 (2004)
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7796
# 0.40001
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7383
# 0.499952
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7099
# 0.599819
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.6716
# 0.800164
def run(beta=0.7796, targetBeta=0.7099, nbatch=64, nbatchValidate=1, batch_size=1, nepoch=16,
        dense0Unit=8, denseOut=6,
        dimChainLen=1, chainLen=1,
        lrInit=0.000001, lrSteps=[(1/4, 0.02), (7/8, 0.02), (1, 0.0)]):
    nthmc.setup(nthmc.Conf())
    t0 = tf.timestamp()
    tf.print('beta:',beta)
    tf.print('targetBeta:',targetBeta)
    tf.print('nbatch:',nbatch)
    tf.print('nbatchValidate:',nbatchValidate)
    tf.print('batch_size:',batch_size)
    tf.print('nepoch:',nepoch)
    tf.print('dense0Unit:',dense0Unit)
    tf.print('denseOut:',denseOut)
    tf.print('dimChainLen:',dimChainLen)
    tf.print('chainLen:',chainLen)
    tf.print('lrInit:',lrInit)
    tf.print('lrSteps:',lrSteps)
    testConf = gauge.readGauge('../../config.dbw2_8t16_b0.7796.m0.lime')
    transformedAct = action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        tl.Dense(units=dense0Unit, activation='swish'),
                        transform.Normalization(),

                        tl.Dense(units=denseOut, activation=None)]),
                    dir=i, is_odd=eo)
                 for _ in range(chainLen) for i in range(4) for _ in range(dimChainLen) for eo in {False,True}]),
            action=action.SU3d4(beta=beta, c1=action.C1DBW2))
    target = action.TransformedActionVectorFromMatrixBase(
            transform=transform.Ident(),
            action=action.SU3d4(beta=targetBeta, c1=action.C1DBW2))
    transformedAct(testConf)
    transformedAct.transform.load_weights('weights')
    for t in transformedAct.transform.chain:
        # t is StoutSmearSlice here with CoefficientNets
        tf.print('coeff dir:',t.dir,'odd' if t.is_odd else 'even')
    dt = tf.timestamp()-t0
    tf.print('# initialization time:',dt,'sec')
    ti = tf.timestamp()
    mappedConf,lndet,iter = transformedAct.transform.inv(testConf)
    tf.print('lndet:',lndet)
    tf.print('iter:',iter)
    tf.print('# inv time:', tf.timestamp()-ti, 'sec')
    ti = tf.timestamp()
    ft,lndet,cs = transformedAct.gradient(mappedConf)
    tf.print('lndet:',lndet)
    tf.print('cs:',cs,summarize=-1)
    tf.print('# transformedAct.gradient time:', tf.timestamp()-ti, 'sec')
    ti = tf.timestamp()
    f0,_,_ = target.gradient(mappedConf)
    tf.print('# target.gradient time:', tf.timestamp()-ti, 'sec')
    ti = tf.timestamp()
    loss = forcetrain.LMEl2Loss()
    lv,res = loss(ft,f0)
    tf.print('loss:', lv)
    loss.printCallResults(res)
    tf.print('# loss time:', tf.timestamp()-ti, 'sec')
    tf.print('# Total time:',tf.timestamp()-t0,'sec')

if __name__=='__main__':
    run()
