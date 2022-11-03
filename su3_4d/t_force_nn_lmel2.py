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
        dense0Unit=8, shift=1, attnHeads=2, attnKey=4, dense1Unit=8, ffsize=16, denseOut=6,
        chainLen=1,
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
    tf.print('shift:',shift)
    tf.print('attnHeads:',attnHeads)
    tf.print('attnKey:',attnKey)
    tf.print('dense1Unit:',dense1Unit)
    tf.print('ffsize:',ffsize)
    tf.print('denseOut:',denseOut)
    tf.print('chainLen:',chainLen)
    tf.print('lrInit:',lrInit)
    tf.print('lrSteps:',lrSteps)
    if shift>0 and attnHeads>0 and chainLen>1:
        raise ValueError('Not enough memory for the hyperparameter set')
    confLoader = dataset.TrainLoader(nbatch=nbatch, batch_size=batch_size)
    validateLoader = dataset.ValidateLoader(nbatch=nbatchValidate, batch_size=batch_size)
    tf.print('Training set',confLoader.set,summarize=-1)
    tf.print('Validation set',validateLoader.set,summarize=-1)
    transformedAct = action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        tl.Dense(units=dense0Unit, activation='swish'),
                        transform.SymmetricShifts(symmetric_shifts=shift),
                        transform.Residue(transform.LocalSelfAttention(num_heads=attnHeads,key_dim=attnKey)) if attnHeads>0 else lambda x:x,
                        transform.FlattenSiteLocal(input_local_rank=2),
                        tl.Dense(units=dense1Unit, activation='swish'),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=ffsize, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=denseOut, activation=None)]),
                    dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4) for _ in range(chainLen)]),
            action=action.SU3d4(beta=beta, c1=action.C1DBW2))
    target = action.TransformedActionVectorFromMatrixBase(
            transform=transform.Ident(),
            action=action.SU3d4(beta=targetBeta, c1=action.C1DBW2))
    transformedAct(confLoader(0))
    dt = tf.timestamp()-t0
    tf.print('# initialization time:',dt,'sec')
    trainer = forcetrain.Trainer(
        loss=forcetrain.LMEl2Loss(),
        opt=tk.optimizers.Adam(learning_rate=0.000001),
        transformedAction=transformedAct,
        targetAction=target,
        nepoch=nepoch)
    trainer(confLoader, validateLoader=validateLoader, validateFrequency=8, saveWeights='weights',
        optChange=forcetrain.LRSteps([(nepoch*nbatch*f,lr) for f,lr in lrSteps], epochSteps=nbatch))
    tf.print('# Total time:',tf.timestamp()-t0,'sec')

if __name__=='__main__':
    run()
