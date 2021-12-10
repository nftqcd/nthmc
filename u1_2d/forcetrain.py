import nthmc, ftr, evolve
import tensorflow as tf
from numpy import inf

class LossFun:
    def __init__(self, action, betaMap=0.0, cNorm2=1.0, cNormInf=1.0):
        tf.print('LossFun init with action', action, summarize=-1)
        self.action = action
        self.plainAction = None
        self.setBetaMap(betaMap)
        self.cNorm2 = cNorm2
        self.cNormInf = cNormInf
    def __call__(self, x):
        f,_,_ = self.action.derivAction(x)
        if self.plainAction is not None:
            f0,_,_ = self.plainAction.derivAction(x)
            f -= f0
        f = tf.reshape(f, [f.shape[0],-1])
        if self.cNorm2 == 0:
            ldf2 = 0
        else:
            ldf2 = tf.reduce_mean(tf.norm(f, ord=2, axis=-1))/self.action.volume
        if self.cNormInf == 0:
            ldfI = 0
        else:
            ldfI = tf.reduce_mean(tf.norm(f, ord=inf, axis=-1))
        return self.cNorm2*ldf2+self.cNormInf*ldfI, (ldf2,ldfI)
    def printCallResults(self, res):
        ldf2,ldfI = res
        if self.cNorm2 != 0:
            tf.print('dfnorm2/v:', ldf2, summarize=-1)
        if self.cNormInf != 0:
            tf.print('dfnormInf:', ldfI, summarize=-1)
    def setBetaMap(self, betaMap):
        if betaMap>0:
            if self.plainAction is not None:
                self.plainAction.beta.assign(betaMap)
                self.plainAction.beta0 = betaMap
            else:
                self.plainAction = self.action.__class__(
                    transform=ftr.Ident(),
                    nbatch=self.action.shape[0],
                    rng=self.action.rng.split()[0],
                    beta=betaMap,
                    beta0=betaMap,
                    size=self.action.size,
                    name=self.action.name+':loss')
        else:
            self.plainAction = None

@tf.function(jit_compile=True)
def inferStep_func(mcmc, loss, x0, p0, accrand, print, detail, forceAccept):
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0, accrand)
    if loss is not None:
        lv, lossRes = loss(x)
    else:
        lv = None
        lossRes = None
    if print:
        inferRes = (lv,lossRes) + nthmc.packMCMCRes(mcmc, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, detail)
    else:
        inferRes = None
    if forceAccept:
        return x1,inferRes
    else:
        return x,inferRes

@tf.function
def printInferResults(lv,lossRes,v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq,topo, loss=None):
    nthmc.printMCMCRes(v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq,topo)
    if loss is not None:
        loss.printCallResults(lossRes)
        tf.print('loss:', lv, summarize=-1)

def inferStep(mcmc, loss, x0, print=True, detail=True, forceAccept=False):
    t0 = tf.timestamp()
    p0 = mcmc.generate.action.randomMom()
    accrand = mcmc.generate.action.rng.uniform([x0.shape[0]], dtype=tf.float64)
    x,inferRes = inferStep_func(mcmc, loss, x0, p0, accrand, print, detail, tf.constant(forceAccept))
    printInferResults(*inferRes, loss)
    dt = tf.timestamp()-t0
    tf.print('# inferStep time:',dt,'sec',summarize=-1)
    return x

def initRun(mcmc, loss, x0, weights):
    tf.print('# run once and set weights')
    t0 = tf.timestamp()
    inferStep(mcmc, loss, x0, print=False)
    mcmc.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# finished autograph run in',dt,'sec',summarize=-1)

def infer(conf, mcmc, loss, weights, x0, detail=True):
    initRun(mcmc, loss, x0, weights)
    x, _, invIter = mcmc.generate.action.transform.inv(x0)
    if invIter >= mcmc.generate.action.transform.invMaxIter:
        tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',mcmc.generate.action.transform.invMaxIter, summarize=-1)
    for epoch in range(conf.nepoch):
        mcmc.changePerEpoch(epoch, conf)
        tf.print('weightsAll:', mcmc.get_weights(), summarize=-1)
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', mcmc.generate.action.beta, summarize=-1)
        for step in range(conf.nstepMixing):
            tf.print('# mixing step:', step, summarize=-1)
            x = inferStep(mcmc, loss, x, detail=False, forceAccept=True)
        dt = tf.timestamp()-t0
        tf.print('-------- done mixing epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepEpoch):
            tf.print('# traj:', step, summarize=-1)
            x = inferStep(mcmc, loss, x, detail=detail)
        dt = tf.timestamp()-t0
        tf.print('-------- end epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
    return x

@tf.function(jit_compile=True)
def trainStep_func(loss, opt, x):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(loss.action.transform.trainable_weights)
        lv, lossRes = loss(x)
    grads = tape.gradient(lv, loss.action.transform.trainable_weights)
    opt.apply_gradients(zip(grads, loss.action.transform.trainable_weights))
    plaqWoT = loss.action.plaquetteWoTrans(x)
    return lossRes,lv,plaqWoT

@tf.function
def printResults(loss,lossRes,lv,plaqWoT):
    loss.printCallResults(lossRes)
    tf.print('loss:', lv, summarize=-1)
    tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
    tf.print('weights:', loss.action.transform.trainable_weights, summarize=-1)

def trainStep(loss, opt, x):
    t0 = tf.timestamp()
    lossRes,lv,plaqWoT = trainStep_func(loss, opt, x)
    printResults(loss, lossRes,lv,plaqWoT)
    dt = tf.timestamp()-t0
    tf.print('# trainStep time:',dt,'sec',summarize=-1)

def initMCMC(conf, action, mcmcFun, weights):
    t0 = tf.timestamp()
    mcmcGen = mcmcFun(action)
    x0 = mcmcGen.generate.action.random()
    p0 = mcmcGen.generate.action.randomMom()
    accrand = mcmcGen.generate.action.rng.uniform([x0.shape[0]], dtype=tf.float64)
    mcmcGen(x0, p0, accrand)
    if weights is not None:
        mcmcGen.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# initMCMC time:',dt,'sec',summarize=-1)
    return mcmcGen

@tf.function(jit_compile=True)
def rebaseFromTo_func(x, base_from, base_to):
    xtarget,_,_ = base_from(x)
    xNew,_,invIter = base_to.inv(xtarget)
    return xNew, invIter

def rebaseFromTo(x, base_from, base_to):
    t = tf.timestamp()
    xNew, invIter = rebaseFromTo_func(x, base_from, base_to)
    dt = tf.timestamp()-t
    tf.print('# rebase transformation time:',dt,'sec max iter:',invIter,summarize=-1)
    if invIter >= base_to.invMaxIter:
        tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',base_to.invMaxIter, summarize=-1)
    return xNew

def train(conf, actionFun, mcmcFun, lossFun, opt, weights=None):
    tbegin = tf.timestamp()
    mcmcGen = initMCMC(conf, actionFun(), mcmcFun, weights)
    x = mcmcGen.generate.action.random()
    loss = lossFun(actionFun())
    loss.action.transform(x)
    loss.action.transform.set_weights(mcmcGen.generate.action.transform.get_weights())
    tuneSimple = evolve.RegressStepTuner(0.6, conf.trajLength, memoryLength=2, minCount=16)
    tuneFine = evolve.RegressStepTuner(conf.accRate, conf.trajLength, memoryLength=3)
    for epoch in range(conf.nepoch):
        mcmcGen.changePerEpoch(epoch, conf)
        loss.action.changePerEpoch(epoch, conf)
        if conf.refreshOpt and len(opt.variables())>0:
            tf.print('# reset optimizer')
            for var in opt.variables():
                var.assign(tf.zeros_like(var))
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', loss.action.beta, summarize=-1)
        for step in range(conf.nstepMixing):
            tf.print('# pre-training inference step with forced acceptance:', step, summarize=-1)
            x = evolve.tuneStep(mcmcGen, x, forceAccept=True)
        dt = tf.timestamp()-t0
        tf.print('-------- done forced acceptance epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepMixing):
            tf.print('# pre-training inference step:', step, summarize=-1)
            x = evolve.tuneStep(mcmcGen, x, stepTuner=tuneSimple)
        dt = tf.timestamp()-t0
        tuneSimple.reset()
        tf.print('-------- done mixing epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        if conf.nconfStepTune>0:
            t0 = tf.timestamp()
            nstepTune = conf.nconfStepTune//conf.nbatch
            for step in range(nstepTune if nstepTune>16 else 16):
                tf.print('# pre-training tuning step:', step, summarize=-1)
                x = evolve.tuneStep(mcmcGen, x, stepTuner=tuneFine)
            dt = tf.timestamp()-t0
            tuneFine.reset()
            tf.print('-------- done step tuning epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepEpoch):
            tf.print('# training inference step:', step, summarize=-1)
            x = evolve.tuneStep(mcmcGen, x)
            tf.print('# training step:', step, summarize=-1)
            # x is in the mapped space of mcmcGen
            xtrain = rebaseFromTo(x, mcmcGen.generate.action.transform, loss.action.transform)
            trainStep(loss, opt, xtrain)
            # tf.print('opt.variables():',len(opt.variables()),opt.variables(),summarize=-1)
        dt = tf.timestamp()-t0
        tf.print('-------- done training epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
        x = rebaseFromTo(x, mcmcGen.generate.action.transform, loss.action.transform)
        mcmcGen.generate.action.transform.set_weights(loss.action.transform.get_weights())
        if conf.nstepPostTrain>0:
            t0 = tf.timestamp()
            for step in range(conf.nstepPostTrain):
                tf.print('# post-training inference step:', step, summarize=-1)
                x = inferStep(mcmcGen, loss, x, detail=False)
            dt = tf.timestamp()-t0
            tf.print('-------- done post-training epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepPostTrain, 'sec/step --------', summarize=-1)
    dt = tf.timestamp()-tbegin
    tf.print('# train time:',dt,'sec',summarize=-1)
    return x, mcmcGen, loss

def run(conf, actionFun, mcmcFun, lossFun, opt, weights=None):
    x, mcmc, loss = train(conf, actionFun, mcmcFun, lossFun, opt, weights=weights)
    tf.print('finalWeightsAll:', mcmc.get_weights(), summarize=-1)
    return x, mcmc, loss

if __name__ == '__main__':
    import ftr
    import tensorflow.keras as tk
    conf = nthmc.Conf(nbatch=4, nepoch=2, nstepEpoch=8, stepPerTraj=4, trainDt=False, trajLength=4.)
    nthmc.setup(conf)
    op0 = (((1,2,-1,-2), (1,-2,-1,2)),
           ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
    # requires different coefficient bounds:
    # (1,2,-1,-2,1,-2,-1,2)
    # (1,2,-1,-2,1,2,-1,-2)
    # (1,-2,-1,2,1,-2,-1,2)
    op1 = (((2,-1,-2,1), (2,1,-2,-1)),
           ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
    fixedP = (1,2,-1,-2)
    convP0 = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    convP1 = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    conv = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    transform = lambda: ftr.TransformChain([
        ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0())], conv()),
        ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0())], conv()),
        ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0())], conv()),
        ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0())], conv()),
        ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1())], conv()),
        ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1())], conv()),
        ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1())], conv()),
        ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1())], conv()),
    ])
    ftr.checkDep(transform())
    rng = tf.random.Generator.from_seed(conf.seed)
    actionFun = lambda: nthmc.U1d2(transform(), conf.nbatch, rng.split()[0])
    lossFun = lambda action: LossFun(action, betaMap=2.5, cNorm2=1.0, cNormInf=1.0)
    mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))
    opt = tk.optimizers.Adam(learning_rate=0.001)
    x, mcmc, loss = run(conf, actionFun, mcmcFun, lossFun, opt)
    x, _, _ = mcmc.generate.action.transform(x)
    infer(conf, mcmc, loss, mcmc.get_weights(), x)
