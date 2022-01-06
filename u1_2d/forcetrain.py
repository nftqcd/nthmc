import nthmc, ftr, evolve
import tensorflow as tf
from numpy import inf

class LossFun:
    def __init__(self, action, betaMap=0.0, cNorm2=1.0, cNorm4=1.0, cNorm6=1.0, cNorm8=1.0, cNorm10=0.0, cNorm12=0.0, cNorm14=0.0, cNormInf=1.0):
        tf.print('LossFun init with action', action, summarize=-1)
        self.action = action
        self.plainAction = None
        self.setBetaMap(betaMap)
        self.cNorm2 = cNorm2
        self.cNorm4 = cNorm4
        self.cNorm6 = cNorm6
        self.cNorm8 = cNorm8
        self.cNorm10 = cNorm10
        self.cNorm12 = cNorm12
        self.cNorm14 = cNorm14
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
            ldf2 = tf.reduce_mean(tf.norm(f, ord=2, axis=-1))/self.action.volume**(1./2.)
        if self.cNorm4 == 0:
            ldf4 = 0
        else:
            ldf4 = tf.reduce_mean(tf.norm(f, ord=4, axis=-1))/self.action.volume**(1./4.)
        if self.cNorm6 == 0:
            ldf6 = 0
        else:
            ldf6 = tf.reduce_mean(tf.norm(f, ord=6, axis=-1))/self.action.volume**(1./6.)
        if self.cNorm8 == 0:
            ldf8 = 0
        else:
            ldf8 = tf.reduce_mean(tf.norm(f, ord=8, axis=-1))/self.action.volume**(1./8.)
        if self.cNorm10 == 0:
            ldf10 = 0
        else:
            ldf10 = tf.reduce_mean(tf.norm(f, ord=10, axis=-1))/self.action.volume**(1./10.)
        if self.cNorm12 == 0:
            ldf12 = 0
        else:
            ldf12 = tf.reduce_mean(tf.norm(f, ord=12, axis=-1))/self.action.volume**(1./12.)
        if self.cNorm14 == 0:
            ldf14 = 0
        else:
            ldf14 = tf.reduce_mean(tf.norm(f, ord=14, axis=-1))/self.action.volume**(1./14.)
        if self.cNormInf == 0:
            ldfI = 0
        else:
            ldfI = tf.reduce_mean(tf.norm(f, ord=inf, axis=-1))
        return self.cNorm2*ldf2 + self.cNorm4*ldf4 + self.cNorm6*ldf6 + self.cNorm8*ldf8 + self.cNorm10*ldf10 + self.cNorm12*ldf12 +self.cNorm14*ldf14 + self.cNormInf*ldfI, (ldf2,ldf4,ldf6,ldf8,ldf10,ldf12,ldf14,ldfI)
    def printCallResults(self, res):
        ldf2,ldf4,ldf6,ldf8,ldf10,ldf12,ldf14,ldfI = res
        if self.cNorm2 != 0:
            tf.print('dfnorm2/v:', ldf2, summarize=-1)
        if self.cNorm4 != 0:
            tf.print('dfnorm4/v:', ldf4, summarize=-1)
        if self.cNorm6 != 0:
            tf.print('dfnorm6/v:', ldf6, summarize=-1)
        if self.cNorm8 != 0:
            tf.print('dfnorm8/v:', ldf8, summarize=-1)
        if self.cNorm10 != 0:
            tf.print('dfnorm10/v:', ldf10, summarize=-1)
        if self.cNorm12 != 0:
            tf.print('dfnorm12/v:', ldf12, summarize=-1)
        if self.cNorm14 != 0:
            tf.print('dfnorm14/v:', ldf14, summarize=-1)
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

class LRDecay:
    def __init__(self, halfLife):
        if halfLife<=0:
            raise ValueError(f'LRDecay.halfLife must be positive, got {halfLife}')
        self.decayRate = 0.5**(1/halfLife)
    def __call__(self, opt, *args):
        opt.learning_rate.assign(self.decayRate*opt.learning_rate)

class LRSteps:
    def __init__(self, stepTargets):
        # stepTargets: [(step, targetLR), (step, targetLR), ...]
        self.stepTargets = sorted(stepTargets)
    def __call__(self, opt, step):
        for i in range(len(self.stepTargets)):
            if step<self.stepTargets[i][0]:
                s,t = self.stepTargets[i]
                break
        l = opt.learning_rate
        d = s-step
        opt.learning_rate.assign((l*d+t)/(d+1))

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
def printResults(loss,lossRes,lv,plaqWoT,printWeights=True):
    loss.printCallResults(lossRes)
    tf.print('loss:', lv, summarize=-1)
    tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
    if printWeights:
        tf.print('weights:', loss.action.transform.trainable_weights, summarize=-1)

def trainStep(loss, opt, x, printWeights=True):
    t0 = tf.timestamp()
    lossRes,lv,plaqWoT = trainStep_func(loss, opt, x)
    printResults(loss, lossRes,lv,plaqWoT,printWeights=printWeights)
    dt = tf.timestamp()-t0
    tf.print('# trainStep time:',dt,'sec',summarize=-1)

def initMCMC(conf, action, mcmcFun, weights, transformWeights=None):
    t0 = tf.timestamp()
    mcmcGen = mcmcFun(action)
    x0 = mcmcGen.generate.action.random()
    p0 = mcmcGen.generate.action.randomMom()
    accrand = mcmcGen.generate.action.rng.uniform([x0.shape[0]], dtype=tf.float64)
    mcmcGen(x0, p0, accrand)
    if weights is not None:
        mcmcGen.set_weights(weights)
    if transformWeights is not None:
        mcmcGen.generate.action.transform.load_weights(transformWeights)
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
        tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
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
        tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
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

@tf.function(jit_compile=True)
def transformBackTo_func(x, base_to):
    xNew,_,invIter = base_to.inv(x)
    return xNew, invIter

def transformBackTo(x, base_to):
    t = tf.timestamp()
    xNew, invIter = transformBackTo_func(x, base_to)
    dt = tf.timestamp()-t
    tf.print('# inverse transformation time:',dt,'sec max iter:',invIter,summarize=-1)
    if invIter >= base_to.invMaxIter:
        tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',base_to.invMaxIter, summarize=-1)
    return xNew

def trainWithConfs(conf, loadConf, actionFun, lossFun, opt, epoch=0, loadWeights=None, saveWeights=None, optChange=None):
    # only runs one epoch
    tbegin = tf.timestamp()
    act = actionFun()
    act.changePerEpoch(epoch, conf)
    x = act.random()
    act.transform(x)
    if loadWeights is not None:
        act.transform.load_weights(loadWeights)
    loss = lossFun(act)
    t0 = tf.timestamp()
    tf.print('Initialization took:', t0-tbegin, 'sec', summarize=-1)
    tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
    tf.print('beta:', act.beta, summarize=-1)
    for step in range(conf.nstepEpoch):
        tf.print('# training step:', step, summarize=-1)
        # x is in the target space
        x = loadConf(step)
        xtrain = transformBackTo(x, act.transform)
        trainStep(loss, opt, xtrain, printWeights=False)
        if optChange is not None:
            optChange(opt, step)
        # tf.print('opt.variables():',len(opt.variables()),opt.variables(),summarize=-1)
    dt = tf.timestamp()-t0
    tf.print('-------- done training epoch', epoch,
        'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
    if saveWeights is not None:
        act.transform.save_weights(saveWeights)
    tend = tf.timestamp()
    tf.print('# train time:',tend-tbegin,'sec',summarize=-1)
    return loss

def runInfer(conf, actionFun, mcmcFun, x0=None, saveFile='', saveFreq=0, weights=None, transformWeights=None):
    # If len(saveFile) > 0, save the beginning and end configs to saveFile + '.b{beta}.n{nbatch}.l{x}_{y}.NNNNN'.
    # If saveFreq > 0, save every saveFreq trajectories.
    # Ignores conf.nstepPostTrain
    tbegin = tf.timestamp()
    mcmcGen = initMCMC(conf, actionFun(), mcmcFun, weights, transformWeights)
    if x0 is None:
        x = mcmcGen.generate.action.random()
    else:
        x = transformBackTo(x0, mcmcGen.generate.action.transform)
    tuneSimple = evolve.RegressStepTuner(0.6, conf.trajLength, memoryLength=2, minCount=16)
    tuneFine = evolve.RegressStepTuner(conf.accRate, conf.trajLength, memoryLength=3)
    for epoch in range(conf.nepoch):
        mcmcGen.changePerEpoch(epoch, conf)
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', mcmcGen.generate.action.beta, summarize=-1)
        if conf.nstepMixing>0:
            tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
            t0 = tf.timestamp()
            for step in range(conf.nstepMixing):
                tf.print('# pre inference step with forced acceptance:', step, summarize=-1)
                x = evolve.tuneStep(mcmcGen, x, forceAccept=True)
            dt = tf.timestamp()-t0
            tf.print('-------- done forced acceptance epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
            t0 = tf.timestamp()
            for step in range(conf.nstepMixing):
                tf.print('# pre inference step:', step, summarize=-1)
                x = evolve.tuneStep(mcmcGen, x, stepTuner=tuneSimple)
            dt = tf.timestamp()-t0
            tuneSimple.reset()
            tf.print('-------- done mixing epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        if conf.nconfStepTune>0:
            tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
            t0 = tf.timestamp()
            nstepTune = conf.nconfStepTune//conf.nbatch
            for step in range(nstepTune if nstepTune>16 else 16):
                tf.print('# pre inference tuning step:', step, summarize=-1)
                x = evolve.tuneStep(mcmcGen, x, stepTuner=tuneFine)
            dt = tf.timestamp()-t0
            tuneFine.reset()
            tf.print('-------- done step tuning epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        tf.print('using',mcmcGen.generate.name,'dt',mcmcGen.generate.dt,'step/traj',mcmcGen.generate.stepPerTraj,summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepEpoch):
            if len(saveFile)>0 and step==0:
                evolve.saveConfs(x, f'{saveFile}.b{mcmcGen.generate.action.beta.numpy()}.n{conf.nbatch}.l{x.shape[2]}_{x.shape[3]}.{step:05d}')
            tf.print('# inference step:', step, summarize=-1)
            x = evolve.tuneStep(mcmcGen, x)
            if len(saveFile)>0 and ((saveFreq>0 and (step+1)%saveFreq==0) or step+1==conf.nstepEpoch):
                evolve.saveConfs(x, f'{saveFile}.b{mcmcGen.generate.action.beta.numpy()}.n{conf.nbatch}.l{x.shape[2]}_{x.shape[3]}.{step+1:05d}')
        dt = tf.timestamp()-t0
        tf.print('-------- done inference epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
    dt = tf.timestamp()-tbegin
    tf.print('# train time:',dt,'sec',summarize=-1)
    return x, mcmcGen

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
