import nthmc, ftr
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
    def __call__(self, x, print=True):
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
        if print:
            if self.cNorm2 != 0:
                tf.print('dfnorm2/v:', ldf2, summarize=-1)
            if self.cNormInf != 0:
                tf.print('dfnormInf:', ldfI, summarize=-1)
        return self.cNorm2*ldf2+self.cNormInf*ldfI
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

@tf.function
def inferStep(mcmc, loss, x0, print=True, detail=True, forceAccept=False, tuningStepSize=False):
    p0 = mcmc.generate.action.randomMom()
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0)
    if loss is not None:
        lv = loss(x, print=print)
    if print:
        plaqWoT = mcmc.generate.action.plaquetteWoTrans(x)
        plaq = mcmc.generate.action.plaquette(x)
        dp2 = tf.math.reduce_mean(tf.math.squared_difference(p1,p0), axis=range(1,len(p0.shape)))
        if detail:
            if loss is not None:
                tf.print('loss:', lv, summarize=-1)
            tf.print('V-old:', v0, summarize=-1)
            tf.print('T-old:', t0, summarize=-1)
            tf.print('V-prp:', v1, summarize=-1)
            tf.print('T-prp:', t1, summarize=-1)
            tf.print('dp2:', dp2, summarize=-1)
            tf.print('force:', f2s, fms, summarize=-1)
            tf.print('coeff:', bs, summarize=-1)
            tf.print('lnJ:', ls, summarize=-1)
            tf.print('dH:', dH, summarize=-1)
            tf.print('exp_mdH:', tf.exp(-dH), summarize=-1)
            tf.print('arand:', arand, summarize=-1)
            tf.print('accept:', acc, summarize=-1)
            tf.print('plaqWoTrans:', plaqWoT, summarize=-1)
            tf.print('plaq:', plaq, summarize=-1)
            tf.print('topo:', mcmc.generate.action.topoCharge(x), summarize=-1)
        else:
            if loss is not None:
                tf.print('loss:', lv, summarize=-1)
            tf.print('V-old:', tf.reduce_mean(v0), summarize=-1)
            tf.print('T-old:', tf.reduce_mean(t0), summarize=-1)
            tf.print('V-prp:', tf.reduce_mean(v1), summarize=-1)
            tf.print('T-prp:', tf.reduce_mean(t1), summarize=-1)
            tf.print('dp2:', tf.reduce_mean(dp2), summarize=-1)
            tf.print('force:', tf.reduce_mean(f2s), tf.reduce_min(f2s), tf.reduce_max(f2s), tf.reduce_mean(fms), tf.reduce_min(fms), tf.reduce_max(fms), summarize=-1)
            if len(bs.shape)>1:
                tf.print('coeff:', tf.reduce_mean(bs, axis=(0,-1)), summarize=-1)
            tf.print('lnJ:', tf.reduce_mean(ls), tf.reduce_min(ls), tf.reduce_max(ls), summarize=-1)
            tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
            tf.print('exp_mdH:', tf.reduce_mean(tf.exp(-dH)), summarize=-1)
            tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
            tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
            tf.print('plaq:', tf.reduce_mean(plaq), tf.reduce_min(plaq), tf.reduce_max(plaq), summarize=-1)
            tf.print('topo:', mcmc.generate.action.topoCharge(x), summarize=-1)
    if tuningStepSize and tf.reduce_mean(tf.cast(acc,tf.float64))<0.5:
        mcmc.generate.dt.assign(mcmc.generate.dt/2.0)
        tf.print('# reduce step size to:',mcmc.generate.dt)
    if forceAccept:
        return x1
    else:
        return x

def initRun(mcmc, loss, x0, weights):
    tf.print('# run once and set weights')
    t0 = tf.timestamp()
    inferStep(mcmc, loss, x0, print=False)
    mcmc.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# finished autograph run in',dt,'sec')

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

@tf.function
def trainStep(loss, opt, x):
    with tf.GradientTape() as tape:
        lv = loss(x)
    grads = tape.gradient(lv, loss.action.transform.trainable_weights)
    opt.apply_gradients(zip(grads, loss.action.transform.trainable_weights))
    plaqWoT = loss.action.plaquetteWoTrans(x)
    tf.print('loss:', lv, summarize=-1)
    tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
    tf.print('weights:', loss.action.transform.trainable_weights, summarize=-1)

def initMCMC(conf, action, mcmcFun, weights):
    t0 = tf.timestamp()
    mcmcGen = mcmcFun(action)
    x0 = mcmcGen.generate.action.random()
    p0 = mcmcGen.generate.action.randomMom()
    mcmcGen(x0, p0)
    if weights is not None:
        mcmcGen.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# finished initialize MCMC in',dt,'sec')
    return mcmcGen

def train(conf, actionFun, mcmcFun, lossFun, opt, weights=None):
    mcmcGen = initMCMC(conf, actionFun(), mcmcFun, weights)
    x = mcmcGen.generate.action.random()
    loss = lossFun(actionFun())
    loss.action.transform(x)
    loss.action.transform.set_weights(mcmcGen.generate.action.transform.get_weights())
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
            x = inferStep(mcmcGen, loss, x, detail=False, forceAccept=True)
        for step in range(conf.nstepMixing):
            tf.print('# pre-training inference step:', step, summarize=-1)
            x = inferStep(mcmcGen, loss, x, detail=False, tuningStepSize=True)
        dt = tf.timestamp()-t0
        tf.print('-------- done mixing epoch', epoch,
            'in', dt, 'sec,', 0.5*dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepEpoch):
            tf.print('# training inference step:', step, summarize=-1)
            x = inferStep(mcmcGen, None, x, detail=False)
            tf.print('# training step:', step, summarize=-1)
            # x is in the mapped space of mcmcGen
            t = tf.timestamp()
            xtarget,_,_ = mcmcGen.generate.action.transform(x)
            dtf = tf.timestamp()-t
            t = tf.timestamp()
            xtrain,_,invIter = loss.action.transform.inv(xtarget)
            dtb = tf.timestamp()-t
            tf.print('# forward time:',dtf,'sec','backward time:',dtb,'sec','max iter:',invIter)
            if invIter >= mcmcGen.generate.action.transform.invMaxIter:
                tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',mcmcGen.generate.action.transform.invMaxIter, summarize=-1)
            trainStep(loss, opt, xtrain)
            # tf.print('opt.variables():',len(opt.variables()),opt.variables())
        dt = tf.timestamp()-t0
        tf.print('-------- done training epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
        if conf.nstepEpoch>0:
            t = tf.timestamp()
            xtarget,_,_ = mcmcGen.generate.action.transform(x)
            dtf = tf.timestamp()-t
            mcmcGen.generate.action.transform.set_weights(loss.action.transform.get_weights())
            t = tf.timestamp()
            x,_,invIter = mcmcGen.generate.action.transform.inv(xtarget)
            dtb = tf.timestamp()-t
            tf.print('# forward time:',dtf,'sec','backward time:',dtb,'sec','max iter:',invIter)
            if invIter >= mcmcGen.generate.action.transform.invMaxIter:
                tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',mcmcGen.generate.action.transform.invMaxIter, summarize=-1)
        if conf.nstepPostTrain>0:
            t0 = tf.timestamp()
            for step in range(conf.nstepPostTrain):
                tf.print('# post-training inference step:', step, summarize=-1)
                x = inferStep(mcmcGen, loss, x, detail=False)
            dt = tf.timestamp()-t0
            tf.print('-------- done post-training epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepPostTrain, 'sec/step --------', summarize=-1)
    return x, mcmcGen, loss

def run(conf, actionFun, mcmcFun, lossFun, opt, weights=None):
    x, mcmc, loss = train(conf, actionFun, mcmcFun, lossFun, opt, weights=weights)
    tf.print('finalWeightsAll:', mcmc.get_weights(), summarize=-1)
    return x, mcmc, loss

if __name__ == '__main__':
    import ftr
    import tensorflow.keras as tk
    conf = nthmc.Conf(nbatch=4, nepoch=2, nstepEpoch=8, initDt=0.1, stepPerTraj = 4)
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
    mcmcFun = lambda action: nthmc.Metropolis(conf, nthmc.LeapFrog(conf, action))
    opt = tk.optimizers.Adam(learning_rate=0.001)
    x, mcmc, loss = run(conf, actionFun, mcmcFun, lossFun, opt)
    x, _, _ = mcmc.generate.action.transform(x)
    infer(conf, mcmc, loss, mcmc.get_weights(), x)
