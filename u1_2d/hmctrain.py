import nthmc
import tensorflow as tf

def initMCMC(conf, mcmcFun, weights):
    t0 = tf.timestamp()
    mcmcTrain = mcmcFun()
    mcmcGen = mcmcFun()
    x0 = mcmcTrain.generate.action.random()
    p0 = mcmcTrain.generate.action.randomMom()
    mcmcTrain(x0, p0)
    mcmcGen(x0, p0)
    if weights is not None:
        mcmcTrain.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# finished initialize MCMC in',dt,'sec')
    return mcmcTrain, mcmcGen

def train(conf, mcmcFun, lossFun, opt, weights=None):
    mcmcTrain, mcmcGen = initMCMC(conf, mcmcFun, weights)
    loss = lossFun(mcmcTrain.generate.action)
    x = mcmcTrain.generate.action.random()
    for epoch in range(conf.nepoch):
        mcmcGenRNGState = mcmcGen.generate.action.rng.state
        mcmcGen.set_weights(mcmcTrain.get_weights())
        mcmcGen.generate.action.rng.reset(mcmcGenRNGState)
        mcmcGen.changePerEpoch(epoch, conf)
        mcmcTrain.changePerEpoch(epoch, conf)
        if conf.refreshOpt and len(opt.variables())>0:
            tf.print('# reset optimizer')
            for var in opt.variables():
                var.assign(tf.zeros_like(var))
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', loss.action.beta, summarize=-1)
        for step in range(conf.nstepMixing):
            tf.print('# pre-training inference step with forced acceptance:', step, summarize=-1)
            x = nthmc.inferStep(mcmcGen, loss, x, detail=False, forceAccept=True)
        for step in range(conf.nstepMixing):
            tf.print('# pre-training inference step:', step, summarize=-1)
            x = nthmc.inferStep(mcmcGen, loss, x, detail=False, tuningStepSize=True)
        dt = tf.timestamp()-t0
        tf.print('-------- done mixing epoch', epoch,
            'in', dt, 'sec,', 0.5*dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        t0 = tf.timestamp()
        mcmcTrain.generate.dt.assign(mcmcGen.generate.dt)
        for step in range(conf.nstepEpoch):
            tf.print('# training inference step:', step, summarize=-1)
            x = nthmc.inferStep(mcmcGen, loss, x, detail=False)
            tf.print('# training step:', step, summarize=-1)
            # x is in the mapped space of mcmcGen
            t = tf.timestamp()
            xtarget,_,_ = mcmcGen.generate.action.transform(x)
            dtf = tf.timestamp()-t
            t = tf.timestamp()
            xtrain,_,invIter = mcmcTrain.generate.action.transform.inv(xtarget)
            dtb = tf.timestamp()-t
            tf.print('# forward time:',dtf,'sec','backward time:',dtb,'sec','max iter:',invIter)
            if invIter >= mcmcTrain.generate.action.transform.invMaxIter:
                tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',mcmcTrain.generate.action.transform.invMaxIter, summarize=-1)
            nthmc.trainStep(mcmcTrain, loss, opt, xtrain)
            # tf.print('opt.variables():',len(opt.variables()),opt.variables())
        dt = tf.timestamp()-t0
        tf.print('-------- done training epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
        if conf.nstepEpoch>0:
            t = tf.timestamp()
            xtarget,_,_ = mcmcGen.generate.action.transform(x)
            dtf = tf.timestamp()-t
            t = tf.timestamp()
            x,_,invIter = mcmcTrain.generate.action.transform.inv(xtarget)
            dtb = tf.timestamp()-t
            tf.print('# forward time:',dtf,'sec','backward time:',dtb,'sec','max iter:',invIter)
            if invIter >= mcmcTrain.generate.action.transform.invMaxIter:
                tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',mcmcTrain.generate.action.transform.invMaxIter, summarize=-1)
        if conf.nstepPostTrain>0:
            t0 = tf.timestamp()
            for step in range(conf.nstepPostTrain):
                tf.print('# post-training inference step:', step, summarize=-1)
                x = nthmc.inferStep(mcmcTrain, loss, x, detail=False)
            dt = tf.timestamp()-t0
            tf.print('-------- done post-training epoch', epoch,
                'in', dt, 'sec,', dt/conf.nstepPostTrain, 'sec/step --------', summarize=-1)
    return x, mcmcTrain, loss

def run(conf, mcmcFun, lossFun, opt, weights=None):
    x, mcmc, loss = train(conf, mcmcFun, lossFun, opt, weights=weights)
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
    lossFun = lambda action: nthmc.LossFun(action, cCosDiff=0.01, cTopoDiff=1.0, cForce2=1.0, dHmin=0.5, topoFourierN=1)
    opt = tk.optimizers.Adam(learning_rate=0.001)
    rng = tf.random.Generator.from_seed(conf.seed)
    mcmcFun = lambda: nthmc.Metropolis(conf, nthmc.LeapFrog(conf, nthmc.U1d2(transform(), conf.nbatch, rng.split()[0])))
    x, mcmc, loss = run(conf, mcmcFun, lossFun, opt)
    x, _, _ = mcmc.generate.action.transform(x)
    nthmc.infer(conf, mcmc, loss, mcmc.get_weights(), x)
