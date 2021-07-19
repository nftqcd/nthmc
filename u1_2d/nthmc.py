import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
from numpy import inf
import math, os
import sys
sys.path.append("../lib")
import group

class Conf:
    def __init__(self,
                             nbatch = 16,
                             nepoch = 4,
                             nstepEpoch = 256,
                             nstepMixing = 16,
                             initDt = 0.2,
                             trainDt = True,
                             stepPerTraj = 10,
                             checkReverse = False,
                             refreshOpt = True,
                             nthr = 4,
                             nthrIop = 1,
                             seed = 9876543211):
        self.nbatch = nbatch
        self.nepoch = nepoch
        self.nstepEpoch = nstepEpoch
        self.nstepMixing = nstepMixing
        self.initDt = initDt
        self.trainDt = trainDt
        self.stepPerTraj = stepPerTraj
        self.checkReverse = checkReverse
        self.refreshOpt = refreshOpt
        self.nthr = nthr
        self.nthrIop = nthrIop
        self.seed = seed

def refreshP(shape):
    return tf.random.normal(shape, dtype=tf.float64)
def kineticEnergy(p):
    return 0.5*tf.reduce_sum(p**2, axis=range(1,len(p.shape)))
def project(x):
    return tf.math.floormod(x+math.pi, 2*math.pi)-math.pi

class U1d2(tl.Layer):
    def __init__(self, transform, beta = 4.0, beta0 = 1.0, size = [8,8], name='U1d2', **kwargs):
        super(U1d2, self).__init__(autocast=False, name=name, **kwargs)
        self.g = group.U1Phase
        self.targetBeta = tf.constant(beta, dtype=tf.float64)
        self.beta = tf.Variable(beta, dtype=tf.float64, trainable=False)
        self.beta0 = beta0
        self.size = size
        self.transform = transform
        self.volume = size[0]*size[1]
    def call(self, x):
        return self.action(x)
    def changePerEpoch(self, epoch, conf):
        self.beta.assign((epoch+1.0)/conf.nepoch*(self.targetBeta-self.beta0)+self.beta0)
    def initState(self, nbatch):
        return tf.Variable(2.0*math.pi*tf.random.uniform((nbatch, 2, *self.size), dtype=tf.float64)-math.pi)    # shape: batch, dim(x,y), x, y
    def plaqPhase(self, x):
        y, _, _ = self.transform(x)
        return self.plaqPhaseWoTrans(y)
    def plaqPhaseWoTrans(self, x):
        return x[:,0] + tf.roll(x[:,1], shift=-1, axis=1) - tf.roll(x[:,0], shift=-1, axis=2) - x[:,1]
    def topoChargeFromPhase(self, p):
        return tf.math.floordiv(0.1 + tf.reduce_sum(project(p), axis=range(1,len(p.shape))), 2*math.pi)
    def topoCharge(self, x):
        return self.topoChargeFromPhase(self.plaqPhase(x))
    def topoChargeFourierFromPhase(self, p, n):
        t = tf.sin(p)
        for i in range(2,n+1):
            f = tf.cast(i, tf.float64)
            dt = tf.sin(f*p)/f
            if i & 1 == 0:
                t -= dt
            else:
                t += dt
        return tf.reduce_sum(t, axis=range(1,len(t.shape)))/math.pi
    def topoChargeFourier(self, x, n):
        return self.topoChargeFourierFromPhase(self.plaqPhase(x), n)
    def plaquetteWoTrans(self, x):
        return tf.reduce_mean(self.g.trace(self.plaqPhaseWoTrans(x)), range(1,len(x.shape)-1))
    def plaquette(self, x):
        return tf.reduce_mean(self.g.trace(self.plaqPhase(x)), range(1,len(x.shape)-1))
    def action(self, x):
        "Returns the action and the log Jacobian."
        y, l, bs = self.transform(x)
        a = self.beta*tf.reduce_sum(1.0-self.g.trace(self.plaqPhaseWoTrans(y)), axis=range(1,len(x.shape)-1))
        return a-l, l, bs
    def derivAction(self, x):
        "Returns the derivative and the log Jacobian."
        with tf.GradientTape() as tape:
            tape.watch(x)
            a, l, bs = self.action(x)
        g = tape.gradient(a, x)
        return g, l, bs
    def showTransform(self, **kwargs):
        self.transform.showTransform(**kwargs)

class Metropolis(tk.Model):
    def __init__(self, conf, generator, name='Metropolis', **kwargs):
        super(Metropolis, self).__init__(autocast=False, name=name, **kwargs)
        tf.print(self.name, 'init with conf', conf, summarize=-1)
        self.generate = generator
        self.checkReverse = conf.checkReverse
    def call(self, x0, p0):
        v0, l0, b0 = self.generate.action(x0)
        t0 = kineticEnergy(p0)
        x1, p1, ls, f2s, fms, bs = self.generate(x0, p0)
        if self.checkReverse:
            self.revCheck(x0, p0, x1, p1)
        v1, l1, b1 = self.generate.action(x1)
        ls.insert(0,l0)
        ls.append(l1)
        ls = tf.stack(ls, -1)
        f2s = tf.stack(f2s, -1)
        fms = tf.stack(fms, -1)
        bs.insert(0,b0)
        bs.append(b1)
        bs = tf.stack(bs, -1)
        t1 = kineticEnergy(p1)
        dH = (v1+t1) - (v0+t0)
        exp_mdH = tf.exp(-dH)
        arand = tf.random.uniform(exp_mdH.shape, dtype=tf.float64)
        acc = tf.reshape(tf.less(arand, exp_mdH), arand.shape + (1,1,1))
        x = tf.where(acc, x1, x0)
        p = tf.where(acc, p1, p0)
        acc = tf.reshape(acc, [-1])
        return (x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs)
    def revCheck(self, x0, p0, x1, p1):
        tol = 1e-10
        xr, pr, _ = self.generate(x1, p1)
        dxr = 1.0-tf.cos(xr-x0)
        dpr = 1.0-tf.cos(pr-p0)
        if tf.math.reduce_euclidean_norm(dxr) > tol or tf.math.reduce_euclidean_norm(dpr) > tol:
            for i in range(0,dxr.shape[0]):
                if tf.math.reduce_euclidean_norm(dxr[i,...]) > tol or tf.math.reduce_euclidean_norm(dpr[i,...]) > tol:
                    tf.print('Failed rev check:', i, summarize=-1)
                    tf.print(dxr[i,...], dpr[i,...], summarize=-1)
                    tf.print(x0[i,...], p0[i,...], summarize=-1)
                    tf.print(x1[i,...], p1[i,...], summarize=-1)
                    tf.print(xr[i,...], pr[i,...], summarize=-1)
    def changePerEpoch(self, epoch, conf):
        self.generate.changePerEpoch(epoch, conf)

class LeapFrog(tl.Layer):
    def __init__(self, conf, action, name='LeapFrog', **kwargs):
        super(LeapFrog, self).__init__(autocast=False, name=name, **kwargs)
        self.dt = self.add_weight(initializer=tk.initializers.Constant(conf.initDt), dtype=tf.float64, trainable=conf.trainDt)
        self.stepPerTraj = conf.stepPerTraj
        self.action = action
        tf.print(self.name, 'init with dt', self.dt, 'step/traj', self.stepPerTraj, summarize=-1)
    def call(self, x0, p0):
        dt = self.dt
        x = x0 + 0.5*dt*p0
        d, l, b = self.action.derivAction(x)
        p = p0 - dt*d
        df = tf.reshape(d, [d.shape[0],-1])
        f2s = [tf.norm(df, ord=2, axis=-1)]
        fms = [tf.norm(df, ord=inf, axis=-1)]
        ls = [l]
        bs = [b]
        for i in range(self.stepPerTraj-1):
            x += dt*p
            d, l, b = self.action.derivAction(x)
            p -= dt*d
            df = tf.reshape(d, [d.shape[0],-1])
            f2s.append(tf.norm(df, ord=2, axis=-1))
            fms.append(tf.norm(df, ord=inf, axis=-1))
            ls.append(l)
            bs.append(b)
        x += 0.5*dt*p
        x = project(x)
        return (x, -p, ls, f2s, fms, bs)
    def changePerEpoch(self, epoch, conf):
        self.action.changePerEpoch(epoch, conf)

class LossFun:
    def __init__(self, action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=1.0, topoFourierN=9, cForce2=0.0):
        tf.print('LossFun init with action', action, summarize=-1)
        self.action = action
        self.cCosDiff = cCosDiff
        self.cTopoDiff = cTopoDiff
        self.cForce2 = cForce2
        self.dHmin = 0.0 if dHmin<0 else dHmin
        self.topoFourierN = topoFourierN
    def __call__(self, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, print=True):
        pp0 = self.action.plaqPhase(x0)
        pp1 = self.action.plaqPhase(x1)
        if self.cCosDiff == 0:
            ldc = 0
        else:
            ldc = tf.math.reduce_mean(1-tf.cos(pp1-pp0), axis=range(1,len(pp0.shape)))
        if self.cTopoDiff == 0:
            ldt = 0
        else:
            ldt = tf.math.squared_difference(
                self.action.topoChargeFourierFromPhase(pp1,self.topoFourierN),
                self.action.topoChargeFourierFromPhase(pp0,self.topoFourierN))
        if self.cForce2 == 0:
            lf2 = 0
        else:
            lf2 = tf.math.reduce_mean(f2s, axis=range(1,len(f2s.shape)))
        lap = tf.exp(-tf.maximum(dH,self.dHmin))
        if print:
            if self.cCosDiff != 0:
                tf.print('cosDiff:', tf.reduce_mean(ldc), summarize=-1)
            if self.cTopoDiff != 0:
                tf.print('topoDiff:', tf.reduce_mean(ldt), summarize=-1)
            if self.cForce2 != 0:
                tf.print('force2:', tf.reduce_mean(lf2), summarize=-1)
            tf.print('accProb:', tf.reduce_mean(lap), summarize=-1)
        return -tf.math.reduce_mean((self.cCosDiff*ldc+self.cTopoDiff*ldt-self.cForce2/self.action.volume*lf2)*lap)

def initRun(mcmc, loss, x0, weights):
    tf.print('# run once and set weights')
    t0 = tf.timestamp()
    inferStep(mcmc, loss, x0, print=False)
    mcmc.set_weights(weights)
    dt = tf.timestamp()-t0
    tf.print('# finished autograph run in',dt,'sec')

@tf.function
def inferStep(mcmc, loss, x0, print=True, detail=True, forceAccept=False, tuningStepSize=False):
    p0 = refreshP(x0.shape)
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0)
    lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, print=print)
    if print:
        plaqWoT = mcmc.generate.action.plaquetteWoTrans(x)
        plaq = mcmc.generate.action.plaquette(x)
        dp2 = tf.math.reduce_mean(tf.math.squared_difference(p1,p0), axis=range(1,len(p0.shape)))
        if detail:
            tf.print('V-old:', v0, summarize=-1)
            tf.print('T-old:', t0, summarize=-1)
            tf.print('V-prp:', v1, summarize=-1)
            tf.print('T-prp:', t1, summarize=-1)
            tf.print('dp2:', dp2, summarize=-1)
            tf.print('force:', f2s, fms, summarize=-1)
            tf.print('coeff:', bs, summarize=-1)
            tf.print('lnJ:', ls, summarize=-1)
            tf.print('dH:', dH, summarize=-1)
            tf.print('arand:', arand, summarize=-1)
            tf.print('accept:', acc, summarize=-1)
            tf.print('loss:', lv, summarize=-1)
            tf.print('plaqWoTrans:', plaqWoT, summarize=-1)
            tf.print('plaq:', plaq, summarize=-1)
            tf.print('topo:', mcmc.generate.action.topoCharge(x), summarize=-1)
        else:
            tf.print('dp2:', tf.reduce_mean(dp2), summarize=-1)
            tf.print('force:', tf.reduce_mean(f2s), tf.reduce_min(f2s), tf.reduce_max(f2s), tf.reduce_mean(fms), tf.reduce_min(fms), tf.reduce_max(fms), summarize=-1)
            tf.print('coeff:', tf.reduce_mean(bs, axis=(0,-1)), summarize=-1)
            tf.print('lnJ:', tf.reduce_mean(ls), tf.reduce_min(ls), tf.reduce_max(ls), summarize=-1)
            tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
            tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
            tf.print('loss:', lv, summarize=-1)
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

def infer(conf, mcmc, loss, weights, x0, detail=True):
    initRun(mcmc, loss, x0, weights)
    x, _ = mcmc.generate.action.transform.inv(x0)
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

def showTransform(conf, mcmc, loss, weights, **kwargs):
    x0 = mcmc.generator.action.initState(conf.nbatch)
    initRun(mcmc, loss, x0, weights)
    mcmc.generator.action.showTransform(**kwargs)

@tf.function
def trainStep(mcmc, loss, opt, x0):
    p0 = refreshP(x0.shape)
    with tf.GradientTape() as tape:
        x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0)
        lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs)
    grads = tape.gradient(lv, mcmc.trainable_weights)
    opt.apply_gradients(zip(grads, mcmc.trainable_weights))
    # tf.print('grads:', grads, summarize=-1)
    # if tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(g)) for g in grads]):
    #     tf.print('*** got grads nan ***')
    # else:
    #     opt.apply_gradients(zip(grads, mcmc.trainable_weights))
    plaqWoT = loss.action.plaquetteWoTrans(x)
    plaq = loss.action.plaquette(x)
    #tf.print('V-old:', v0, summarize=-1)
    #tf.print('T-old:', t0, summarize=-1)
    #tf.print('V-prp:', v1, summarize=-1)
    #tf.print('T-prp:', t1, summarize=-1)
    tf.print('dp2:', tf.reduce_mean(tf.math.squared_difference(p1,p0)), summarize=-1)
    tf.print('force:', tf.reduce_mean(f2s), tf.reduce_min(f2s), tf.reduce_max(f2s), tf.reduce_mean(fms), tf.reduce_min(fms), tf.reduce_max(fms), summarize=-1)
    tf.print('coeff:', tf.reduce_mean(bs, axis=(0,-1)), summarize=-1)
    tf.print('lnJ:', tf.reduce_mean(ls), tf.reduce_min(ls), tf.reduce_max(ls), summarize=-1)
    tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
    #tf.print('arand:', arand, summarize=-1)
    tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
    tf.print('weights:', mcmc.trainable_weights, summarize=-1)
    tf.print('loss:', lv, summarize=-1)
    tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
    tf.print('plaq:', tf.reduce_mean(plaq), tf.reduce_min(plaq), tf.reduce_max(plaq), summarize=-1)
    tf.print('topo:', loss.action.topoCharge(x), summarize=-1)
    return x

def train(conf, mcmc, loss, opt, x0, weights=None, requireInv=False):
    if weights is not None:
        initRun(mcmc, loss, x0, weights)
        if requireInv:
            x0, _ = loss.action.transform.inv(x0)
    elif requireInv:
        raise ValueError('Inverse transform required without weights.')
    x = x0
    for epoch in range(conf.nepoch):
        mcmc.changePerEpoch(epoch, conf)
        if conf.refreshOpt and len(opt.variables())>0:
            tf.print('# reset optimizer')
            for var in opt.variables():
                var.assign(tf.zeros_like(var))
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', loss.action.beta, summarize=-1)
        for step in range(conf.nstepMixing):
            tf.print('# inference step with forced acceptance:', step, summarize=-1)
            x = inferStep(mcmc, loss, x, detail=False, forceAccept=True)
        for step in range(conf.nstepMixing):
            tf.print('# inference step:', step, summarize=-1)
            x = inferStep(mcmc, loss, x, detail=False)
        dt = tf.timestamp()-t0
        tf.print('-------- done mixing epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepMixing, 'sec/step --------', summarize=-1)
        t0 = tf.timestamp()
        for step in range(conf.nstepEpoch):
            tf.print('# training step:', step, summarize=-1)
            x = trainStep(mcmc, loss, opt, x)
            # tf.print('opt.variables():',len(opt.variables()),opt.variables())
        dt = tf.timestamp()-t0
        tf.print('-------- end epoch', epoch,
            'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
    return x

def run(conf, mcmc, loss, opt, x0, weights=None, requireInv=False):
    x = train(conf, mcmc, loss, opt, x0, weights=weights, requireInv=requireInv)
    tf.print('finalWeightsAll:', mcmc.get_weights(), summarize=-1)
    return x

def setup(conf):
    tf.random.set_seed(conf.seed)
    tk.backend.set_floatx('float64')
    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(False)
    tf.config.threading.set_inter_op_parallelism_threads(conf.nthrIop)    # ALCF suggests number of socket
    tf.config.threading.set_intra_op_parallelism_threads(conf.nthr)    # ALCF suggests number of physical cores
    os.environ["OMP_NUM_THREADS"] = str(conf.nthr)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

if __name__ == '__main__':
    import ftr
    conf = Conf(nbatch=4, nepoch=2, nstepEpoch=8, initDt=0.1, stepPerTraj = 4)
    setup(conf)
    op0 = (((1,2,-1,-2), (1,-2,-1,2)),
           ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
    # requires different coefficient bounds:
    # (1,2,-1,-2,1,-2,-1,2)
    # (1,2,-1,-2,1,2,-1,-2)
    # (1,-2,-1,2,1,-2,-1,2)
    op1 = (((2,-1,-2,1), (2,1,-2,-1)),
           ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
    fixedP = (1,2,-1,-2)
    fixedR0 = (2,2,1,-2,-2,-1)
    fixedR1 = (1,1,2,-1,-1,-2)
    convP0 = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    convP1 = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    convR = lambda pad: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ), pad)
    conv = lambda: ftr.PeriodicConv((
        tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    ))
    transform = lambda: ftr.TransformChain([
        ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ])
    ftr.checkDep(transform())
    action = U1d2(transform())
    loss = LossFun(action, cCosDiff=0.01, cTopoDiff=1.0, cForce2=1.0, dHmin=0.5, topoFourierN=1)
    opt = tk.optimizers.Adam(learning_rate=0.001)
    x0 = action.initState(conf.nbatch)
    mcmc = Metropolis(conf, LeapFrog(conf, action))
    x, _, _ = action.transform(run(conf, mcmc, loss, opt, x0))
    infer(conf, mcmc, loss, mcmc.get_weights(), x)
