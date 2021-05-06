import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import math, os
import sys
sys.path.append("../lib")
import field, group

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
                             seed = 1331):
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
    def call(self, x):
        return self.action(x)
    def changePerEpoch(self, epoch, conf):
        self.beta.assign((epoch+1.0)/conf.nepoch*(self.targetBeta-self.beta0)+self.beta0)
    def initState(self, nbatch):
        return tf.Variable(2.0*math.pi*tf.random.uniform((nbatch, 2, *self.size), dtype=tf.float64)-math.pi)    # shape: batch, dim(x,y), x, y
    def plaqPhase(self, x):
        y, _ = self.transform(x)
        return self.plaqPhaseNoTrans(y)
    def plaqPhaseNoTrans(self, x):
        return x[:,0,:,:] + tf.roll(x[:,1,:,:], shift=-1, axis=1) - tf.roll(x[:,0,:,:], shift=-1, axis=2) - x[:,1,:,:]
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
    def plaquette(self, x):
        return tf.reduce_mean(tf.cos(self.plaqPhase(x)), range(1,len(x.shape)-1))
    def action(self, x):
        y, l = self.transform(x)
        a = self.beta*tf.reduce_sum(1.0-tf.cos(self.plaqPhaseNoTrans(y)), axis=range(1,len(x.shape)-1))
        return a-l
    def derivAction(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            a = self.action(x)
        g = tape.gradient(a, x)
        return g
    def showTransform(self, **kwargs):
        self.transform.showTransform(**kwargs)

class Ident(tl.Layer):
    def __init__(self, name='Ident', **kwargs):
        super(Ident, self).__init__(autocast=False, name=name, **kwargs)
    def call(self, x):
        return (x, 0.0)
    def inv(self, y):
        return (y, 0.0)
    def showTransform(self, **kwargs):
        tf.print(self.name, **kwargs)

class Scalar(tl.Layer):
    def __init__(self, output, name='Scalar', **kwargs):
        self.xs = tf.Variable((0.0,)*output, dtype=tf.float64)
        super(Scalar, self).__init__(autocast=False, name=name, **kwargs)
    def call(self, _):
        return self.xs

class PeriodicConv(tl.Layer):
    def __init__(self, layers, name='PeriodicConv', **kwargs):
        """
        Periodic pad the input to its tail side and call layers on it,
        assuming 2D x y dimensions are the fastest moving axis.
        Conv2D layers are assumed to be channels_last.
        Channels_first causes error: Conv2DCustomBackpropInputOp only supports NHWC.
        """
        super(PeriodicConv, self).__init__(autocast=False, name=name, **kwargs)
        self.layers = layers
    def call(self, x):
        n0 = sum([l.kernel_size[0]-1 for l in self.layers])
        n1 = sum([l.kernel_size[1]-1 for l in self.layers])
        # U(1) specific
        y = tf.concat((x, x[...,:n0,:,:]), axis=-3)
        y = tf.concat((y, y[...,:n1,:]), axis=-2)
        for l in self.layers:
            y = l(y)
        return y

class TransformChain(tl.Layer):
    def __init__(self, transforms, name='TransformChain', **kwargs):
        super(TransformChain, self).__init__(autocast=False, name=name, **kwargs)
        self.chain = transforms
    def call(self, x):
        y = x
        l = tf.zeros(x.shape[0], dtype=tf.float64)
        for f in self.chain:
            y, t = f(y)
            l += t
        return (y, l)
    def inv(self, y):
        x = y
        l = tf.zeros(y.shape[0], dtype=tf.float64)
        for f in reversed(self.chain):
            x, t = f.inv(x)
            l += t
        return (x, l)
    def showTransform(self, **kwargs):
        n = len(self.chain)
        tf.print(self.name, '[', n, ']', **kwargs)
        for i in range(n):
            tf.print(i, ':', end='')
            self.chain[i].showTransform(**kwargs)

class GenericStoutSmear(tl.Layer):
    def __init__(self, ordpaths, alphalayer, alphamap, alphamasks=(), first=(0,0), repeat=(2,2), gauge=group.U1Phase, invAbsR2=1E-30, name='GenericStoutSmear', **kwargs):
        """
        ordpaths: OrdPaths(), derivatives are wrt the first link in each path
        first: the link to update as (x, y)
        repeat: tile repeat pattern (dx, dy)
        TODO: check for repetitions in each path to determine the coefficient bound for beta().
        Specify the loop with directions {+/-1, +/-2}, forward/backward following the link
        on the first or second dimension.
        Uses the specified loop (TODO: and its mirror image about the first link).
        It updates the first link using the derivative of the sum of the pair of loops
        with respect to the first link, with coefficients determined by alphalayer and alphamap.
        Each element in alphamasks relative to first corresponds to the ordered path passed to alphalayer.
        """
        super(GenericStoutSmear, self).__init__(autocast=False, name=name, **kwargs)
        self.alphaLayer = alphalayer
        self.alphamap = alphamap
        self.alphamasks = alphamasks
        self.op = ordpaths
        self.difflooplen = sum(alphamap)
        self.first = first
        self.repeat = repeat
        self.gauge = gauge
        self.dir = abs(ordpaths.paths[0].flatten()[0])-1
        self.invAbsR2 = invAbsR2
        if self.difflooplen+len(self.alphamasks) != len(self.op.paths):
            raise ValueError('Layer arguments alphamap and alphamasks does not match ordpaths.')
    def build(self, shape):
        m = tf.scatter_nd((self.first,), tf.constant((1.,), dtype=tf.float64), self.repeat)
        r = [shape[2+i]//rep for i,rep in enumerate(self.repeat)]
        self.mask = tf.tile(m, r)
        self.unmask = [
            1.0 - tf.tile(
                tf.roll(
                    tf.scatter_nd(
                        cm, tf.constant((1.,)*len(cm), dtype=tf.float64), self.repeat
                        ), self.first, range(len(self.first))
                    ), r)
            for cm in self.alphamasks]
        self.dataShape = shape
        self.updateIndex = tf.constant([[i,self.dir] for i in range(shape[0])])
        super(GenericStoutSmear, self).build(shape)
    def call(self, x):
        ps = field.OrdProduct(self.op, x, self.gauge).prodList()
        bs = self.beta(ps)
        d = 0.0
        f = 0.0
        b = 0
        for k,a in enumerate(self.alphamap):
            fa = 0.0
            da = 0.0
            for i in range(b, b+a):
                fa += self.gauge.diffTrace(ps[i])
                da -= self.gauge.trace(ps[i])  # diff@diff@trace is -trace in U(1) if link appears once
            b = a
            f += bs[...,k]*self.mask*fa
            d += bs[...,k]*self.mask*da
        f = self.expanddir(f)
        return (self.gauge.compatProj(x+f), tf.math.reduce_sum(tf.math.log1p(d), range(1,len(d.shape))))
    def beta(self,ps):
        "Leading dimension, or the length of self.alphamap, matches the ordered paths."
        # 2*atan(a)/pi/nloop makes it in (-1/nloop,1/nloop)
        if len(ps)>self.difflooplen:
            y = tf.stack([m*self.gauge.trace(p) for m,p in zip(self.unmask,ps[self.difflooplen:])], axis=-1)
        else:
            y = []
        return 2.0/self.difflooplen/math.pi*tf.math.atan(self.alphaLayer(y))
    def inv(self, y):
        x = y
        for i in range(1024):
            ps = field.OrdProduct(self.op, x, self.gauge).prodList()
            bs = self.beta(ps)
            f = 0.0
            b = 0
            for k,a in enumerate(self.alphamap):
                fa = 0.0
                for i in range(b, b+a):
                    fa += self.gauge.diffTrace(ps[i])
                b = a
                f += bs[...,k]*self.mask*fa
            f = self.expanddir(f)
            if self.invAbsR2 > tf.math.reduce_mean(tf.math.squared_difference(f, y-x)):
                x = self.gauge.compatProj(y-f)
                _, l = self(x)
                return (x, -l)
            x = y-f
        raise ValueError(f'Failed to converge in inverting from {y}, current {x} with delta {f}.')
    def expanddir(self, x):
        return tf.scatter_nd(self.updateIndex, x, self.dataShape)
    def showTransform(self, **kwargs):
        tf.print(self.name, 'α:', self.alphaLayer.trainable_weights, 'T:', self.first, ':', self.repeat, **kwargs)

class Metropolis(tk.Model):
    def __init__(self, conf, generator, name='Metropolis', **kwargs):
        super(Metropolis, self).__init__(autocast=False, name=name, **kwargs)
        tf.print(self.name, 'init with conf', conf, summarize=-1)
        self.generate = generator
        self.checkReverse = conf.checkReverse
    def call(self, x0, p0):
        v0 = self.generate.action(x0)
        t0 = kineticEnergy(p0)
        x1, p1 = self.generate(x0, p0)
        if self.checkReverse:
            self.revCheck(x0, p0, x1, p1)
        v1 = self.generate.action(x1)
        t1 = kineticEnergy(p1)
        dH = (v1+t1) - (v0+t0)
        exp_mdH = tf.exp(-dH)
        arand = tf.random.uniform(exp_mdH.shape, dtype=tf.float64)
        acc = tf.reshape(tf.less(arand, exp_mdH), arand.shape + (1,1,1))
        x = tf.where(acc, x1, x0)
        p = tf.where(acc, p1, p0)
        acc = tf.reshape(acc, [-1])
        return (x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand)
    def revCheck(self, x0, p0, x1, p1):
        tol = 1e-10
        xr, pr = self.generate(x1, p1)
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
        p = p0 - dt*self.action.derivAction(x)
        for i in range(self.stepPerTraj-1):
            x += dt*p
            p -= dt*self.action.derivAction(x)
        x += 0.5*dt*p
        x = project(x)
        return (x, -p)
    def changePerEpoch(self, epoch, conf):
        self.action.changePerEpoch(epoch, conf)

class LossFun:
    def __init__(self, action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=1.0, topoFourierN=9):
        tf.print('LossFun init with action', action, summarize=-1)
        self.action = action
        self.cCosDiff = cCosDiff
        self.cTopoDiff = cTopoDiff
        self.dHmin = dHmin
        self.topoFourierN = topoFourierN
    def __call__(self, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, print=True):
        #tf.print('LossFun called with', x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, summarize=-1)
        pp0 = self.action.plaqPhase(x0)
        pp1 = self.action.plaqPhase(x1)
        ldc = tf.math.reduce_mean(1-tf.cos(pp1-pp0), axis=range(1,len(pp0.shape)))
        ldt = tf.math.squared_difference(
            self.action.topoChargeFourierFromPhase(pp1,self.topoFourierN),
            self.action.topoChargeFourierFromPhase(pp0,self.topoFourierN))
        lap = tf.exp(-tf.maximum(dH,self.dHmin))
        if print:
            tf.print('cosDiff:', tf.reduce_mean(ldc), summarize=-1)
            tf.print('topoDiff:', tf.reduce_mean(ldt), summarize=-1)
            tf.print('accProb:', tf.reduce_mean(lap), summarize=-1)
        return -tf.math.reduce_mean((self.cCosDiff*ldc+self.cTopoDiff*ldt)*lap)

def initRun(mcmc, loss, x0, weights):
    tf.print('# run once and set weights')
    inferStep(mcmc, loss, x0, print=False)
    mcmc.set_weights(weights)
    tf.print('# finished autograph run')

@tf.function
def inferStep(mcmc, loss, x0, print=True, detail=True):
    p0 = refreshP(x0.shape)
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand = mcmc(x0, p0)
    lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, print=print)
    if print:
        if detail:
            tf.print('V-old:', v0, summarize=-1)
            tf.print('T-old:', t0, summarize=-1)
            tf.print('V-prp:', v1, summarize=-1)
            tf.print('T-prp:', t1, summarize=-1)
            tf.print('dH:', dH, summarize=-1)
            tf.print('arand:', arand, summarize=-1)
            tf.print('accept:', acc, summarize=-1)
            tf.print('loss:', lv, summarize=-1)
            tf.print('plaq:', loss.action.plaquette(x), summarize=-1)
            tf.print('topo:', loss.action.topoCharge(x), summarize=-1)
        else:
            tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
            tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
            tf.print('loss:', lv, summarize=-1)
            tf.print('plaq:', tf.reduce_mean(loss.action.plaquette(x)), summarize=-1)
            tf.print('topo:', loss.action.topoCharge(x), summarize=-1)
    return x

def infer(conf, mcmc, loss, weights, x0, detail=True):
    initRun(mcmc, loss, x0, weights)
    x, _ = loss.action.transform.inv(x0)
    for epoch in range(conf.nepoch):
        mcmc.changePerEpoch(epoch, conf)
        tf.print('weightsAll:', mcmc.get_weights(), summarize=-1)
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', loss.action.beta, summarize=-1)
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
        x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand = mcmc(x0, p0)
        lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand)
    grads = tape.gradient(lv, mcmc.trainable_weights)
    tf.print('grads:', grads, summarize=-1)
    if tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(g)) for g in grads]):
        tf.print('*** got grads nan ***')
    else:
        opt.apply_gradients(zip(grads, mcmc.trainable_weights))
    #tf.print('V-old:', v0, summarize=-1)
    #tf.print('T-old:', t0, summarize=-1)
    #tf.print('V-prp:', v1, summarize=-1)
    #tf.print('T-prp:', t1, summarize=-1)
    tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
    #tf.print('arand:', arand, summarize=-1)
    tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
    tf.print('weights:', mcmc.trainable_weights, summarize=-1)
    tf.print('loss:', lv, summarize=-1)
    tf.print('plaq:', tf.reduce_mean(loss.action.plaquette(x)), summarize=-1)
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
    optw = None
    for epoch in range(conf.nepoch):
        mcmc.changePerEpoch(epoch, conf)
        if optw is not None:
            #tf.print('setOptWeights:', optw)
            opt.set_weights(optw)
        t0 = tf.timestamp()
        tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
        tf.print('beta:', loss.action.beta, summarize=-1)
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
            if conf.refreshOpt and optw is None:
                optw = opt.get_weights()
                for i in range(len(optw)):
                    optw[i] = tf.zeros_like(optw[i])
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
    conf = Conf(nbatch=16, nepoch=16, nstepEpoch=32, initDt=0.1)
    #conf = Conf(nbatch=4, nepoch=2, nstepEpoch=2048, initDt=0.1)
    #conf = Conf()
    setup(conf)
    #action = U1d2(TransformChain([Ident()]))
    op0 = field.OrdPaths(
        field.topath(((1,2,-1,-2), (1,-2,-1,2),  # for derivative
            (1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2),  # for derivative
            (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1),  # for derivative
            (-1,-2,1,2))))  # for alphalayer
    # requires different coefficient bounds:
    # (1,2,-1,-2,1,-2,-1,2)
    # (1,2,-1,-2,1,2,-1,-2)
    # (1,-2,-1,2,1,-2,-1,2)
    op1 = field.OrdPaths(
        field.topath(((2,-1,-2,1), (2,1,-2,-1),  # for derivative
            (2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1),  # for derivative
            (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2),  # for derivative
            (-1,-2,1,2))))  # for alphalayer
    pathmap=(2,4)
    #action = U1d2(TransformChain([
    #    GenericStoutSmear(ordpaths=op0, alphalayer=Scalar(2), alphamap=pathmap, first=(0,0), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op0, alphalayer=Scalar(2), alphamap=pathmap, first=(0,1), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op0, alphalayer=Scalar(2), alphamap=pathmap, first=(1,0), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op0, alphalayer=Scalar(2), alphamap=pathmap, first=(1,1), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op1, alphalayer=Scalar(2), alphamap=pathmap, first=(0,0), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op1, alphalayer=Scalar(2), alphamap=pathmap, first=(1,0), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op1, alphalayer=Scalar(2), alphamap=pathmap, first=(0,1), repeat=(2,2)),
    #    GenericStoutSmear(ordpaths=op1, alphalayer=Scalar(2), alphamap=pathmap, first=(1,1), repeat=(2,2)),
    #    ]))
    def conv0():
        return PeriodicConv((
            tk.layers.Conv2D(4, (3,2), activation='gelu', kernel_initializer=tk.initializers.Constant(0.0001), bias_initializer=tk.initializers.Constant(0.0001)),
            tk.layers.Conv2D(2, 1, activation='gelu', kernel_initializer=tk.initializers.Constant(0.0001), bias_initializer=tk.initializers.Constant(0.0001)),
            ))
    def conv1():
        return PeriodicConv((
            tk.layers.Conv2D(4, (2,3), activation='gelu', kernel_initializer=tk.initializers.Constant(0.0001), bias_initializer=tk.initializers.Constant(0.0001)),
            tk.layers.Conv2D(2, 1, activation='gelu', kernel_initializer=tk.initializers.Constant(0.0001), bias_initializer=tk.initializers.Constant(0.0001)),
            ))
    am0 = ([(1,0),(1,1)],)
    am1 = ([(0,1),(1,1)],)
    action = U1d2(TransformChain([
        GenericStoutSmear(ordpaths=op0, alphalayer=conv0(), alphamasks=am0, alphamap=pathmap, first=(0,0), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op0, alphalayer=conv0(), alphamasks=am0, alphamap=pathmap, first=(0,1), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op0, alphalayer=conv0(), alphamasks=am0, alphamap=pathmap, first=(1,0), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op0, alphalayer=conv0(), alphamasks=am0, alphamap=pathmap, first=(1,1), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op1, alphalayer=conv1(), alphamasks=am0, alphamap=pathmap, first=(0,0), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op1, alphalayer=conv1(), alphamasks=am0, alphamap=pathmap, first=(1,0), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op1, alphalayer=conv1(), alphamasks=am0, alphamap=pathmap, first=(0,1), repeat=(2,2)),
        GenericStoutSmear(ordpaths=op1, alphalayer=conv1(), alphamasks=am0, alphamap=pathmap, first=(1,1), repeat=(2,2)),
        ]))
    loss = LossFun(action, cCosDiff=0.01, cTopoDiff=1.0, dHmin=0.5, topoFourierN=1)
    opt = tk.optimizers.Adam(learning_rate=0.001)
    x0 = action.initState(conf.nbatch)
    mcmc = Metropolis(conf, LeapFrog(conf, action))
    run(conf, mcmc, loss, opt, x0)
