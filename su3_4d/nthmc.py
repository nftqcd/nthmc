import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
from numpy import inf
import datetime, math, os
import sys
import ftr
sys.path.append("../lib")
import group

class Conf:
    def __init__(self,
            nbatch = 16,
            nepoch = 4,
            nstepEpoch = 256,
            nstepMixing = 16,
            nstepPostTrain = 0,
            nconfStepTune = 512,
            initDt = None,
            trainDt = True,
            stepPerTraj = 10,
            trajLength = None,
            accRate = 0.8,
            checkReverse = False,
            refreshOpt = True,
            nthr = 4,
            nthrIop = 1,
            softPlace = True,
            xlaCluster = False,
            seed = 9876543211):
        if initDt is None and trajLength is None:
            raise ValueError('missing argument: initDt or trajLength')
        self.nbatch = nbatch
        self.nepoch = nepoch
        self.nstepEpoch = nstepEpoch
        self.nstepMixing = nstepMixing
        self.nstepPostTrain = nstepPostTrain
        self.nconfStepTune = nconfStepTune
        self.initDt = trajLength/stepPerTraj if initDt is None else initDt
        self.trainDt = trainDt
        self.stepPerTraj = stepPerTraj
        self.trajLength = initDt*stepPerTraj if trajLength is None else trajLength
        self.accRate = accRate
        self.checkReverse = checkReverse
        self.refreshOpt = refreshOpt
        self.nthr = nthr
        self.nthrIop = nthrIop
        self.softPlace = softPlace
        self.xlaCluster = xlaCluster
        self.seed = seed

C1Symanzik = -1.0/12.0  # tree-level
C1Iwasaki = -0.331
C1DBW2 = -1.4088

class SU3d4(tl.Layer):
    def __init__(self, rng, nbatch=1, transform=ftr.Ident(), beta = 6.0, beta0 = 3.0, c1 = 0, size = [4,4,4,4], name='SU3d4', **kwargs):
        super(SU3d4, self).__init__(autocast=False, name=name, **kwargs)
        self.g = group.SU3
        self.targetBeta = tf.constant(beta, dtype=tf.float64)
        self.beta = tf.Variable(beta, dtype=tf.float64, trainable=False, name=name+'.beta')
        self.beta0 = beta0
        self.c1 = tf.constant(c1, dtype=tf.float64)
        self.size = size
        self.transform = transform
        self.volume = size[0]*size[1]*size[2]*size[3]
        self.shape = (nbatch, 4, *size, *self.g.size)
        self.rng = rng
    def call(self, x):
        return self.action(x)
    def build(self, shape):
        if self.shape != shape:
            raise ValueError(f'shape mismatch: init with {self.shape}, bulid with {shape}')
    def changePerEpoch(self, epoch, conf):
        self.beta.assign((epoch+1.0)/conf.nepoch*(self.targetBeta-self.beta0)+self.beta0)
    def updateGauge(self, x, p):
        return self.g.mul(self.g.exp(p), x)
    def random(self):
        return self.g.random(self.shape, self.rng)
    def randomMom(self):
        return self.g.randomMom(self.shape, self.rng)
    def momEnergy(self, p):
        return self.g.momEnergy(p)
#    def plaqPhase(self, x):
#        y, _, _ = self.transform(x)
#        return self.plaqPhaseWoTrans(y)
#    def plaqPhaseWoTrans(self, x):
#        return x[:,0] + tf.roll(x[:,1], shift=-1, axis=1) - tf.roll(x[:,0], shift=-1, axis=2) - x[:,1]
#    def topoChargeFromPhase(self, p):
#        return tf.math.floordiv(0.1 + tf.reduce_sum(self.compatProj(p), axis=range(1,len(p.shape))), 2*math.pi)
    def topoCharge(self, x):
        # FIXME
        return 0.0
    def coeffs(self):
        "Coefficients for the plaquette and rectangle terms."
        return ((1.0-8.0*self.c1)*self.beta, self.c1*self.beta)
    def plaqFieldsWoTrans(self, x, needsRect=False):
        ps = []
        rs = []
        for mu in range(1,4):
            for nu in range(0,mu):
                xmunu = self.g.mul(x[:,mu], tf.roll(x[:,nu], shift=-1, axis=mu+1))
                xnumu = self.g.mul(x[:,nu], tf.roll(x[:,mu], shift=-1, axis=nu+1))
                ps.append(self.g.trace(self.g.mul(xmunu, xnumu, adjoint_r=True)))
                if needsRect:
                    uu = self.g.mul(x[:,nu], xmunu, adjoint_l=True)
                    ur = self.g.mul(x[:,mu], xnumu, adjoint_l=True)
                    ul = self.g.mul(xmunu, tf.roll(x[:,mu], shift=-1, axis=nu+1), adjoint_r=True)
                    ud = self.g.mul(xnumu, tf.roll(x[:,nu], shift=-1, axis=mu+1), adjoint_r=True)
                    rs.append(self.g.trace(self.g.mul(ul, tf.roll(ur, shift=1, axis=mu+1), adjoint_r=True)))
                    rs.append(self.g.trace(self.g.mul(uu, tf.roll(ud, shift=1, axis=nu+1), adjoint_r=True)))
        return ps, rs
    def plaquetteListWoTrans(self, x):
        ps,_ = self.plaqFieldsWoTrans(x)
        return [tf.reduce_mean(p, axis=range(1,len(p.shape)))/3.0 for p in ps]
    def plaquetteList(self, x):
        y, _, _ = self.transform(x)
        return self.plaquetteListWoTrans(y)
    def plaquetteWoTrans(self, x):
        ps,_ = self.plaqFieldsWoTrans(x)
        psum = 0.0
        for p in ps:
            psum += tf.reduce_sum(tf.math.real(p), axis=range(1,len(p.shape)))
        return psum/(6*3*self.volume)
    def plaquette(self, x):
        y, _, _ = self.transform(x)
        return self.plaquetteWoTrans(y)
    def action(self, x):
        "Returns the action, the log Jacobian, and extra info from transform."
        y, l, bs = self.transform(x)
        cp, cr = self.coeffs()
        ps, rs = self.plaqFieldsWoTrans(y, needsRect=True)
        psum = 0.0
        for p in ps:
            psum += tf.reduce_sum(tf.math.real(p), axis=range(1,len(p.shape)))
        a = cp*psum
        if cr!=0.0:
            rsum = 0.0
            for r in rs:
                rsum += tf.reduce_sum(tf.math.real(r), axis=range(1,len(r.shape)))
            a += cr*rsum
        a = (-1.0/3.0)*a
        return a-l, l, bs
    def derivActionPlaqManual(self, x0):
        cp = tf.cast(self.beta/3.0, x0.dtype)
        x, l, bs = self.transform(x0)
        # qex.gauge.staples.makeStaples
        stf = [[None]*4 for _ in range(4)]
        stu = [[None]*4 for _ in range(4)]
        for mu in range(1,4):
            for nu in range(0,mu):
                umu = tf.roll(x[:,mu], shift=-1, axis=nu+1)
                unu = tf.roll(x[:,nu], shift=-1, axis=mu+1)
                umunu = self.g.mul(umu, unu, adjoint_r=True)
                stf[mu][nu] = self.g.mul(x[:,nu], umunu)
                stf[nu][mu] = self.g.mul(x[:,mu], umunu, adjoint_r=True)
                unumu = self.g.mul(x[:,nu], x[:,mu], adjoint_l=True)
                stu[mu][nu] = self.g.mul(unumu, unu)
                stu[nu][mu] = self.g.mul(unumu, umu, adjoint_l=True)
        # qex.gauge.gaugeAction.gaugeForce
        f = [0.0]*4
        for mu in range(1,4):
            for nu in range(0,mu):
                f[mu] += stf[mu][nu] + tf.roll(stu[mu][nu], shift=1, axis=nu+1)
                f[nu] += stf[nu][mu] + tf.roll(stu[nu][mu], shift=1, axis=mu+1)
        for mu in range(4):
            f[mu] = self.g.projectTAH(cp*self.g.mul(x[:,mu], f[mu], adjoint_r=True))
        return tf.stack(f, axis=1), l, bs
    def derivAction(self, x):
        "Returns the derivative of the action, the log Jacobian, and extra info from transform."
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            a, l, bs = self.action(x)
        g = self.g.projectTAH(self.g.mul(tape.gradient(a, x), x, adjoint_r=True))
        return g, l, bs
    def showTransform(self, **kwargs):
        self.transform.showTransform(**kwargs)

class Metropolis(tk.Model):
    def __init__(self, conf, generator, name='Metropolis', **kwargs):
        super(Metropolis, self).__init__(autocast=False, name=name, **kwargs)
        tf.print(self.name, 'init with conf', conf, summarize=-1)
        self.generate = generator
        self.checkReverse = conf.checkReverse
    def call(self, x0, p0, accrand):
        v0, l0, b0 = self.generate.action(x0)
        t0 = self.generate.action.momEnergy(p0)
        x1, p1, ls, f2s, fms, bs = self.generate(x0, p0)
        if self.checkReverse:
            self.revCheck(x0, p0, x1, p1)
        v1, l1, b1 = self.generate.action(x1)
        ls = tf.concat([tf.expand_dims(l0, 0), ls, tf.expand_dims(l1, 0)], 0)
        bs = tf.concat([tf.expand_dims(b0, 0), bs, tf.expand_dims(b1, 0)], 0)
        t1 = self.generate.action.momEnergy(p1)
        dH = (v1+t1) - (v0+t0)
        exp_mdH = tf.exp(-dH)
        acc = tf.reshape(tf.less(accrand, exp_mdH), accrand.shape + (1,)*(len(x0.shape)-1))
        x = tf.where(acc, x1, x0)
        p = tf.where(acc, p1, p0)
        acc = tf.reshape(acc, [-1])
        return (x, p, x1, p1, v0, t0, v1, t1, dH, acc, accrand, ls, f2s, fms, bs)
    def revCheck(self, x0, p0, x1, p1):
        tol = 1e-10
        fac = 2.0*(3*3+1)*self.generate.action.volume
        v0,_,_ = self.generate.action.action(x0)
        t0 = self.generate.action.momEnergy(p0)
        xr, pr, _, _, _, _ = self.generate(x1, p1)
        vr,_,_ = self.generate.action.action(xr)
        tr = self.generate.action.momEnergy(pr)
        dvr = tf.math.abs(vr-v0)/self.generate.action.volume
        dtr = tf.math.abs(tr-t0)/self.generate.action.volume
        dxr = self.generate.action.g.mul(xr,x0,adjoint_r=True) - tf.eye(3,batch_shape=[1]*(len(x0.shape)-2),dtype=x0.dtype)
        dpr = pr-p0
        n_dxr = tf.math.real(tf.math.reduce_euclidean_norm(dxr))/fac
        n_dpr = tf.math.real(tf.math.reduce_euclidean_norm(dpr))/tf.math.real(tf.math.reduce_euclidean_norm(p0))
        tf.print('revCheck: delta_x',n_dxr,'delta_p',n_dpr,'delta_v',dvr,'delta_t',dtr, summarize=-1)
        if n_dxr > tol or n_dpr > tol:
            for i in range(0,dxr.shape[0]):
                if tf.math.real(tf.math.reduce_euclidean_norm(dxr[i,...]))/fac > tol or tf.math.real(tf.math.reduce_euclidean_norm(dpr[i,...]))/tf.math.real(tf.math.reduce_euclidean_norm(p0[i,...])) > tol:
                    tf.print('Failed rev check:', i, summarize=-1)
                    tf.print(dxr[i,0,0,0,0,0], dpr[i,0,0,0,0,0], summarize=-1)
                    tf.print(x0[i,0,0,0,0,0], p0[i,0,0,0,0,0], summarize=-1)
                    tf.print(x1[i,0,0,0,0,0], p1[i,0,0,0,0,0], summarize=-1)
                    tf.print(xr[i,0,0,0,0,0], pr[i,0,0,0,0,0], summarize=-1)
    def changePerEpoch(self, epoch, conf):
        self.generate.changePerEpoch(epoch, conf)

def mean_min_max(x,**kwargs):
    return tf.reduce_mean(x,**kwargs),tf.reduce_min(x,**kwargs),tf.reduce_max(x,**kwargs)

def packMCMCRes(mcmc, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, detail=False):
    plaqWoT = mcmc.generate.action.plaquetteWoTrans(x)
    plaq = mcmc.generate.action.plaquette(x)
    dp2 = tf.math.reduce_mean(tf.math.real(tf.math.squared_difference(p1,p0)), axis=range(1,len(p0.shape)))
    if detail:
        inferRes = (v0,),(t0,),(v1,),(t1,),(dp2,),(f2s,),(fms,),((bs,) if len(bs.shape)>1 else None),(ls,),(dH,),(tf.exp(-dH),),(arand,),acc,(plaqWoT,),(plaq,),mcmc.generate.action.topoCharge(x)
    else:
        inferRes = mean_min_max(v0),mean_min_max(t0),mean_min_max(v1),mean_min_max(t1),mean_min_max(dp2),mean_min_max(f2s),mean_min_max(fms),(mean_min_max(bs,axis=(0,1)) if len(bs.shape)>1 else None),mean_min_max(ls),mean_min_max(dH),mean_min_max(tf.exp(-dH)),mean_min_max(arand),tf.reduce_mean(tf.cast(acc,tf.float64)),mean_min_max(plaqWoT),mean_min_max(plaq),mcmc.generate.action.topoCharge(x)
    return inferRes

def printMCMCRes(v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq,topo):
    tf.print('V-old:', *v0, summarize=-1)
    tf.print('T-old:', *t0, summarize=-1)
    tf.print('V-prp:', *v1, summarize=-1)
    tf.print('T-prp:', *t1, summarize=-1)
    tf.print('dp2:', *dp2, summarize=-1)
    tf.print('force:', *f2s, *fms, summarize=-1)
    if bs is not None:
        tf.print('coeff:', *bs, summarize=-1)
    tf.print('lnJ:', *ls, summarize=-1)
    tf.print('dH:', *dH, summarize=-1)
    tf.print('exp_mdH:', *expmdH, summarize=-1)
    tf.print('arand:', *arand, summarize=-1)
    tf.print('accept:', acc, summarize=-1)
    tf.print('plaqWoTrans:', *plaqWoT, summarize=-1)
    tf.print('plaq:', *plaq, summarize=-1)
    tf.print('topo:', topo, summarize=-1)

def showTransform(conf, mcmc, loss, weights, **kwargs):
    x0 = mcmc.generator.action.initState(conf.nbatch)
    initRun(mcmc, loss, x0, weights)
    mcmc.generator.action.showTransform(**kwargs)

def setup(conf):
    tf.random.set_seed(conf.seed)
    tk.backend.set_floatx('float64')
    tf.config.set_soft_device_placement(conf.softPlace)
    if conf.xlaCluster:
        tf.config.optimizer.set_jit('autoclustering')    # changes RNG behavior
    else:
        tf.config.optimizer.set_jit(False)
    tf.config.threading.set_inter_op_parallelism_threads(conf.nthrIop)    # ALCF suggests number of socket
    tf.config.threading.set_intra_op_parallelism_threads(conf.nthr)    # ALCF suggests number of physical cores
    os.environ["OMP_NUM_THREADS"] = str(conf.nthr)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    tf.print('NTHMC', str(datetime.datetime.now()), summarize=-1)
    tf.print('Python', sys.version, summarize=-1)
    tf.print('TensorFlow', tf.version.VERSION, tf.version.GIT_VERSION, tf.version.COMPILER_VERSION, summarize=-1)
    tf.print('Device', *tf.config.get_visible_devices(), summarize=-1)
    tf.print('RandomSeed', conf.seed, summarize=-1)

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
    action = U1d2(transform(), conf.nbatch, tf.random.Generator.from_seed(conf.seed))
    loss = LossFun(action, cCosDiff=0.01, cTopoDiff=1.0, cForce2=1.0, dHmin=0.5, topoFourierN=1)
    opt = tk.optimizers.Adam(learning_rate=0.001)
    x0 = action.random()
    mcmc = Metropolis(conf, LeapFrog(conf, action))
    x, _, _ = action.transform(run(conf, mcmc, loss, opt, x0))
    infer(conf, mcmc, loss, mcmc.get_weights(), x)
