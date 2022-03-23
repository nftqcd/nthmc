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
