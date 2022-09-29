import tensorflow as tf
import tensorflow.keras as tk
import datetime, os
import gauge
import sys

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
            seed = 876543211):
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
        v0, l0, b0 = self.generate.dynamics.V(x0)
        t0 = self.generate.dynamics.T(p0)
        x1, p1, ls, f2s, fms, bs = self.generate(x0, p0)
        if self.checkReverse:
            self.revCheck(x0, p0, x1, p1)
        v1, l1, b1 = self.generate.dynamics.V(x1)
        ls = tf.stack([l0, ls, l1], 0)
        bs = tf.stack([b0, bs, b1], 0)
        t1 = self.generate.dynamics.T(p1)
        dH = (v1+t1) - (v0+t0)
        exp_mdH = tf.exp(-dH)
        acc = tf.less(accrand, exp_mdH)
        x = x0.batch_update(acc, x1)
        p = p0.batch_update(acc, p1)
        return (x, p, x1, p1, v0, t0, v1, t1, dH, acc, accrand, ls, f2s, fms, bs)
    def revCheck(self, x0, p0, x1, p1):
        tol = 1e-10
        vol = x0.full_volume()
        fac = 2.0*(3*3+1)
        v0,_,_ = self.generate.dynamics.V(x0)
        t0 = self.generate.dynamics.T(p0)
        xr, pr, _, _, _, _ = self.generate(x1, p1)
        vr,_,_ = self.generate.dynamics.V(xr)
        tr = self.generate.dynamics.T(pr)
        dvr = tf.math.abs(vr-v0)/vol
        dtr = tf.math.abs(tr-t0)/vol
        xrx0adj = xr(x0.adjoint())
        dxr = xrx0adj - xrx0adj.unit()
        dpr = pr-p0
        n_dxr = tf.sqrt(dxr.norm2(scope='site').reduce_mean()/fac)
        n_dpr = tf.sqrt(dpr.norm2()/p0.norm2())
        tf.print('revCheck: delta_x',n_dxr,'delta_p',n_dpr,'delta_v',dvr,'delta_t',dtr, summarize=-1)
        if tf.math.reduce_mean(n_dxr) > tol or tf.math.reduce_mean(n_dpr) > tol:
            for i in range(dxr.batch_size()):
                if n_dxr[i] > tol or n_dpr[i] > tol:
                    tf.print('Failed rev check:', i, summarize=-1)
                    get0 = lambda z: z[0].get_batch(i).get_site(0,0,0,0)
                    def show(a,b):
                        tf.print(get0(a),get0(b),summarize=-1)
                    show(dxr,dpr)
                    show(x0,p0)
                    show(x1,p1)
                    show(xr,pr)

def mean_min_max(x,**kwargs):
    return tf.reduce_mean(x,**kwargs),tf.reduce_min(x,**kwargs),tf.reduce_max(x,**kwargs)

def packMCMCRes(mcmc, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, detail=False):
    plaqWoT = gauge.plaq(x)
    plaq = gauge.plaq(mcmc.generate.dynamics.V.transform(x)[0])
    dp2 = (p1-p0).norm2(scope='site').reduce_mean()
    if detail:
        inferRes = (v0,),(t0,),(v1,),(t1,),(dp2,),(f2s,),(fms,),((bs,) if len(bs.shape)>1 else None),(ls,),(dH,),(tf.exp(-dH),),(arand,),acc,(plaqWoT,),(plaq,)
    else:
        inferRes = mean_min_max(v0),mean_min_max(t0),mean_min_max(v1),mean_min_max(t1),mean_min_max(dp2),mean_min_max(f2s),mean_min_max(fms),mean_min_max(bs,axis=range(len(bs.shape)-1)),mean_min_max(ls),mean_min_max(dH),mean_min_max(tf.exp(-dH)),mean_min_max(arand),tf.reduce_mean(tf.cast(acc,tf.float64)),mean_min_max(plaqWoT),mean_min_max(plaq)
    return inferRes

def printMCMCRes(v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq):
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

def setup(conf):
    tk.utils.set_random_seed(conf.seed)
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
