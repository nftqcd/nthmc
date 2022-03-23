import nthmc
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import numpy
from numpy import inf
import sys
sys.path.append("../lib")
import group

# tf.function would fail and claim some tensor cannot be accessed, if we use class and functions.
# so we just use tuple here.
def newEvolveStat(size):
    ls = tf.TensorArray(tf.float64, size=size, tensor_array_name='lndet')
    f2s = tf.TensorArray(tf.float64, size=size, tensor_array_name='fnorm2')
    fms = tf.TensorArray(tf.float64, size=size, tensor_array_name='fnormInf')
    bs = tf.TensorArray(tf.float64, size=size, tensor_array_name='coeffs')
    return (ls,f2s,fms,bs)
def recordEvolveStat(stat, i, force, lndet, coeffs):
    (ls,f2s,fms,bs) = stat
    f2 = group.norm2(force)
    df = tf.reshape(f2, [f2.shape[0],-1])
    f2s = f2s.write(i, tf.math.reduce_mean(df, axis=-1))
    fms = fms.write(i, tf.norm(df, ord=inf, axis=-1))
    ls = ls.write(i, lndet)
    bs = bs.write(i, coeffs)
    return (ls,f2s,fms,bs)
def collectEvolveStat(stat):
    (ls,f2s,fms,bs) = stat
    ls = ls.stack()
    f2s = f2s.stack()
    fms = fms.stack()
    bs = bs.stack()
    return (ls, f2s, fms, bs)

class LeapFrog(tl.Layer):
    def __init__(self, conf, action, name='LeapFrog', **kwargs):
        super(LeapFrog, self).__init__(autocast=False, name=name, **kwargs)
        self.dt = self.add_weight(initializer=tk.initializers.Constant(conf.initDt), dtype=tf.float64, trainable=conf.trainDt)
        self.stepPerTraj = self.add_weight(initializer=tk.initializers.Constant(conf.stepPerTraj), dtype=tf.int64, trainable=False)    # TF error with int32, https://github.com/tensorflow/tensorflow/issues/53192
        self.action = action
        tf.print(self.name, 'init with dt', self.dt, 'step/traj', self.stepPerTraj, summarize=-1)
    def call(self, x0, p0):
        n = tf.cast(self.stepPerTraj, tf.int32)
        stat = newEvolveStat(n)
        dt = tf.cast(self.dt, x0.dtype)
        x = self.action.updateGauge(x0, 0.5*dt*p0)
        d, l, b = self.action.derivAction(x)
        stat = recordEvolveStat(stat,0,d,l,b)
        p = p0 - dt*d
        for i in tf.range(1,n):
            x = self.action.updateGauge(x, dt*p)
            d, l, b = self.action.derivAction(x)
            stat = recordEvolveStat(stat,i,d,l,b)
            p -= dt*d
        x = self.action.updateGauge(x, 0.5*dt*p)
        return (x, -p, *collectEvolveStat(stat))
    def changePerEpoch(self, epoch, conf):
        self.action.changePerEpoch(epoch, conf)

class Omelyan2MN(tl.Layer):
    def __init__(self, conf, action, c_lambda=0.1931833275037836, name='Omelyan2MN', **kwargs):
        # Omelyan et. al. (2003), equation (31)
        super(Omelyan2MN, self).__init__(autocast=False, name=name, **kwargs)
        self.dt = self.add_weight(initializer=tk.initializers.Constant(conf.initDt), dtype=tf.float64, trainable=conf.trainDt)
        self.stepPerTraj = self.add_weight(initializer=tk.initializers.Constant(conf.stepPerTraj), dtype=tf.int64, trainable=False)    # TF error with int32, https://github.com/tensorflow/tensorflow/issues/53192
        self.c_lambda = self.add_weight(initializer=tk.initializers.Constant(c_lambda), dtype=tf.float64, trainable=False)
        self.action = action
        tf.print(self.name, 'init with dt', self.dt, 'step/traj', self.stepPerTraj, summarize=-1)
    def call(self, x0, p0):
        n = tf.cast(2*self.stepPerTraj, tf.int32)
        stat = newEvolveStat(n)
        dt = self.dt
        m_dt_2 = tf.cast(-0.5*dt, x0.dtype)
        dt_lambda = tf.cast(self.c_lambda*dt, x0.dtype)
        dt_lambda2 = 2.*dt_lambda
        dt_1_lambda2 = tf.cast(dt, x0.dtype)-dt_lambda2

        x = self.action.updateGauge(x0, dt_lambda*p0)

        d, l, b = self.action.derivAction(x)
        stat = recordEvolveStat(stat,0,d,l,b)
        p = p0 + m_dt_2*d

        x = self.action.updateGauge(x, dt_1_lambda2*p)

        d, l, b = self.action.derivAction(x)
        stat = recordEvolveStat(stat,1,d,l,b)
        p += m_dt_2*d

        for i in tf.range(2, n, 2):

            x = self.action.updateGauge(x, dt_lambda2*p)

            d, l, b = self.action.derivAction(x)
            stat = recordEvolveStat(stat,i,d,l,b)
            p += m_dt_2*d
            i += 1

            x = self.action.updateGauge(x, dt_1_lambda2*p)

            d, l, b = self.action.derivAction(x)
            stat = recordEvolveStat(stat,i,d,l,b)
            p += m_dt_2*d

        x = self.action.updateGauge(x, dt_lambda*p)
        return (x, -p, *collectEvolveStat(stat))
    def changePerEpoch(self, epoch, conf):
        self.action.changePerEpoch(epoch, conf)

@tf.function(jit_compile=True)
def tuneStep_func(mcmc, x0, p0, accrand, detail, forceAccept):
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0, accrand)
    inferRes = nthmc.packMCMCRes(mcmc, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs, detail)
    return (tf.cond(forceAccept, lambda:x1, lambda:x),inferRes,dH)

@tf.function
def printTuneStepResults(v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq,topo,mcmc,stepTuner,stepTunerDH):
    nthmc.printMCMCRes(v0,t0,v1,t1,dp2,f2s,fms,bs,ls,dH,expmdH,arand,acc,plaqWoT,plaq,topo)
    if stepTuner is not None:
        if stepTuner(mcmc, stepTunerDH):
            tf.print('# TuneStep set',mcmc.generate.name,'dt',mcmc.generate.dt,'step/traj',mcmc.generate.stepPerTraj, summarize=-1)
        else:
            tf.print('# TuneStep skipped')

def tuneStep(mcmc, x0, stepTuner=None, detail=False, forceAccept=False):
    # stepTuner uses boolean_mask that has where op, which is not supported in jit_compile yet
    # https://github.com/tensorflow/tensorflow/issues/52905
    # run it in printTuneStepResults
    t0 = tf.timestamp()
    p0 = mcmc.generate.action.randomMom()
    accrand = mcmc.generate.action.rng.uniform([x0.shape[0]], dtype=tf.float64)
    t1 = tf.timestamp()
    x,inferRes,dH = tuneStep_func(mcmc, x0, p0, accrand, detail, tf.constant(forceAccept))
    t2 = tf.timestamp()
    printTuneStepResults(*inferRes, mcmc, stepTuner, dH)
    te = tf.timestamp()
    tf.print('# tuneStep time:',te-t0,'sec (',t1-t0,'+',t2-t1,'+',te-t2,')',summarize=-1)
    return x

class RegressStepTuner:
    def __init__(self, targetAccRate, trajLength, memoryLength=3, minCount=32):
        if targetAccRate<=0 or targetAccRate>=1:
            raise ValueError(f'RegressTuner:targetAccRate must be in (0,1), got {targetAccRate}')
        self.targetAccRate = tf.constant(targetAccRate, dtype=tf.float64)
        self.trajLength = tf.constant(trajLength, dtype=tf.float64)
        self.memoryLength = tf.constant(memoryLength, dtype=tf.int64)
        self.minCount = tf.constant(minCount, dtype=tf.int64)
        self.savedAccRate = tf.Variable(tf.zeros(memoryLength, dtype=tf.float64), trainable=False)
        self.savedAccRateErr = tf.Variable(tf.zeros(memoryLength, dtype=tf.float64), trainable=False)
        self.savedStepPerTraj = tf.Variable(tf.zeros(memoryLength, dtype=tf.int64), trainable=False)
        self.savedCount = tf.Variable(tf.zeros(memoryLength, dtype=tf.int64), trainable=False)
        self.tempAccRate = tf.Variable(0, dtype=tf.float64, trainable=False)
        self.tempAccRateErr = tf.Variable(0, dtype=tf.float64, trainable=False)
        self.tempStepPerTraj = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.tempCount = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.tol = 1e-4
    def reset(self):
        self.savedAccRate.assign(tf.zeros(self.memoryLength, dtype=tf.float64))
        self.savedAccRateErr.assign(tf.zeros(self.memoryLength, dtype=tf.float64))
        self.savedStepPerTraj.assign(tf.zeros(self.memoryLength, dtype=tf.int64))
        self.savedCount.assign(tf.zeros(self.memoryLength, dtype=tf.int64))
        self.tempAccRate.assign(0)
        self.tempAccRateErr.assign(0)
        self.tempStepPerTraj.assign(0)
        self.tempCount.assign(0)
    def __call__(self, mcmc, dH):
        n = dH.shape[0]
        exp_mdH = tf.exp(-dH)
        aa = tf.where(exp_mdH>1., tf.constant(1., dtype=tf.float64), exp_mdH)
        a = tf.reduce_mean(aa)
        aa -= a
        ae = tf.constant(0., dtype=tf.float64) if n==1 else tf.math.sqrt(tf.reduce_sum(aa*aa)/tf.cast(n*(n-1), tf.float64))
        m = tf.equal(self.savedStepPerTraj, mcmc.generate.stepPerTraj)
        def combineWith(sn,s,se):
            se2 = se*se
            new_n = sn+n
            new_a = (s*tf.cast(sn,tf.float64)+a*tf.cast(n,tf.float64))/tf.cast(sn+n,tf.float64)
            new_ae = tf.math.sqrt((((se*se*tf.cast(sn-1,tf.float64)+s*s)*tf.cast(sn,tf.float64) + (ae*ae*tf.cast(n-1,tf.float64)+a*a)*tf.cast(n,tf.float64)) / tf.cast(new_n, tf.float64) - new_a*new_a) / tf.cast(new_n-1, tf.float64))
            return (new_n,new_a,new_ae)
        if tf.math.reduce_any(m):
            s = tf.boolean_mask(self.savedAccRate, m)[0]
            se = tf.boolean_mask(self.savedAccRateErr, m)[0]
            sn = tf.boolean_mask(self.savedCount, m)[0]
            new_n,new_a,new_ae = combineWith(sn,s,se)
            self.savedAccRate.assign(tf.where(m, new_a, self.savedAccRate))
            self.savedAccRateErr.assign(tf.where(m, new_ae, self.savedAccRateErr))
            self.savedCount.assign(tf.where(m, new_n, self.savedCount))
        elif n>=self.minCount:
            self.savedAccRate.assign(tf.roll(self.savedAccRate, 1, 0))
            self.savedAccRateErr.assign(tf.roll(self.savedAccRateErr, 1, 0))
            self.savedStepPerTraj.assign(tf.roll(self.savedStepPerTraj, 1, 0))
            self.savedCount.assign(tf.roll(self.savedCount, 1, 0))
            self.savedAccRate[0].assign(a)
            self.savedAccRateErr[0].assign(ae)
            self.savedStepPerTraj[0].assign(mcmc.generate.stepPerTraj)
            self.savedCount[0].assign(n)
        elif mcmc.generate.stepPerTraj==self.tempStepPerTraj:
            s = self.tempAccRate
            se = self.tempAccRateErr
            sn = self.tempCount
            new_n,new_a,new_ae = combineWith(sn,s,se)
            if new_n>=self.minCount:
                self.savedAccRate.assign(tf.roll(self.savedAccRate, 1, 0))
                self.savedAccRateErr.assign(tf.roll(self.savedAccRateErr, 1, 0))
                self.savedStepPerTraj.assign(tf.roll(self.savedStepPerTraj, 1, 0))
                self.savedCount.assign(tf.roll(self.savedCount, 1, 0))
                self.savedAccRate[0].assign(new_a)
                self.savedAccRateErr[0].assign(new_ae)
                self.savedStepPerTraj[0].assign(mcmc.generate.stepPerTraj)
                self.savedCount[0].assign(new_n)
                self.tempAccRate.assign(0)
                self.tempAccRateErr.assign(0)
                self.tempStepPerTraj.assign(0)
                self.tempCount.assign(0)
            else:
                self.tempAccRate.assign(new_a)
                self.tempAccRateErr.assign(new_ae)
                self.tempCount.assign(new_n)
        else:
            self.tempAccRate.assign(a)
            self.tempAccRateErr.assign(ae)
            self.tempStepPerTraj.assign(mcmc.generate.stepPerTraj)
            self.tempCount.assign(n)
        # tf.print('# RegressTuner: acceptance rate:',a,'+/-',ae, summarize=-1)
        # tf.print('# RegressTuner: tempAccRate',self.tempAccRate,'tempAccRateErr',self.tempAccRateErr,'tempStepPerTraj',self.tempStepPerTraj,'tempCount',self.tempCount, summarize=-1)
        # tf.print('# RegressTuner: savedAccRate',self.savedAccRate,'savedAccRateErr',self.savedAccRateErr,'savedStepPerTraj',self.savedStepPerTraj,'savedCount',self.savedCount, summarize=-1)
        hasMinCount = self.minCount <= self.savedCount
        if tf.math.reduce_any(hasMinCount):
            ar = tf.boolean_mask(self.savedAccRate, hasMinCount)
            are_shift = self.tol*(self.tol+tf.math.top_k(tf.boolean_mask(self.savedAccRateErr, hasMinCount)).values[0])
            are = tf.where(self.savedAccRateErr>are_shift, self.savedAccRateErr, self.savedAccRateErr+are_shift)
            si2 = 1./tf.boolean_mask(are, hasMinCount)**2
            spt = tf.boolean_mask(self.savedStepPerTraj, hasMinCount)
            x = self.trajLength/tf.cast(spt, tf.float64)
            # linear fit
            sum_si2 = tf.reduce_sum(si2)
            x_si2 = x*si2
            x_si2_1 = tf.reduce_sum(x_si2)
            x_si2_x = tf.reduce_sum(x_si2*x)
            A = 1. - (x_si2_1/x_si2_x)*x
            A2 = tf.reduce_sum(A*A)
            B = 1. - (sum_si2/x_si2_1)*x
            B2 = tf.reduce_sum(B*B)
            # tf.print('x_si2_1',x_si2_1,'si2',tf.reduce_sum(si2),'x_si2_x',x_si2_x,'det',x_si2_1*x_si2_1-tf.reduce_sum(si2)*x_si2_x)
            # tf.print('A',A,'norm2',A2,'B',B,'norm2',B2)
            if A2 < self.tol or B2 < self.tol:
                # single datum or singular matrix
                cA = tf.reduce_sum(si2*ar)/tf.reduce_sum(si2)
                # rough approximation: dH = -ln(a) = c*x^2 => X = x sqrt(ln(t)/ln(a))
                if cA==0:
                    X = 0.618*tf.sort(x)[0]
                else:
                    X = tf.sort(x)[0] * tf.math.sqrt(tf.math.log(self.targetAccRate)/tf.math.log(cA))
                # tf.print('# RegressTuner: singular, X',X, summarize=-1)
            else:
                A_si2 = A*si2
                A_si2_1 = tf.reduce_sum(A_si2)
                A_si2_y = tf.reduce_sum(A_si2*ar)
                A_si2_yY = tf.reduce_sum(A_si2*(ar-self.targetAccRate))
                B_si2_x = tf.reduce_sum(B*x_si2)
                B_si2_y = tf.reduce_sum(B*si2*ar)
                cA = A_si2_y/A_si2_1
                cB = B_si2_y/B_si2_x
                X = (self.targetAccRate-cA)/cB
                # dXdy = (B_si2_x/(A_si2_1*B_si2_y**2)) * (A_si2_yY*B - B_si2_y*A)    # without the sigma^-2
                # dX = tf.math.sqrt(tf.reduce_sum(dXdy*si2*dXdy))
                # tf.print('# RegressTuner: fit, cA',cA,'cB',cB,'X',X,'+/-',dX, summarize=-1)
            # For stability, restrict the size
            sortedx = tf.sort(x)
            if X < 0.809*sortedx[0]:
                X = 0.809*sortedx[0]
            elif X > 1.618*sortedx[-1]:
                X = 1.618*sortedx[-1]
            new_spt = tf.cast(tf.round(self.trajLength/X), tf.int64)
            if new_spt < 1:
                new_spt = tf.constant(1, dtype=tf.int64)
            mcmc.generate.stepPerTraj.assign(new_spt)
            mcmc.generate.dt.assign(self.trajLength/tf.cast(new_spt, tf.float64))
            changed = True
        else:
            changed = False
        return changed

def saveConfs(x, file):
    t0 = tf.timestamp()
    numpy.save(file, x.numpy(), allow_pickle=False, fix_imports=False)
    tf.print('# save to',file,' time:',tf.timestamp()-t0,'sec',summarize=-1)
