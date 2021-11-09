import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
from numpy import inf

class Omelyan2MN(tl.Layer):
    def __init__(self, conf, action, c_lambda=0.1931833275037836, name='Omelyan2MN', **kwargs):
        # Omelyan et. al. (2003), equation (31)
        super(Omelyan2MN, self).__init__(autocast=False, name=name, **kwargs)
        self.dt = self.add_weight(initializer=tk.initializers.Constant(conf.initDt), dtype=tf.float64, trainable=conf.trainDt)
        self.stepPerTraj = self.add_weight(initializer=tk.initializers.Constant(conf.stepPerTraj), dtype=tf.int32, trainable=False)
        self.c_lambda = self.add_weight(initializer=tk.initializers.Constant(c_lambda), dtype=tf.float64, trainable=False)
        self.action = action
        tf.print(self.name, 'init with dt', self.dt, 'step/traj', self.stepPerTraj, summarize=-1)
    def call(self, x0, p0):
        n = 2*self.stepPerTraj
        dt = self.dt
        m_dt_2 = -0.5*dt
        dt_lambda = self.c_lambda*dt
        dt_lambda2 = 2.*dt_lambda
        dt_1_lambda2 = dt-dt_lambda2
        f2s = tf.TensorArray(tf.float64, size=n)
        fms = tf.TensorArray(tf.float64, size=n)
        ls = tf.TensorArray(tf.float64, size=n)
        bs = tf.TensorArray(tf.float64, size=n)

        x = x0 + dt_lambda*p0

        d, l, b = self.action.derivAction(x)
        p = p0 + m_dt_2*d
        df = tf.reshape(d, [d.shape[0],-1])
        f2s = f2s.write(0, tf.norm(df, ord=2, axis=-1))
        fms = fms.write(0, tf.norm(df, ord=inf, axis=-1))
        ls = ls.write(0, l)
        bs = bs.write(0, b)

        x += dt_1_lambda2*p

        d, l, b = self.action.derivAction(x)
        p += m_dt_2*d
        df = tf.reshape(d, [d.shape[0],-1])
        f2s = f2s.write(1, tf.norm(df, ord=2, axis=-1))
        fms = fms.write(1, tf.norm(df, ord=inf, axis=-1))
        ls = ls.write(1, l)
        bs = bs.write(1, b)

        i = 2
        while i<n:

            x += dt_lambda2*p

            d, l, b = self.action.derivAction(x)
            p += m_dt_2*d
            df = tf.reshape(d, [d.shape[0],-1])
            f2s = f2s.write(i, tf.norm(df, ord=2, axis=-1))
            fms = fms.write(i, tf.norm(df, ord=inf, axis=-1))
            ls = ls.write(i, l)
            bs = bs.write(i, b)
            i += 1

            x += dt_1_lambda2*p

            d, l, b = self.action.derivAction(x)
            p += m_dt_2*d
            df = tf.reshape(d, [d.shape[0],-1])
            f2s = f2s.write(i, tf.norm(df, ord=2, axis=-1))
            fms = fms.write(i, tf.norm(df, ord=inf, axis=-1))
            ls = ls.write(i, l)
            bs = bs.write(i, b)
            i += 1

        x += dt_lambda*p
        x = self.action.compatProj(x)
        return (x, -p, ls, f2s, fms, bs)
    def changePerEpoch(self, epoch, conf):
        self.action.changePerEpoch(epoch, conf)

@tf.function
def tuneStep(mcmc, x0, stepTuner=None, print=True, detail=False, forceAccept=False):
    p0 = mcmc.generate.action.randomMom()
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x0, p0)
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
            tf.print('exp_mdH:', tf.exp(-dH), summarize=-1)
            tf.print('arand:', arand, summarize=-1)
            tf.print('accept:', acc, summarize=-1)
            tf.print('plaqWoTrans:', plaqWoT, summarize=-1)
            tf.print('plaq:', plaq, summarize=-1)
            tf.print('topo:', mcmc.generate.action.topoCharge(x), summarize=-1)
        else:
            tf.print('V-old:', tf.reduce_mean(v0), summarize=-1)
            tf.print('T-old:', tf.reduce_mean(t0), summarize=-1)
            tf.print('V-prp:', tf.reduce_mean(v1), summarize=-1)
            tf.print('T-prp:', tf.reduce_mean(t1), summarize=-1)
            tf.print('dp2:', tf.reduce_mean(dp2), summarize=-1)
            tf.print('force:', tf.reduce_mean(f2s), tf.reduce_min(f2s), tf.reduce_max(f2s), tf.reduce_mean(fms), tf.reduce_min(fms), tf.reduce_max(fms), summarize=-1)
            if len(bs.shape)>1:
                tf.print('coeff:', tf.reduce_mean(bs, axis=(0,1)), summarize=-1)
            tf.print('lnJ:', tf.reduce_mean(ls), tf.reduce_min(ls), tf.reduce_max(ls), summarize=-1)
            tf.print('dH:', tf.reduce_mean(dH), summarize=-1)
            tf.print('exp_mdH:', tf.reduce_mean(tf.exp(-dH)), summarize=-1)
            tf.print('accept:', tf.reduce_mean(tf.cast(acc,tf.float64)), summarize=-1)
            tf.print('plaqWoTrans:', tf.reduce_mean(plaqWoT), tf.reduce_min(plaqWoT), tf.reduce_max(plaqWoT), summarize=-1)
            tf.print('plaq:', tf.reduce_mean(plaq), tf.reduce_min(plaq), tf.reduce_max(plaq), summarize=-1)
            tf.print('topo:', mcmc.generate.action.topoCharge(x), summarize=-1)
    if stepTuner is not None:
        stepTuner(mcmc, dH, print=print)
    if forceAccept:
        return x1
    else:
        return x

class HalfSteps:
    def __init__(self, minAccRate, trajLength):
        self.minAccRate = tf.constant(minAccRate, dtype=tf.float64)
        self.trajLength = tf.constant(trajLength, dtype=tf.float64)
    def __call__(self, mcmc, dH, print=True):
        exp_mdH = tf.exp(-dH)
        a = tf.reduce_mean(tf.where(exp_mdH>1., tf.constant(1., dtype=tf.float64), exp_mdH))
        if a < self.minAccRate:
            mcmc.generate.stepPerTraj.assign(mcmc.generate.stepPerTraj*2)
            mcmc.generate.dt.assign(self.trajLength/tf.cast(mcmc.generate.stepPerTraj, tf.float64))
            tf.print('# HalfSteps: acceptance rate:',a,'smaller than minAccRate:',self.minAccRate,'reducing step size', summarize=-1)
            tf.print('Set',mcmc.generate.name,'dt',mcmc.generate.dt,'step/traj',mcmc.generate.stepPerTraj, summarize=-1)

class RegressTuner:
    def __init__(self, targetAccRate, trajLength, memoryLength=3, minCount=32):
        if targetAccRate<=0 or targetAccRate>=1:
            raise ValueError(f'RegressTuner:targetAccRate must be in (0,1), got {targetAccRate}')
        self.targetAccRate = tf.constant(targetAccRate, dtype=tf.float64)
        self.trajLength = tf.constant(trajLength, dtype=tf.float64)
        self.memoryLength = tf.constant(memoryLength, dtype=tf.int32)
        self.minCount = tf.constant(minCount, dtype=tf.int32)
        self.savedAccRate = tf.Variable(tf.zeros(memoryLength, dtype=tf.float64), trainable=False)
        self.savedAccRateErr = tf.Variable(tf.zeros(memoryLength, dtype=tf.float64), trainable=False)
        self.savedStepPerTraj = tf.Variable(tf.zeros(memoryLength, dtype=tf.int32), trainable=False)
        self.savedCount = tf.Variable(tf.zeros(memoryLength, dtype=tf.int32), trainable=False)
        self.tempAccRate = tf.Variable(0, dtype=tf.float64, trainable=False)
        self.tempAccRateErr = tf.Variable(0, dtype=tf.float64, trainable=False)
        self.tempStepPerTraj = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.tempCount = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.tol = 1e-4
    def __call__(self, mcmc, dH, print=True):
        n = dH.shape[0]
        exp_mdH = tf.exp(-dH)
        aa = tf.where(exp_mdH>1., tf.constant(1., dtype=tf.float64), exp_mdH)
        a = tf.reduce_mean(aa)
        aa -= a
        ae = tf.constant(0., dtype=tf.float64) if n==1 else tf.math.sqrt(tf.reduce_sum(aa*aa)/tf.cast(n*(n-1), tf.float64))
        m = self.savedStepPerTraj == mcmc.generate.stepPerTraj
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
        tf.print('# RegressTuner: acceptance rate:',a,'+/-',ae, summarize=-1)
        tf.print('# RegressTuner: tempAccRate',self.tempAccRate,'tempAccRateErr',self.tempAccRateErr,'tempStepPerTraj',self.tempStepPerTraj,'tempCount',self.tempCount, summarize=-1)
        tf.print('# RegressTuner: savedAccRate',self.savedAccRate,'savedAccRateErr',self.savedAccRateErr,'savedStepPerTraj',self.savedStepPerTraj,'savedCount',self.savedCount, summarize=-1)
        hasMinCount = self.minCount <= self.savedCount
        if tf.math.reduce_any(hasMinCount):
            ar = tf.boolean_mask(self.savedAccRate, hasMinCount)
            are_shift = self.tol*(self.tol+tf.math.top_k(tf.boolean_mask(self.savedAccRateErr, hasMinCount)).values[0])
            are = tf.where(self.savedAccRateErr>are_shift, self.savedAccRateErr, self.savedAccRateErr+are_shift)
            si2 = 1./tf.boolean_mask(are, hasMinCount)**2
            x = self.trajLength/tf.cast(tf.boolean_mask(self.savedStepPerTraj, hasMinCount), tf.float64)
            # linear fit
            sum_si2 = tf.reduce_sum(si2)
            x_si2 = x*si2
            x_si2_1 = tf.reduce_sum(x_si2)
            x_si2_x = tf.reduce_sum(x_si2*x)
            A = 1. - (x_si2_1/x_si2_x)*x
            A2 = tf.reduce_sum(A*A)
            B = 1. - (sum_si2/x_si2_1)*x
            B2 = tf.reduce_sum(B*B)
            tf.print('x_si2_1',x_si2_1,'si2',tf.reduce_sum(si2),'x_si2_x',x_si2_x,'det',x_si2_1*x_si2_1-tf.reduce_sum(si2)*x_si2_x)
            tf.print('A',A,'norm2',A2,'B',B,'norm2',B2)
            if A2 < self.tol or B2 < self.tol:
                # single datum or singular matrix
                cA = tf.reduce_sum(si2*ar)/tf.reduce_sum(si2)
                # rough approximation: dH = -ln(a) = c*x^2 => X = x sqrt(ln(t)/ln(a))
                if cA==0:
                    X = 0.5*tf.sort(x)[0]
                else:
                    X = tf.sort(x)[0] * tf.math.sqrt(tf.math.log(self.targetAccRate)/tf.math.log(cA))
                tf.print('# RegressTuner: singular, X',X, summarize=-1)
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
                dXdy = (B_si2_x/(A_si2_1*B_si2_y**2)) * (A_si2_yY*B - B_si2_y*A)    # without the sigma^-2
                dX = tf.math.sqrt(tf.reduce_sum(dXdy*si2*dXdy))
                tf.print('# RegressTuner: fit, cA',cA,'cB',cB,'X',X,'+/-',dX, summarize=-1)
            # For stability
            sortedx = tf.sort(x)
            if X < 0.5*sortedx[0]:
                X = 0.5*sortedx[0]
            elif X > 2.*sortedx[-1]:
                X = 2.*sortedx[-1]
            new_steps = tf.cast(tf.round(self.trajLength/X), tf.int32)
            if new_steps < 1:
                new_steps = 1
            mcmc.generate.stepPerTraj.assign(new_steps)
            mcmc.generate.dt.assign(self.trajLength/tf.cast(new_steps, tf.float64))
            tf.print('# RegressTuner: Set',mcmc.generate.name,'dt',mcmc.generate.dt,'step/traj',mcmc.generate.stepPerTraj, summarize=-1)
        else:
            tf.print('# RegressTuner: skip without minCount')
