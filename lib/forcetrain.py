import tensorflow as tf

class NormLoss:
    def __init__(self, cNormList=[1.0], cNormInf=0.0):
        tf.print('NormLoss init with cNormList', cNormList, 'cNormInf', cNormInf, summarize=-1)
        self.cNormList = cNormList
        self.cNormInf = cNormInf
    def __call__(self, x, y):
        df = (x-y).norm2(scope='site')    # TODO: try element level squared, too.
        dfn = df
        ldf = []
        loss = 0.0
        for i,c in enumerate(self.cNormList):
            s = tf.reduce_mean(dfn.reduce_mean()**(1./(2.*(i+1))))
            dfn *= df
            ldf.append(s)
            loss += c*s
        if self.cNormInf == 0:
            ldfI = 0
        else:
            ldfI = tf.reduce_mean(tf.math.sqrt(df.reduce_max()))
            loss += self.cNormInf*ldfI
        return loss, (ldf,ldfI)
    def printCallResults(self, res, label=""):
        ldf,ldfI = res
        for i,l in enumerate(ldf):
            tf.print(label+f'dfRMS{(i+1)*2}:', l, summarize=-1)
        if self.cNormInf != 0:
            tf.print(label+'dfnormInf:', ldfI, summarize=-1)

class LSELoss:
    def __call__(self, x, y):
        v = x.full_volume()
        df = (x-y).norm2(scope='site')
        loss = tf.reduce_mean(df.reduce_logsumexp())    # mean over batch, LSE over lattice
        loss -= tf.math.log(df.typecast(v))    # remove the volume factor
        df2 = tf.reduce_mean(df.reduce_mean())
        dfM = tf.reduce_mean(df.reduce_max())
        return loss, (df2,dfM)
    def printCallResults(self, res, label=""):
        df2,dfM = res
        tf.print(label+'dfnorm2:', df2, summarize=-1)
        tf.print(label+'dfnormInf:', dfM, summarize=-1)

class LRDecay:
    def __init__(self, halfLife):
        if halfLife<=0:
            raise ValueError(f'LRDecay.halfLife must be positive, got {halfLife}')
        self.decayRate = 0.5**(1/halfLife)
    def __call__(self, opt, *args):
        l = opt.learning_rate
        nl = self.decayRate*l
        tf.print('new learning rate:',nl,'previous',l,summarize=-1)
        l.assign(nl)

class LRSteps:
    def __init__(self, stepTargets, epochSteps=0):
        # stepTargets: [(step, targetLR), (step, targetLR), ...]
        self.stepTargets = sorted(stepTargets)
        self.epochSteps = epochSteps
        self.epoch = 0
    def __call__(self, opt, step_num):
        step = self.epoch*self.epochSteps+step_num
        for i in range(len(self.stepTargets)):
            if step<self.stepTargets[i][0]:
                s,t = self.stepTargets[i]
                break
        l = opt.learning_rate
        d = s-step
        # (t-l)(s-step-1) = (t-nl)(s-step)
        nl = (l*(d-1)+t)/d
        tf.print('new learning rate:',nl,'previous',l,summarize=-1)
        l.assign(nl)
    def setEpoch(self, epoch):
        self.epoch = epoch

class Trainer:
    def __init__(self, loss, opt, transformedAction, targetAction=None, nepoch=1, memInfo=True):
        self.loss = loss
        self.opt = opt
        self.transformedAction = transformedAction
        self.targetAction = targetAction
        self.nepoch = nepoch
        self.memInfo = memInfo
        self.from_tensors = None
        self.trainStep_func = None
        self.validateStep_func = None
    def __call__(self, confLoader, validateLoader=None, validateFrequency=0, loadWeights=None, saveWeights=None, optChange=None, optChangeEpoch=None, printWeights=False):
        if loadWeights is not None:
            t0 = tf.timestamp()
            self.transformedAction.transform.load_weights(loadWeights)
            tf.print('# loadWeights time:',tf.timestamp()-t0,'sec',summarize=-1)
        for epoch in range(self.nepoch):
            t0 = tf.timestamp()
            tf.print('-------- begin epoch', epoch, '@', t0, '--------', summarize=-1)
            if optChange is not None:
                optChange.setEpoch(epoch)
            self.trainWithConfs(confLoader, validateLoader=validateLoader, validateFrequency=validateFrequency, optChange=optChange, printWeights=printWeights)
            if validateFrequency==0 and validateLoader is not None:    # trainWithConfs validates if frequency>0
                self.validate(validateLoader)
            if optChangeEpoch is not None:
                optChangeEpoch(self.opt, epoch)
            # tf.print('self.opt.variables():',len(self.opt.variables()),self.opt.variables(),summarize=-1)
            t1 = tf.timestamp()
            tf.print('-------- end epoch', epoch, '@', t1, '--------', summarize=-1)
            tf.print('# epoch time:',t1-t0,'sec',summarize=-1)
        if saveWeights is not None:
            t1 = tf.timestamp()
            self.transformedAction.transform.save_weights(saveWeights)
            tf.print('# saveWeights time:',tf.timestamp()-t1,'sec',summarize=-1)
    def trainWithConfs(self, confLoader, validateLoader=None, validateFrequency=0, optChange=None, printWeights=False):
        # only runs one epoch
        tbegin = tf.timestamp()
        step = 0
        lv = 0.0
        while True:
            # x is in the target space
            x = confLoader(step)
            if x is None:
                break
            tf.print('# training step:', step, summarize=-1)
            if self.from_tensors==None:
                self.from_tensors = x.from_tensors
            y = self.transformBack(x)
            step_lv = self.trainStep(y, printWeights=printWeights)
            if optChange is not None:
                optChange(self.opt, step)
            # tf.print('self.opt.variables():',len(self.opt.variables()),self.opt.variables(),summarize=-1)
            step += 1
            lv += (step_lv-lv)/step
            if validateLoader is not None and validateFrequency>0 and step%validateFrequency==0:
                self.validate(validateLoader)
        tf.print('average training loss:', lv, summarize=-1)
        dt = tf.timestamp()-tbegin
        tf.print('# average time:', dt/step, 'sec/step', summarize=-1)
    def validate(self, validateLoader):
        tbegin = tf.timestamp()
        step = 0
        lv = 0.0
        while True:
            # x is in the target space
            x = validateLoader(step)
            if x is None:
                break
            tf.print('# validation step:', step, summarize=-1)
            if self.from_tensors==None:
                self.from_tensors = x.from_tensors
            y = self.transformBack(x)
            step_lv = self.validateStep(y)
            step += 1
            lv += (step_lv-lv)/step
        tf.print('average validation loss:', lv, summarize=-1)
        dt = tf.timestamp()-tbegin
        tf.print('# average validation time:', dt/step, 'sec/step', summarize=-1)
    @tf.function(jit_compile=True)
    def transformBack_func(self, x_):
        xNew,_,invIter = self.transformedAction.transform.inv(self.from_tensors(x_))
        return xNew.to_tensors(), invIter
    def transformBack(self, x):
        t = tf.timestamp()
        xNew_, invIter = self.transformBack_func(x.to_tensors())
        xNew = self.from_tensors(xNew_)
        dt = tf.timestamp()-t
        tf.print('# inverse transformation time:',dt,'sec max iter:',invIter,summarize=-1)
        if invIter >= self.transformedAction.transform.invMaxIter:
            tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',self.transformedAction.transform.invMaxIter, summarize=-1)
        return xNew
    def computeLoss(self, x):
        f,_,_ = self.transformedAction.gradient(x)
        if self.targetAction is not None:
            f0,_,_ = self.targetAction.gradient(x)
        else:
            f0 = 0.0
        return self.loss(f,f0)
    def trainStep_func_(self, x_):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.transformedAction.trainable_weights)
            lv, lossRes = self.computeLoss(self.from_tensors(x_))
        grads = tape.gradient(lv, self.transformedAction.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.transformedAction.trainable_weights))
        return lossRes,lv
    def validateStep_func_(self, x_):
        lv, lossRes = self.computeLoss(self.from_tensors(x_))
        return lossRes,lv
    @tf.function
    def printResults(self,lossRes,lv,printWeights=True,label=""):
        self.loss.printCallResults(lossRes,label=label)
        tf.print(label+'loss:', lv, summarize=-1)
        if printWeights:
            tf.print(label+'weights:', self.transformedAction.trainable_weights, summarize=-1)
    def trainStep(self, x, printWeights=True):
        t0 = tf.timestamp()
        if self.memInfo and tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats('GPU:0')
        if self.trainStep_func is None:
            # func = tf.function(self.trainStep_func_, jit_compile=True)    # FIXME xla always recompiles the function?
            func = tf.function(self.trainStep_func_)
            t1 = tf.timestamp()
            tf.print('# call tf.function time:',t1-t0,'sec', summarize=-1)
            self.trainStep_func = func.get_concrete_function(x.to_tensors())
            tf.print('# get_concrete_function time:',tf.timestamp()-t1,'sec', summarize=-1)
        lossRes,lv = self.trainStep_func(x.to_tensors())
        self.printResults(lossRes,lv,printWeights=printWeights)
        dt = tf.timestamp()-t0
        tf.print('# trainStep time:',dt,'sec',summarize=-1)
        if self.memInfo and tf.config.list_physical_devices('GPU'):
            mi = tf.config.experimental.get_memory_info('GPU:0')
            tf.print('# trainStep mem: peak',int(mi['peak']/(1024*1024)),'MiB current',int(mi['current']/(1024*1024)),'MiB')
            tf.config.experimental.reset_memory_stats('GPU:0')
        return lv
    def validateStep(self, x):
        t0 = tf.timestamp()
        if self.memInfo and tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats('GPU:0')
        if self.validateStep_func is None:
            func = tf.function(self.validateStep_func_, jit_compile=True)
            t1 = tf.timestamp()
            tf.print('# call tf.function time:',t1-t0,'sec', summarize=-1)
            self.validateStep_func = func.get_concrete_function(x.to_tensors())
            tf.print('# get_concrete_function time:',tf.timestamp()-t1,'sec', summarize=-1)
        lossRes,lv = self.validateStep_func(x.to_tensors())
        self.printResults(lossRes,lv,printWeights=False,label='validation ')
        dt = tf.timestamp()-t0
        tf.print('# validateStep time:',dt,'sec',summarize=-1)
        if self.memInfo and tf.config.list_physical_devices('GPU'):
            mi = tf.config.experimental.get_memory_info('GPU:0')
            tf.print('# validateStep mem: peak',int(mi['peak']/(1024*1024)),'MiB current',int(mi['current']/(1024*1024)),'MiB')
            tf.config.experimental.reset_memory_stats('GPU:0')
        return lv

if __name__ == '__main__':
    import action, gauge, transform, nthmc
    import tensorflow.keras as tk
    conf = nthmc.Conf(nbatch=1, nepoch=512, stepPerTraj=4, trainDt=False, trajLength=0.25)
    nthmc.setup(conf)
    trainingConf = gauge.readGauge('../su3_4d/data/lat/dbw2_8t16_b0.7796/s0/config.01024.lime')
    validator = gauge.readGauge('../su3_4d/data/lat/dbw2_8t16_b0.7796/s3/config.08176.lime')
    confLoader = lambda i: trainingConf if i==0 else None
    validateLoader = lambda i: validator if i==0 else None
    transformedAct = action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(coeff=transform.CoefficientVariable(0.04), dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]),
            action=action.SU3d4(beta=0.7796, c1=action.C1DBW2))
    # a/r₀, Eq (4.11), S. Necco, Nucl. Phys. B683, 137 (2004)
    #    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7796
    # 0.40001
    #    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7099
    # 0.599819
    #    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.6716
    # 0.800164
    target = action.TransformedActionVectorFromMatrixBase(
            transform=transform.Ident(),
            action=action.SU3d4(beta=0.6716, c1=action.C1DBW2))
    t0 = tf.timestamp()
    transformedAct(confLoader(0))
    dt = tf.timestamp()-t0
    tf.print('# initialize transformedAct time:',dt,'sec')
    trainer = Trainer(
        loss=NormLoss(cNormList=[1.0,1.0], cNormInf=1.0),
        opt=tk.optimizers.Adam(learning_rate=0.00001),
        transformedAction=transformedAct,
        targetAction=target,
        nepoch=conf.nepoch)
    trainer(confLoader, validateLoader=validateLoader, printWeights=True,
        optChangeEpoch=LRSteps([(conf.nepoch/4, 0.1), (conf.nepoch/2, 0.01), (conf.nepoch, 0.0000001)]))
