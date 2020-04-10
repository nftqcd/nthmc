import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import math, os

class Conf:
  def __init__(self,
               nbatch = 32,
               nepoch = 64,
               nstepEpoch = 2048,
               nstepMixing = 64,
               initDt = 0.2,
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
    self.stepPerTraj = stepPerTraj
    self.checkReverse = checkReverse
    self.refreshOpt = refreshOpt
    self.nthr = nthr
    self.nthrIop = nthrIop
    self.seed = seed

def refreshP(shape):
  return tf.random.normal(shape, dtype=tf.float64)
def kineticEnergy(p):
  return 0.5*tf.reduce_sum(tf.reshape(p, [p.shape[0], -1])**2, axis=1)
def regularize(x):
  return tf.math.floormod(x+math.pi, 2*math.pi)-math.pi

class OneD(tl.Layer):
  def __init__(self, transform, beta = 5, size = 64, name='OneD', **kwargs):
    super(OneD, self).__init__(autocast=False, name=name, **kwargs)
    self.targetBeta = tf.constant(beta, dtype=tf.float64)
    self.beta = tf.Variable(beta, dtype=tf.float64, trainable=False)
    self.size = size
    self.transform = transform
  def call(self, x):
    return self.action(x)
  def changePerEpoch(self, epoch, conf):
    self.beta.assign((epoch+1.0)/conf.nepoch*(self.targetBeta-1.0)+1.0)
  def initState(self, nbatch):
    return tf.Variable(2.0*math.pi*tf.random.uniform((nbatch, self.size), dtype=tf.float64)-math.pi)
  def plaqPhase(self, x):
    y, _ = self.transform(x)
    return self.plaqPhaseNoTrans(y)
  def plaqPhaseNoTrans(self, x):
    return tf.roll(x, shift=-1, axis=1) - x
  def topoChargeFromPhase(self, p):
    return tf.math.floordiv(0.1 + tf.reduce_sum(regularize(p), axis=1), 2*math.pi)
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
    return tf.reduce_sum(t, axis=1)/math.pi
  def topoChargeFourier(self, x, n):
    return self.topoChargeFourierFromPhase(self.plaqPhase(x), n)
  def plaquette(self, x):
    return tf.reduce_mean(tf.cos(self.plaqPhase(x)), axis=1)
  def action(self, x):
    y, l = self.transform(x)
    a = self.beta*tf.reduce_sum(1.0-tf.cos(self.plaqPhaseNoTrans(y)), axis=1)
    return a-l
  def derivAction(self, x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      a = self.action(x)
    g = tape.gradient(a, x)
    return g

class Ident(tl.Layer):
  def __init__(self, name='Ident', **kwargs):
    super(Ident, self).__init__(autocast=False, name=name, **kwargs)
  def call(self, x):
    return (x, 0.0)
  def inv(self, y):
    return (y, 0.0)

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

class OneDNeighbor(tl.Layer):
  def __init__(self, distance=1, alpha=0.0, order=1, mask='even', invAbsR2=1E-30, name='OneDNeighbor', **kwargs):
    super(OneDNeighbor, self).__init__(autocast=False, name=name, **kwargs)
    self.alpha = tf.Variable(alpha, dtype=tf.float64)
    self.mask = mask
    self.distance = distance
    self.order = order
    self.invAbsR2 = invAbsR2
  def build(self, shape):
    l = shape[1]
    if l % (2*self.distance) != 0:
      raise ValueError(f'OneDNeighbor with distance={self.distance} does not fit periodicly to model length {l}')
    o = tf.cast(tf.less(self.distance-1, tf.math.floormod(tf.range(l),2*self.distance)), tf.float64)
    if self.mask == 'even':
      self.mask = 1.0-o
    elif self.mask == 'odd':
      self.mask = o
    else:
      raise ValueError(f'OneDNeighbor with unknown mask={self.mask}.  Valid options are: even, odd.')
    super(OneDNeighbor, self).build(shape)
  def call(self, x):
    b = self.beta()
    f = self.shift(b, x)
    c = tf.cos((tf.roll(x, shift=-self.distance, axis=1) - x)*self.order)
    d = b*self.mask*(tf.roll(c, shift=self.distance, axis=1) + c)
    return (regularize(x+f), tf.reduce_sum(tf.math.log1p(d), axis=1))
  def beta(self):
    return tf.math.atan(self.alpha)/math.pi
  def shift(self, beta, x):
    s = tf.sin((tf.roll(x, shift=-self.distance, axis=1) - x)*self.order)
    f = beta/self.order*self.mask*(tf.roll(s, shift=self.distance, axis=1) - s)
    return f
  def inv(self, y):
    b = self.beta()
    x = y
    while True:
      f = self.shift(b, x)
      if self.invAbsR2 > tf.reduce_mean(tf.math.squared_difference(f, y-x)):
        x = regularize(y-f)
        _, l = self(x)
        return (x, -l)
      x = y-f

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
    acc = tf.expand_dims(tf.less(arand, exp_mdH), -1)
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
    self.dt = None
    self.dt = self.add_weight(initializer=tk.initializers.Constant(conf.initDt), dtype=tf.float64)
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
    x = regularize(x)
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
    ldc = tf.math.reduce_mean(1-tf.cos(pp1-pp0), axis=1)
    ldt = tf.math.squared_difference(
      self.action.topoChargeFourierFromPhase(pp1,self.topoFourierN),
      self.action.topoChargeFourierFromPhase(pp0,self.topoFourierN))
    lap = tf.exp(-tf.maximum(dH,self.dHmin))
    if print:
      tf.print('cosDiff:', tf.reduce_mean(ldc), summarize=-1)
      tf.print('topoDiff:', tf.reduce_mean(ldt), summarize=-1)
      tf.print('accProb:', tf.reduce_mean(lap), summarize=-1)
    return -tf.math.reduce_mean((self.cCosDiff*ldc+self.cTopoDiff*ldt)*lap)

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

def infer(conf, mcmc, loss, weights, x0):
  tf.print('# run once and set weights')
  inferStep(mcmc, loss, x0, print=False)
  mcmc.set_weights(weights)
  tf.print('# finished autograph run')
  x, _ = loss.action.transform.inv(x0)
  for epoch in range(conf.nepoch):
    mcmc.changePerEpoch(epoch, conf)
    tf.print('weightsAll:', mcmc.get_weights())
    t0 = tf.timestamp()
    tf.print('-------- start epoch', epoch, '@', t0, '--------', summarize=-1)
    tf.print('beta:', loss.action.beta, summarize=-1)
    for step in range(conf.nstepEpoch):
      tf.print('# traj:', step, summarize=-1)
      x = inferStep(mcmc, loss, x)
    dt = tf.timestamp()-t0
    tf.print('-------- end epoch', epoch,
      'in', dt, 'sec,', dt/conf.nstepEpoch, 'sec/step --------', summarize=-1)
  return x

def runInfer(conf, action, loss, weights, x0):
  mcmc = Metropolis(conf, LeapFrog(conf, action))
  x = infer(conf, mcmc, loss, weights, x0)
  return x

@tf.function
def trainStep(mcmc, loss, opt, x0):
  p0 = refreshP(x0.shape)
  with tf.GradientTape() as tape:
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand = mcmc(x0, p0)
    lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand)
  grads = tape.gradient(lv, mcmc.trainable_weights)
  tf.print('grads:', grads, summarize=-1)
  if tf.math.reduce_any(tf.math.is_nan(grads)):
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

def train(conf, mcmc, loss, opt, x0):
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

def run(conf, action, loss, opt, x0):
  mcmc = Metropolis(conf, LeapFrog(conf, action))
  x = train(conf, mcmc, loss, opt, x0)
  tf.print('finalWeightsAll:', mcmc.get_weights())
  return x

def setup(conf):
  tf.random.set_seed(conf.seed)
  tk.backend.set_floatx('float64')
  tf.config.set_soft_device_placement(True)
  tf.config.optimizer.set_jit(True)
  tf.config.threading.set_inter_op_parallelism_threads(conf.nthrIop)  # ALCF suggests number of socket
  tf.config.threading.set_intra_op_parallelism_threads(conf.nthr)  # ALCF suggests number of physical cores
  os.environ["OMP_NUM_THREADS"] = str(conf.nthr)
  os.environ["KMP_BLOCKTIME"] = "0"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

if __name__ == '__main__':
  conf = Conf(nbatch=4, nepoch=2, nstepEpoch=20, initDt=0.1)
  #conf = Conf(nbatch=4, nepoch=2, nstepEpoch=2048, initDt=0.1)
  #conf = Conf()
  setup(conf)
  #action = OneD(TransformChain([Ident()]))
  action = OneD(TransformChain([
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd')]))
  loss = LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.5, topoFourierN=9)
  opt = tk.optimizers.Adam(learning_rate=0.001)
  x0 = action.initState(conf.nbatch)
  run(conf, action, loss, opt, x0)
