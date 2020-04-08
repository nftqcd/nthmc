import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import math, os

class Conf:
  def __init__(self,
               nbatch = 32,
               nepoch = 64,
               nstepEpoch = 2048,
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

class OneD(tl.Layer):
  def __init__(self, transforms, beta = 5, size = 64, name='OneD', **kwargs):
    super(OneD, self).__init__(autocast=False, name=name, **kwargs)
    self.targetBeta = tf.constant(beta, dtype=tf.float64)
    self.beta = tf.Variable(beta, dtype=tf.float64, trainable=False)
    self.size = size
    self.transforms = transforms
  def call(self, x):
    return self.action(x)
  def changePerEpoch(self, epoch, conf):
    self.beta.assign((epoch+1.0)/conf.nepoch*(self.targetBeta-1.0)+1.0)
  def initState(self, nbatch):
    return tf.Variable(2.0*math.pi*tf.random.uniform((nbatch, self.size), dtype=tf.float64)-math.pi)
  def regularize(self, x):
    return tf.math.floormod(x+math.pi, 2*math.pi)-math.pi
  def plaqPhase(self, y):
    x = y
    for tran in self.transforms:
      x = tran(x)
    return tf.roll(x, shift=-1, axis=1) - x
  def topoChargeFromPhase(self, p):
    return tf.math.floordiv(0.1 + tf.reduce_sum(self.regularize(p), axis=1), 2*math.pi)
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
    a = self.beta*tf.reduce_sum(1.0-tf.cos(self.plaqPhase(x)), axis=1)
    for tran in self.transforms:
      a -= tran.logDetJacob(x)
    return a
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
    return x
  #def jacob(self, x, g):
  #  return g
  def logDetJacob(self, x):
    return 0.0
  #def derivLogDetJacob(self, x):
  #  return 0.0

class OneDNeighbor(tl.Layer):
  def __init__(self, distance=1, mask='even', name='OneDNeighbor', **kwargs):
    super(OneDNeighbor, self).__init__(autocast=False, name=name, **kwargs)
    self.alpha = tf.Variable(0.0, dtype=tf.float64)
    self.mask = mask
    self.distance = distance
  def build(self, shape):
    l = shape[1]
    if l % self.distance != 0:
      raise ValueError(f'OneDNeighbor with distance={self.distance} does not fit periodicly to model length {l}')
    o = tf.cast(tf.less(self.distance-1, tf.math.floormod(tf.range(l),2*self.distance)), tf.float64)
    if self.mask == 'even':
      self.mask = 1.0-o
    elif self.mask == 'odd':
      self.mask = o
  def call(self, x):
    s = tf.sin(tf.roll(x, shift=-1, axis=1) - x)
    f = self.beta()*self.mask*(tf.roll(s, shift=1, axis=1) - s)
    return tf.math.floormod(x+f+math.pi, 2*math.pi)-math.pi
  def beta(self):
    return tf.math.atan(self.alpha)/math.pi
  def logDetJacob(self, x):
    s = tf.cos(tf.roll(x, shift=-1, axis=1) - x)
    f = self.beta()*self.mask*(tf.roll(s, shift=1, axis=1) + s)
    return tf.reduce_sum(tf.math.log1p(f), axis=1)

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
    self.initDt = conf.initDt
    self.stepPerTraj = conf.stepPerTraj
    self.action = action
    tf.print(self.name, 'init with dt', self.initDt, 'step/traj', self.stepPerTraj, summarize=-1)
  def build(self, input_shape):
    tf.print(self.name, 'got input_shape', input_shape, summarize=-1)
    if self.dt is None:
      self.dt = self.add_weight(initializer=tk.initializers.Constant(self.initDt), dtype=tf.float64)
  def call(self, x0, p0):
    dt = self.dt
    x = x0 + 0.5*dt*p0
    p = p0 - dt*self.action.derivAction(x)
    for i in range(self.stepPerTraj-1):
      x += dt*p
      p -= dt*self.action.derivAction(x)
    x += 0.5*dt*p
    x = self.action.regularize(x)
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
  def __call__(self, x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand):
    #tf.print('LossFun called with', x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand, summarize=-1)
    pp0 = self.action.plaqPhase(x0)
    pp1 = self.action.plaqPhase(x1)
    ldc = tf.math.reduce_mean(1-tf.cos(pp1-pp0), axis=1)
    ldt = tf.math.squared_difference(
      self.action.topoChargeFourierFromPhase(pp1,self.topoFourierN),
      self.action.topoChargeFourierFromPhase(pp0,self.topoFourierN))
    lap = tf.exp(-tf.maximum(dH,self.dHmin))
    tf.print('cosDiff:', ldc, summarize=-1)
    tf.print('topoDiff:', ldt, summarize=-1)
    tf.print('accProb:', lap, summarize=-1)
    return -tf.math.reduce_mean((self.cCosDiff*ldc+self.cTopoDiff*ldt)*lap)

@tf.function
def trainStep(mcmc, loss, opt, x0):
  p0 = refreshP(x0.shape)
  with tf.GradientTape() as tape:
    x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand = mcmc(x0, p0)
    lv = loss(x, p, x0, p0, x1, p1, v0, t0, v1, t1, dH, acc, arand)
  grads = tape.gradient(lv, mcmc.trainable_weights)
  tf.print('grads:', grads)
  if tf.math.reduce_any(tf.math.is_nan(grads)):
    tf.print('*** got grads nan ***')
  else:
    opt.apply_gradients(zip(grads, mcmc.trainable_weights))
  tf.print('V-old:', v0, summarize=-1)
  tf.print('T-old:', t0, summarize=-1)
  tf.print('V-prp:', v1, summarize=-1)
  tf.print('T-prp:', t1, summarize=-1)
  tf.print('dH:', dH, summarize=-1)
  tf.print('arand:', arand, summarize=-1)
  tf.print('accept:', acc, summarize=-1)
  tf.print('weights:', mcmc.trainable_weights, summarize=-1)
  tf.print('loss:', lv, summarize=-1)
  tf.print('plaq:', loss.action.plaquette(x), summarize=-1)
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
  #action = OneD(transforms=[Ident()])
  action = OneD(transforms=[
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd'),
    OneDNeighbor(mask='even'), OneDNeighbor(mask='odd')])
  loss = LossFun(action, cCosDiff=1.0, cTopoDiff=1.0, dHmin=0.5, topoFourierN=9)
  opt = tk.optimizers.Adam(learning_rate=0.001)
  x0 = action.initState(conf.nbatch)
  run(conf, action, loss, opt, x0)
