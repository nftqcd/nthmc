import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import ftr
import sys
sys.path.append("../lib")
import group

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
        self.c1 = c1
        self.size = size
        self.transform = transform
        self.volume = size[0]*size[1]*size[2]*size[3]
        self.shape = (nbatch, 4, *size, *self.g.size)
        self.rng = rng
        tf.print('SU3d4 init with beta',self.beta,'coeff',self.coeffs(),'shape',self.shape)
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
                    rs.append(self.g.trace(self.g.mul(ur, tf.roll(ul, shift=-1, axis=mu+1), adjoint_r=True)))
                    rs.append(self.g.trace(self.g.mul(uu, tf.roll(ud, shift=-1, axis=nu+1), adjoint_r=True)))
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
        ps, rs = self.plaqFieldsWoTrans(y, needsRect=self.c1!=0)
        psum = 0.0
        for p in ps:
            psum += tf.reduce_sum(tf.math.real(p), axis=range(1,len(p.shape)))
        # tf.print('action plaq',cp*psum)
        a = cp*psum
        if self.c1!=0:
            rsum = 0.0
            for r in rs:
                rsum += tf.reduce_sum(tf.math.real(r), axis=range(1,len(r.shape)))
            a += cr*rsum
            # tf.print('action rect',cr*rsum)
        a = (-1.0/3.0)*a
        return a-l, l, bs
    def derivActionPlaq(self, x):
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
            f[mu] = self.g.projectTAH((1.0/3.0)*self.g.mul(x[:,mu], f[mu], adjoint_r=True))
        return tf.stack(f, axis=1)
    def derivAction(self, x):
        "Returns the derivative of the action, the log Jacobian, and extra info from transform."
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            a, l, bs = self.action(x)
        g = self.g.projectTAH(self.g.mul(tape.gradient(a, x), x, adjoint_r=True))
        return g, l, bs
    def showTransform(self, **kwargs):
        self.transform.showTransform(**kwargs)


if __name__ == '__main__':
    import nthmc, ftr, evolve
    conf = nthmc.Conf(nbatch=1, nepoch=2, nstepEpoch=8, trajLength=4.0, stepPerTraj=128)
    nthmc.setup(conf)
    # tf.config.run_functions_eagerly(True)
    action = SU3d4(tf.random.Generator.from_seed(conf.seed), conf.nbatch,
        transform=ftr.Ident(),
        beta=0.7796, beta0=0.7796, c1=C1DBW2,
        size = [8,8,8,16])
    x0 = action.random()
    with tf.GradientTape(watch_accessed_variables=False) as tape:
       tape.watch(x0)
       a,_,_ = action.action(x0)
    g = tape.gradient(a, x0)
    g = action.g.projectTAH(action.g.mul(g,x0,adjoint_r=True))
    tf.print('g',tf.math.real(g[0,0,0,0,0,0]),tf.math.imag(g[0,0,0,0,0,0]))
    ge,_ = tf.linalg.eig(g[0,0,0,0,0,0])
    tf.print('geig',tf.math.real(ge),tf.math.imag(ge))
    f = action.derivActionPlaq(x0)
    tf.print('f',tf.math.real(f[0,0,0,0,0,0]),tf.math.imag(f[0,0,0,0,0,0]))
    fe,_ = tf.linalg.eig(f[0,0,0,0,0,0])
    tf.print('feig',tf.math.real(fe),tf.math.imag(fe))

    expf = action.g.exp(f)
    expfe,_ = tf.linalg.eig(expf[0,0,0,0,0,0])
    tf.print('expfeig',tf.math.real(expfe),tf.math.imag(expfe))

    tf.print('x0R',tf.math.real(x0[0,0,0,0,0,0]))
    tf.print('x0I',tf.math.imag(x0[0,0,0,0,0,0]))
    plaq = action.plaquetteList(x0)
    for pl in plaq:
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)
    p0 = action.randomMom()
    tf.print('p0R',tf.math.real(p0[0,0,0,0,0,0]))
    tf.print('p0I',tf.math.imag(p0[0,0,0,0,0,0]))
    v0,_,_ = action.action(x0)
    t0 = action.momEnergy(p0)
    tf.print('H0', v0+t0, v0, t0)

    mdLF = evolve.LeapFrog(conf, action)
    tbegin = tf.timestamp()
    x, p, _, _, _, _ = mdLF(x0,p0)
    tf.print('# MD time:',tf.timestamp()-tbegin,'sec',summarize=-1)

    tf.print('x1R',tf.math.real(x[0,0,0,0,0,0]))
    tf.print('x1I',tf.math.imag(x[0,0,0,0,0,0]))
    plaq = action.plaquetteList(x)
    for pl in plaq:
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)
    tf.print('p1R',tf.math.real(p[0,0,0,0,0,0]))
    tf.print('p1I',tf.math.imag(p[0,0,0,0,0,0]))
    v1,_,_ = action.action(x)
    t1 = action.momEnergy(p)
    tf.print('H1', v1+t1, v1, t1)
    tf.print('dH', v1+t1-(v0+t0))

    conf.stepPerTraj = 64
    mdOM = evolve.Omelyan2MN(conf, action)
    tbegin = tf.timestamp()
    x, p, _, _, _, _ = mdOM(x0,p0)
    tf.print('# MD time:',tf.timestamp()-tbegin,'sec',summarize=-1)

    tf.print('x1R',tf.math.real(x[0,0,0,0,0,0]))
    tf.print('x1I',tf.math.imag(x[0,0,0,0,0,0]))
    plaq = action.plaquetteList(x)
    for pl in plaq:
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)
    tf.print('p1R',tf.math.real(p[0,0,0,0,0,0]))
    tf.print('p1I',tf.math.imag(p[0,0,0,0,0,0]))
    v1,_,_ = action.action(x)
    t1 = action.momEnergy(p)
    tf.print('H1', v1+t1, v1, t1)
    tf.print('dH', v1+t1-(v0+t0))

    mcmc = nthmc.Metropolis(conf, mdOM)
    mcmcFun = tf.function(mcmc)    # jit_compile=True is very slow

    for i in range(16):
        tbegin = tf.timestamp()
        p = action.randomMom()
        xn, pn, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmcFun(x, p, action.rng.uniform([x0.shape[0]], dtype=tf.float64))
        tf.print('# mcmc step',i,'time:',tf.timestamp()-tbegin,'sec',summarize=-1)
        nthmc.printMCMCRes(*nthmc.packMCMCRes(mcmc, xn, pn, x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs))
        x, p = xn, pn

    conf.checkReverse=True
    mcmcChRev = nthmc.Metropolis(conf, mdOM)
    mcmcChRev(x,p,action.rng.uniform([x0.shape[0]], dtype=tf.float64))
