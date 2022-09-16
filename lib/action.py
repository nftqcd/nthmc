import tensorflow as tf
import tensorflow.keras.layers as tl
import gauge, lattice

C1Symanzik = -1.0/12.0  # tree-level
C1Iwasaki = -0.331
C1DBW2 = -1.4088

class SU3d4:
    def __init__(self, beta = 6.0, c1 = 0):
        self.beta = beta
        self.c1 = c1
        tf.print('SU3d4 init with beta',self.beta,'coeff',self.coeffs())
    def __call__(self, x):
        "Returns the action"
        computeRect = self.c1!=0
        cp, cr = self.coeffs()
        ps, rs = gauge.plaquetteField(x, computeRect=computeRect)
        psum = 0.0
        for p in ps:
            psum += p.lattice.real().trace().reduce_sum()
        # tf.print('action plaq',cp*psum)
        a = cp*psum
        if computeRect:
            rsum = 0.0
            for r in rs:
                rsum += r.lattice.real().trace().reduce_sum()
            a += cr*rsum
            # tf.print('action rect',cr*rsum)
        a = (-1.0/3.0)*a
        return a
    def coeffs(self):
        "Coefficients for the plaquette and rectangle terms."
        return ((1.0-8.0*self.c1)*self.beta, self.c1*self.beta)
    def gradientPlaq(self, x):
        # qex.gauge.staples.makeStaples
        stf = [[None]*4 for _ in range(4)]
        stu = [[None]*4 for _ in range(4)]
        for mu in range(1,4):
            for nu in range(0,mu):
                umu = x[mu].shift(nu, 1)
                unu = x[nu].shift(mu, 1)
                umunu = umu(unu.adjoint())
                stf[mu][nu] = x[nu](umunu)
                stf[nu][mu] = x[mu](umunu.adjoint())
                unumu = x[nu].adjoint()(x[mu])
                stu[mu][nu] = unumu(unu)
                stu[nu][mu] = unumu.adjoint()(umu)
        # qex.gauge.gaugeAction.gaugeForce
        f = [0.0]*4
        for mu in range(1,4):
            for nu in range(0,mu):
                f[mu] += stf[mu][nu] + stu[mu][nu].shift(nu, -1)
                f[nu] += stf[nu][mu] + stu[nu][mu].shift(mu, -1)
        cp, _ = self.action.coeffs()
        for mu in range(4):
            f[mu] = ((cp/3.0)*x[mu](f[mu].adjoint())).projectTangent()
        return gauge.Tangent(f)

class TransformedActionMatrixBase(tl.Layer):
    def __init__(self, transform, action, name='TransformedActionMatrixBase', **kwargs):
        super(TransformedActionMatrixBase, self).__init__(autocast=False, name=name, **kwargs)
        self.action = action
        self.transform = transform
    def call(self, x):
        "Returns the action, the log Jacobian, and extra info from transform."
        y, l, bs = self.transform(x)
        a = self.action(y)
        return a-l, l, bs
    def gradient(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x.to_tensors())
            v, l, b = self(x)
        d = x.from_tensors(tape.gradient(v, x.to_tensors()))(x.adjoint()).projectTangent()
        return d, l, b

class TransformedActionVectorBase(TransformedActionMatrixBase):
    def __init__(self, name='TransformedActionVectorBase', **kwargs):
        super(TransformedActionVectorBase, self).__init__(name=name, **kwargs)
    def gradient(self, x):
        sv = x.zeroTangentVector()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(sv.to_tensors())
            v, l, b = self(sv.exp()(x))
        d = sv.from_tensors(tape.gradient(v, sv.to_tensors()))
        return d, l, b

class TransformedActionVectorFromMatrixBase(TransformedActionMatrixBase):
    def __init__(self, name='TransformedActionVectorFromMatrixBase', **kwargs):
        super(TransformedActionVectorFromMatrixBase, self).__init__(name=name, **kwargs)
    def gradient(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x.to_tensors())
            v, l, b = self(x)
        # factor of 0.5 due to TF convention of complex derivatives
        d = 0.5*x.from_tensors(tape.gradient(v, x.to_tensors()))(x.adjoint()).projectTangent().to_tangentVector()
        return d, l, b

class QuadraticMomentum:
    def __call__(self, p):
        return p.energy()
    def gradient(self, p):
        return p

class Dynamics:
    def __init__(self, V, T=QuadraticMomentum()):
        """
        For splittable Hamiltonian, H(x,p) = T(p) + V(x)
        stepP: p(τ+ε) = p(τ) - ε ∂/∂x H(x, p(τ)) = p(τ) - ε d/dx V(x)
        stepX: x(τ+ε) = x(τ) + ε ∂/∂p H(x(τ), p) = x(τ) + ε d/dp T(p)
        For recording purposes, stepP returns more information,
        We will save those directly to Dynamics later.
        stepP(eps, x, p) => newp, (d/dx V), logJ, coeffs
        stepX(eps, x, p) => newx
        """
        self.V = V
        self.T = T
    def stepP(self, eps, x, p):
        d, l, b = self.V.gradient(x)
        newp = (-eps)*d + p
        return newp, d, l, b
    def stepX(self, eps, x, p):
        d = self.T.gradient(p)
        return (eps*d).exp()(x)

if __name__ == '__main__':
    import sys, os
    import nthmc, evolve, transform
    conf = nthmc.Conf(nbatch=1, nepoch=2, nstepEpoch=8, trajLength=4.0, stepPerTraj=128)
    nthmc.setup(conf)
    mom = QuadraticMomentum()
    act = SU3d4(beta=0.7796, c1=C1DBW2)
    # act = SU3d4(beta=6.0, c1=0)
    tact = TransformedActionVectorBase(transform=transform.Ident(), action=act)
    rng = tf.random.Generator.from_seed(conf.seed)

    if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
        x0 = gauge.readGauge(sys.argv[1])
    else:
        x0 = gauge.random(rng, [8,8,8,16])
    # x0 = x0.hypercube_partition()

    for pl in gauge.plaquette(x0):
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)

    v0,_,_ = tact(x0)
    tf.print('action', v0)
    g,_,_ = tact.gradient(x0)
    for i,gd in enumerate(g):
        tf.print('force norm2 dim',i,':',gd.norm2())
    # ge,_ = tf.linalg.eig(g[0].get_batch(0).get_site(0,0,0,0))
    # tf.print('geig',tf.math.real(ge),tf.math.imag(ge))

    p0 = x0.randomTangentVector(rng)
    t0 = mom(p0)
    tf.print('H0', v0+t0, v0, t0)

    @tf.function(jit_compile=True)
    def mdfun_(x_,p_,md):
        x = x0.from_tensors(x_)
        p = p0.from_tensors(p_)
        xn,pn,a,b,c,d = md(x,p)
        return xn.to_tensors(),pn.to_tensors(),a,b,c,d
    def mdfun(x,p,md):
        x_,p_,a,b,c,d = mdfun_(x.to_tensors(),p.to_tensors(),md)
        return x.from_tensors(x_),p.from_tensors(p_),a,b,c,d

    dyn = Dynamics(V=tact, T=mom)

    mdLF = evolve.LeapFrog(conf, dyn)
    for i in range(3):
        tbegin = tf.timestamp()
        x, p, _, _, _, _ = mdfun(x0,p0,mdLF)
        tf.print('# MD time:',i,tf.timestamp()-tbegin,'sec',summarize=-1)
    for pl in gauge.plaquette(x):
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)
    v1,_,_ = tact(x)
    t1 = mom(p)
    tf.print('H1', v1+t1, v1, t1)
    tf.print('dH', v1+t1-(v0+t0))

    conf.stepPerTraj = 64
    mdOM = evolve.Omelyan2MN(conf, dyn)
    for i in range(3):
        tbegin = tf.timestamp()
        x, p, _, _, _, _ = mdfun(x0,p0,mdOM)
        tf.print('# MD time:',i,tf.timestamp()-tbegin,'sec',summarize=-1)
    for pl in gauge.plaquette(x):
        tf.print(tf.math.real(pl), tf.math.imag(pl), summarize=-1)
    v1,_,_ = tact(x)
    t1 = mom(p)
    tf.print('H1', v1+t1, v1, t1)
    tf.print('dH', v1+t1-(v0+t0))

    @tf.function(jit_compile=True)
    def mcmcfun_(x_,p_,r):
        x = x0.from_tensors(x_)
        p = p0.from_tensors(p_)
        xn, pn, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x,p,r)
        return xn.to_tensors(),pn.to_tensors(),x1.to_tensors(),p1.to_tensors(), v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs
    def mcmcfun(x,p,r):
        x_, p_, x1_, p1_, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmcfun_(x.to_tensors(),p.to_tensors(),r)
        return x.from_tensors(x_),p.from_tensors(p_),x.from_tensors(x1_),p.from_tensors(p1_), v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs
    mcmc = nthmc.Metropolis(conf, mdOM)

    for i in range(16):
        tbegin = tf.timestamp()
        p = x.randomTangentVector(rng)
        xn, pn, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmcfun(x, p, rng.uniform([], dtype=tf.float64))
        tf.print('# mcmc step',i,'time:',tf.timestamp()-tbegin,'sec',summarize=-1)
        nthmc.printMCMCRes(*nthmc.packMCMCRes(mcmc, xn, pn, x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs))
        x, p = xn, pn

    conf.checkReverse=True
    mcmcChRev = nthmc.Metropolis(conf, mdOM)
    mcmcChRev(x,p,rng.uniform([], dtype=tf.float64))
