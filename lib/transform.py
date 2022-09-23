import tensorflow as tf
import tensorflow.keras.layers as tl
import gauge
from lattice import evenodd_mask, SubSet, SubSetEven, SubSetOdd, combine_subsets

class TransformBase(tl.Layer):
    def __init__(self, name='TransformerBase', **kwargs):
        super(TransformBase, self).__init__(autocast=False, name=name, **kwargs)
        self.invMaxIter = 1
    def __call__(self, x):
        """
        To make build get shape, we have to get bare.
        We need to wrap in call.
        """
        self.from_tensors = x.from_tensors
        y,l,b = super(TransformBase, self).__call__(x.to_tensors())
        return x.from_tensors(y),l,b
    def call(self, x):
        """
        This is the call with x in tensors.
        Children of this class should implement transform instead.
        """
        y,l,b = self.transform(self.from_tensors(x))
        return y.to_tensors(),l,b
    def showTransform(self, **kwargs):
        tf.print(self.name, **kwargs)

class Ident(TransformBase):
    def __init__(self, name='Ident', **kwargs):
        super(Ident, self).__init__(name=name, **kwargs)
    def transform(self, x):
        return (x, tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
    def inv(self, y):
        return (y, tf.constant(0.0, dtype=tf.float64), 0)

class StoutSmearSlice(TransformBase):
    """
    Smear a single direction and even/odd slice of the gauge field,
    such that the Jacobian matrix of the transformation is diagonal in volume.
    """
    def __init__(self, coeff, dir, is_odd, invAbsR2=1E-24, invMaxIter=8192, name='StoutSmearSlice', **kwargs):
        super(StoutSmearSlice, self).__init__(name=name, **kwargs)
        self.coeff = coeff
        self.dir = dir
        self.is_odd = is_odd
        if is_odd:
            self.subset_fix = SubSetEven
            self.subset_upd = SubSetOdd
        else:
            self.subset_fix = SubSetOdd
            self.subset_upd = SubSetEven
        self.invAbsR2 = invAbsR2
        self.invMaxIter = invMaxIter
    def build(self, shape):
        # assuming the last two dimensions are on site
        # 4 dimension lattice volume
        # 1 optional leading batch dimension
        assert(isinstance(shape, list) and len(shape)==4)
        sh = shape[0]
        if isinstance(sh, tf.TensorShape):
            if len(sh)==6:
                dim = sh[:4]
                maskshape = dim+(1,1)
            elif len(sh)==7:
                dim = sh[1:5]
                maskshape = (1,)+dim+(1,1)
            else:
                raise ValueError(f'unimplemented for lattice shape {shape}')
            mask = tf.reshape(evenodd_mask(dim)%2, maskshape)
            if not self.is_odd:
                mask = 1-mask
            self.filter = tf.cast(mask, tf.complex128)  # 1 for update
            self.parted = False
        elif isinstance(sh, list) and len(sh)==16 and isinstance(sh[0], tf.TensorShape):
            # hypercube partitioned
            self.parted = True
        else:
            raise ValueError(f'unimplemented for shape {shape}')
    def transform(self, x):
        f,l,b = self.compute_change(x)
        return self.slice_apply(f,x),l,b
    def filter_fixed_update(self, xin):
        if self.parted:
            f = xin.wrap([t.get_subset(self.subset_fix) if d==self.dir else t for d,t in enumerate(xin.unwrap())])
            u = xin[self.dir].get_subset(self.subset_upd)
            return f,u
        else:
            f = xin.wrap([t*t.from_tensors(1-self.filter) if d==self.dir else t for d,t in enumerate(xin.unwrap())])
            u = xin[self.dir]*xin[self.dir].from_tensors(self.filter)
            return f,u
    def slice_apply(self, f, x):
        if self.parted:
            return x.wrap([t.wrap(combine_subsets(t.get_subset(self.subset_fix).unwrap(), f(t.get_subset(self.subset_upd)).unwrap())) if d==self.dir else t for d,t in enumerate(x.unwrap())])
        else:
            return x.wrap([f(t) if d==self.dir else t for d,t in enumerate(x.unwrap())])
    def scale_coeff(self, x, max):
        pi_2 = 0.63661977236758134307    # 2/pi
        if not tf.is_tensor(x):
            x = tf.math.atan(tf.convert_to_tensor(x,dtype=tf.float64))
        return tf.constant(pi_2*max,dtype=tf.float64) * tf.math.atan(x)
    def compute_change(self, xin, change_only=False):
        mu = self.dir
        x,xupd = self.filter_fixed_update(xin)
        stf = [None]*4
        stu = [None]*4
        for nu in range(4):
            if nu!=mu:
                umu = x[mu].shift(nu, 1)
                unu = x[nu].shift(mu, 1)
                umunu = umu(unu.adjoint())
                stf[nu] = x[nu](umunu)
                unumu = x[nu].adjoint()(x[mu])
                stu[nu] = unumu(unu)
        f = 0.0
        for nu in range(4):
            if nu!=mu:
                f += stf[nu] + stu[nu].shift(nu, -1)
        if callable(self.coeff):
            # pass symmetrized plaq field to self.coeff
            # coeff = self.coeff(sym_plaq_field)
            raise ValueError('TODO callable self.coeff')
        else:
            coeff = self.coeff
        cplaq = self.scale_coeff(coeff[0],0.125)
        f *= f.typecast(cplaq)    # 3/4 * 1/6  for 6 SU(3) terms in f
        if len(coeff)>1:
            # 6-edge terms: 1,2,3,-2,-1,-3
            raise ValueError('TODO len(coeff)>1')
        if change_only:
            fexp = f(xupd.adjoint()).projectTangent().exp()
            return fexp
        else:
            fexp,logdet = f.smearIndepLogDetJacobian(xupd)
            return fexp, logdet, cplaq
    def inv_iter(self, x):
        return self.compute_change(x, change_only=True)
    def inv(self, y):
        # NOTE: inv batch at the same time.  Does it worth inv single lattice at a time?
        ydir = y[self.dir]
        x = y
        f = self.inv_iter(x)
        u = f.unit()
        c = self.invAbsR2*2*(3*3+1)
        def cond(i_,x_,f_):
            xt = x.from_tensors(x_)
            ft = f.from_tensors(f_)
            d = tf.math.reduce_mean((ft(xt[self.dir])(ydir.adjoint())-u).norm2(scope='site').reduce_mean())
            # tf.print('inv',i_,d)
            return i_<self.invMaxIter and d>c
        def body(i_,x_,f_):
            xt = x.from_tensors(x_)
            xn = self.slice_apply(f.from_tensors(f_).adjoint(),y)
            return i_+1, xn.to_tensors(), self.inv_iter(xn).to_tensors()
        i,x_,f_ = tf.while_loop(cond, body, [0, x.to_tensors(), f.to_tensors()])
        x = x.from_tensors(x_)
        f = f.from_tensors(f_)
        x = self.slice_apply(f.adjoint(),y)
        _, l, _ = self.compute_change(x)
        return (x, -l, i)
    def showTransform(self, **kwargs):
        tf.print(self.name,self.dir,self.is_odd, **kwargs)

class TransformChain(TransformBase):
    def __init__(self, transforms, name='TransformChain', **kwargs):
        super(TransformChain, self).__init__(name=name, **kwargs)
        self.chain = transforms
        self.invMaxIter = 0
        for t in transforms:
            if hasattr(t,'invMaxIter'):
                if self.invMaxIter < t.invMaxIter:
                    self.invMaxIter = t.invMaxIter
    def transform(self, x):
        y = x
        l = 0.0
        bs = 0.0
        for f in self.chain:
            y, t, b = f(y)
            l += t
            bs += b
        bs /= len(self.chain)
        return (y, l, bs)
    def inv(self, y):
        x = y
        l = 0.0
        imax = 0
        for f in reversed(self.chain):
            x, t, i = f.inv(x)
            l += t
            if imax < i:
                imax = i
        return (x, l, imax)
    def showTransform(self, **kwargs):
        n = len(self.chain)
        tf.print(self.name, '[', n, ']', **kwargs)
        for i in range(n):
            tf.print(i, ':', end='')
            self.chain[i].showTransform(**kwargs)

if __name__ == '__main__':
    import sys, os
    from math import pi
    import nthmc, action, evolve, gauge, lattice, transform
    conf = nthmc.Conf(nbatch=1, nepoch=2, nstepEpoch=8, trajLength=0.1, stepPerTraj=4)
    nthmc.setup(conf)
    rng = tf.random.Generator.from_seed(conf.seed)

    if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
        x0 = gauge.readGauge(sys.argv[1])
    else:
        x0 = gauge.random(rng, [2,2,2,4])
    x0 = x0.hypercube_partition()

    dyn = action.Dynamics(
        action.TransformedActionVectorBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(coeff=[tf.math.tan(tf.constant(0.1*4*pi,dtype=tf.float64))], dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]),
            action=action.SU3d4(beta=0.7796, c1=action.C1DBW2)))
    dyn.V.transform.showTransform()

    def show_plaq(x):
        for i,pl in enumerate(gauge.plaquette(x)):
            tf.print('base',i,':',tf.math.real(pl), tf.math.imag(pl), summarize=-1)
        x,_,_ = dyn.V.transform(x)
        for i,pl in enumerate(gauge.plaquette(x)):
            tf.print('trns',i,':',tf.math.real(pl), tf.math.imag(pl), summarize=-1)
        t = tf.timestamp()
        x,_,i = dyn.V.transform.inv(x)
        t = tf.timestamp()-t
        tf.print('inv_iter',i,'time',t,'sec')
        for i,pl in enumerate(gauge.plaquette(x)):
            tf.print('invt',i,':',tf.math.real(pl), tf.math.imag(pl), summarize=-1)

    show_plaq(x0)
    if isinstance(dyn.V, action.TransformedActionMatrixBase):
        p0 = x0.randomTangent(rng)
    else:
        p0 = x0.randomTangentVector(rng)

    md = evolve.LeapFrog(conf, dyn)
    def run(steps):
        md.setStepPerTraj(steps)
        v0,l,b = md.dynamics.V(x0)
        tf.print('logdet',l)
        tf.print('coeff',b)
        t0 = md.dynamics.T(p0)
        tf.print('H0',v0+t0,v0,t0)
        x1,p1,_,_,_,_ = md(x0,p0)
        v1,l,b = md.dynamics.V(x1)
        tf.print('logdet',l)
        tf.print('coeff',b)
        t1 = md.dynamics.T(p1)
        tf.print('H1',v1+t1,v1,t1)
        tf.print('dH',v1+t1-v0-t0)
        show_plaq(x1)

    run(4)
    run(8)
