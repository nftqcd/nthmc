import tensorflow as tf
import tensorflow.keras.layers as tl
import gauge
from lattice import Lattice, evenodd_mask, SubSetAll, SubSetEven, SubSetOdd, combine_subsets

def scale_coeff(x, max):
    pi_2 = 0.63661977236758134307    # 2/pi
    if not tf.is_tensor(x):
        x = tf.math.atan(tf.convert_to_tensor(x,dtype=tf.float64))
    return tf.constant(pi_2*max,dtype=tf.float64) * tf.math.atan(x)

def power3_trace(x):
    x2 = x(x)
    x3 = x(x2)
    return x.trace(),x2.trace(),x3.trace()

class TransformBase(tl.Layer):
    def __init__(self, name='TransformerBase', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
        self.invMaxIter = 1
    def __call__(self, x):
        """
        To make build get shape, we have to get bare.
        We need to wrap in call.
        """
        self.from_tensors = x.from_tensors
        y,l,b = super().__call__(x.to_tensors())
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
        super().__init__(name=name, **kwargs)
    def transform(self, x):
        bs = x.batch_size()
        if bs==0:
            return (x, tf.zeros([], dtype=tf.float64), tf.zeros([], dtype=tf.float64))
        else:
            return (x, tf.zeros([bs], dtype=tf.float64), tf.zeros([bs], dtype=tf.float64))
    def inv(self, y):
        bs = y.batch_size()
        if bs==0:
            return (y, tf.zeros([], dtype=tf.float64), tf.zeros([], dtype=tf.float64))
        else:
            return (y, tf.zeros([bs], dtype=tf.float64), tf.zeros([bs], dtype=tf.float64))

class StoutSmearSlice(TransformBase):
    """
    Smear a single direction and even/odd slice of the gauge field,
    such that the Jacobian matrix of the transformation is diagonal in volume.
    """
    def __init__(self, coeff, dir, is_odd, invAbsR2=1E-24, invMaxIter=8192, name='StoutSmearSlice', **kwargs):
        super().__init__(name=name, **kwargs)
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
                raise ValueError(f'unsupported lattice shape {shape}')
            mask = tf.reshape(evenodd_mask(dim)%2, maskshape)
            if not self.is_odd:
                mask = 1-mask
            self.filter = tf.cast(mask, tf.complex128)  # 1 for update
            self.parted = False
        elif isinstance(sh, list) and len(sh)==16 and isinstance(sh[0], tf.TensorShape):
            # hypercube partitioned
            self.parted = True
        else:
            raise ValueError(f'unsupported lattice shape {shape}')
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
    def stack_tensors(self, loop_filter, loop_others):
        if self.parted:    # loop_filter are E/O subset, loop_others are full lattices
            zero = loop_others[0].zeros()
            loop_filter = [l+zero for l in loop_filter]
        treal = []
        for l in loop_filter+loop_others:
            if self.parted:
                l = l.combine_hypercube()
            t = l.to_tensors()
            treal += [tf.math.real(t),tf.math.imag(t)]
        return tf.stack(treal, axis=-1)    # 306
    def compute_change(self, xin, change_only=False):
        n_plaq = 6
        n_chair = 48
        mu = self.dir
        x,xupd = self.filter_fixed_update(xin)
        Umu = [None]*4
        Unu = [None]*4
        R = [None]*4
        L = [None]*4
        stf = [None]*4
        stu = [None]*4
        for nu in range(4):
            if nu!=mu:
                Umu[nu] = x[mu].shift(nu, 1)
                Unu[nu] = x[nu].shift(mu, 1)
                R[nu] = Umu[nu](Unu[nu].adjoint())
                stf[nu] = x[nu](R[nu])
                L[nu] = x[nu].adjoint()(x[mu])
                stu[nu] = L[nu](Unu[nu])
        loop_wo_mu = []
        for nu in range(4):
            if nu!=mu:
                loop_wo_mu.append(stf[nu])
                loop_wo_mu.append(stu[nu].shift(nu, -1))
        Sf = [[None]*4 for _ in range(4)]
        Sb = [[None]*4 for _ in range(4)]
        if callable(self.coeff) or self.coeff.shape[-1]==n_plaq+n_chair:    # compute staples off mu
            # pass symmetrized wilson loops to self.coeff if callable
            # 1. plaquette perpendicular to mu, 3 dirs
            # 2. rectangle perpendicular to mu, 2 types, 3 dirs
            loop_perp = []
            for nu in range(1,4):
                if nu==mu:
                    continue
                for xi in range(nu):
                    if xi==mu:
                        continue
                    xxi = x[xi].shift(nu, 1)
                    xnu = x[nu].shift(xi, 1)
                    Unuxi = x[nu](xxi)
                    Uxinu = x[xi](xnu)
                    Sb[xi][nu] = x[xi].adjoint()(Unuxi).shift(xi, -1)
                    Sb[nu][xi] = x[nu].adjoint()(Uxinu).shift(nu, -1)
                    Sf[nu][xi] = Unuxi(xnu.adjoint())
                    Sf[xi][nu] = Uxinu(xxi.adjoint())
                    if callable(self.coeff):
                        # PLAQUETTE
                        plaq = power3_trace(Unuxi(Uxinu.adjoint()))
                        # 3 more, symmetrical around the point
                        plaq_nu = [p.shift(nu, -1) for p in plaq]
                        plaq_xi = [p.shift(xi, -1) for p in plaq]
                        plaq_nuxi = [p.shift(xi, -1) for p in plaq_nu]
                        # RECTANGLE
                        rect_lr = power3_trace(Sf[nu][xi](Sb[nu][xi].adjoint()))
                        # 1 from back, symmetrical around the point
                        rect_lr_xi = [r.shift(xi, -1) for r in rect_lr]
                        rect_du = power3_trace(Sf[xi][nu](Sb[xi][nu].adjoint()))
                        # 1 from back, symmetrical around the point
                        rect_du_nu = [r.shift(nu, -1) for r in rect_du]
                        # SAVE
                        loop_perp += [*plaq, *plaq_nu, *plaq_xi, *plaq_nuxi]    # 12 numbers
                        loop_perp += [*rect_lr, *rect_lr_xi, *rect_du, *rect_du_nu]    # 12 numbers
        if callable(self.coeff):
            # symmetrize about the link
            loop_perp_mu = [loop.shift(mu, 1) for loop in loop_perp]
            loop_perp += loop_perp_mu    # 144 numbers in total
            # 3. rectangles parallel to mu, 1 type, 3 dirs, from half of the links in mu
            loop_para = []
            for nu in range(4):
                if nu!=mu:
                    loop_para += power3_trace(stf[nu](stu[nu].adjoint()))    # already symmetrical about the link
            sym_field = self.stack_tensors(loop_para, loop_perp)
            coeff = self.coeff(sym_field)
        else:
            coeff = self.coeff
        c_scaled_real = scale_coeff(coeff,0.75)    # 3/4 for each SU(3) terms in f
        c_scaled = loop_wo_mu[0].typecast(c_scaled_real)
        if c_scaled.shape[-1]==n_plaq+n_chair:
            # 6-edge terms: mu,nu,-mu,xi,-nu,-xi or mu,xi,nu,-xi,-mu,-nu (6 staples along mu x 2 sides x 4 staples/side)
            for nu in range(4):
                if nu==mu:
                    continue
                Unumu = x[nu](Umu[nu])
                Umunu = x[mu](Unu[nu])
                for xi in range(4):
                    if xi==mu or xi==nu:
                        continue
                    # Unumu is nu-mu
                    loop_wo_mu.append(Unumu(Sf[xi][nu].adjoint()))    # Sf[xi][nu] is xi-nu-xi, adjoint makes it -nu dir
                    loop_wo_mu.append(Unumu(Sb[xi][nu].adjoint()))    # Sb[xi][nu] is (-xi)-nu-xi, adjoint makes it -nu dir
                    # L[nu] is (-nu)-mu
                    loop_wo_mu.append(L[nu](Sf[xi][nu]).shift(nu, -1))
                    loop_wo_mu.append(L[nu](Sb[xi][nu]).shift(nu, -1))
                    # Umunu is mu-nu
                    loop_wo_mu.append(Sf[xi][nu].adjoint()(Umunu).shift(nu, -1))
                    loop_wo_mu.append(Sb[xi][nu].adjoint()(Umunu).shift(nu, -1))
                    # R[nu] is mu-(-nu)
                    loop_wo_mu.append(Sf[xi][nu](R[nu]))
                    loop_wo_mu.append(Sb[xi][nu](R[nu]))
        elif c_scaled.shape[-1]!=n_plaq:
            raise ValueError(f'unsupported coeff shape {c_scaled.shape}, site dim must be n_plaq or n_plaq+n_chair')
        f = 0.0
        for i,s in enumerate(loop_wo_mu):
            if len(c_scaled.shape)==1:    # global coefficients
                f += s*c_scaled[i]
            elif self.parted and len(c_scaled.shape) in {7,8}:    # (1 batch_dim +) 4 dim + 2 mat + 1 coeff_dim
                f += s*Lattice(c_scaled[...,i], nd=s.unwrap().nd, batch_dim=s.unwrap().batch_dim).hypercube_partition()
            else:    # hope for the best
                f += s*c_scaled[...,i]
        f /= f.typecast(len(loop_wo_mu))
        if change_only:
            fexp = f(xupd.adjoint()).projectTangent().exp()
            return fexp
        else:
            fexp,logdet = f.smearIndepLogDetJacobian(xupd)
            return fexp, logdet, c_scaled_real
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
        super().__init__(name=name, **kwargs)
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

class CoefficientNets(tl.Layer):
    def __init__(self, sequence, name='CoefficientNets', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
        self.chain = sequence
    def call(self, x):
        y = x
        for nn in self.chain:
            y = nn(y)
        return tf.expand_dims(tf.expand_dims(y,-2),-2)

class SymmetricShifts:
    """
    Symmetrically shift 4 dimensions of the lattice one step at a time for max symmetric_shifts times.
    Stack the resulting tensors along the last but 1 axis.
    Always assume site rank 1, a vector on each site.
    Result has one additional dimension at -2, with size (1+2*shifts)^4.
    """
    def __init__(self, symmetric_shifts):
        if (not isinstance(symmetric_shifts,int)) or symmetric_shifts<0:
            raise ValueError(f'unsupported symmetric_shifts={symmetric_shifts})')
        self.symmetric_shifts = symmetric_shifts
    def __call__(self, x):
        # assuming the batch_dim always comes before TZYXC.
        xs = [x]
        for a in range(-4,0):
            xs = [tf.roll(lat, shift=s, axis=a)
                for s in range(-self.symmetric_shifts,1+self.symmetric_shifts)
                for lat in xs]
        y = tf.stack(xs,axis=-2)
        return y

class Normalization(tl.Layer):
    """
    Simaliar to keras.LayerNormalization,
    except that the gamma and beta are site local,
    and share across the lattice.
    """
    def __init__(self, epsilon=1e-12, name='Normalization', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=input_shape[-1:], initializer='zeros', trainable=True)
    def call(self, x):
        # assuming the batch_dim always comes before TZYXC.
        lat_rank = 5
        lat_axis = range(-lat_rank, 0)
        mean = tf.reduce_mean(x, axis=lat_axis)
        xm = x - tf.reshape(mean, mean.shape+(1,)*lat_rank)
        var = tf.reduce_mean(xm**2, axis=lat_axis)
        xnormalized = xm / tf.reshape(tf.sqrt(var+self.epsilon), var.shape+(1,)*lat_rank)
        y = xnormalized * self.gamma + self.beta
        return y

class LocalSelfAttention(tl.MultiHeadAttention):
    """
    Local attention for lattice field.  Assume that input site rank is 2.
    """
    def __init__(self, name='LocalSelfAttention', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
    def call(self, x):
        xflat = tf.reshape(x, shape=(-1,)+tuple(x.shape[-2:]))
        yflat = super().call(query=xflat, value=xflat, key=xflat)
        return tf.reshape(yflat, x.shape)

class Residue(tl.Layer):
    "Add the input to the output of one or a series of procedures."
    def __init__(self, procedure, name='Residue', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
        self.procedure = procedure
    def call(self, x):
        if isinstance(self.procedure, (list,tuple)):
            y = x
            for p in self.procedure:
                y = p(y)
        else:
            y = self.procedure(x)
        x += y
        return x

class LocalFeedForward(tl.Layer):
    def __init__(self, inner_size, inner_activation='swish', name='LocalFeedForward', **kwargs):
        super().__init__(autocast=False, name=name, **kwargs)
        self.inner_size = inner_size
        self.inner_activation = inner_activation
    def build(self, shape):
        self.dense_in = tl.Dense(units=self.inner_size, activation=self.inner_activation)
        self.dense_out = tl.Dense(units=shape[-1], activation=None)
    def call(self, x):
        y = self.dense_in(x)
        z = self.dense_out(y)
        return z

class FlattenSiteLocal:
    def __init__(self, input_local_rank):
        self.input_local_rank = input_local_rank
    def __call__(self, x):
        return tf.reshape(x, tuple(x.shape[:-self.input_local_rank])+(-1,))

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
    # x0 = x0.hypercube_partition()

    """
    dyn = action.Dynamics(
        action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(coeff=tf.math.tan(tf.constant(0.1*4*pi,dtype=tf.float64))*tf.ones([6],tf.float64), dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]),
            action=action.SU3d4(beta=0.7796, c1=action.C1DBW2)))
    dyn = action.Dynamics(
        action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        transform.SymmetricShifts(symmetric_shifts=1),
                        transform.Residue(transform.LocalSelfAttention(num_heads=3,key_dim=5)),
                        tl.Dense(units=16, activation='swish'),
                        transform.FlattenSiteLocal(input_local_rank=2),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=1024, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=6, activation=None)]),
                    dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]),
            action=action.SU3d4(beta=0.7796, c1=action.C1DBW2)))
    """
    dyn = action.Dynamics(
        action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        transform.SymmetricShifts(symmetric_shifts=1),
                        transform.Residue(transform.LocalSelfAttention(num_heads=3,key_dim=5)),
                        tl.Dense(units=16, activation='swish'),
                        transform.FlattenSiteLocal(input_local_rank=2),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=1024, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=54, activation=None)]),
                    dir=i, is_odd=eo)
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
        tf.print('coeff',tf.reduce_mean(b))
        t0 = md.dynamics.T(p0)
        tf.print('H0',v0+t0,v0,t0)
        x1,p1,_,_,_,_ = md(x0,p0)
        v1,l,b = md.dynamics.V(x1)
        tf.print('logdet',l)
        tf.print('coeff',tf.reduce_mean(b))
        t1 = md.dynamics.T(p1)
        tf.print('H1',v1+t1,v1,t1)
        tf.print('dH',v1+t1-v0-t0)
        show_plaq(x1)

    run(4)
    tf.print('non_trainable_variables')
    for i,v in enumerate(dyn.V.transform.non_trainable_variables):
        tf.print(i,v.shape)
    tf.print('trainable_variables')
    for i,v in enumerate(dyn.V.transform.trainable_variables):
        tf.print(i,v.shape)
    tf.print('variables')
    for i,v in enumerate(dyn.V.transform.variables):
        tf.print(i,v.shape)
    tf.print('weights')
    for i,v in enumerate(dyn.V.transform.get_weights()):
        tf.print(i,v.shape)
    run(8)
