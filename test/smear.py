from math import pi
from . import testutil as tu
from ..lib import action, transform
from ..lib import gauge as g
from ..lib.lattice import SubSetEven, SubSetOdd
import tensorflow as tf

class TestSmearSlice(tu.LatticeTest):
    def setUp(self):
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)
        self.test_shapes = [
            (-1, [4,4,2,2,2,3,3]),
            ( 0, [1,4,2,6,2,2,3,3]),
            ( 0, [3,4,2,2,2,2,3,3]),
        ]
        self.w = action.SU3d4(beta=0.12)


    def test_smear_shape0_dir0_even(self):
        self.smear_shape(*self.test_shapes[0],dir=0,is_odd=False)
    def test_smear_shape0_dir1_even(self):
        self.smear_shape(*self.test_shapes[0],dir=1,is_odd=False)
    def test_smear_shape0_dir2_even(self):
        self.smear_shape(*self.test_shapes[0],dir=2,is_odd=False)
    def test_smear_shape0_dir3_even(self):
        self.smear_shape(*self.test_shapes[0],dir=3,is_odd=False)
    def test_smear_shape0_dir0_odd(self):
        self.smear_shape(*self.test_shapes[0],dir=0,is_odd=True)
    def test_smear_shape0_dir1_odd(self):
        self.smear_shape(*self.test_shapes[0],dir=1,is_odd=True)
    def test_smear_shape0_dir2_odd(self):
        self.smear_shape(*self.test_shapes[0],dir=2,is_odd=True)
    def test_smear_shape0_dir3_odd(self):
        self.smear_shape(*self.test_shapes[0],dir=3,is_odd=True)

    def test_smear_shape1_dir0_even(self):
        self.smear_shape(*self.test_shapes[1],dir=0,is_odd=False)
    def test_smear_shape1_dir1_even(self):
        self.smear_shape(*self.test_shapes[1],dir=1,is_odd=False)
    def test_smear_shape1_dir2_even(self):
        self.smear_shape(*self.test_shapes[1],dir=2,is_odd=False)
    def test_smear_shape1_dir3_even(self):
        self.smear_shape(*self.test_shapes[1],dir=3,is_odd=False)
    def test_smear_shape1_dir0_odd(self):
        self.smear_shape(*self.test_shapes[1],dir=0,is_odd=True)
    def test_smear_shape1_dir1_odd(self):
        self.smear_shape(*self.test_shapes[1],dir=1,is_odd=True)
    def test_smear_shape1_dir2_odd(self):
        self.smear_shape(*self.test_shapes[1],dir=2,is_odd=True)
    def test_smear_shape1_dir3_odd(self):
        self.smear_shape(*self.test_shapes[1],dir=3,is_odd=True)

    def test_smear_shape2_dir0_even(self):
        self.smear_shape(*self.test_shapes[2],dir=0,is_odd=False)
    def test_smear_shape2_dir1_even(self):
        self.smear_shape(*self.test_shapes[2],dir=1,is_odd=False)
    def test_smear_shape2_dir2_even(self):
        self.smear_shape(*self.test_shapes[2],dir=2,is_odd=False)
    def test_smear_shape2_dir3_even(self):
        self.smear_shape(*self.test_shapes[2],dir=3,is_odd=False)
    def test_smear_shape2_dir0_odd(self):
        self.smear_shape(*self.test_shapes[2],dir=0,is_odd=True)
    def test_smear_shape2_dir1_odd(self):
        self.smear_shape(*self.test_shapes[2],dir=1,is_odd=True)
    def test_smear_shape2_dir2_odd(self):
        self.smear_shape(*self.test_shapes[2],dir=2,is_odd=True)
    def test_smear_shape2_dir3_odd(self):
        self.smear_shape(*self.test_shapes[2],dir=3,is_odd=True)

    def smear_shape(self,bd,dims,dir,is_odd):
        mkmap = lambda:transform.StoutSmearSlice(coeff=tf.math.tan(tf.constant(self.w.beta/3*4*pi,dtype=tf.float64))*tf.ones([6],tf.float64), dir=dir, is_odd=is_odd)
        lat = self.random(dims)
        gauge = g.from_tensor(lat, batch_dim=bd).projectSU()
        ss = SubSetOdd if is_odd else SubSetEven
        v = gauge.hypercube_partition().zeroTangentVector()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(v[dir].get_subset(ss).to_tensors())
            vg = v.combine_hypercube().exp()(gauge)
            fplaq = self.w.gradientPlaq(vg)
            gmapref = fplaq.exp().adjoint()(vg)
            gup = gmapref[dir].get_subset(ss)
            gup_nograd = gup.from_tensors(tf.stop_gradient(gup.to_tensors()))
            ug = tf.stack(gup(gup_nograd.adjoint()).unwrap().to_su3vector().to_tensors(), axis=bd+1)
            if bd<0:
                uv = tf.reshape(ug,[-1])
            else:
                uv = tf.reshape(ug,[ug.shape[0],-1])
        jacob = tape.jacobian(uv,v[dir].get_subset(ss).to_tensors())
        if bd<0:
            js = tf.stack(jacob,axis=1)
            js = tf.reshape(js,[js.shape[0],-1])
        else:
            js = tf.stack(jacob,axis=3)
            js = tf.reshape(js,[js.shape[0],js.shape[1],js.shape[2],-1])
            js = tf.transpose(js, perm=[0,2,1,3])
        logdetjref = tf.math.log(tf.linalg.det(js))
        if bd==0:
            logdetjref = tf.linalg.diag_part(logdetjref)
        with self.subTest(part=False):
            self.check_update_slices(mkmap(), gauge, gmapref, logdetjref, dir, is_odd)
        with self.subTest(part=True):
            self.check_update_slices(mkmap(), gauge.hypercube_partition(), gmapref.hypercube_partition(), logdetjref, dir, is_odd)

    def check_update_slices(self, tmap, gauge, gmapref, logdetjref, dir, is_odd):
        gmap,l,b = tmap(gauge)
        with self.subTest(quantity='logDetJ'):
            self.check_eqv(logdetjref, l, tol=1e-11, rtol=1e-10)    # prec loss in direct det?
        with self.subTest(quantity='coeff'):
            scaled_coeff = tf.reduce_mean(0.75*2/pi*tf.math.atan(tmap.coeff))
            scaled_coeff = tf.stack([scaled_coeff,0,scaled_coeff,0])
            bs = gauge.batch_size()
            if bs==0:
                self.check_eqv(scaled_coeff, b)
            else:
                for i in range(bs):
                    with self.subTest(batch=i):
                        self.check_eqv(scaled_coeff, b[i])
        if is_odd:
            ss,ssfix = SubSetOdd,SubSetEven
        else:
            ss,ssfix = SubSetEven,SubSetOdd
        ss = SubSetOdd if is_odd else SubSetEven
        with self.subTest(update=False):
            for d in range(4):
                if d!=dir:
                    with self.subTest(checkdir=d):
                        self.check_eqv(gauge[d], gmap[d])
            with self.subTest(checkdir=dir,subset=ssfix):
                self.check_eqv(
                    gauge[dir].lattice.get_subset(ssfix),
                    gmap[dir].lattice.get_subset(ssfix))
        with self.subTest(update=True,checkdir=dir,subset=ss):
            self.check_eqv(
                gmapref[dir].lattice.get_subset(ss),
                gmap[dir].lattice.get_subset(ss))
        gmap_inv,l_inv,itr = tmap.inv(gmap)    # tmap.invAbsR2=1E-24 by default, thus the tol/rtol here
        with self.subTest(quantity='inv gauge'):
            self.check_eqv(gauge, gmap_inv, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='inv logDetJ'):
            self.check_eqv(l, -l_inv, tol=1e-12, rtol=1e-11)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
