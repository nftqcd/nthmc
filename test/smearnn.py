from math import pi
import tensorflow as tf
import tensorflow.keras.layers as tl
from . import testutil as tu
from ..lib import action, transform
from ..lib import gauge as g
from ..lib.lattice import SubSetEven, SubSetOdd

class TestSmearSliceNN(tu.LatticeTest):
    def setUp(self):
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)
        self.test_shapes = [
            (-1, [4,4,2,2,2,3,3]),
            ( 0, [1,4,2,6,2,2,3,3]),
            ( 0, [3,4,2,2,2,2,3,3]),
        ]

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
        for n in {6,54}:
            with self.subTest(n_coeff=n):
                self.smear_shape_coeff(bd,dims,dir,is_odd,n)

    def smear_shape_coeff(self,bd,dims,dir,is_odd,n_coeff):
        coeff = self.rng.normal([n_coeff], dtype=tf.float64)
        coeffnn=transform.CoefficientNets([
            transform.SymmetricShifts(symmetric_shifts=1),
            transform.Residue(transform.LocalSelfAttention(num_heads=3,key_dim=5)),
            transform.FlattenSiteLocal(input_local_rank=2),
            tl.Dense(units=64, activation='swish'),
            transform.Normalization(),
            transform.Residue(transform.LocalFeedForward(inner_size=128, inner_activation='swish')),
            transform.Normalization(),
            tl.Dense(units=n_coeff, activation=None, kernel_initializer='zeros')])
        latshape = dims[2+bd:6+bd]+[306,]
        coeffnn(tf.zeros(latshape,tf.float64))
        coeffnn.chain[-1].bias.assign(coeff)
        mkmap = lambda:transform.StoutSmearSlice(coeff=coeff, dir=dir, is_odd=is_odd)
        mkmapnn = lambda:transform.StoutSmearSlice(coeff=coeffnn, dir=dir, is_odd=is_odd)
        gauge = g.from_tensor(self.random(dims), batch_dim=bd).projectSU()
        with self.subTest(part=False):
            self.check(mkmap(), mkmapnn(), gauge)
        with self.subTest(part=True):
            self.check(mkmap(), mkmapnn(), gauge.hypercube_partition())

    def check(self, tmap, tmapnn, gauge):
        gmap,l,b = tmap(gauge)
        gmapnn,lnn,bnn = tmapnn(gauge)
        with self.subTest(quantity='logDetJ'):
            self.check_eqv(l,lnn)
        with self.subTest(quantity='coeff'):
            self.check_eqv(b,bnn)
        with self.subTest(quantity='gauge'):
            self.check_eqv(gmap,gmapnn)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
