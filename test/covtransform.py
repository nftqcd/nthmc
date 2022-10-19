from math import pi
from . import testutil as tu
from ..lib import action, lattice, field, transform
from ..lib import gauge as g
import tensorflow as tf
import tensorflow.keras.layers as tl

class TestSmearSliceCovariance(tu.LatticeTest):
    def setUp(self):
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)

    def test0(self):
        self.check(-1, [4,2,4,2,2,3,3])

    def test1(self):
        self.check(0, [1,4,2,2,6,4,3,3])

    def test2(self):
        self.check(0, [3,4,2,2,2,2,3,3])

    def check(self, bd, dims):
        gauge = g.from_tensor(self.random(dims), batch_dim=bd).projectSU()
        Glat = lattice.Lattice(self.random(dims[:bd+1]+dims[bd+2:]), batch_dim=bd)
        G = g.Transporter(Glat, path=field.Path()).projectSU()
        gaugeT = gauge.wrap([G(g(G.adjoint())) for g in gauge])
        tmap = transform.TransformChain(
            [transform.StoutSmearSlice(
                coeff=transform.CoefficientNets([
                    transform.SymmetricShifts(symmetric_shifts=1),
                    transform.Residue(transform.LocalSelfAttention(num_heads=8,key_dim=4)),
                    tl.Dense(units=9, activation='swish'),
                    transform.FlattenSiteLocal(input_local_rank=2),
                    transform.Normalization(),
                    transform.Residue(transform.LocalFeedForward(inner_size=108, inner_activation='swish')),
                    transform.Normalization(),
                    tl.Dense(units=54, activation=None)]),
                dir=i, is_odd=eo)
             for eo in {False,True} for i in range(4)])
        gmap,l,b = tmap(gauge)
        gTmap,lT,bT = tmap(gaugeT)
        gTmapT = gTmap.wrap([G.adjoint()(g(G)) for g in gTmap])
        with self.subTest(quantity='logDetJ'):
            self.check_eqv(l,lT, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='coeff'):
            self.check_eqv(b,bT)
        with self.subTest(quantity='gauge'):
            self.check_eqv(gmap,gTmapT)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
