from . import testutil as tu
from ..lib import lattice as l
from ..lib import group as p
from ..lib import gauge as g
import tensorflow as tf

class TestGaugeOp(tu.LatticeTest):
    """ Examples:
    def setUp(self):
        print('setUp')

    def tearDown(self):
        print('tearDown')

    def test_example(self):
        self.assertEqual('foo'.upper(), 'FOO')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    """
    def setUp(self):
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)
        self.test_shapes = [
            (-1, [4,16,8,6,8,3,3]),
            ( 0, [3,4,14,8,8,6,3,3]),
        ]

    def test_checkSU(self):
        for bd,dims in self.test_shapes:
            with self.subTest(bd=bd,dims=dims):
                lat = self.random(dims)
                if bd<0:
                    rp = p.checkSU(tf.expand_dims(lat, 0))
                else:
                    rp = p.checkSU(lat)
                gauge = g.from_tensor(lat, batch_dim=bd)
                resg = gauge.checkSU()
                gaugeP = gauge.hypercube_partition()
                resp = gaugeP.checkSU()
                self.check_eqv(rp, resg)
                self.check_eqv(rp, resp)
                if bd<0:
                    with self.subTest(extra='batch of 1'):
                        latE = tf.expand_dims(lat, 0)
                        gauge = g.from_tensor(latE, batch_dim=0)
                        resg = gauge.checkSU()
                        gaugeP = gauge.hypercube_partition()
                        resp = gaugeP.checkSU()
                        self.check_eqv(rp, resg)
                        self.check_eqv(rp, resp)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
