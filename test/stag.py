from . import testutil as tu
from ..lib import gauge, stagD
import tensorflow as tf

class TestGauge(tu.LatticeTest):
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
            (0, [8,8,8,16], 3),
            (3, [8,8,8,16], 3),
            (3, [8,8,8,16], 5),
        ]

    def testBC(self):
        for nb,dims,nc in self.test_shapes:
            with self.subTest(nb=nb,dims=dims,nc=nc):
                lat = gauge.unit(dims, nbatch=nb, nc=nc)
                latbc = gauge.setBC(lat)
                latP = lat.hypercube_partition()
                latbcP = gauge.setBC(latP)
                latPbc = latbcP.combine_hypercube()
                self.check_eqv(latbc, latPbc)

    def testPhase(self):
        for nb,dims,nc in self.test_shapes:
            with self.subTest(nb=nb,dims=dims,nc=nc):
                lat = gauge.unit(dims, nbatch=nb, nc=nc)
                latph = stagD.phase(lat)
                latP = lat.hypercube_partition()
                latphP = stagD.phase(latP)
                latPph = latphP.combine_hypercube()
                self.check_eqv(latph, latPph)

if __name__ == '__main__':
    tu.main()
