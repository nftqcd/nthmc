import sys
sys.path.append("../lib")
import lattice as l
import group as g
import stagD as s
import tensorflow as tf
import testutil as tu

class TestHypercube(tu.LatticeTest):
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
        self.rng = tf.random.Generator.from_seed(7654321)
        self.test_shapes = [
            (0, [4,16,8,8,8,3,3]),
            (1, [4,3,16,8,8,8,3,3]),
        ]

    def testBC(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = g.unit(dims)
                latbc = l.setBC(lat, batch_dims=nb)
                latP = l.hypercube_partition(lat, batch_dims=nb+1)
                latbcP = l.setBC(latP, batch_dims=nb)
                latPbc = l.combine_hypercube(latbcP, batch_dims=nb+1)
                self.check_eqv(latbc, latPbc)

    def testPhase(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = g.unit(dims)
                latph = s.phase(lat, batch_dims=nb)
                latP = l.hypercube_partition(lat, batch_dims=nb+1)
                latphP = s.phase(latP, batch_dims=nb)
                latPph = l.combine_hypercube(latphP, batch_dims=nb+1)
                self.check_eqv(latph, latPph)

if __name__ == '__main__':
    tu.main()
