import sys
sys.path.append("../lib")
import lattice as l
import group as g
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
            (0, [16,8,8,8,3]),
            (0, [16,8,8,8,3,3]),
            (0, [16,8,8,8,1]),
            (1, [4,16,8,8,8,3,3]),
            (0, [16,8,8,6,3]),
            (0, [16,8,8,6,3]),
            (0, [16,8,6,8,3]),
            (0, [16,6,8,8,3]),
            (0, [14,8,8,8,3]),
            (1, [3,16,8,8,6,3]),
            (1, [3,16,8,8,6,3]),
            (1, [3,16,8,6,8,3]),
            (1, [3,16,6,8,8,3]),
            (1, [3,14,8,8,8,3]),
            (1, [3,16,6,8,6,3]),
            (1, [3,16,8,6,6,3]),
            (1, [3,16,8,6,8,3]),
            (1, [3,16,6,6,8,3]),
            (1, [3,14,8,8,6,3]),
            (2, [3,4,16,8,8,8,3,3]),
        ]

    def test_ident(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = self.random(dims)
                latP = l.hypercube_partition(lat, nd=4, batch_dims=nb)
                latN = l.combine_hypercube(latP, nd=4, batch_dims=nb)
                self.check_eqv(latN, lat)
                if nb==0:
                    with self.subTest(extra='batch of 1'):
                        latE = tf.expand_dims(lat, 0)
                        latNP = l.hypercube_partition(latE, nd=4, batch_dims=1)
                        latNN = l.combine_hypercube(latNP, nd=4, batch_dims=1)
                        self.check_eqv(latNN, latE)

    def test_shift(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = self.random(dims)
                for d in range(4):
                    for n in range(-4,5):
                        with self.subTest(dir=d,len=n):
                            latS = l.shift(lat, d, n, nd=4, batch_dims=nb)
                            latP = l.hypercube_partition(lat, nd=4, batch_dims=nb)
                            latPS = l.shift(latP, d, n, nd=4, batch_dims=nb)
                            latN = l.combine_hypercube(latPS, nd=4, batch_dims=nb)
                            with self.subTest(subset='all'):
                                self.check_eqv(latN, latS)
                            with self.subTest(subset='even/odd'):
                                latE = l.get_even(latP)
                                latO = l.get_odd(latP)
                                latNE = l.get_even(latPS)
                                latNO = l.get_odd(latPS)
                                latES = l.shift(latE, d, n, nd=4, batch_dims=nb)
                                latOS = l.shift(latO, d, n, nd=4, batch_dims=nb)
                                with self.subTest(check='even'):
                                    if n%2==0:
                                        self.check_eqv(latES, latNE)
                                    else:
                                        self.check_eqv(latOS, latNE)
                                with self.subTest(check='odd'):
                                    if n%2==0:
                                        self.check_eqv(latOS, latNO)
                                    else:
                                        self.check_eqv(latES, latNO)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
