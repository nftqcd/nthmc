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
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)
        self.test_shapes = [
            (-1, [16,8,8,8,3]),
            (-1, [16,8,8,8,3,3]),
            (-1, [16,8,8,8,1]),
            ( 0, [4,16,8,8,8,3,3]),
            (-1, [16,8,8,6,3]),
            (-1, [16,8,8,6,3]),
            (-1, [16,8,6,8,3]),
            (-1, [16,6,8,8,3]),
            (-1, [14,8,8,8,3]),
            ( 0, [3,16,8,8,6,3]),
            ( 0, [3,16,8,8,6,3]),
            ( 0, [3,16,8,6,8,3]),
            ( 0, [3,16,6,8,8,3]),
            ( 0, [3,14,8,8,8,3]),
            ( 0, [3,16,6,8,6,3]),
            ( 0, [3,16,8,6,6,3]),
            ( 0, [3,16,8,6,8,3]),
            ( 0, [3,16,6,6,8,3]),
            ( 0, [3,14,8,8,6,3]),
        ]

    def test_ident(self):
        for bd,dims in self.test_shapes:
            with self.subTest(bd=bd,dims=dims):
                lat = self.random(dims)
                latC = l.Lattice(lat, nd=4, batch_dim=bd)
                latP = latC.hypercube_partition()
                latN = latP.combine_hypercube()
                self.check_eqv(latN, latC)
                if bd<0:
                    with self.subTest(extra='batch of 1'):
                        latE = l.Lattice(tf.expand_dims(lat, 0), nd=4, batch_dim=0)
                        latNP = latE.hypercube_partition()
                        latNN = latNP.combine_hypercube()
                        self.check_eqv(latNN, latE)

    def test_tensor_conv(self):
        for bd,dims in self.test_shapes:
            with self.subTest(bd=bd,dims=dims):
                lat = self.random(dims)
                latC = l.Lattice(lat, nd=4, batch_dim=bd)
                latP = latC.hypercube_partition()
                Z = latP.zeros()
                latN = Z.from_tensors(latP.to_tensors())
                self.check_eqv(latN, latP)
                if bd<0:
                    with self.subTest(extra='batch of 1'):
                        latE = l.Lattice(tf.expand_dims(lat, 0), nd=4, batch_dim=0)
                        latNP = latE.hypercube_partition()
                        Z = latNP.zeros()
                        latNN = Z.from_tensors(latNP.to_tensors())
                        self.check_eqv(latNN, latNP)

    def test_shift(self):
        for bd,dims in self.test_shapes:
            with self.subTest(bd=bd,dims=dims):
                latC = l.Lattice(self.random(dims), nd=4, batch_dim=bd)
                for d in range(4):
                    for n in range(-4,5):
                        with self.subTest(dir=d,len=n):
                            latS = latC.shift(d, n)
                            latP = latC.hypercube_partition()
                            latPS = latP.shift(d, n)
                            latN = latPS.combine_hypercube()
                            with self.subTest(subset='all'):
                                self.check_eqv(latN, latS)
                            with self.subTest(subset='even/odd'):
                                latE = latP.get_subset(l.SubSetEven)
                                latO = latP.get_subset(l.SubSetOdd)
                                latNE = latPS.get_subset(l.SubSetEven)
                                latNO = latPS.get_subset(l.SubSetOdd)
                                latES = latE.shift(d, n)
                                latOS = latO.shift(d, n)
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
