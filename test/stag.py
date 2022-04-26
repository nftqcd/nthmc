import sys
sys.path.append("../lib")
import lattice as l
import group as g
import tensorflow as tf
import unittest as ut

class TestEvenOdd(ut.TestCase):
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
            (1, [3,4,16,8,8,8,3,3]),
        ]

    def testBC(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = g.unit(dims)
                latbc = l.setBC(lat, isEO=False, batch_dims=nb)
                latEO = l.evenodd_partition(lat, batch_dims=nb+1)
                latbcEO = l.setBC(latEO, isEO=True, batch_dims=nb)
                latEObc = l.combine_evenodd(latbcEO, batch_dims=nb+1)
                self.checkEqv(latbc, latEObc)

    def checkEqv(self,a,b,tol=1e-13,rtol=1e-13,alwaysPrint=False,printDetail=False):
        d = a-b
        v = 1
        for l in d.shape:
            v *= l
        if d.dtype == tf.complex128:
            v *= 2
        axis = range(len(d.shape))
        m = tf.sqrt(g.norm2(d,axis)/v)
        ma = tf.sqrt(g.norm2(b+d,axis)/v)    # broadcast to the same shape
        mb = tf.sqrt(g.norm2(a-d,axis)/v)
        mn = ma if ma>mb else mb
        if alwaysPrint:
            if self._subtest is None:
                print(f'{self.id()}: sq diff abs: {m}  rel {m/mn}')
            else:
                print(f'{self._subtest.id()}: sq diff abs: {m}  rel {m/mn}')
        if m>=tol or (ma>0 and mb>0 and m/mn>=rtol):
            if printDetail:
                print(f'received {a}')
                print(f'expected {b}')
            if not alwaysPrint:
                print(f'abs sq diff: {m}')
                if ma>0 and mb>0:
                    print(f'rel sq diff: {m/mn}')
        with self.subTest(tolerance='absolute'):
            self.assertLess(m, tol)
        if ma>0 and mb>0:
            with self.subTest(tolerance='relative'):
                self.assertLess(m/mn, rtol)

if __name__ == '__main__':
    ut.main()
