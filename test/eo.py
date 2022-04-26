import sys
sys.path.append("../lib")
import lattice as l
import group as g
import tensorflow as tf
import unittest as ut

mul = tf.linalg.matmul
mulv = tf.linalg.matvec
tr = tf.linalg.trace
det = tf.linalg.det
re = tf.math.real
im = tf.math.imag
adj = tf.linalg.adjoint
conj = tf.math.conj
cmplx = tf.dtypes.complex
toCmplx = lambda x: tf.cast(x,tf.complex128)

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

    def testEOIdent(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = self.random(dims)
                latEO = l.evenodd_partition(lat, nd=4, batch_dims=nb)
                latN = l.combine_evenodd(latEO, nd=4, batch_dims=nb)
                self.checkEqv(latN, lat)

    def testEOshift(self):
        for nb,dims in self.test_shapes:
            with self.subTest(nb=nb,dims=dims):
                lat = self.random(dims)
                for d in range(4):
                    for n in range(-4,5):
                        with self.subTest(dir=d,len=n):
                            latS = l.shift(lat, d, n, nd=4, isEO=False, batch_dims=nb)
                            latEO = l.evenodd_partition(lat, nd=4, batch_dims=nb)
                            latEOS = l.shift(latEO, d, n, nd=4, isEO=True, batch_dims=nb)
                            latN = l.combine_evenodd(latEOS, nd=4, batch_dims=nb)
                            with self.subTest(subset='all'):
                                self.checkEqv(latN, latS)
                            with self.subTest(subset='even/odd'):
                                latE = l.get_even(latEO, nb)
                                latO = l.get_odd(latEO, nb)
                                latNE = l.get_even(latEOS, nb)
                                latNO = l.get_odd(latEOS, nb)
                                latES = l.shift(latE, d, n, nd=4, isEO=False, subset='even', batch_dims=nb)
                                latOS = l.shift(latO, d, n, nd=4, isEO=False, subset='odd', batch_dims=nb)
                                with self.subTest(check='even'):
                                    if n%2==0:
                                        self.checkEqv(latES, latNE)
                                    else:
                                        self.checkEqv(latOS, latNE)
                                with self.subTest(check='odd'):
                                    if n%2==0:
                                        self.checkEqv(latOS, latNO)
                                    else:
                                        self.checkEqv(latES, latNO)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

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
