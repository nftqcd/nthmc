import sys
sys.path.append("../lib")
import parts
import group as g
import tensorflow as tf
import unittest as ut

class LatticeTest(ut.TestCase):
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
    def check_eqv(self,a,b,tol=1e-13,rtol=1e-13,alwaysPrint=False,printDetail=False):
        if isinstance(a, parts.HypercubeParts) and isinstance(b, parts.HypercubeParts) and a.subset==b.subset and len(a)==len(b):
            for i,x,y in zip(range(len(a)),a,b):
                with self.subTest(idx=i):
                    self.check_eqv(x,y,tol,rtol,alwaysPrint,printDetail)
        elif isinstance(a, parts.HypercubeParts) or isinstance(b, parts.HypercubeParts):
            raise ValueError(f'a and b must both be instance of HypercubeParts')
        else:
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

main = ut.main
