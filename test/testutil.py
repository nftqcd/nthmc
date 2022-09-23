import unittest as ut
import sys
sys.path.append("../lib")
import lattice, gauge
import group as g
import tensorflow as tf

ut.TestCase.defaultTestResult = None

class LatticeTest(ut.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(4321)
        tf.keras.backend.set_floatx('float64')

    def check_eqv(self,a,b,tol=1e-13,rtol=1e-13,alwaysPrint=False,printDetail=False):
        if isinstance(a, (lattice.Lattice,gauge.Transporter)) or isinstance(b, (lattice.Lattice,gauge.Transporter)):
            if a.is_compatible(b):
                self.check_eqv(a.unwrap(), b.unwrap(), tol,rtol,alwaysPrint,printDetail)
            else:
                raise ValueError(f'a and b are not compatible {a.__class__} vs {b.__class__}')
        elif isinstance(a, (list,tuple,gauge.Gauge)) and isinstance(b, (list,tuple,gauge.Gauge)) and len(a)==len(b):
            for i,x,y in zip(range(len(a)),a,b):
                with self.subTest(idx=i):
                    self.check_eqv(x,y,tol,rtol,alwaysPrint,printDetail)
        elif isinstance(a, (list,tuple,gauge.Gauge)) or isinstance(b, (list,tuple,gauge.Gauge)):
            raise ValueError(f'different classes {a.__class__} vs. {b.__class__}')
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
                if self._subself is None:
                    print(f'{self.id()}: sq diff abs: {m}  rel {m/mn}')
                else:
                    print(f'{self._subself.id()}: sq diff abs: {m}  rel {m/mn}')
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

    def test_check_eqv(self):
        self.check_eqv(tf.constant(0.1,tf.float64), tf.constant(0.10000000000001,tf.float64))
        self.check_eqv(tf.constant([0.1],tf.float64), tf.constant([0.10000000000001],tf.float64))
        self.check_eqv(tf.constant([0.1,0.1],tf.float64), tf.constant([0.10000000000001,0.10000000000001],tf.float64))

    @ut.expectedFailure
    def test_check_eqv_failure(self):
        self.check_eqv(tf.constant(0.1,tf.float64), tf.constant(0.1000000000001,tf.float64))
        self.check_eqv(tf.constant([0.1],tf.float64), tf.constant([0.1000000000001],tf.float64))
        self.check_eqv(tf.constant([0.1,0.1],tf.float64), tf.constant([0.1000000000001,0.10000000000001],tf.float64))

def main(*args, **kwargs):
    ut.main(*args, verbosity=2, **kwargs)

if __name__=='__main__':
    main()
