import sys
sys.path.append("../lib")
import field as f
import group as g
import tensorflow as tf
import unittest as ut

class TestOrderedPaths(ut.TestCase):
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
        self.x = tf.reshape(tf.linspace(0, 1, 3*2*4*4), [3,2,4,4])
        p = f.OrdPaths(
            f.topath(((1,2,-1,-2), (1,-2,-1,2),
                (1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2),
                (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1))))
        self.o = f.OrdProduct(p, self.x, g.U1Phase)
        self.tol = 1E-14

    def test_path12_1_2(self):
        a = self.o.prod((1,2,-1,-2))
        b = self.x[:,0] + tf.roll(self.x[:,1], shift=-1, axis=1) - tf.roll(self.x[:,0], shift=-1, axis=2) - self.x[:,1]
        self.checkEqv(a, b)

    def test_path1_2_12(self):
        a = self.o.prod((1,-2,-1,2))
        b = self.x[:,0] - tf.roll(self.x[:,1], shift=[-1,1], axis=[1,2]) - tf.roll(self.x[:,0], shift=1, axis=2) + tf.roll(self.x[:,1], shift=1, axis=2)
        self.checkEqv(a, b)

    def test_path112_1_1_2(self):
        a = self.o.prod((1,1,2,-1,-1,-2))
        b = self.x[:,0] + tf.roll(self.x[:,0], shift=-1, axis=1) + tf.roll(self.x[:,1], shift=-2, axis=1) - tf.roll(self.x[:,0], shift=[-1,-1], axis=[1,2]) - tf.roll(self.x[:,0], shift=-1, axis=2) - self.x[:,1]
        self.checkEqv(a, b)

    def test_path11_2_1_12(self):
        a = self.o.prod((1,1,-2,-1,-1,2))
        b = self.x[:,0] + tf.roll(self.x[:,0], shift=-1, axis=1) - tf.roll(self.x[:,1], shift=[-2,1], axis=[1,2]) - tf.roll(self.x[:,0], shift=[-1,1], axis=[1,2]) - tf.roll(self.x[:,0], shift=1, axis=2) + tf.roll(self.x[:,1], shift=1, axis=2)
        self.checkEqv(a, b)

    def test_path12_1_1_21(self):
        a = self.o.prod((1,2,-1,-1,-2,1))
        b = self.x[:,0] + tf.roll(self.x[:,1], shift=-1, axis=1) - tf.roll(self.x[:,0], shift=-1, axis=2) - tf.roll(self.x[:,0], shift=[1,-1], axis=[1,2]) - tf.roll(self.x[:,1], shift=1, axis=1) + tf.roll(self.x[:,0], shift=1, axis=1)
        self.checkEqv(a, b)

    def test_path1_2_1_121(self):
        a = self.o.prod((1,-2,-1,-1,2,1))
        b = self.x[:,0] - tf.roll(self.x[:,1], shift=[-1,1], axis=[1,2]) - tf.roll(self.x[:,0], shift=1, axis=2) - tf.roll(self.x[:,0], shift=[1,1], axis=[1,2]) + tf.roll(self.x[:,1], shift=[1,1], axis=[1,2]) + tf.roll(self.x[:,0], shift=1, axis=1)
        self.checkEqv(a, b)

    def checkEqv(self,a,b):
        m = tf.reduce_mean(tf.math.squared_difference(a, b))
        if self.tol <= m:
            print(f'received {a}')
            print(f'expected {b}')
        self.assertGreater(self.tol, m)

if __name__ == '__main__':
    ut.main()
