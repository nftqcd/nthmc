import unittest
import tensorflow as tf
class MyLayer:
    def __init__(self, c):
        self.c = c
    def __call__(self, x):
        c = tf.cast(self.c, dtype=tf.complex128)
        return c*x,self.c*x

class TestCast(unittest.TestCase):
    @unittest.expectedFailure
    def test_cast(self):
        mylayer = MyLayer(0.1)
        x = tf.constant(0.1, tf.complex128)
        y,z = mylayer(x)
        self.assertEqual(y,z,'TF issue #57779 #35938')

class TestOps(unittest.TestCase):
    @unittest.expectedFailure
    def test_det(self):
        m = tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float64)
        det = tf.function(tf.linalg.det,jit_compile=True).get_concrete_function(m)
        d = det(m)
        self.assertEqual(1,d.numpy(),'TF issue #57807')
    def test_logdet(self):
        m = tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float64)
        logdet = tf.function(tf.linalg.logdet,jit_compile=True).get_concrete_function(m)
        d = logdet(m)
        self.assertEqual(0,d.numpy())

if __name__ == '__main__':
    unittest.main()
