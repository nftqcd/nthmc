import tensorflow as tf
import tensorflow.keras.layers as tl

class Ident(tl.Layer):
    def __init__(self, name='Ident', **kwargs):
        super(Ident, self).__init__(autocast=False, name=name, **kwargs)
        self.invMaxIter = 1
    def call(self, x):
        return (x, tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
    def inv(self, y):
        return (y, tf.constant(0.0, dtype=tf.float64), 0)
    def showTransform(self, **kwargs):
        tf.print(self.name, **kwargs)
