import tensorflow as tf
import tensorflow.keras as tk

class Scale(tk.layers.Layer):
    def __init__(self, c, dt):
        super(Scale, self).__init__(name='Scale')
        self.c = self.add_weight(initializer=tf.keras.initializers.Constant(c), dtype=dt)
    @tf.function(jit_compile=True)
    def call(self, x):
        return self.c*x

print('float32:', Scale(2, tf.float32)(tf.ones(4, dtype=tf.float32)))
print('int64:', Scale(2, tf.int64)(tf.ones(4, dtype=tf.int64)))
print('int32:', Scale(2, tf.int32)(tf.ones(4, dtype=tf.int32)))
