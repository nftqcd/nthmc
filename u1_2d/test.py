import nthmc, ftr
import tensorflow as tf
import tensorflow.keras as tk
import math, os, unittest
import sys
sys.path.append("../lib")
import field, group

class TestGenericStoutSmear(unittest.TestCase):

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
        pi = tf.constant(math.pi, dtype=tf.float64)
        op0 = (((1,2,-1,-2), (1,-2,-1,2)),
                    ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
        op1 = (((2,-1,-2,1), (2,1,-2,-1)),
                    ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
        self.testShape = (3,2,6,8)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.get_global_generator().uniform(self.testShape, -math.pi, math.pi, dtype=tf.float64)
        self.testMask = tf.constant([[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0]], dtype=tf.float64)
        self.ss = [
            ftr.GenericStoutSmear(((0,0),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,1),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,0),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,1),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,0),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,1),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,0),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,1),(2,2)), op1, [], ftr.Scalar(2)),
        ]
        for i,s in enumerate(self.ss):
            s.build(self.testShape)
            s.layerCoefficient.xs.assign([1.0+0.1*i, 1.0+0.01*i])

    def test_mask(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                self.assertTrue(tf.reduce_all(s.maskUpdate == tf.roll(self.testMask, shift=(i//2,i), axis=(0,1))))

    def test_call(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, _, _ = s(self.testField)
                m = 1 - s.maskUpdate
                self.assertTrue(tf.reduce_all(m*y == m*self.testField))

    def test_jacob(self):
        v = tf.math.reduce_prod(self.testShape[1:])
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                x = self.testField
                with tf.GradientTape(persistent=True) as t:    # persistent for jacobian without pfor
                    t.watch(x)
                    y, ld, _ = s(x)
                j = t.batch_jacobian(y, x, experimental_use_pfor=False)    # pfor fails for roll op
                for b in range(self.testShape[0]):
                    ldj = 0.
                    for mu in range(self.testShape[1]):
                        for x in range(self.testShape[2]):
                            for y in range(self.testShape[3]):
                               ldj += tf.math.log(j[b,mu,x,y,mu,x,y])
                    with self.subTest(b=b):
                        with self.subTest(test='diagonal'):
                            self.assertAlmostEqual(ld[b].numpy(), ldj.numpy(), places=14)
                        with self.subTest(test='full matrix'):
                            self.assertAlmostEqual(ld[b].numpy(), tf.math.log(tf.linalg.det(tf.reshape(j[b], (v,v)))).numpy(), places=14)

    def test_inv(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, l, _ = s(self.testField)
                z, m, invIter = s.inv(y)
                if invIter >= s.invMaxIter:
                    tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',s.invMaxIter, summarize=-1)
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(z, self.testField)), 1E-28)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(l, -m)), 1E-28)

    def test_symmetry_translation(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld, _ = s(self.testField)
                sx = tf.roll(self.testField, (2,4), (2,3))
                sy, sld, _ = s(sx)
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, tf.roll(sy, (-2,-4), (2,3)))), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseX(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld, _ = s(self.testField)
                sx = tf.reverse(self.testField, [2])
                sx = tf.stack([tf.roll(-sx[:,0], -1, 1), sx[:,1]], 1)
                if s.linkDir != 1:  # first link is on other directions
                    s.maskUpdate = tf.roll(s.maskUpdate, -1, 0)
                sy, sld, _ = s(sx)
                sy = tf.reverse(tf.stack([tf.roll(-sy[:,0], 1, 1), sy[:,1]], 1), [2])
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseY(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld, _ = s(self.testField)
                sx = tf.reverse(self.testField, [3])
                sx = tf.stack([sx[:,0], tf.roll(-sx[:,1], -1, 2)], 1)
                if s.linkDir != 2:  # first link is on other directions
                    s.maskUpdate = tf.roll(s.maskUpdate, -1, 1)
                sy, sld, _ = s(sx)
                sy = tf.reverse(tf.stack([sy[:,0], tf.roll(-sy[:,1], 1, 2)], 1), [3])
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_gauge(self):
        G = tf.random.get_global_generator().uniform(self.latticeShape, -math.pi, math.pi, dtype=tf.float64)
        u = group.U1Phase
        tx = tf.stack(
            [u.mul(u.mul(G,self.testField[:,d]), tf.roll(G, -1, axis=1+d), adjoint_r=True) for d in range(2)],
            1)
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld, _ = s(self.testField)
                ty = tf.stack(
                    [u.mul(u.mul(G,y[:,d]), tf.roll(G, -1, axis=1+d), adjoint_r=True) for d in range(2)],
                    1)
                sy, sld, _ = s(tx)
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(u.compatProj(ty), u.compatProj(sy))), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

class TestChain(unittest.TestCase):

    def setUp(self):
        pi = tf.constant(math.pi, dtype=tf.float64)
        op0 = (((1,2,-1,-2), (1,-2,-1,2)),
                    ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
        op1 = (((2,-1,-2,1), (2,1,-2,-1)),
                    ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
        self.testShape = (3,2,6,8)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.get_global_generator().uniform(self.testShape, -math.pi, math.pi, dtype=tf.float64)
        self.ss = ftr.TransformChain([
            ftr.GenericStoutSmear(((0,0),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,1),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,0),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,1),(2,2)), op0, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,0),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((0,1),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,0),(2,2)), op1, [], ftr.Scalar(2)),
            ftr.GenericStoutSmear(((1,1),(2,2)), op1, [], ftr.Scalar(2)),
        ])
        self.ss.build(self.testShape)
        for i,s in enumerate(self.ss.chain):
            s.layerCoefficient.xs.assign([1.0+0.2*i, 1.0+0.04*i])

    def test_jacob(self):
        v = tf.math.reduce_prod(self.testShape[1:])
        x = self.testField
        with tf.GradientTape(persistent=True) as t:    # persistent for jacobian without pfor
            t.watch(x)
            y, ld, _ = self.ss(x)
        j = t.batch_jacobian(y, x, experimental_use_pfor=False)    # pfor fails for roll op
        for b in range(self.testShape[0]):
            with self.subTest(b=b):
                self.assertAlmostEqual(ld[b].numpy(), tf.math.log(tf.linalg.det(tf.reshape(j[b], (v,v)))).numpy(), places=13)

    def test_inv(self):
        y, l, _ = self.ss(self.testField)
        z, m, invIter = self.ss.inv(y)
        if invIter >= self.ss.invMaxIter:
            tf.print('WARNING: max inverse iteration reached',invIter,'with invMaxIter',self.ss.invMaxIter, summarize=-1)
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(z, self.testField)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(l, -m)), 1E-24)

    def test_symmetry_translation(self):
        y, ld, _ = self.ss(self.testField)
        sx = tf.roll(self.testField, (2,4), (2,3))
        sy, sld, _ = self.ss(sx)
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, tf.roll(sy, (-2,-4), (2,3)))), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseX(self):
        y, ld, _ = self.ss(self.testField)
        sx = tf.reverse(self.testField, [2])
        sx = tf.stack([tf.roll(-sx[:,0], -1, 1), sx[:,1]], 1)
        for s in self.ss.chain:
            if s.linkDir != 1:  # first link is on other directions
                s.maskUpdate = tf.roll(s.maskUpdate, -1, 0)
                s.unmaskFixedLoop = [tf.roll(m, 1, 0) for m in s.unmaskFixedLoop]
        sy, sld, _ = self.ss(sx)
        sy = tf.reverse(tf.stack([tf.roll(-sy[:,0], 1, 1), sy[:,1]], 1), [2])
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseY(self):
        y, ld, _ = self.ss(self.testField)
        sx = tf.reverse(self.testField, [3])
        sx = tf.stack([sx[:,0], tf.roll(-sx[:,1], -1, 2)], 1)
        for s in self.ss.chain:
            if s.linkDir != 2:  # first link is on other directions
                s.maskUpdate = tf.roll(s.maskUpdate, -1, 1)
                s.unmaskFixedLoop = [tf.roll(m, 1, 1) for m in s.unmaskFixedLoop]
        sy, sld, _ = self.ss(sx)
        sy = tf.reverse(tf.stack([sy[:,0], tf.roll(-sy[:,1], 1, 2)], 1), [3])
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_gauge(self):
        G = tf.random.get_global_generator().uniform(self.latticeShape, -math.pi, math.pi, dtype=tf.float64)
        u = group.U1Phase
        tx = tf.stack(
            [u.mul(u.mul(G,self.testField[:,d]), tf.roll(G, -1, axis=1+d), adjoint_r=True) for d in range(2)],
            1)
        y, ld, _ = self.ss(self.testField)
        ty = tf.stack(
            [u.mul(u.mul(G,y[:,d]), tf.roll(G, -1, axis=1+d), adjoint_r=True) for d in range(2)],
            1)
        sy, sld, _ = self.ss(tx)
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(u.compatProj(ty), u.compatProj(sy))), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

class TestConvChain(TestChain):

    def setUp(self):
        pi = tf.constant(math.pi, dtype=tf.float64)
        op0 = (((1,2,-1,-2), (1,-2,-1,2)),
                    ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
        op1 = (((2,-1,-2,1), (2,1,-2,-1)),
                    ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
        self.testShape = (3,2,8,6)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.get_global_generator().uniform(self.testShape, -math.pi, math.pi, dtype=tf.float64)
        fixedP = (1,2,-1,-2)
        fixedR0 = (2,2,1,-2,-2,-1)
        fixedR1 = (1,1,2,-1,-1,-2)
        convP0 = lambda: ftr.PeriodicConv((
            tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        ))
        convP1 = lambda: ftr.PeriodicConv((
            tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        ))
        convR = lambda pad: ftr.PeriodicConv((
            tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        ), pad)
        conv = lambda: ftr.PeriodicConv((
            tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
            tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
        ))
        self.ss = ftr.TransformChain([
            ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
            ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
            ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
            ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
            ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
            ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
            ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
            ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        ])
        self.ss.build(self.testShape)

    def test_symmetry_reverseX(self):
        # need to flip conv layer weights
        pass
    def test_symmetry_reverseY(self):
        # need to flip conv layer weights
        pass

if __name__ == '__main__':
    tf.random.set_seed(9876543211)
    tf.keras.backend.set_floatx('float64')
    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(False)
    tf.config.threading.set_inter_op_parallelism_threads(1)    # ALCF suggests number of socket
    tf.config.threading.set_intra_op_parallelism_threads(4)    # ALCF suggests number of physical cores
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    unittest.main()
