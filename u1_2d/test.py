import nthmc
import tensorflow as tf
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
        self.op0 = field.OrdPaths(
            field.topath(((1,2,-1,-2), (1,-2,-1,2),
                (1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2),
                (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1))))
        self.op1 = field.OrdPaths(
            field.topath(((2,-1,-2,1), (2,1,-2,-1),
                (2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1),
                (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2))))
        self.pathmap = (2,4)
        self.testShape = (3,2,6,8)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.uniform(self.testShape, dtype=tf.float64)*(2*pi)-pi
        self.testMask = tf.constant([[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,0,0,0,0,0,0,0]], dtype=tf.float64)
        self.ss = [
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            ]
        for i,s in enumerate(self.ss):
            s.build(self.testShape)
            s.alphaLayer.xs.assign([1.0+0.1*i, 1.0+0.01*i])

    def test_mask(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                self.assertTrue(tf.reduce_all(s.mask == tf.roll(self.testMask, shift=(i//2,i), axis=(0,1))))

    def test_call(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                # makes sure masks were applied as we intended with non-trivial alphaLayer
                s.alphaLayer.xs.assign([1.0+0.1*i, 1.0+0.01*i])
                y, _ = s(self.testField)
                m = 1 - s.mask
                self.assertTrue(tf.reduce_all(m*y == m*self.testField))

    def test_jacob(self):
        v = tf.math.reduce_prod(self.testShape[1:])
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                x = self.testField
                with tf.GradientTape(persistent=True) as t:    # persistent for jacobian without pfor
                    t.watch(x)
                    y, ld = s(x)
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
                y, l = s(self.testField)
                z, m = s.inv(y)
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(z, self.testField)), 1E-28)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(l, -m)), 1E-28)

    def test_symmetry_translation(self):
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld = s(self.testField)
                sx = tf.roll(self.testField, (2,4), (2,3))
                sy, sld = s(sx)
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, tf.roll(sy, (-2,-4), (2,3)))), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseX(self):
        act = nthmc.U1d2(nthmc.Ident(), size=self.testShape[2:])
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld = s(self.testField)
                sx = tf.reverse(self.testField, [2])
                sx = tf.stack([tf.roll(-sx[:,0], -1, 1), sx[:,1]], 1)
                if s.op.paths[0].flatten()[0] != 1:  # first link is on other directions
                    s.mask = tf.roll(s.mask, -1, 0)
                sy, sld = s(sx)
                sy = tf.reverse(tf.stack([tf.roll(-sy[:,0], 1, 1), sy[:,1]], 1), [2])
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseY(self):
        act = nthmc.U1d2(nthmc.Ident(), size=self.testShape[2:])
        for i,s in enumerate(self.ss):
            with self.subTest(i=i):
                y, ld = s(self.testField)
                sx = tf.reverse(self.testField, [3])
                sx = tf.stack([sx[:,0], tf.roll(-sx[:,1], -1, 2)], 1)
                if s.op.paths[0].flatten()[0] != 2:  # first link is on other directions
                    s.mask = tf.roll(s.mask, -1, 1)
                sy, sld = s(sx)
                sy = tf.reverse(tf.stack([sy[:,0], tf.roll(-sy[:,1], 1, 2)], 1), [3])
                with self.subTest(test='field'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
                with self.subTest(test='logdet'):
                    self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_gauge(self):
        pass

class TestChain(unittest.TestCase):

    def setUp(self):
        pi = tf.constant(math.pi, dtype=tf.float64)
        self.op0 = field.OrdPaths(
            field.topath(((1,2,-1,-2), (1,-2,-1,2),
                (1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2),
                (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1))))
        self.op1 = field.OrdPaths(
            field.topath(((2,-1,-2,1), (2,1,-2,-1),
                (2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1),
                (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2))))
        self.pathmap = (2,4)
        self.testShape = (3,2,6,8)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.uniform(self.testShape, dtype=tf.float64)*(2*pi)-pi
        self.ss = nthmc.TransformChain([
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=nthmc.Scalar(2), alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            ])
        self.ss.build(self.testShape)
        for i,s in enumerate(self.ss.chain):
            s.alphaLayer.xs.assign([1.0+0.2*i, 1.0+0.04*i])

    def test_jacob(self):
        v = tf.math.reduce_prod(self.testShape[1:])
        x = self.testField
        with tf.GradientTape(persistent=True) as t:    # persistent for jacobian without pfor
            t.watch(x)
            y, ld = self.ss(x)
        j = t.batch_jacobian(y, x, experimental_use_pfor=False)    # pfor fails for roll op
        for b in range(self.testShape[0]):
            with self.subTest(b=b):
                self.assertAlmostEqual(ld[b].numpy(), tf.math.log(tf.linalg.det(tf.reshape(j[b], (v,v)))).numpy(), places=14)

    def test_inv(self):
        y, l = self.ss(self.testField)
        z, m = self.ss.inv(y)
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(z, self.testField)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(l, -m)), 1E-24)

    def test_symmetry_translation(self):
        y, ld = self.ss(self.testField)
        sx = tf.roll(self.testField, (2,4), (2,3))
        sy, sld = self.ss(sx)
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, tf.roll(sy, (-2,-4), (2,3)))), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseX(self):
        y, ld = self.ss(self.testField)
        sx = tf.reverse(self.testField, [2])
        sx = tf.stack([tf.roll(-sx[:,0], -1, 1), sx[:,1]], 1)
        for s in self.ss.chain:
            if s.op.paths[0].flatten()[0] != 1:  # first link is on other directions
                s.mask = tf.roll(s.mask, -1, 0)
                s.unmask = [tf.roll(m, 1, 0) for m in s.unmask]
        sy, sld = self.ss(sx)
        sy = tf.reverse(tf.stack([tf.roll(-sy[:,0], 1, 1), sy[:,1]], 1), [2])
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_reverseY(self):
        y, ld = self.ss(self.testField)
        sx = tf.reverse(self.testField, [3])
        sx = tf.stack([sx[:,0], tf.roll(-sx[:,1], -1, 2)], 1)
        for s in self.ss.chain:
            if s.op.paths[0].flatten()[0] != 2:  # first link is on other directions
                s.mask = tf.roll(s.mask, -1, 1)
                s.unmask = [tf.roll(m, 1, 1) for m in s.unmask]
        sy, sld = self.ss(sx)
        sy = tf.reverse(tf.stack([sy[:,0], tf.roll(-sy[:,1], 1, 2)], 1), [3])
        with self.subTest(test='field'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(y, sy)), 1E-26)
        with self.subTest(test='logdet'):
            self.assertLess(tf.reduce_mean(tf.math.squared_difference(ld, sld)), 1E-26)

    def test_symmetry_gauge(self):
        pass

class TestConvChain(TestChain):

    def setUp(self):
        pi = tf.constant(math.pi, dtype=tf.float64)
        def conv0():
            return nthmc.PeriodicConv((
                tf.keras.layers.Conv2D(4, (3,2), activation='gelu', kernel_initializer=tf.keras.initializers.Constant(0.5), bias_initializer=tf.keras.initializers.Constant(1.2)),
                tf.keras.layers.Conv2D(2, 1, activation='gelu', kernel_initializer=tf.keras.initializers.Constant(0.1), bias_initializer=tf.keras.initializers.Constant(-0.07)),
                ))
        def conv1():
            return nthmc.PeriodicConv((
                tf.keras.layers.Conv2D(4, (2,3), activation='gelu', kernel_initializer=tf.keras.initializers.Constant(0.6), bias_initializer=tf.keras.initializers.Constant(1.4)),
                tf.keras.layers.Conv2D(2, 1, activation='gelu', kernel_initializer=tf.keras.initializers.Constant(0.3), bias_initializer=tf.keras.initializers.Constant(-0.08)),
                ))
        self.op0 = field.OrdPaths(
            field.topath((
                # for derivatives
                (1,2,-1,-2), (1,-2,-1,2),
                (1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2),
                (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1),
                # for alphalayer
                (-1,-2,1,2),
                )))
        self.op1 = field.OrdPaths(
            field.topath((
                # for derivatives
                (2,-1,-2,1), (2,1,-2,-1),
                (2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1),
                (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2),
                # for alphalayer
                (-1,-2,1,2),
                )))
        self.am0 = ([(1,0),(1,1)],
                   )
        self.am1 = ([(0,1),(1,1)],
                   )
        self.pathmap = (2,4)
        self.testShape = (3,2,6,8)
        self.latticeShape = (self.testShape[0],)+self.testShape[2:]
        self.testField = tf.random.uniform(self.testShape, dtype=tf.float64)*(2*pi)-pi
        self.ss = nthmc.TransformChain([
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=conv0(), alphamasks=self.am0, alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=conv0(), alphamasks=self.am0, alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=conv0(), alphamasks=self.am0, alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op0, alphalayer=conv0(), alphamasks=self.am0, alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=conv1(), alphamasks=self.am1, alphamap=self.pathmap, first=(0,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=conv1(), alphamasks=self.am1, alphamap=self.pathmap, first=(1,0), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=conv1(), alphamasks=self.am1, alphamap=self.pathmap, first=(0,1), repeat=(2,2)),
            nthmc.GenericStoutSmear(ordpaths=self.op1, alphalayer=conv1(), alphamasks=self.am1, alphamap=self.pathmap, first=(1,1), repeat=(2,2)),
            ])
        self.ss.build(self.testShape)

if __name__ == '__main__':
    tf.random.set_seed(987654321)
    tf.keras.backend.set_floatx('float64')
    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_inter_op_parallelism_threads(1)    # ALCF suggests number of socket
    tf.config.threading.set_intra_op_parallelism_threads(4)    # ALCF suggests number of physical cores
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    unittest.main()
