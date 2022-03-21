import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tl
import math
import sys
sys.path.append("../lib")
import field, group

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

class Scalar(tl.Layer):
    def __init__(self, output, init=0.0, name='Scalar', **kwargs):
        if isinstance(init, (tuple, list)):
            if len(init)!=output:
                raise ValueError(f'expecting len(init) == {output}, but got: {init}')
            self.xs = tf.Variable(init, dtype=tf.float64)
        else:
            self.xs = tf.Variable((init,)*output, dtype=tf.float64)
        super(Scalar, self).__init__(autocast=False, name=name, **kwargs)
    def call(self, _):
        return self.xs

class PeriodicConv(tl.Layer):
    def __init__(self, layers, pad=None, name='PeriodicConv', **kwargs):
        """
        Periodic pad the input periodically and call layers on it,
        assuming 2D x y dimensions are the -3 and -2 dimension,
        with the last dimension channels, assuming all
        Conv2D layers to be channels_last.
        Channels_first causes error: Conv2DCustomBackpropInputOp only supports NHWC.
        layers are Conv2D layers that provide kernel_size.
        By default the padding is symmetric on both sides, with the extra one from
        the odd padding size placed on the tail side, and the head side is given by
        (sum([l.kernel_size[dim]-1 for l in layers])+1)//2.
        Passing pad to set the default head padding.
        """
        super(PeriodicConv, self).__init__(autocast=False, name=name, **kwargs)
        self.layers = layers
        self.n0 = sum([l.kernel_size[0]-1 for l in layers])
        self.n1 = sum([l.kernel_size[1]-1 for l in layers])
        if pad is None:
            self.pad = ((self.n0+1)//2, (self.n1+1)//2)
        else:
            self.pad = pad
        # print(f'PeriodicConv: total padding {self.n0} {self.n1} left pad {self.pad}')
    def call(self, x):
        # 2D U(1) specific
        # Rotate and pad to center the neural networks.
        # tf.print('PeriodicConv:x',x,summarize=-1)
        # The shift is the padding on the head side.
        y = tf.roll(x, shift=self.pad, axis=(-3,-2))
        # tf.print('PeriodicConv:y_roll',y,summarize=-1)
        # then we pad on the tail side.
        y = tf.concat((y, y[...,:self.n0,:,:]), axis=-3)
        # tf.print('PeriodicConv:y_pad0R',y,summarize=-1)
        y = tf.concat((y, y[...,:self.n1,:]), axis=-2)
        # tf.print('PeriodicConv:y_pad1R',y,summarize=-1)
        for l in self.layers:
            y = l(y)
        # tf.print('PeriodicConv:y_return',y,summarize=-1)
        return y

class TransformChain(tk.Model):
    def __init__(self, transforms, name='TransformChain', **kwargs):
        super(TransformChain, self).__init__(autocast=False, name=name, **kwargs)
        self.chain = transforms
        self.invMaxIter = 0
        for t in transforms:
            if hasattr(t,'invMaxIter'):
                if self.invMaxIter < t.invMaxIter:
                    self.invMaxIter = t.invMaxIter
    def call(self, x):
        y = x
        l = tf.zeros(x.shape[0], dtype=tf.float64)
        bs = 0.0
        for f in self.chain:
            y, t, b = f(y)
            l += t
            bs += b
        bs /= len(self.chain)
        return (y, l, bs)
    def inv(self, y):
        x = y
        l = tf.zeros(y.shape[0], dtype=tf.float64)
        imax = 0
        for f in reversed(self.chain):
            x, t, i = f.inv(x)
            l += t
            if imax < i:
                imax = i
        return (x, l, imax)
    def showTransform(self, **kwargs):
        n = len(self.chain)
        tf.print(self.name, '[', n, ']', **kwargs)
        for i in range(n):
            tf.print(i, ':', end='')
            self.chain[i].showTransform(**kwargs)

class GenericStoutSmear(tk.Model):
    def __init__(self, linkSubset, updatedLoops, fixedLoopLayers, coefficientLayer,
            gauge=group.U1Phase, invAbsR2=1E-30, invMaxIter=8192, name='GenericStoutSmear', **kwargs):
        """
        linkSubset: ((x,y),(dx,dy))
            to-be-updated link subset as x+n*dx,y+m*dy, for int n,m
        updatedLoops: ((loop_a_0,loop_a_1,...),(loop_b_0,loop_b_1,...),...)
            Wilson loops for updating the link similar to stout smearing
        fixedLoopLayers: ((loop0, NNLayer0), (loop1, NNLayer1), ...)
            Each NNLayer applies to the Wison loop, and then stack them together
            with the new dimension the channel for coefficientLayer
        coefficientLayer: NNLayer
            The stacked result of fixedLoopLayers pass through this NNLayer,
            the resulting channels are the coefficients for each group of Wilson loops
            used in the stout smearing

        Input: a batch of gauge fields, shape as (batch, dim, x, y)
        Output: (fields, Log Det Jacobian)
            same gauge fields with the linkSubset updated as
                U_x <- exp(sum_l atan(C_x_l)/(N_l pi/2) sum_a T_a d_a/dU_x sum_n Tr loop_l_n) U_x
            where
                loop_l_n are given in updatedLoops
                C_x_l are from each channel of the output of coefficientLayer
                T_a are the group generators

        Loops are specified by a list of integers denoting direction, {±1, ±2, ...}
        The first link in each loop of updatedLoops should use the same direction,
        and it is the link we are updating.
        """
        super(GenericStoutSmear, self).__init__(autocast=False, name=name, **kwargs)
        # TODO: check arguments
        self.linkFirst, self.linkRepeat = linkSubset
        self.updatedLoops = updatedLoops
        self.loopsDiff = sum(updatedLoops,())
        self.linkDir = self.loopsDiff[0][0]
        if self.linkDir<1:
            raise ValueError(f'expecting positive direction in the first link, but got: {updatedLoops}')
        if not all([self.linkDir == x[0] for x in self.loopsDiff]):
            raise ValueError(f'expecting same direction in the first links, but got: {updatedLoops}')
        self.fixedLoopLayers = fixedLoopLayers
        self.loopsFixed = tuple([x[0] for x in fixedLoopLayers])
        self.layersFixedLoop = tuple([x[1] for x in fixedLoopLayers])
        self.pathsAll = field.OrdPaths(field.topath(self.loopsDiff + self.loopsFixed))
        self.listNumLoopsDiff = tuple([len(x) for x in updatedLoops])
        self.numLoopsDiff = len(self.loopsDiff)
        self.numLoopsFixed = len(self.loopsFixed)
        self.baseMaskFixedLoop = [
            tuple(set([
                tuple([(-x)%r for x,r in zip(coord.x, self.linkRepeat)])
                for coord in field.coordAtPathWithDir(loop, self.linkDir, 2)]))
            for loop in self.loopsFixed]
        self.layerCoefficient = coefficientLayer
        self.gauge = gauge
        self.invAbsR2 = invAbsR2
        self.invMaxIter = invMaxIter
        # print(f'self.baseMaskFixedLoop: {self.baseMaskFixedLoop}')
    def build(self, shape):
        m = tf.scatter_nd((self.linkFirst,), tf.constant((1.,), dtype=tf.float64), self.linkRepeat)
        # print(f'm: {m}')
        r = [shape[2+i]//rep for i,rep in enumerate(self.linkRepeat)]
        # print(f'r: {r}')
        self.maskUpdate = tf.tile(m, r)
        # print(f'self.maskUpdate: {self.maskUpdate}')
        self.unmaskFixedLoop = [
            1.0 - tf.tile(
                tf.roll(
                    tf.scatter_nd(cm, tf.constant((1.,)*len(cm), dtype=tf.float64), self.linkRepeat),
                    self.linkFirst,
                    range(len(self.linkFirst))),
                r)
            for cm in self.baseMaskFixedLoop]
        # print(f'self.unmaskFixedLoop: {self.unmaskFixedLoop}')
        self.dataShape = shape
        self.updateIndex = tf.constant([[i,self.linkDir-1] for i in range(shape[0])])  # -1 to count from 0
        super(GenericStoutSmear, self).build(shape)
    def call(self, x):
        ps = field.OrdProduct(self.pathsAll, x, self.gauge).prodList()
        # tf.print('ps:\n',ps, summarize=-1)
        bs = self.beta(ps[self.numLoopsDiff:])
        d = 0.0
        f = 0.0
        b = 0
        for k,a in enumerate(self.listNumLoopsDiff):
            fa = 0.0
            da = 0.0
            for i in range(b, b+a):
                fa += self.gauge.diffTrace(ps[i])
                da += self.gauge.diff2Trace(ps[i])
            b = a
            f += bs[...,k]*self.maskUpdate*fa
            d += bs[...,k]*self.maskUpdate*da
        f = self.expanddir(f)
        if self.numLoopsFixed>0:
            bs = tf.math.reduce_mean(bs, range(1,len(bs.shape)-1))
        return (
            self.gauge.compatProj(x+f),
            tf.math.reduce_sum(tf.math.log1p(d), range(1,len(d.shape))),
            bs)
    def beta(self,ps):
        # 2*atan(a)/pi/nloop makes it in (-1/nloop,1/nloop)
        # tf.print('beta:len(self.layersFixedLoop) ',len(self.layersFixedLoop))
        # tf.print('beta:len(self.unmaskFixedLoop) ',len(self.unmaskFixedLoop))
        # tf.print('beta:len(ps) ',len(ps))
        # tf.print('beta:ps',ps)
        if self.numLoopsFixed>0:
            y = self.layerCoefficient(
                tf.concat(
                    [l(tf.expand_dims(m*self.gauge.trace(p),-1))
                     for l,m,p in
                     zip(self.layersFixedLoop,self.unmaskFixedLoop,ps)],
                    axis=-1))
        else:
            y = self.layerCoefficient(0)
        # tf.print('beta:y ',y, summarize=-1)
        return 2.0/self.numLoopsDiff/math.pi*tf.math.atan(y)
    def inv_iter(self, x):
        ps = field.OrdProduct(self.pathsAll, x, self.gauge).prodList()
        bs = self.beta(ps[self.numLoopsDiff:])
        f = 0.0
        b = 0
        for k,a in enumerate(self.listNumLoopsDiff):
            fa = 0.0
            for i in range(b, b+a):
                fa += self.gauge.diffTrace(ps[i])
            b = a
            f += bs[...,k]*self.maskUpdate*fa
        return self.expanddir(f)
    def inv(self, y):
        x = y
        i = 0
        f = self.inv_iter(x)
        while self.invMaxIter > i and self.invAbsR2 < tf.math.reduce_mean(tf.math.squared_difference(f, y-x)):
            i += 1
            x = y-f
            f = self.inv_iter(x)
        # tf.Assert(i<self.invMaxIter, ['', y, 'current', x, 'with delta', f], summarize=-1)
        x = self.gauge.compatProj(y-f)
        _, l, _ = self(x)
        return (x, -l, i)
    def expanddir(self, x):
        return tf.scatter_nd(self.updateIndex, x, self.dataShape)
    def showTransform(self, show_weights=True, **kwargs):
        if show_weights:
            tf.print(self.name, 'α:', self.trainable_weights, 'T:', self.linkFirst, ':', self.linkRepeat, **kwargs)
        else:
            tf.print(self.name, 'T:', self.linkFirst, ':', self.linkRepeat, **kwargs)

def checkDep(tr, defaultShape = (16,16)):
    """
    Check the dependence of a GenericStoutSmear.
    This will build and initialize the model for a specific size if it has not already built,
    and it may not be usable for other data shape.
    """
    if isinstance(tr, TransformChain):
        for t in tr.chain:
            checkDep(t, defaultShape)
        return
    elif not isinstance(tr, GenericStoutSmear):
        raise ValueErro(f'Accepts a GenericStoutSmear or a TransformChain of GenericStoutSmear, but got {tr}')
    tr.showTransform(show_weights=False)
    if hasattr(tr, 'dataShape'):
        # already built
        testShape = tr.dataShape[-2:]
    else:
        testShape = defaultShape
    testPoint = [(s//(2*r))*r+f for s,r,f in zip(testShape,tr.linkRepeat,tr.linkFirst)]
    testDim = tr.linkDir-1
    x = group.U1Phase.random((1,2)+testShape, tf.random.get_global_generator())
    tr.build(x.shape)
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        y, _, _ = tr(x)
        z = y[0,testDim,testPoint[0],testPoint[1]]
    dzdx = g.gradient(z, x)[0]
    # tf.print('y: ',y, summarize=-1)
    # tf.print('l: ',l, summarize=-1)
    # tf.print('dzdx: ',dzdx, summarize=-1)
    dep = tf.cast(dzdx!=0,tf.float64)
    fail = 0
    if tf.norm((1-tf.scatter_nd([testPoint],[tf.constant(1,dtype=tf.float64)],testShape))*tr.maskUpdate*dep[testDim])!=0:
        print(f'Error: links depend on other links to be updated!')
        tf.print('Masked dependence: ',tr.maskUpdate*dep[testDim],summarize=-1)
        tf.print('Mask: ',tr.maskUpdate,summarize=-1)
        fail += 1
    depi = tf.cast(dep,tf.int32)
    # search for the minimal rectangle dependent region by masking everything and then shrinking
    def search(mat):
        x0,y0 = (0,0)
        xl,yl = testShape
        mask = lambda x0,y0,xl,yl: 1 - tf.pad(tf.ones((xl,yl),dtype=tf.int32), [[x0, testShape[0]-x0-xl], [y0, testShape[1]-y0-yl]])
        while tf.math.reduce_sum(mat * mask(x0,y0,xl-1,yl)) == 0:
            xl -= 1
        while tf.math.reduce_sum(mat * mask(x0,y0,xl,yl-1)) == 0:
            yl -= 1
        while tf.math.reduce_sum(mat * mask(x0+1,y0,xl-1,yl)) == 0:
            x0 += 1
            xl -= 1
        while tf.math.reduce_sum(mat * mask(x0,y0+1,xl,yl-1)) == 0:
            y0 += 1
            yl -= 1
        return tf.constant(((x0,y0),(xl,yl)))
    rect = tf.stack([search(depi[0]), search(depi[1])], 0)
    # tf.print('dependent links rect: ',rect,summarize=-1)
    # TODO: this check is naive.
    if tf.math.reduce_any(rect[0,0]!=rect[1,0]):
        print(f'Error: origin of the dependent links differ between directions.')
        tf.print('dzdx: ',dzdx,summarize=-1)
        tf.print('dependent links rect: ',rect,summarize=-1)
        fail += 1
    elif tf.math.reduce_any(rect[testDim,1]%2!=(1,1)) or tf.math.reduce_any(rect[1-testDim,1]%2!=(0,0)) or \
         rect[testDim,1,testDim]+1!=rect[1-testDim,1,testDim] or rect[testDim,1,1-testDim]-1!=rect[1-testDim,1,1-testDim]:
        printf(f'Error: dependent links does not form a rectangle.')
        tf.print('dzdx: ',dzdx,summarize=-1)
        tf.print('dependent links rect: ',rect,summarize=-1)
        fail += 1
    elif tf.math.reduce_any(rect[testDim,0]+rect[testDim,1]//2!=testPoint):
        print(f'Error: dependent links does not center around the update link.')
        tf.print('dzdx: ',dzdx,summarize=-1)
        tf.print('dependent links rect: ',rect,summarize=-1)
        print(f'testPoint: {testPoint}')
        fail += 1
    else:
        tf.print('Dependent links forms',rect[0,1,0],'x',rect[1,1,1],'plaquettes.')
    if fail>0:
        raise ValueError(f'Dependency check failed {fail} tests.')

if __name__ == '__main__':
    tf.random.set_seed(9876543211)
    tk.backend.set_floatx('float64')
    op0 = (((1,2,-1,-2), (1,-2,-1,2)),
           ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
    # requires different coefficient bounds:
    # (1,2,-1,-2,1,-2,-1,2)
    # (1,2,-1,-2,1,2,-1,-2)
    # (1,-2,-1,2,1,-2,-1,2)
    op1 = (((2,-1,-2,1), (2,1,-2,-1)),
           ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
    fixedP = (1,2,-1,-2)
    fixedR0 = (2,2,1,-2,-2,-1)
    fixedR1 = (1,1,2,-1,-1,-2)
    convP0 = lambda: PeriodicConv((
        tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.Constant(0.01), bias_initializer=tk.initializers.Constant(0.01)),
    ))
    convP1 = lambda: PeriodicConv((
        tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.Constant(0.01), bias_initializer=tk.initializers.Constant(0.01)),
    ))
    convR = lambda pad: PeriodicConv((
        tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.Constant(0.01), bias_initializer=tk.initializers.Constant(0.01)),
    ), pad)
    conv = lambda: PeriodicConv((
        tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.Constant(0.01), bias_initializer=tk.initializers.Constant(0.01)),
        tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.Constant(0.01), bias_initializer=tk.initializers.Constant(0.01)),
    ))

    checkDep(GenericStoutSmear(((1,1),(2,2)), op0, [], Scalar(2,[1,1])))
    checkDep(GenericStoutSmear(((1,1),(2,2)), op1, [], Scalar(2,[1,1])))
    checkDep(TransformChain([
        GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
        GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
        GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ]))
