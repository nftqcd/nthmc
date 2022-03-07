import sys
sys.path.append("../lib")
import group as g
import tensorflow as tf
import unittest as ut

mul = tf.linalg.matmul
mulv = tf.linalg.matvec
tr = tf.linalg.trace
re = tf.math.real
im = tf.math.imag
T = g.su3gen()
comm = [[mul(T[a],T[b])-mul(T[b],T[a]) for b in range(8)] for a in range(8)]
acom = [[mul(T[a],T[b])+mul(T[b],T[a]) for b in range(8)] for a in range(8)]

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
        self.x = tf.constant([0.04,0.06,0.1,0.16,0.26,0.42,0.68,1.1], dtype=tf.float64)
        self.X = g.su3fromvec(self.x)
        self.y = tf.constant([-0.574, -0.56, -0.508, -0.442, -0.324, -0.14, 0.162, 0.648], dtype=tf.float64)
        self.Y = g.su3fromvec(self.y)

    def test_xata(self):
        # X = X^a T^a
        X = 0
        for a in range(8):
            with self.subTest(a=a):
                X += tf.cast(self.x[a],tf.complex128)*T[a]
        self.checkEqv(X, self.X)

    def test_trtatb(self):
        # tr[T^a T^b] = -1/2 delta^ab
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    t = tr(mul(T[a],T[b]))
                    if a==b:
                        self.checkEqv(t, -0.5)
                    else:
                        self.checkEqv(t, 0)

    def test_fabctc(self):
        # [T^a, T^b] = f^abc T^c
        fabctc = g.su3fabc(T)
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    self.checkEqv(fabctc[...,a,b], comm[a][b])

    def test_anticomm(self):
        # {T^a, T^b} = - 1/3 delta^ab + i d^abc T^c
        dabctc = g.su3dabc(T)
        i = tf.dtypes.complex(tf.constant(0,dtype=tf.float64), tf.constant(1,dtype=tf.float64))
        dab3 = (-1.0/3.0)*tf.eye(3, dtype=tf.complex128)
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    if a==b:
                        self.checkEqv(dab3+i*dabctc[...,a,b], acom[a][b])
                    else:
                        self.checkEqv(i*dabctc[...,a,b], acom[a][b])

    def test_adxy(self):
        adxy = g.su3adapply(g.su3ad(self.X), self.Y)
        self.checkEqv(adxy, mul(self.X,self.Y)-mul(self.Y,self.X))

    def test_adyx(self):
        adyx = g.su3adapply(g.su3ad(self.Y), self.X)
        self.checkEqv(adyx, mul(self.Y,self.X)-mul(self.X,self.Y))

    def test_exp0(self):
        for i in range(8):
            with self.subTest(a=i):
                self.exp_helper(T[i])

    def test_exp(self):
        with self.subTest(v='x'):
            self.exp_helper(self.X)
        with self.subTest(v='y'):
            self.exp_helper(self.X)

    def test_diffexp0(self):
        v = tf.constant([1,0,0,0,0,0,0,0],dtype=tf.float64)
        for i in range(8):
            with self.subTest(a=i):
                self.diffexp_helper(v)
                v = tf.roll(v,shift=1,axis=0)

    def test_diffexp(self):
        with self.subTest(v='x'):
            self.diffexp_helper(self.x)
        with self.subTest(v='y'):
            self.diffexp_helper(self.y)

    def exp_helper(self, x):
        ex = g.exp(x)
        exm = tf.linalg.expm(x)
        with self.subTest(target='det'):
            self.checkEqv(tf.linalg.det(ex), 1, tol=1e-22)    # error from series expansion
        with self.subTest(target='mat'):
            self.checkEqv(ex, exm, tol=1e-22)    # error from series expansion

    def diffexp_helper(self, v):
        exm = g.exp(g.su3fromvec(-v))
        # exm = tf.linalg.expm(g.su3fromvec(-v))
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(v)
            m = g.su3fromvec(v)
            ex = g.su3vec(mul(exm,g.exp(m)))
            # ex = g.su3vec(mul(exm,tf.linalg.expm(m)))
        # print(f'norm2(x): {g.norm2(m)}')
        jx = g.diffexp(g.su3ad(m))
        jpx = t.jacobian(ex, v, experimental_use_pfor=False)
        with self.subTest(target='det'):
            self.checkEqv(tf.linalg.det(jx), tf.linalg.det(jpx), tol=1e-19)    # error from series expansion
        with self.subTest(target='mat'):
            self.checkEqv(jx, jpx, tol=1e-19)    # error from series expansion

    def checkEqv(self,a,b,tol=1e-28,rtol=1e-14):
        d = a-b
        axis = range(len(d.shape))
        m = g.norm2(d,axis)
        ma = g.norm2(d+b,axis)    # broadcast to the same shape
        mb = g.norm2(a-d,axis)
        mn = ma if ma>mb else mb
        if m>=tol or (ma>0 and mb>0 and m/mn>=rtol):
            print(f'received {a}')
            print(f'expected {b}')
            print(f'abs sq diff: {m}')
            if ma>0 and mb>0:
                print(f'rel sq diff: {m/mn}')
        self.assertLess(m, tol)
        if ma>0 and mb>0:
            self.assertLess(m/mn, rtol)

if __name__ == '__main__':
    ut.main()
