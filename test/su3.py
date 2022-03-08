import sys
sys.path.append("../lib")
import group as g
import tensorflow as tf
import unittest as ut

mul = tf.linalg.matmul
mulv = tf.linalg.matvec
tr = tf.linalg.trace
det = tf.linalg.det
re = tf.math.real
im = tf.math.imag
adj = tf.linalg.adjoint
conj = tf.math.conj
cmplx = tf.dtypes.complex
toCmplx = lambda x: tf.cast(x,tf.complex128)
I = cmplx(tf.constant(0,dtype=tf.float64), tf.constant(1,dtype=tf.float64))
T = g.su3gen()
comm = [[mul(T[a],T[b])-mul(T[b],T[a]) for b in range(8)] for a in range(8)]
acom = [[mul(T[a],T[b])+mul(T[b],T[a]) for b in range(8)] for a in range(8)]

class TestSU3(ut.TestCase):
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
                X += toCmplx(self.x[a])*T[a]
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

    def test_projectTAH(self):
        """
        projectTAH(M)
            = - T^a tr[T^a (M - M†)]
            = 1/2 { δ_il δ_jk (M - M†)_lk - 1/3 δ_ij δ_kl (M - M†)_lk }
            = 1/2 { (M - M†)_ij - 1/3 δ_ij tr(M - M†) }
        Note:
            T^a_ij T^a_kl = - 1/2 { δ_il δ_jk - 1/3 δ_ij δ_kl }
        """
        m = g.exp(self.X)
        p = g.projectTAH(m)
        q = g.su3fromvec(g.su3vec(m-adj(m))/2)
        self.checkEqv(p,q)

    def test_diffProjTAH(self):
        """
        F = T^b ∂_b tr[M + M†]
          = T^b tr[T^b M - M† T^b]
          = T^b tr[T^b (M - M†)]
        F^a = tr[T^a (M - M†)]
        ∂_c f^a = tr[T^a (T^c M + M† T^c)]
                = 1/2 tr[{T^a,T^c} (M+M†) + [T^a,T^c] (M-M†)]
                = 1/2 tr[d^acb T^b i (M+M†) - 1/3 δ^ac (M+M†) + f^acb T^b (M-M†)]
                = 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) + f^acb F^b }
                = 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) - adF^ac }
        """
        X = g.exp(self.X)
        Y = g.exp(self.Y)
        ep = 0.12
        M = None
        def f(x):
            nonlocal Y, M
            M = ep * mul(x,Y,adjoint_b=True)
            return -g.projectTAH(M)
        F,tj = g.SU3JacobianTF(X, f, is_SU3=False)

        adF = g.su3ad(F)
        Ms = M+adj(M)
        trMs = tr(Ms)
        j = 0.5*(g.su3dabc(-0.5*g.su3vec(I*Ms)) + (-re(trMs)/3.0)*g.eyeOf(adF) - adF)
        with self.subTest(target='det'):
            self.checkEqv(det(j), det(tj))
        with self.subTest(target='mat'):
            self.checkEqv(j, tj)

    def test_fabctc(self):
        # [T^a, T^b] = f^abc T^c
        fabctc = g.su3fabc(tf.transpose(T,perm=list(range(1,len(T.shape)))+[0]))
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    self.checkEqv(fabctc[...,a,b], comm[a][b])

    def test_anticomm(self):
        # {T^a, T^b} = - 1/3 delta^ab + i d^abc T^c
        dabctc = g.su3dabc(tf.transpose(T,perm=list(range(1,len(T.shape)))+[0]))
        dab3 = (-1.0/3.0)*tf.eye(3, dtype=tf.complex128)
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    if a==b:
                        self.checkEqv(dab3+I*dabctc[...,a,b], acom[a][b])
                    else:
                        self.checkEqv(I*dabctc[...,a,b], acom[a][b])

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

    def exp_helper(self, x):
        ex = g.exp(x)
        exm = tf.linalg.expm(x)
        with self.subTest(target='det'):
            self.checkEqv(det(ex), 1, tol=1e-22)    # error from series expansion
        with self.subTest(target='mat'):
            self.checkEqv(ex, exm, tol=1e-22)    # error from series expansion

    def test_diffexp(self):
        with self.subTest(v='x'):
            self.diffexp_helper(self.x)
        with self.subTest(v='y'):
            self.diffexp_helper(self.y)

    def diffexp_helper(self, v):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(v)
            m = g.su3fromvec(v)
            em = g.exp(m)
            ex = g.su3vec(mul(tf.stop_gradient(em),em,adjoint_a=True))
        # print(f'norm2(x): {g.norm2(m)}')
        jx = g.diffexp(g.su3ad(m))
        jpx = t.jacobian(ex, v, experimental_use_pfor=False)
        with self.subTest(target='det'):
            self.checkEqv(det(jx), det(jpx), tol=1e-19)    # error from series expansion
        with self.subTest(target='mat'):
            self.checkEqv(jx, jpx, tol=1e-19)    # error from series expansion

    def test_expfmulu(self):
        """
        Z = exp(ε T^b ∂_b tr[X Y† + Y X†]) X,  for X,Y in G, and ∂_b X = T_b X
        M = ε X Y†,  M in G
        Z = exp(F) X,  F in g
        F = T^b ∂_b tr[M + M†]
          = T^b tr[T^b M - M† T^b]
          = T^b tr[T^b (M - M†)]
        ∂_c Z = exp(F) T^c X + exp(F) J(F)[∂_c F] X
              = exp(F) { T^c + exp(F) J(F)[∂_c F] } exp(-F) Z
              = exp(adF)[T^c + J(F)[∂_c F]] Z
              = exp(adF)[T^c + T^e J(F)^eb [∂_c F]^b] Z
              = T^a { exp(adF)^ac + [exp(adF)J(F)]^ab [∂_c F]^b } Z
              = T^a { exp(adF)^ac + [(exp(adF)-1)/adF]^ab [∂_c F]^b } Z
              = T^a { exp(adF)^ac + J(-F)^ab [∂_c F]^b } Z
        [∂_c F]^b = ∂_c ∂_b tr[M + M†]
                  = tr[T^b T^c M + M† T^c T^b]
                  = 1/2 tr[{T^b,T^c} (M + M†)] + 1/2 tr[[T^b,T^c] (M - M†)]
                  = 1/2 { d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†] + f^bcd tr[T^d (M - M†)] }
                  = 1/2 { d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†] + f^bcd F^d }
                  = 1/2 { d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†] - adF^bc }
        -2 tr[(∂_c Z) Z† T^a]
            = exp(adF)^ac + J(-F)^ab [∂_c F]^b
            = exp(adF)^ac
              + 1/2 [(exp(adF)-1)/adF]^ab {d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†]}
              - 1/2 [(exp(adF)-1)]^ac
            = 1/2 { [(exp(adF)+1)]^ac + [(exp(adF)-1)/adF]^ab {d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†]} }
            = 1/2 { [(exp(adF)+1)]^ac + J(-F)^ab {d^bcd tr[T^d i (M + M†)] - 1/3 δ^bc tr[M + M†]} }
        Note:
            exp(adX) Y = exp(X) Y exp(-X)  for X,Y in g
        """

        X = g.exp(self.X)
        Y = g.exp(self.Y)
        ep = 0.12
        M = None
        F = None
        def f(x):
            nonlocal Y, M, F
            M = ep * mul(x,Y,adjoint_b=True)
            F = -g.projectTAH(M)
            return mul(g.exp(F),x)
        Z,tj = g.SU3JacobianTF(X, f)

        adF = g.su3ad(F)
        Ms = M+adj(M)
        trMs = tr(Ms)
        with self.subTest(equation='chain rule'):
            dF = 0.5*(g.su3dabc(-0.5*g.su3vec(I*Ms)) + (-re(trMs)/3.0)*g.eyeOf(adF) - adF)
            j = g.exp(adF) + mul(g.diffexp(-adF), dF)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj), tol=1e-20)
            with self.subTest(target='mat'):
                self.checkEqv(j, tj, tol=1e-20)
        with self.subTest(equation='combined'):
            K = (-re(trMs)/3.0)*g.eyeOf(adF) + g.su3dabc(-0.5*g.su3vec(I*Ms))
            j = 0.5*(g.exp(adF)+g.eyeOf(adF) + mul(g.diffexp(-adF), K))
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj), tol=1e-20)
            with self.subTest(target='mat'):
                self.checkEqv(j, tj, tol=1e-20)

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
