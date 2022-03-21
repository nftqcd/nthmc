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
        self.z = tf.constant([0.481, -0.755, 0.009, 0.773, -0.463, 0.301, -0.916, -0.172], dtype=tf.float64)
        self.Z = g.su3fromvec(self.z)

    def test_xata(self):
        # X = X^a T^a
        X = 0
        for a in range(8):
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

    def test_projectTAHFromDiff(self):
        """
        projectTAH(M) = T^a ∂_a (- tr[M + M†]) = T^a ∂_a (-2 ReTr M)
        """
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(self.x)
            m = g.exp(g.su3fromvec(self.x))
            r = -2*re(tr(m))
        d = t.gradient(r, self.x)
        p = g.su3vec(g.projectTAH(m))
        self.checkEqv(p,d)

    def test_projectTAHFromDiffSU3(self):
        """
        Testing TensorFlow's gradient, and our normalization convention.
        projectTAH(M) = T^a ∂_a (- tr[M + M†]) = T^a ∂_a (-2 ReTr M)
        """
        m = g.exp(self.X)
        r,d = g.SU3GradientTF(lambda x: -2*re(tr(x)), m)
        p = g.projectTAH(m)
        self.checkEqv(p,g.su3fromvec(d))

    def test_projectTAHFromDiffSU3Mat(self):
        """
        Testing TensorFlow's gradient, and our normalization convention.
        projectTAH(M) = T^a ∂_a (- tr[M + M†]) = T^a ∂_a (-2 ReTr M)
        """
        m = g.exp(self.X)
        r,d = g.SU3GradientTFMat(lambda x: -2*re(tr(x)), m)
        p = g.projectTAH(m)
        self.checkEqv(p,d)

    def test_diffprojectTAH(self):
        X = g.exp(self.X)
        Y = g.exp(self.Y)
        ep = 0.12
        M = None
        def f(x):
            nonlocal Y, M
            M = ep * mul(x,Y,adjoint_b=True)
            return g.projectTAH(M)
        P,tj = g.SU3JacobianTF(f, X, is_SU3=False)

        with self.subTest(project='given'):
            j = g.diffprojectTAH(M, P)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(project='recompute'):
            j = g.diffprojectTAH(M)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)

    def test_diffprojectTAHMat(self):
        """
        Same as test_diffprojectTAH but uses SU3JacobianTFMat.
        """
        X = g.exp(self.X)
        Y = g.exp(self.Y)
        ep = 0.12
        M = None
        def f(x):
            nonlocal Y, M
            M = ep * mul(x,Y,adjoint_b=True)
            return g.projectTAH(M)
        P,tj = g.SU3JacobianTFMat(f, X, is_SU3=False)

        with self.subTest(project='given'):
            j = g.diffprojectTAH(M, P)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(project='recompute'):
            j = g.diffprojectTAH(M)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(name='consistency with diffprojectTAHCross'):
            j = g.diffprojectTAHCross(M, Adx=tf.eye(8, dtype=tf.float64), p=P)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)

    def test_diffprojectTAHCross(self):
        """
        ∂_Y^b ∂_X^a (-2) ReTr[ X (Z Y)† ] = - ∂_Y^b ∂_X^a tr[ X Y† Z† + Z Y X† ]
            = - 2 ReTr[T^a (- X Y†) T^b Z†]
        Note the extra negative sign from ∂_Y^b.
        """
        X = g.projectSU(g.exp(self.X))
        Y = g.projectSU(g.exp(self.Y))
        Z = g.projectSU(g.exp(self.Z))
        ep = 0.12
        M = None
        def f(y):
            nonlocal M
            M = ep * mul(X,mul(Z,y),adjoint_b=True)
            return g.projectTAH(M)
        P,tj = g.SU3JacobianTF(f, Y, is_SU3=False)

        with self.subTest(project='given', Adx='given'):
            j = -g.diffprojectTAHCross(M, Adx=g.SU3Ad(mul(X,Y,adjoint_b=True)), p=P)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(project='recompute', Adx='given'):
            j = -g.diffprojectTAHCross(M, Adx=g.SU3Ad(mul(X,Y,adjoint_b=True)))
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(project='given', Adx='recompute'):
            j = -g.diffprojectTAHCross(M, x=mul(X,Y,adjoint_b=True), p=P)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(project='recompute', Adx='recompute'):
            j = -g.diffprojectTAHCross(M, x=mul(X,Y,adjoint_b=True))
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)

    def test_diff2projectTAH(self):
        """
        P^a = -tr[T^a (M - M†)]
        ∂_c P^a = -tr[T^a (T^c M + M† T^c)]
                = -1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) + f^acb F^b }
        ∂_d ∂_c P^a = -tr[T^a T^c T^d M - T^c T^a M† T^d]
        Use diffProjTAH, but use T^d M for M.
        """
        X = g.exp(self.X)
        Y = g.exp(self.Y)
        Z = g.exp(self.Z)
        ep = 0.12
        """ For 2nd derivative here,
        [d/dSU(3)] { [d/dSU(3)] su(3) }
            = (T^c Y])_mn ∂_Y_mn { (T^b X)_ij (∂_X_ij F(X,Y)^a) }
            = T^c_ml Y_ln ∂_Y_mn { T^b_ik X_kj (∂_X_ij F(X,Y)^a) }
            = T^c_ml Y_ln { ∂Y ∂X F(X,Y) }^ab_mn
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(Y)
            def f(x):
                nonlocal Y, Z
                M = ep * mul(x,mul(Z,Y),adjoint_b=True)
                return g.projectTAH(M)
            F,tj = g.SU3JacobianTF(f, X, is_SU3=False)
            tjC = tf.cast(tj, tf.complex128)
        tj2m = t.jacobian(tjC, Y, experimental_use_pfor=False)
        tj2 = re(tf.einsum('cml,ln,abmn->abc', T, Y, conj(tj2m)))
        for i in range(8):
            M = ep * mul(X,mul(Z,mul(T[i],Y)),adjoint_b=True)
            j = g.diffprojectTAH(M)
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj2[...,i]))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj2[...,i])

    def test_fabctc(self):
        # [T^a, T^b] = f^abc T^c
        fabctc = g.su3fabc(tf.transpose(T,perm=list(range(1,len(T.shape)))+[0]))
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    self.checkEqv(fabctc[...,a,b], comm[a][b])

    def test_anticomm(self):
        # {T^a, T^b} = - 1/3 δ^ab + i d^abc T^c
        dabctc = g.su3dabc(tf.transpose(T,perm=list(range(1,len(T.shape)))+[0]))
        dab3 = (-1.0/3.0)*tf.eye(3, dtype=tf.complex128)
        for a in range(8):
            for b in range(8):
                with self.subTest(a=a,b=b):
                    if a==b:
                        self.checkEqv(dab3+I*dabctc[...,a,b], acom[a][b])
                    else:
                        self.checkEqv(I*dabctc[...,a,b], acom[a][b])

    def test_tatb(self):
        """
        T^a T^b = 1/2 ([T^a, T^b] + {T^a, T^b})
                = 1/2 (f^abc + i d^abc) T^c - 1/6 δ^ab
        """
        tc = tf.transpose(T,perm=list(range(1,len(T.shape)))+[0])
        fabctc = g.su3fabc(tc)
        dabctc = g.su3dabc(tc)
        one = tf.eye(3, dtype=tf.float64)
        dab = tf.eye(8, dtype=tf.float64)
        dabone = tf.cast(tf.einsum('ij,kl->ijkl', one, dab), tf.complex128)
        tatb = tf.einsum('aij,bjk->ikab', T, T)
        self.checkEqv(0.5*(fabctc+I*dabctc+(-1.0/3.0)*dabone), tatb)

    def test_adxy(self):
        adxy = g.su3adapply(g.su3ad(self.X), self.Y)
        self.checkEqv(adxy, mul(self.X,self.Y)-mul(self.Y,self.X))

    def test_adyx(self):
        adyx = g.su3adapply(g.su3ad(self.Y), self.X)
        self.checkEqv(adyx, mul(self.Y,self.X)-mul(self.X,self.Y))

    def test_AdX(self):
        """
        exp(adx) = Ad[exp(x)], for x in su(3) algebra.
        """
        with self.subTest(x='x'):
            self.checkEqv(g.SU3Ad(g.exp(self.X,order=15)), g.exp(g.su3ad(self.X),order=19))
        with self.subTest(x='y'):
            self.checkEqv(g.SU3Ad(g.exp(self.Y,order=15)), g.exp(g.su3ad(self.Y),order=19))
        with self.subTest(x='z'):
            self.checkEqv(g.SU3Ad(g.exp(self.Z,order=15)), g.exp(g.su3ad(self.Z),order=19))

    def test_exp0(self):
        for i in range(8):
            with self.subTest(a=i):
                self.exp_helper(T[i],tol=1e-12,rtol=1e-12)    # error from series expansion

    def test_exp(self):
        with self.subTest(v='x'):
            self.exp_helper(self.X,tol=1e-11,rtol=1e-11)    # error from series expansion
        with self.subTest(v='y'):
            self.exp_helper(self.Y,tol=1e-12,rtol=1e-12)    # error from series expansion
        with self.subTest(v='z'):
            self.exp_helper(self.Z,tol=1e-10,rtol=1e-10)    # error from series expansion

    def exp_helper(self, x, tol, rtol):
        ex = g.exp(x)
        exm = tf.linalg.expm(x)
        with self.subTest(target='det'):
            self.checkEqv(det(ex), 1, tol=tol, rtol=rtol)
        with self.subTest(target='mat'):
            self.checkEqv(ex, exm, tol=tol, rtol=rtol)

    def test_diffexp0(self):
        v = tf.constant([1,0,0,0,0,0,0,0],dtype=tf.float64)
        for i in range(8):
            with self.subTest(a=i):
                self.diffexp_helper(v,tol=1e-11,rtol=1e-11)    # error from series expansion
                v = tf.roll(v,shift=1,axis=0)

    def test_diffexp(self):
        with self.subTest(v='x'):
            self.diffexp_helper(self.x,tol=1e-9,rtol=1e-9)    # error from series expansion
        with self.subTest(v='y'):
            self.diffexp_helper(self.y,tol=1e-10,rtol=1e-10)    # error from series expansion
        with self.subTest(v='z'):
            self.diffexp_helper(self.z,tol=1e-9,rtol=1e-8)    # error from series expansion

    def diffexp_helper(self, v, tol, rtol):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(v)
            m = g.su3fromvec(v)
            em = g.exp(m)
            ex = g.su3vec(mul(tf.stop_gradient(em),em,adjoint_a=True))
        # print(f'norm2(x): {g.norm2(m)}')
        jx = g.diffexp(g.su3ad(m))
        jpx = t.jacobian(ex, v, experimental_use_pfor=False)
        with self.subTest(target='det'):
            self.checkEqv(det(jx), det(jpx), tol=tol, rtol=rtol)
        with self.subTest(target='mat'):
            self.checkEqv(jx, jpx, tol=tol, rtol=rtol)

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
        det(-2 tr[(∂_c Z) Z† T^a])
            = det(exp(adF)^ac + J(-F)^ab [∂_c F]^b)
            = det(δ^ac + J(F)^ab [∂_c F]^b)
        Note:
            exp(adX) Y = exp(X) Y exp(-X)  for X,Y in g
            det(exp(adX)) = exp(tr(ln(exp(adX)))) = exp(tr(adX)) = exp(0) = 1
        """

        X = g.projectSU(g.exp(self.X))
        Y = g.projectSU(g.exp(self.Y))
        ep = 0.12
        M = None
        F = None
        def f(x):
            nonlocal Y, M, F
            M = ep * mul(x,Y,adjoint_b=True)
            F = -g.projectTAH(M)
            return mul(g.exp(F),x)
        Z,tj = g.SU3JacobianTF(f, X)

        adF = g.su3ad(F)
        Ms = M+adj(M)
        trMs = re(tr(Ms))
        with self.subTest(equation='combined'):
            K = (-1.0/3.0)*tf.reshape(trMs,trMs.shape+[1,1])*g.eyeOf(adF) + g.su3dabc(-0.5*g.su3vec(I*Ms))
            j = 0.5*(g.exp(adF)+g.eyeOf(adF) + mul(g.diffexp(-adF), K))
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        dF = g.diffprojectTAH(-M,F)
        j = g.exp(adF) + mul(g.diffexp(-adF), dF)
        with self.subTest(equation='chain rule'):
            with self.subTest(target='det'):
                self.checkEqv(det(j), det(tj))
            with self.subTest(target='mat'):
                self.checkEqv(j, tj)
        with self.subTest(equation='det simplified'):
            self.checkEqv(det(g.eyeOf(adF) + mul(g.diffexp(adF), dF)), det(j))

    def test_difflndetexpfmuluDiag(self):
        """
        ∇_d ln det {δ^ac + J(F)^ab [∂_c F^b]}    # ∇ can act on different links, ∂ only on the updating link
            = m^{-1}^ca {[∇_d J(F)^ab] [∂_c F^b] + J(F)^ab [∇_d ∂_c F^b]}
        where
            m^ac = δ^ac + J(F)^ab [∂_c F^b]
        This test is for ∇ = ∂.
        ∇_d adF^ce = ∇_d (- f^ceg F^g) = - f^ceg [∇_d F^g]
        ∇_d J(F)^ab
            = Σ_{k=0} 1/(k+1)! (-1)^k ∇_d [(adF)^k]^ab
            = Σ_{k=1} 1/(k+1)! (-1)^k Σ_{j=0}^{k-1} [adF^j]^ac (∇_d adF^ce) [adF^(k-j-1)]^eb
            = Σ_{k=1} 1/(k+1)! (-1)^k Σ_{j=0}^{k-1} [adF^j]^ac (- f^cea [∇_d F^a]) [adF^(k-j-1)]^eb
            = Σ_{k=0} 1/(k+2)! (-1)^k Σ_{j=0}^k [adF^j]^ac f^ceg [∇_d F^g] [adF^(k-j)]^eb
            = 1/2 ( dadF
              + (-1)/3 ( adF dadF + dadF adF
              + (-1)/4 ( adF^2 dadF + adF dadF adF + dadF adF^2
              + (-1)/5 ( adF^3 dadF + adF^2 dadF adF + adF dadF adF^2 + dadF adF^3
              + (-1)/6 ( adF^4 dadF + adF^3 dadF adF + adF^2 dadF adF^2 + adF dadF adF^3 + dadF adF^4
              + (-1)/7 ( adF^5 dadF + adF^4 dadF adF + adF^3 dadF adF^2 + adF^2 dadF adF^3 + adF dadF adF^4 + dadF adF^5 + ... ))))))
            = 1/2 ( {dadF, 1/2 + (-1)/3 adF (1 + (-1)/4 adF (1 + (-1)/5 adF (1 + (-1)/6 adF (1 + (-1)/7 adF (1 + ...)))))}
              + (-1)/3 (-1)/4 adF ( {dadF, 1/2 + (-1)/5 adF (1 + (-1)/6 adF (1 + (-1)/7 adF (1 + ...)))}
              + (-1)/5 (-1)/6 adF ( {dadF, 1/2 + (-1)/7 adF (1 + ... )} + ... ) adF ) adF )
            = 1/2 ( {dadF, f(2)} + 1/(3*4) adF ( {dadF, f(4)} + 1/(5*6) adF ( {dadF, f(6)} + ... ) adF ) adF )
            = 1/2 g(2)
        where
        f(n) = 1/2 + Σ_{k=1} 1/(k+n)! (-1)^k adF^k
             = 1/2 + (-1/(n+1)) adF (1 + (-1/(n+2)) adF (1/2 + f(n+2)))
        g(n) = {dadF, f(n)} + 1/((n+1)(n+2)) adF g(n+2) adF
        """

        X = g.exp(self.X)
        Y = g.exp(self.Y)
        ep = 0.12
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(X)
            M = ep * mul(X,Y,adjoint_b=True)
            F = -g.projectTAH(M)
            adF = g.su3ad(F)
            dF = g.diffprojectTAH(-M, F)
            JF = g.diffexp(adF)
            m = g.eyeOf(adF) + mul(JF, dF)
            j = tf.math.log(det(m))
        fx = 0.5*g.projectTAH(mul(t.gradient(j,X), X, adjoint_b=True))    # Factor of 0.5
        ndF = len(dF.shape)
        d2F = g.diffprojectTAH(-mul(T,tf.expand_dims(M,-3)))    # []_dcb = ∇_d ∂_c F^b
        dadF = g.su3fabc(tf.transpose(dF, perm=list(range(ndF-2))+[ndF-1,ndF-2]))    # []_dce = - ∇_d adF^ce

        order = 12.0   # must be an even number
        one = g.eyeOf(adF)
        half = 0.5*one
        fn = half
        gn = g.eyeOf(dadF)
        while order>3.0:
            fn = half + (-1.0/(order-1.0)) * mul(adF, (one + (-1.0/order) * mul(adF, half + fn)))
            gn = mul(dadF,fn) + mul(fn,dadF) + (1.0/(order*(order-1.0))) * mul(adF, mul(gn, adF))
            order -= 2.0
        dJ = 0.5 * gn

        f = tf.einsum('ca,dac->d', tf.linalg.inv(m), mul(dJ,dF) + mul(JF,d2F))
        with self.subTest(space='su(3)'):
            self.checkEqv(fx,g.su3fromvec(f))
        with self.subTest(space='R'):
            self.checkEqv(g.su3vec(fx),f)

    def test_difflndetexpfmuluOffDiag(self):
        """
        ∇_d ln det {δ^ac + J(F)^ab [∂_c F^b]}    # ∇ can act on different links, ∂ only on the updating link
            = m^{-1}^ca {[∇_d J(F)^ab] [∂_c F^b] + J(F)^ab [∇_d ∂_c F^b]}
        where
            m^ac = δ^ac + J(F)^ab [∂_c F^b]
        This test is for ∇ ≠ ∂, with ∇_d = ∇_Y^d and ∂_c = ∂_X^c, M = X (Z Y)†
        ∇_d adF^ce = ∇_d (- f^ceg F^g) = - f^ceg [∇_d F^g]
        ∇_d F^g = ∇_d tr[T^g X (Z Y)† - Z Y X† T^g]
                = - tr[T^g X Y† T^d Z† + Z T^d Y X† T^g]
        ∇_d ∂_c F^b = ∇_d tr[T^b T^c X (Z Y)† + Z Y X† T^c T^b]
                    = tr[- T^b T^c X Y† T^d Z† + Z T^d Y X† T^c T^b]
                    = tr[- T^b T^c X Y† T^d (X Y†)† X Y† Z† + Z (X Y†)† X Y† T^d Y X† T^c T^b]
                    = tr[- T^b T^c T^f X Y† Z† + Z Y X† T^f T^c T^b] Ad(X Y†)^fd
        ∇_d J(F)^ab
            = Σ_{k=0} 1/(k+1)! (-1)^k ∇_d [(adF)^k]^ab
            = Σ_{k=1} 1/(k+1)! (-1)^k Σ_{j=0}^{k-1} [adF^j]^ac (∇_d adF^ce) [adF^(k-j-1)]^eb
            = Σ_{k=1} 1/(k+1)! (-1)^k Σ_{j=0}^{k-1} [adF^j]^ac (- f^cea [∇_d F^a]) [adF^(k-j-1)]^eb
            = Σ_{k=0} 1/(k+2)! (-1)^k Σ_{j=0}^k [adF^j]^ac f^ceg [∇_d F^g] [adF^(k-j)]^eb
            = 1/2 ( dadF
              + (-1)/3 ( adF dadF + dadF adF
              + (-1)/4 ( adF^2 dadF + adF dadF adF + dadF adF^2
              + (-1)/5 ( adF^3 dadF + adF^2 dadF adF + adF dadF adF^2 + dadF adF^3
              + (-1)/6 ( adF^4 dadF + adF^3 dadF adF + adF^2 dadF adF^2 + adF dadF adF^3 + dadF adF^4
              + (-1)/7 ( adF^5 dadF + adF^4 dadF adF + adF^3 dadF adF^2 + adF^2 dadF adF^3 + adF dadF adF^4 + dadF adF^5 + ... ))))))
            = 1/2 ( {dadF, 1/2 + (-1)/3 adF (1 + (-1)/4 adF (1 + (-1)/5 adF (1 + (-1)/6 adF (1 + (-1)/7 adF (1 + ...)))))}
              + (-1)/3 (-1)/4 adF ( {dadF, 1/2 + (-1)/5 adF (1 + (-1)/6 adF (1 + (-1)/7 adF (1 + ...)))}
              + (-1)/5 (-1)/6 adF ( {dadF, 1/2 + (-1)/7 adF (1 + ... )} + ... ) adF ) adF )
            = 1/2 ( {dadF, f(2)} + 1/(3*4) adF ( {dadF, f(4)} + 1/(5*6) adF ( {dadF, f(6)} + ... ) adF ) adF )
            = 1/2 g(2)
        where
        f(n) = 1/2 + Σ_{k=1} 1/(k+n)! (-1)^k adF^k
             = 1/2 + (-1/(n+1)) adF (1 + (-1/(n+2)) adF (1/2 + f(n+2)))
        g(n) = {dadF, f(n)} + 1/((n+1)(n+2)) adF g(n+2) adF
        """

        X = g.projectSU(g.exp(self.X))
        Y = g.projectSU(g.exp(self.Y))
        Z = g.projectSU(g.exp(self.Z))
        ep = 0.12
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
            t.watch(Y)
            M = ep * mul(X,mul(Z,Y),adjoint_b=True)
            F = -g.projectTAH(M)
            adF = g.su3ad(F)
            dF = g.diffprojectTAH(-M, F)    # ∂_c F^b
            JF = g.diffexp(adF)
            m = g.eyeOf(adF) + mul(JF, dF)
            j = tf.math.log(det(m))
        fx = 0.5*g.projectTAH(mul(t.gradient(j,Y), Y, adjoint_b=True))    # Factor of 0.5
        Adxy = g.SU3Ad(mul(X,Y,adjoint_b=True))
        dydF = g.diffprojectTAHCross(M, p=-F, Adx=Adxy)    # ∇_d F^g
        ndF = len(dF.shape)
        dadF = g.su3fabc(tf.transpose(dydF, perm=list(range(ndF-2))+[ndF-1,ndF-2]))    # []_dce = - ∇_d adF^ce
        d2F = tf.einsum('fcb,fd->dcb', g.diffprojectTAH(mul(T,tf.expand_dims(M,-3))), Adxy)    # []_dcb = ∇_d ∂_c F^b

        order = 12.0   # must be an even number
        one = g.eyeOf(adF)
        half = 0.5*one
        fn = half
        gn = g.eyeOf(dadF)
        while order>3.0:
            fn = half + (-1.0/(order-1.0)) * mul(adF, (one + (-1.0/order) * mul(adF, half + fn)))
            gn = mul(dadF,fn) + mul(fn,dadF) + (1.0/(order*(order-1.0))) * mul(adF, mul(gn, adF))
            order -= 2.0
        dJ = 0.5 * gn

        f = tf.einsum('ca,dac->d', tf.linalg.inv(m), mul(dJ,dF) + mul(JF,d2F))
        with self.subTest(space='su(3)'):
            self.checkEqv(fx,g.su3fromvec(f))
        with self.subTest(space='R'):
            self.checkEqv(g.su3vec(fx),f)

    def checkEqv(self,a,b,tol=1e-13,rtol=1e-13):
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
        if m>=tol or (ma>0 and mb>0 and m/mn>=rtol):
            print(f'received {a}')
            print(f'expected {b}')
            print(f'abs sq diff: {m}')
            if ma>0 and mb>0:
                print(f'rel sq diff: {m/mn}')
        with self.subTest(tolerance='absolute'):
            self.assertLess(m, tol)
        if ma>0 and mb>0:
            with self.subTest(tolerance='relative'):
                self.assertLess(m/mn, rtol)

if __name__ == '__main__':
    ut.main()
