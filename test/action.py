from . import testutil as tu
from ..lib import action, lattice, field, transform
from ..lib import gauge as g
import tensorflow as tf
import tensorflow.keras.layers as tl

class TestAction(tu.LatticeTest):
    def setUp(self):
        super().setUp()
        self.rng = tf.random.Generator.from_seed(7654321)
        self.act = action.SU3d4(beta=0.7796, c1=action.C1DBW2)

    def test0_ident(self):
        self.check(-1, [4,2,4,2,2,3,3], transform.Ident())

    def test0_chain(self):
        self.check(-1, [4,2,4,2,2,3,3],
            transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        transform.SymmetricShifts(symmetric_shifts=1),
                        transform.Residue(transform.LocalSelfAttention(num_heads=8,key_dim=4)),
                        tl.Dense(units=9, activation='swish'),
                        transform.FlattenSiteLocal(input_local_rank=2),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=108, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=54, activation=None)]),
                    dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]))

    def test1_ident(self):
        self.check(0, [1,4,2,2,6,4,3,3], transform.Ident())

    def test1_chain(self):
        self.check(0, [1,4,2,2,6,4,3,3],
            transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        transform.SymmetricShifts(symmetric_shifts=1),
                        transform.Residue(transform.LocalSelfAttention(num_heads=8,key_dim=4)),
                        tl.Dense(units=9, activation='swish'),
                        transform.FlattenSiteLocal(input_local_rank=2),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=108, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=54, activation=None)]),
                    dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]))

    def test2_ident(self):
        self.check(0, [3,4,2,2,2,2,3,3], transform.Ident())

    def test2_chain(self):
        self.check(0, [3,4,2,2,2,2,3,3],
            transform.TransformChain(
                [transform.StoutSmearSlice(
                    coeff=transform.CoefficientNets([
                        transform.SymmetricShifts(symmetric_shifts=1),
                        transform.Residue(transform.LocalSelfAttention(num_heads=8,key_dim=4)),
                        tl.Dense(units=9, activation='swish'),
                        transform.FlattenSiteLocal(input_local_rank=2),
                        transform.Normalization(),
                        transform.Residue(transform.LocalFeedForward(inner_size=108, inner_activation='swish')),
                        transform.Normalization(),
                        tl.Dense(units=54, activation=None)]),
                    dir=i, is_odd=eo)
                 for eo in {False,True} for i in range(4)]))

    def check(self, bd, dims, transform):
        gauge = g.from_tensor(self.random(dims), batch_dim=bd).projectSU()
        tavec = action.TransformedActionVectorBase(transform=transform, action=self.act)
        tavfm = action.TransformedActionVectorFromMatrixBase(transform=transform, action=self.act)
        tamat = action.TransformedActionMatrixBase(transform=transform, action=self.act)
        svec,lvec,bvec = tavec(gauge)
        svfm,lvfm,bvfm = tavfm(gauge)
        smat,lmat,bmat = tamat(gauge)
        with self.subTest(quantity='action vec==vfm'):
            self.check_eqv(svec,svfm, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='logdet vec==vfm'):
            self.check_eqv(lvec,lvfm, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='coeffs vec==vfm'):
            self.check_eqv(bvec,bvfm, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='action vec==mat'):
            self.check_eqv(svec,smat, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='logdet vec==mat'):
            self.check_eqv(lvec,lmat, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='coeffs vec==mat'):
            self.check_eqv(bvec,bmat, tol=1e-12, rtol=1e-11)
        gvec,lvec,bvec = tavec.gradient(gauge)
        gvfm,lvfm,bvfm = tavfm.gradient(gauge)
        gmat,lmat,bmat = tamat.gradient(gauge)
        with self.subTest(quantity='force vec==vfm'):
            self.check_eqv(gvec.to_tensors(),gvfm.to_tensors(), tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='gradient.logdet vec==vfm'):
            self.check_eqv(lvec,lvfm, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='gradient.coeffs vec==vfm'):
            self.check_eqv(bvec,bvfm, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='force vec==mat'):
            self.check_eqv(gvec.to_tangent().to_tensors(),gmat.to_tensors(), tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='force.exp vec.exp==mat.exp'):
            self.check_eqv(gvec.exp().to_tensors(),gmat.exp().to_tensors(), tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='gradient.logdet vec==mat'):
            self.check_eqv(lvec,lmat, tol=1e-12, rtol=1e-11)
        with self.subTest(quantity='gradient.coeffs vec==mat'):
            self.check_eqv(bvec,bmat, tol=1e-12, rtol=1e-11)

    def random(self, shape):
        r = self.rng.normal(shape, dtype=tf.float64)
        i = self.rng.normal(shape, dtype=tf.float64)
        return tf.dtypes.complex(r,i)

if __name__ == '__main__':
    tu.main()
