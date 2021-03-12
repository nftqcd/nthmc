import sys
sys.path.append("../lib")
import field as f
import unittest as ut

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
    def test_adjoint(self):
        p = f.OrderedPaths(2, ())
        self.assertEqual(p.adjoint(1), -1)
        self.assertEqual(p.adjoint(-2), 2)
        self.assertEqual(p.adjoint([]), [])
        self.assertEqual(p.adjoint((1, True)), (1, False))
        self.assertEqual(p.adjoint((1, False)), (1, True))
        self.assertEqual(p.adjoint((-1, True)), (-1, False))
        self.assertEqual(p.adjoint((-1, False)), (-1, True))
        self.assertEqual(p.adjoint(((1, 2), False)), ((1, 2), True))
        self.assertEqual(p.adjoint((((1, 2), False), ((-2,-1), True))), (((-2,-1), False), ((1, 2), True)))
        self.assertEqual(p.adjoint([((-2,-1), True), ((1, 2), False)]), [((1, 2), True), ((-2,-1), False)])
    def test_flatpath(self):
        p = f.OrderedPaths(2, ())
        self.assertEqual(p.flatpath((-2,1)), (-2,1))
        self.assertEqual(p.flatpath((-2,((1,-2), False))), (-2,1,-2))
        self.assertEqual(p.flatpath((-2,((1,-2), True))), (-2,2,-1))
        self.assertEqual(p.flatpath((-2,((1,((1,2), False)), True))), (-2,-2,-1,-1))
        self.assertEqual(p.flatpath((-2,1),True), (-1,2))
        self.assertEqual(p.flatpath((-2,((1,-2), False)),True), (2,-1,2))
        self.assertEqual(p.flatpath((-2,((1,-2), True)),True), (1,-2,2))
        self.assertEqual(p.flatpath((-2,((1,((1,2), False)), True)),True), (1,1,2,2))

if __name__ == '__main__':
    ut.main()
