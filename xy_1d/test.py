import nthmc
import tensorflow as tf
import math, unittest

class TestOneDNeighbor(unittest.TestCase):

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

  def test_mask(self):
    o1e = nthmc.OneDNeighbor(distance=1, mask='even')
    o1e.build([5,8])
    self.assertTrue(tf.reduce_all(
      o1e.mask == tf.constant([1,0,1,0,1,0,1,0],dtype=tf.float64)))
    o2e = nthmc.OneDNeighbor(distance=2, mask='even')
    o2e.build([5,8])
    self.assertTrue(tf.reduce_all(
      o2e.mask == tf.constant([1,1,0,0,1,1,0,0],dtype=tf.float64)))
    o4e = nthmc.OneDNeighbor(distance=4, mask='even')
    o4e.build([5,8])
    self.assertTrue(tf.reduce_all(
      o4e.mask == tf.constant([1,1,1,1,0,0,0,0],dtype=tf.float64)))
    o1o = nthmc.OneDNeighbor(distance=1, mask='odd')
    o1o.build([5,8])
    self.assertTrue(tf.reduce_all(
      o1o.mask == tf.constant([0,1,0,1,0,1,0,1],dtype=tf.float64)))
    o2o = nthmc.OneDNeighbor(distance=2, mask='odd')
    o2o.build([5,8])
    self.assertTrue(tf.reduce_all(
      o2o.mask == tf.constant([0,0,1,1,0,0,1,1],dtype=tf.float64)))

  def test_call(self):
    tol = 1E-12
    o = nthmc.OneDNeighbor(distance=3, mask='odd')
    m = tf.constant([1,1,1,0,0,0,1,1,1,0,0,0],dtype=tf.float64)

    x = tf.random.uniform((1024,12),dtype=tf.float64)*2*math.pi - math.pi
    y, ld = o(x)
    self.assertTrue(tf.reduce_all(y == x))
    self.assertGreater(tol, tf.math.reduce_euclidean_norm(ld))

    o.alpha.assign(1.0)
    y, _ = o(x)
    self.assertTrue(tf.reduce_all(m*y == m*x))

  def test_jacob(self):
    tol = 1E-10
    n = 32
    for order in range(1,5):
      for d in range(1,4):
        with self.subTest(distance=d,order=order):
          o = nthmc.OneDNeighbor(distance=d,order=order)
          o.alpha.assign(-2.1)
          x = tf.random.uniform((n,12),dtype=tf.float64)*2*math.pi - math.pi
          with tf.GradientTape(persistent=True) as t:  # persistent for jacobian without pfor
            t.watch(x)
            y, ld = o(x)
          j = t.jacobian(y,x,experimental_use_pfor=False)  # pfor fails for roll op

          for i in range(n):
            for k in range(n):
              if i != k:
                with self.subTest(i=i, k=k):
                  self.assertEqual(0, tf.math.reduce_euclidean_norm(j[i,:,k,:]))

          #tf.print(ld, summarize=-1)
          for i in range(n):
            with self.subTest(i=i):
              #tf.print('j',i,j[i,:,i,:], summarize=-1)
              dj = tf.linalg.det(j[i,:,i,:])
              #tf.print('det j',i,dj,summarize=-1)
              self.assertGreater(tol, tf.math.squared_difference(ld[i], tf.math.log(dj)))

  def test_inv(self):
    tol = 1E-30
    n = 32
    for order in range(1,8,2):
      for d in range(1,4):
        with self.subTest(distance=d,order=order):
          o = nthmc.OneDNeighbor(distance=d,order=order,mask='odd',invAbsR2=tol)
          o.alpha.assign(-2.1)
          x = tf.random.uniform((n,12),dtype=tf.float64)*2*math.pi - math.pi
          y, l = o(x)
          z, m = o.inv(y)
          self.assertGreater(1E2*tol, tf.reduce_mean(tf.math.squared_difference(x, z)))
          self.assertGreater(1E4*tol, tf.reduce_mean(tf.math.squared_difference(l, -m)))

if __name__ == '__main__':
  unittest.main()
