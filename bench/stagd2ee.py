import tensorflow as tf
import os, sys
from benchutil import bench, time
sys.path.append("../lib")
from lattice import SubSetEven, norm2
from stagD import phase, D2ee
from gauge import readGauge, random, setBC

rng = tf.random.Generator.from_seed(9876543211)

if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
    gconf = readGauge(sys.argv[1])
elif len(sys.argv)==5:
    lat = [int(x) for x in sys.argv[1:]]
    if any([x<2 or x%2==1 for x in lat]):
        raise ValueError(f'received incorrect lattice size {sys.argv[1:]}')
    gconf = random(rng, lat)
else:
    gconf = random(rng, [8,8,8,16])

gconf = gconf.hypercube_partition()
gconf = setBC(gconf)
gconf = phase(gconf)

v1e = gconf[0].lattice.get_subset(SubSetEven).randomNormal(rng, new_site_shape=(3,))

m = 0.1
m2 = tf.constant(m*m, dtype=tf.complex128)

# NOTE: tf.function hates custom objects.
def D2ee_eager(vt, m):
    v = v1e.from_tensors(vt)
    y = D2ee(gconf, v, m)
    return y.to_tensors()

f_fun = tf.function(D2ee_eager)
f_jit = tf.function(D2ee_eager,jit_compile=True)

D2ee_fun,_ = time('fun get concrete',lambda:f_fun.get_concrete_function(v1e.to_tensors(), m2))
D2ee_jit,_ = time('jit get concrete',lambda:f_jit.get_concrete_function(v1e.to_tensors(), m2))

def run_fun(f):
    global v2e
    x = v2e.to_tensors()
    y = f(x, m2)
    v2e = v2e.from_tensors(y)
    return v2e

print('Benchmark eager')
v2e = v1e
v2e,_ = bench('D2ee eager',lambda:run_fun(D2ee_eager))
print(f'{norm2(v2e)}')

print('Benchmark function')
v2e = v1e
v2e,_ = bench('D2ee fun',lambda:run_fun(D2ee_fun))
print(f'{norm2(v2e)}')

print('Benchmark jit_compile')
v2e = v1e
v2e,_ = bench('D2ee jit',lambda:run_fun(D2ee_jit))
print(f'{norm2(v2e)}')
