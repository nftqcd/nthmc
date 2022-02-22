import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, forcetrain
import sys
sys.path.append('../lib')
import field

conf = nthmc.Conf(nbatch=3, nepoch=2, nstepEpoch=2, nstepMixing=2, nstepPostTrain=2, initDt=0.05, stepPerTraj=4, nconfStepTune=0, nthr=2, nthrIop=1, xlaCluster=False)
nthmc.setup(conf)
#tf.config.run_functions_eagerly(True)
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
transform = lambda: ftr.TransformChain([
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
])
ftr.checkDep(transform())
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=4.0, beta0=2.0, size=(16,16), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])
loss = lambda action: forcetrain.LossFun(action, betaMap=2.5, cNorm2=1.0/16.0, cNorm4=0.0, cNorm6=0.0, cNorm8=0.0, cNormInf=1.0)
opt = tk.optimizers.Adam(learning_rate=0.002)
mcmc = lambda action: nthmc.Metropolis(conf, nthmc.LeapFrog(conf, action))
x, mc, ls = forcetrain.run(conf, actionFun, mcmc, loss, opt)
po_mc = mc.generate.action.plaquetteWoTrans(x)
p_mc = mc.generate.action.plaquette(x)
po_ls = ls.action.plaquetteWoTrans(x)
p_ls = ls.action.plaquette(x)
tf.print('Final plaquette without transformation:')
tf.print('from MCMC:',po_mc,summarize=-1)
tf.print('from loss:',po_ls,summarize=-1)
tf.print('Final plaquette after transformation:')
tf.print('from MCMC:',p_mc,summarize=-1)
tf.print('from loss:',p_ls,summarize=-1)

def check(x, y, expected):
    good = True
    if len(x)!=len(y):
        tf.print('Error: lengths of results do not match:',x,y,summarize=-1)
        good = False
    if len(x)!=len(expected):
        tf.print('Error: unexpected length:',x,'expected:',expected,summarize=-1)
        good = False
    if len(y)!=len(expected):
        tf.print('Error: unexpected length:',y,'expected:',expected,summarize=-1)
        good = False
    CT = 1e-14
    if any([abs(a-b)>CT for a,b in zip(x,y)]):
        tf.print('Error: results do not match:',x,y,summarize=-1)
        good = False
    if any([abs(a-b)>CT for a,b in zip(x,expected)]):
        tf.print('Error: unexpected results:',x,'expected:',expected,summarize=-1)
        good = False
    if any([abs(a-b)>CT for a,b in zip(y,expected)]):
        tf.print('Error: unexpected results:',y,'expected:',expected,summarize=-1)
        good = False
    return good

good = check(po_mc, po_ls, [0.78018070391738314, 0.74491830557848471, 0.76857616055902422])
good = good and check(p_mc, p_ls, [0.78472570547365139, 0.75066954223482651, 0.77384157030129985])

if not good:
    sys.exit('Test failed.')
