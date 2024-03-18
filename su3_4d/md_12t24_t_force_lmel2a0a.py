import tensorflow as tf
if __name__=='__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.normpath(path.join(path.dirname(__file__),'../..')))
    __package__ = 'nthmc.su3_4d'
from . import dataset
from ..lib import action, evolve, fieldio, gauge, transform, nthmc, forcetrain

# a/r₀, Eq (4.11), S. Necco, Nucl. Phys. B683, 137 (2004)
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.8895
# 0.265118
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7796
# 0.40001
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7383
# 0.499952
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.7099
# 0.599819
#    ^-+/1.6007 2.3179 0.8020 19.8509*0 1 2 3^~<:0.6716
# 0.800164
def run(beta=0.8895, targetBeta=0.7099, nbatch=64, nbatchValidate=1, batch_size=1, nepoch=8,
        cPlaqInit=0.001, cChairInit=0.0001, dimChainLen=1, chainLen=1,
        lrInit=0.000001, lrSteps=[(1/4, 0.01), (7/8, 0.01), (1, 0.0)],
        tau=[0.0025,0.005,0.01,0.02], test_i=0):
    conf = nthmc.Conf(nthr=24, nthrIop=1)
    conf.seed += test_i
    nthmc.setup(conf)
    tinit = tf.timestamp()
    tf.print('beta:',beta)
    tf.print('targetBeta:',targetBeta)
    tf.print('nbatch:',nbatch)
    tf.print('nbatchValidate:',nbatchValidate)
    tf.print('batch_size:',batch_size)
    tf.print('nepoch:',nepoch)
    tf.print('cPlaqInit:',cPlaqInit)
    tf.print('cChairInit:',cChairInit)
    tf.print('dimChainLen:',dimChainLen)
    tf.print('chainLen:',chainLen)
    tf.print('lrInit:',lrInit)
    tf.print('lrSteps:',lrSteps)

    rng = tf.random.Generator.from_seed(conf.seed)
    initgaugeLoader = dataset.ValidateLoader_12t24(batch_size=1)    # use validation set to init
    tf.print('initgauge:',initgaugeLoader.set,summarize=-1)

    initgauge = initgaugeLoader(test_i)
    tf.print('plaquette:', gauge.plaq(initgauge), summarize=-1)

    transformedAct = action.TransformedActionVectorFromMatrixBase(
            transform=transform.TransformChain(
                [transform.StoutSmearSlice(coeff=transform.CoefficientVariable(cPlaqInit, chair=cChairInit, rng=rng), dir=i, is_odd=eo)
                 for _ in range(chainLen) for i in range(4) for _ in range(dimChainLen) for eo in {False,True}]),
            action=action.SU3d4(beta=beta, c1=action.C1DBW2))
    target = action.TransformedActionVectorFromMatrixBase(
            transform=transform.Ident(),
            action=action.SU3d4(beta=targetBeta, c1=action.C1DBW2))

    transformedAct(initgauge)
    transformedAct.transform.load_weights('weights')
    for t in transformedAct.transform.chain:
        # t is StoutSmearSlice here with simple coeff
        tf.print('coeff dir:',t.dir,'odd' if t.is_odd else 'even',transform.scale_coeff(t.coeff,0.75),summarize=-1)
    dt = tf.timestamp()-tinit
    tf.print('# initialization time:',dt,'sec')
    ti = tf.timestamp()
    mappedConf,lndet,iter = transformedAct.transform.inv(initgauge)
    tf.print('lndet:',lndet)
    tf.print('iter:',iter)
    tf.print('# inv time:', tf.timestamp()-ti, 'sec')
    tf.print('mapped plaquette:', gauge.plaq(mappedConf), summarize=-1)

    @tf.function
    def tact_gradient_func_(xt):
        ft,lndet,cs = transformedAct.gradient(mappedConf.from_tensors(xt))
        return ft.to_tensors(),lndet,cs
    def tact_gradient_func(x):
        ftt,lndet,cs = tact_gradient_func_(x.to_tensors())
        return mappedConf.tangentVector_from_tensors(ftt),lndet,cs

    @tf.function
    def target_gradient_func_(xt):
        ft,lndet,cs = target.gradient(mappedConf.from_tensors(xt))
        return ft.to_tensors(),lndet,cs
    def target_gradient_func(x):
        ftt,lndet,cs = target_gradient_func_(x.to_tensors())
        return mappedConf.tangentVector_from_tensors(ftt),lndet,cs

    ti = tf.timestamp()
    ft,lndet,cs = tact_gradient_func(mappedConf)
    tf.print('lndet:',lndet)
    tf.print('cs:',cs,summarize=-1)
    tf.print('# [init]transformedAct.gradient time:', tf.timestamp()-ti, 'sec')

    ti = tf.timestamp()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')
    ft,lndet,cs = tact_gradient_func(mappedConf)
    tf.print('lndet:',lndet)
    tf.print('cs:',cs,summarize=-1)
    tf.print('# transformedAct.gradient time:', tf.timestamp()-ti, 'sec')
    if tf.config.list_physical_devices('GPU'):
        mi = tf.config.experimental.get_memory_info('GPU:0')
        tf.print('# mem: peak',int(mi['peak']/(1024*1024)),'MiB current',int(mi['current']/(1024*1024)),'MiB')
        tf.config.experimental.reset_memory_stats('GPU:0')

    ti = tf.timestamp()
    f0,_,_ = target_gradient_func(mappedConf)
    tf.print('# [init]target.gradient time:', tf.timestamp()-ti, 'sec')
    ti = tf.timestamp()

    ti = tf.timestamp()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')
    f0,_,_ = target_gradient_func(mappedConf)
    tf.print('# target.gradient time:', tf.timestamp()-ti, 'sec')
    ti = tf.timestamp()
    if tf.config.list_physical_devices('GPU'):
        mi = tf.config.experimental.get_memory_info('GPU:0')
        tf.print('# mem: peak',int(mi['peak']/(1024*1024)),'MiB current',int(mi['current']/(1024*1024)),'MiB')
        tf.config.experimental.reset_memory_stats('GPU:0')

    loss = forcetrain.LMEl2Loss()
    lv,res = loss(ft,f0)
    tf.print('loss:', lv)
    loss.printCallResults(res)
    tf.print('# loss time:', tf.timestamp()-ti, 'sec')

    tf.print('tau:',tau)

    mom = action.QuadraticMomentum()
    dyn = action.Dynamics(V=transformedAct, T=mom)

    md = evolve.Omelyan2MN(dynamics=dyn, trajLength=tau[0], stepPerTraj=1)
    p0 = initgauge.randomTangentVector(rng)

    # @tf.function(jit_compile=True)
    @tf.function
    def mcmcfun_(x_,p_,r):
        x = initgauge.from_tensors(x_)
        p = p0.from_tensors(p_)
        xn, pn, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmc(x,p,r)
        return xn.to_tensors(),pn.to_tensors(),x1.to_tensors(),p1.to_tensors(), v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs
    def mcmcfun(x,p,r):
        x_, p_, x1_, p1_, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmcfun_(x.to_tensors(),p.to_tensors(),r)
        return x.from_tensors(x_),p.from_tensors(p_),x.from_tensors(x1_),p.from_tensors(p1_), v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs

    mcmc = nthmc.Metropolis(md)
    alwaysAccept = tf.constant(0.0, dtype=tf.float64)

    tf.print('Test:',test_i)
    x = initgaugeLoader(test_i)
    tf.print('plaq:', gauge.plaq(x), summarize=-1)
    ti = tf.timestamp()
    x = transformedAct.transform.inv(x)[0]
    tf.print('# inv time:', tf.timestamp()-ti, 'sec')
    tf.print('plaqWoTrans:', gauge.plaq(x), summarize=-1)

    for tlen in tau:
        tbegin = tf.timestamp()
        md.trajLength.assign(tlen)
        md.dt.assign(md.trajLength)
        tf.print('dt:', tlen)
        p = initgauge.randomTangentVector(rng)
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats('GPU:0')
        xn, pn, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs = mcmcfun(x, p, alwaysAccept)
        tf.print('# mcmc step time:',tf.timestamp()-tbegin,'sec',summarize=-1)
        if tf.config.list_physical_devices('GPU'):
            mi = tf.config.experimental.get_memory_info('GPU:0')
            tf.print('# mem: peak',int(mi['peak']/(1024*1024)),'MiB current',int(mi['current']/(1024*1024)),'MiB')
            tf.config.experimental.reset_memory_stats('GPU:0')
        nthmc.printMCMCRes(*nthmc.packMCMCRes(mcmc, xn, pn, x, p, x1, p1, v0, t0, v1, t1, dH, acc, arand, ls, f2s, fms, bs))
        newx = tf.stack(transformedAct.transform(xn)[0].to_tensors())
        fieldio.writeLattice(newx.numpy(), f'12t24/t{tlen}.gauge.test{test_i}.lime')

    tf.print('# Total time:',tf.timestamp()-tinit,'sec')

if __name__=='__main__':
    run()
