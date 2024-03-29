import tensorflow as tf

def time(s,f,n=1,niter=1):
    mint = None
    for i in range(n):
        t0 = tf.timestamp()
        for iter in range(niter):
            ret = f()
        dt = tf.timestamp()-t0
        if mint is None or mint>dt/niter:
            mint = dt/niter
        if n>1:
            if niter>1:
                print(f'{s} run {i}: Time {dt} sec / {niter} = {dt/niter} sec', flush=True)
            else:
                print(f'{s} run {i}: Time {dt} sec', flush=True)
        else:
            if niter>1:
                print(f'{s}: Time {dt} sec / {niter} = {dt/niter} sec', flush=True)
            else:
                print(f'{s}: Time {dt} sec', flush=True)
    return ret, mint

def mem(s=''):
    if tf.config.list_physical_devices('GPU'):
        s = 'mem' if s=='' else s+' mem'
        tf.print(s,tf.config.experimental.get_memory_info('GPU:0'))
        tf.config.experimental.reset_memory_stats('GPU:0')

def bench(s,f):
    r,t = time(s,f,n=3)
    r,t = time(s,f,n=3,niter=1+int(0.5/t))
    mem(s)
    return r,t
