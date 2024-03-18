import random
from ..lib import gauge

StreamNum = 4
ConfNum = 512

def get(confs, lat_label, conf_step):
    fs = []
    for s,t in confs:
        fs.append(f'data/lat/{lat_label}/s{s}/config.{conf_step*t:05d}.lime')
    if len(fs)==1:
        return gauge.readGauge(fs[0])
    else:
        return gauge.readGauge(fs)

TrainingSet = [(s,t) for s in range(StreamNum-1) for t in range(64,ConfNum)]
ValidateSet = [(StreamNum-1,t) for t in range(64,ConfNum)]

ValidateSet_12t24 = [(s,t) for s in range(2) for t in range(1,17)]

def getDistributedSet(s, n):
    ns = len(s)
    d = ns//n
    b = d//2
    r = list(s[b:b+n*d:d])
    if len(r)!=n:
        raise ValueError(f'unable to get {n} samples out of {len(s)}')
    return r

class ConfLoader:
    def __init__(self, nbatch, batch_size, dataset, lat_label='dbw2_8t16_b0.7796', conf_step=16):
        self.nbatch = nbatch
        self.batch_size = batch_size
        self.set = getDistributedSet(dataset, nbatch*batch_size)
        self.lat_label = lat_label
        self.conf_step = conf_step
    def __call__(self, batch_id):
        if batch_id<0 or batch_id>=self.nbatch:
            random.shuffle(self.set)
            return None
        confs = self.set[batch_id*self.batch_size:batch_id*self.batch_size+self.batch_size]
        return get(confs, lat_label=self.lat_label, conf_step=self.conf_step)

class TrainLoader(ConfLoader):
    def __init__(self, nbatch, batch_size):
        super().__init__(nbatch=nbatch, batch_size=batch_size, dataset=TrainingSet)

class ValidateLoader(ConfLoader):
    def __init__(self, nbatch, batch_size):
        super().__init__(nbatch=nbatch, batch_size=batch_size, dataset=ValidateSet)

class ValidateLoader_12t24(ConfLoader):
    def __init__(self, batch_size, nbatch=32):
        super().__init__(nbatch=nbatch, batch_size=batch_size, dataset=ValidateSet_12t24, lat_label='dbw2_12t24_b0.8895', conf_step=256)
