import random
from ..lib import gauge

StreamNum = 4
ConfNum = 512

def get(confs):
    fs = []
    for s,t in confs:
        if s<0 or s>=StreamNum or t<=0 or t>ConfNum:
            raise ValueError(f'unavailable config stream {s} id {t}')
        fs.append(f'data/lat/dbw2_8t16_b0.7796/s{s}/config.{16*t:05d}.lime')
    if len(fs)==1:
        return gauge.readGauge(fs[0])
    else:
        return gauge.readGauge(fs)

TrainingSet = [(s,t) for s in range(StreamNum-1) for t in range(64,ConfNum)]
ValidateSet = [(StreamNum-1,t) for t in range(64,ConfNum)]

def getDistributedSet(s, n):
    ns = len(s)
    d = ns//n
    b = d//2
    r = list(s[b:b+n*d:d])
    if len(r)!=n:
        raise ValueError(f'unable to get {n} samples out of {len(s)}')
    return r

class ConfLoader:
    def __init__(self, nbatch, batch_size, dataset):
        self.nbatch = nbatch
        self.batch_size = batch_size
        self.set = getDistributedSet(dataset, nbatch*batch_size)
    def __call__(self, batch_id):
        if batch_id<0 or batch_id>=self.nbatch:
            random.shuffle(self.set)
            return None
        confs = self.set[batch_id*self.batch_size:batch_id*self.batch_size+self.batch_size]
        return get(confs)

class TrainLoader(ConfLoader):
    def __init__(self, nbatch, batch_size):
        super().__init__(nbatch=nbatch, batch_size=batch_size, dataset=TrainingSet)

class ValidateLoader(ConfLoader):
    def __init__(self, nbatch, batch_size):
        super().__init__(nbatch=nbatch, batch_size=batch_size, dataset=ValidateSet)
