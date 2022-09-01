import tensorflow as tf

def maxValueKey(d):
    return max(d, key=d.get)

class Coord:
    def __init__(self, x = []):
        "x: [int, ...] or (int, ...)"
        self.x = tuple(x)
    def __eq__(self, x):
        if isinstance(x, Coord):
            return self.x == x.x
        else:
            return False
    def __hash__(self):
        return hash(self.x)
    def __str__(self):
        return f'Coord{self.x}'
    def __repr__(self):
        return f'Coord#{id(self)}{self.x}'
    def __len__(self):
        return len(self.x)
    def __getitem__(self, key):
        return self.x[key]
    def __add__(self, y):
        nx = self.nd()
        ny = y.nd()
        n = max(nx, ny)
        m = min(nx, ny)
        d = n*[0]
        for i in range(m):
            d[i] = self.x[i] + y.x[i]
        for i in range(m, nx):
            d[i] = self.x[i]
        for i in range(m, ny):
            d[i] = y.x[i]
        return Coord(d)
    def __sub__(self, y):
        nx = self.nd()
        ny = y.nd()
        n = max(nx, ny)
        m = min(nx, ny)
        d = n*[0]
        for i in range(m):
            d[i] = self.x[i] - y.x[i]
        for i in range(m, nx):
            d[i] = self.x[i]
        for i in range(m, ny):
            d[i] = -y.x[i]
        return Coord(d)
    def __neg__(self):
        n = self.nd()
        d = n*[0]
        for i in range(n):
            d[i] = -self.x[i]
        return Coord(d)
    def __pos__(self):
        n = self.nd()
        d = n*[0]
        for i in range(n):
            d[i] = self.x[i]
        return Coord(d)
    def nd(self):
        return len(self.x)

def coordAtPathWithDir(path, linkDir, dim=0):
    """
    Return a list of coordinates where the path from origin goes through the link in linkDir direction.
    Both path and linkDir counts from 1.
    """
    theDir = abs(linkDir)
    xs = []
    x = Coord([0]*dim)
    if len(path)>0:
        if path[0]<0:
            x += Coord([0]*(abs(path[0])-1) + [1])
    for dir in path:
        d = [0]*(abs(dir)-1) + [1 if dir>0 else -1]
        if dir==theDir:
            xs.append(x)
        x += Coord(d)
        if -dir==theDir:
            xs.append(x)
    return xs

class Path:
    def __init__(self, nd, *dirs):
        "dirs: [+/-d] for d = 1,2,3,... corresponding to x,y,z,..."
        for d in dirs:
            if d==0 or abs(d)>nd:
                raise ValueError(f'invalid dir, got {d}')
        self.dirs = dirs
        self.nd = nd
    def adjoint(self):
        return Path(self.nd, *[-d for d in reversed(self.dirs)])
    def deltaX(self):
        "Return the difference from the end to the start of the path."
        x = self.nd * [0]
        for d in self.dirs:
            if d>0:
                x[d-1] +=  1
            else:
                x[-d-1] -=  1
        return Coord(x)
    def __len__(self):
        return len(self.dirs)
    def __getitem__(self, key):
        return self.dirs[key]
    def __add__(self, o):
        if not isinstance(o, Path):
            raise ValueError(f'must be Path, but got {o}')
        if self.nd!=o.nd:
            raise ValueError(f'different nd: {self.nd} vs. {o.nd}')
        return Path(self.nd, *self.dirs+o.dirs)

class OrdPathInt:
    def __init__(self, path):
        "path: ±int"
        if isinstance(path, int) and path != 0:
            self.d = path
        else:
            raise ValueError(f'path should be ±int: {path}.')
    def __eq__(self, x):
        if isinstance(x, OrdPathInt):
            return self.d == x.d
        else:
            return False
    def __hash__(self):
        return hash(self.d)
    def __str__(self):
        return f'{self.d}'
    def __repr__(self):
        return f'OrdPathInt#{id(self)}({repr(self.d)})'
    def adjoint(self):
        return OrdPathInt(-self.d)
    def flatten(self):
        return (self.d,)
    def position(self):
        "return: Coord relative to the left most starting point."
        if self.d > 0:
            x = []    # empty is zero
        else:
            p = -self.d
            x = p*[0]
            x[p-1] = -1
        return Coord(x)
    def deltaX(self):
        p = abs(self.d)
        d = p*[0]
        if self.d > 0:
            d[p-1] = 1
        else:
            d[p-1] = -1
        return Coord(d)

class OrdPathPair:
    def __init__(self, left, right):
        self.l = left
        self.r = right
    def __eq__(self, x):
        if isinstance(x, OrdPathPair):
            return self.l == x.l and self.r == x.r
        else:
            return False
    def __hash__(self):
        return hash((self.l,self.r))
    def __str__(self):
        return f'Pair({self.l}, {self.r})'
    def __repr__(self):
        return f'OrdPathPair#{id(self)}({repr(self.l)}, {repr(self.r)})'
    def adjoint(self):
        return OrdPathPair(self.r.adjoint(), self.l.adjoint())
    def flatten(self):
        return self.l.flatten() + self.r.flatten()
    def position(self):
        "We always multiply left by a shifted right."
        return self.l.position()
    def deltaX(self):
        return self.l.deltaX() + self.r.deltaX()

class OrdPathList:
    def __init__(self, path):
        if isinstance(path, list):
            self.s = path
        else:
            raise ValueError(f'OrdPathList accepts a list but got: {path}')
    def __eq__(self, x):
        if isinstance(x, OrdPathList):
            return self.s == x.s
        else:
            return False
    def __hash__(self):
        return hash(self.s)
    def __str__(self):
        return f'Path({self.s})'
    def __repr__(self):
        return f'OrdPathSeg#{id(self)}({repr(self.s)})'
    def adjoint(self):
        s = []
        for p in reversed(self.s):
            s.append(p.adjoint())
        return OrdPathList(s)
    def flatten(self):
        s = ()
        for l in self.s:
            s = s + l.flatten()
        return s
    def position(self):
        if len(self.s)>0:
            return self.s[0].position()
        else:
            return Coord([])
    def deltaX(self):
        d = Coord([])
        for l in self.s:
            d += l.deltaX()
        return d

class OrdPathAdj:
    def __init__(self, path):
        if isinstance(path, (OrdPathInt, OrdPathPair, OrdPathList)):
            self.p = path
        else:
            raise ValueError(f'path should be one of OrdPathInt, OrdPathPair, OrdPathList, but got: {path}.')
    def __eq__(self, x):
        if isinstance(x, OrdPathAdj):
            return self.p == x.p
        else:
            return False
    def __hash__(self):
        return hash(self.p)
    def __str__(self):
        return f'Adj({self.p})'
    def __repr__(self):
        return f'OrdPathAdj#{id(self)}({repr(self.p)})'
    def adjoint(self):
        return self.p
    def flatten(self):
        return self.p.adjoint().flatten()
    def position(self):
        return self.p.position() - self.p.deltaX()
    def deltaX(self):
        return -self.p.deltaX()

def topath(xs):
    "Convert possibly nested int sequences to OrdPath*().  We do not nest OrdPathList, but leave outer lists as lists."
    if isinstance(xs, int):
        return OrdPathInt(xs)
    elif isinstance(xs, (tuple, list)):
        ps = [topath(x) for x in xs]
        if isinstance(ps[0], OrdPathList):
            return ps
        else:
            return OrdPathList(ps)
    elif isinstance(xs, (OrdPathInt, OrdPathPair, OrdPathList)):
        return xs
    else:
        raise ValueError(f'topath: cannot convert: {xs}')

def adjointOp(path):
    """ Like adjoint(), but create new OrdPathAdj if it's not already OrdPathAdj or OrdPathInt . """
    if isinstance(path, OrdPathInt):
        return path.adjoint()
    elif isinstance(path, OrdPathAdj):
        return path.p
    else:
        return OrdPathAdj(path)

def mostSharedPair(paths):
    """
    Receives paths and search each element of OrdPathList.
    Return an OrdPathPair occured most frequently among all the paths,
    and its count.
    If there are no pairs, return OrdPath(k:opList) with len 0, and 0.
    """
    pc = {}
    while True:
        p = None
        pa = None
        fp = None
        fpa = None
        c = 0
        ca = 0
        for ps in paths:
            if not (isinstance(ps, OrdPathList) and len(ps.s)>1):
                continue
            i = 0
            while i<len(ps.s)-1:
                i += 1
                t = OrdPathPair(ps.s[i-1],ps.s[i])
                ft = t.flatten()
                if c==0:
                    p = t
                    pa = OrdPathPair(adjointOp(t.r), adjointOp(t.l))
                    if p in pc or pa in pc:
                        continue
                    fp = p.flatten()
                    fpa = tuple([-x for x in reversed(fp)])
                    c = 1
                    i += 1
                elif ft == fp:
                    c += 1
                    i += 1
                elif ft == fpa:
                    ca += 1
                    i += 1
        if c==0:
            # no new pairs
            break
        else:
            ct = c+ca
            if c>=ca:
                pc[p] = ct
            else:
                pc[pa] = ct
    c = 0
    for k,v in pc.items():
        # If there are multiple paris with the max count,
        # which one we return depends on implementation of Table.
        if v>c:
            c = v
            p = k
    if c==0:
        return (OrdPathList([]), 0)
    else:
        return (p, c)

def groupByPair(paths, pair):
    fp = pair.flatten()
    afp = tuple([-x for x in reversed(fp)])
    pairAdj = OrdPathAdj(pair)
    newpaths = []
    for ps in paths:
        if not isinstance(ps, OrdPathList):
            newpaths.append(ps)
            continue
        p = []
        i = 0
        n = len(ps.s)
        while i<n-1:
            t = ps.s[i].flatten() + ps.s[i+1].flatten()
            if t == fp:
                p.append(pair)
                i += 2
            elif t == afp:
                p.append(pairAdj)
                i += 2
            else:
                p.append(ps.s[i])
                i += 1
        if i==n-1:
            p.append(ps.s[i])
        newpaths.append(p[0] if len(p)==1 else OrdPathList(p))
    return newpaths

def minimizeShifts(paths, segments, counts):
    # print(f'minimizeShifts: paths: {paths}')
    # print(f'minimizeShifts: segments: {segments}')
    # print(f'minimizeShifts: counts: {counts}')
    # TODO
    return paths

class OrdPaths:
    """ Build a tree of operations.  Runtime: O(N^3) with N=len(paths). """
    def __init__(self, paths):
        """
        paths: (OrdPathList, ...)
        """
        pgroup = paths
        segments = []
        counts = []
        p,c = mostSharedPair(pgroup)
        while isinstance(p, OrdPathPair):
            segments.append(p)
            counts.append(c)
            pgroup = groupByPair(pgroup, p)
            p,c = mostSharedPair(pgroup)
        pgroup = minimizeShifts(pgroup, segments, counts)
        self.paths = pgroup
        self.segments = segments
        self.counts = counts
        #print(f'OrdPaths.paths: {self.paths}')
        #print(f'OrdPaths,segments: {self.segments}')
    def __eq__(self, x):
        if isinstance(x, OrdPaths):
            return self.paths == x.paths and self.segments == x.segments and self.counts == x.counts
        else:
            return False
    def __hash__(self):
        return hash(tuple(self.paths))
    def __str__(self):
        s = 'OrdPaths:\n    Paths:'
        for p in self.paths:
            s += f'\n        {p}'
        s += '\n    Segments:'
        for i in range(len(self.segments)):
            s += f'\n        {self.segments[i]}  {self.counts[i]}'
        return s
    def __repr__(self):
        return f'OrdPaths#{id(self)}({repr(self.paths)}, {repr(self.segments)}, {repr(self.counts)})'
    def optimal(self, path):
        """
        path: (±dim, ...)
        return: (optimal splitting tree in OrdPath*, isadjoint)
        """
        apath = tuple([-d for d in reversed(path)])
        for p in self.segments:
            if path == p.flatten():
                return p, False
            elif apath == p.flatten():
                return p, True
        raise ValueError(f'OrdPaths.optimal: Do not know how to optimize path: {path}')

class OrdProduct:
    def __init__(self, ordPaths, x, g):
        """
        ordPaths: ∈ OrdPaths
        Uses precomputed OrdPaths to compute all.
        Products of segments are memoized.
        TODO: automatically symmetrize?
        """
        self.ordPaths = ordPaths
        self.x = x
        self.g = g
        self.segments = {}
        for s in ordPaths.segments:
            if isinstance(s, OrdPathPair):
                l,la = self.fetch(s.l)
                r,ra = self.fetch(s.r)
                sh = (s.l.position() - s.l.deltaX() - s.r.position()).x
                self.segments[s.flatten()] = self.g.mul(l, tf.roll(r, shift=(0,)+sh, axis=tuple(range(len(sh)+1))), la, ra)
            else:
                raise ValueError(f'Internal error: OrdProduct got a non-OrdPathPair segment: {s}')

    def prodList(self):
        res = []
        for p in self.ordPaths.paths:
            res.append(self.finishProd(p))
        return res
    def prod(self, pathRaw):
        path, isadjoint = self.ordPaths.optimal(pathRaw)
        return self.finishProd(path, isadjoint)
    def finishProd(self, p, adj=False):
        if isinstance(p, OrdPathAdj):
            f = self.segments[p.p.flatten()]
            if not adj:
                f = self.g.adjoint(f)
        else:
            f = self.segments[p.flatten()]
            if adj:
                f = self.g.adjoint(f)
        pos = (-p.position()).x
        if any([x != 0 for x in pos]):
            f = tf.roll(f, shift=(0,)+pos, axis=tuple(range(len(pos)+1)))
        return f
    def fetch(self, s):
        if isinstance(s, OrdPathInt):
            if s.d>0:
                return (self.x[:,s.d-1], False)
            else:
                return (self.x[:,-s.d-1], True)
        elif isinstance(s, OrdPathPair):
            return (self.segments[s.flatten()], False)
        elif isinstance(s, OrdPathList):
            raise ValueError(f'fetch does not work with OrdPathlist: {s}')
        else:  # OrdPathAdj
            f,a = self.fetch(s.p)
            return (f, a != True)

if __name__ == '__main__':
    ps = OrdPaths(topath([[1,1,2,2,-1,-1,-2,-2]]))
    print(ps)
    ps = OrdPaths(topath([
        [2,-1,-2,1],
        [3,-1,-3,1],
        [3,-2,-3,2],
        [4,-1,-4,1],
        [4,-2,-4,2],
        [4,-3,-4,3],
        [-1,-2,1,2],
        [-1,-3,1,3],
        [-2,-3,2,3],
        [-1,-4,1,4],
        [-2,-4,2,4],
        [-3,-4,3,4],
        [-2,1,2,-1],
        [-3,1,3,-1],
        [-3,2,3,-2],
        [-4,1,4,-1],
        [-4,2,4,-2],
        [-4,3,4,-3],
    ]))
    print(ps)
    ps = OrdPaths(topath([
        [-1,-1,2,2,1,1,-2,-2],
        [2,2,1,1,-2,-2,-1,-1],
        [1,1,-2,-2,-1,-1,2,2],
        [-2,-2,-1,-1,2,2,1,1],
    ]))
    print(ps)
    ps = OrdPaths(topath([
      [1,2,-1,-2], [2,1,-2,-1], [1,-2,-1,2], [2,-1,-2,1], [-1,-2,1,2],
      [1,2,-1,-2,1,-2,-1,2], [-1,-2,1,2,-1,2,1,-2], [-2,-1,2,1,1,-2,-1,2], [-2,1,2,-1,-1,-2,1,2],
      [2,-1,-2,1,-2,-1,2,1]]))
    print(ps)
    ps = OrdPaths(topath([
      [1,1,1,1,1,1,1,1],
      [-1,-1,-1,-1,-1,-1,-1,-1],
      [-1,-1],
      [-1,-1,1],
      [1,1,-1],
      [1,-1,-1],
      [-1,-1,-1]]))
    print(ps)
