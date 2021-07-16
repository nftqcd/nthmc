import tensorflow as tf
from operator import itemgetter

class Field:
    def __init__(self, group, beta=6.0, size=(8,8), name='Field'):
        self.beta=beta
        self.size=size
        self.nd=len(size)
        self.name=name
        self.g=group

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

class OrdPathInt:
    def __init__(self, path):
        "path: ±int"
        if isinstance(path, int) and path != 0:
            self.path = path
            if path > 0:
                self.isadjoint = False
            else:
                self.isadjoint = True
        else:
            raise ValueError(f'path should be ±int: {path}.')
    def __eq__(self, x):
        if isinstance(x, OrdPathInt):
            return self.path == x.path
        else:
            return False
    def __hash__(self):
        return hash(self.path)
    def __str__(self):
        return f'{self.path}'
    def __repr__(self):
        return f'OrdPathInt#{id(self)}({repr(self.path)})'
    def adjoint(self):
        return OrdPathInt(-self.path)
    def flatten(self):
        return (self.path,)
    def position(self):
        "return: Coord relative to the left most starting point."
        if self.path > 0:
            x = []    # empty is zero
        else:
            p = -self.path
            x = p*[0]
            x[p-1] = -1
        return Coord(x)
    def deltax(self):
        p = abs(self.path)
        d = p*[0]
        if self.path > 0:
            d[p-1] = 1
        else:
            d[p-1] = -1
        return Coord(d)

class OrdPathPair:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __eq__(self, x):
        if isinstance(x, OrdPathPair):
            return self.left == x.left and self.right == x.right
        else:
            return False
    def __hash__(self):
        return hash((self.left,self.right))
    def __str__(self):
        return f'topath(({self.left}, {self.right}))'
    def __repr__(self):
        return f'OrdPathPair#{id(self)}({repr(self.left)}, {repr(self.right)})'
    def adjoint(self):
        return OrdPathPair(self.right.adjoint(), self.left.adjoint())
    def flatten(self):
        return self.left.flatten() + self.right.flatten()
    def position(self):
        "We always multiply left by a shifted right."
        return self.left.position()
    def deltax(self):
        return self.left.deltax() + self.right.deltax()
    # only for pair
    def deltalr(self):
        d = self.left.position() - (self.left.deltax() + self.right.position())
        return d

class OrdPathSeg:
    def __init__(self, path, isadjoint):
        self.path = path
        self.isadjoint = isadjoint
    def __eq__(self, x):
        if isinstance(x, OrdPathSeg):
            return self.path == x.path and self.isadjoint == x.isadjoint
        else:
            return False
    def __hash__(self):
        return hash((self.path,self.isadjoint))
    def __str__(self):
        return f'topath(({self.path}, {self.isadjoint}))'
    def __repr__(self):
        return f'OrdPathSeg#{id(self)}({repr(self.path)}, {repr(self.isadjoint)})'
    def adjoint(self):
        return OrdPathSeg(self.path, not self.isadjoint)
    def flatten(self):
        if self.isadjoint:
            return self.path.adjoint().flatten()
        else:
            return self.path.flatten()
    def position(self):
        "return: Coord relative to the left most starting point."
        d = self.path.position()
        if self.isadjoint:
            d -= self.path.deltax()
        return d
    def deltax(self):
        d = self.path.deltax()
        if self.isadjoint:
            return -d
        else:
            return d

def topath(xs):
    "Convert possibly nested int sequences to OrdPath*()."
    if isinstance(xs, int):
        return OrdPathInt(xs)
    elif isinstance(xs, tuple) and len(xs) == 2:
        if isinstance(xs[1], bool):
            return OrdPathSeg(topath(xs[0]), xs[1])
        else:
            return OrdPathPair(*xs)
    elif isinstance(xs, (tuple, list)):
        return [topath(x) for x in xs]
    elif isinstance(xs, (OrdPathInt, OrdPathPair, OrdPathSeg)):
        return xs
    else:
        raise ValueError(f'topath: cannot convert: {xs}')

class OrdPaths:
    def __init__(self, paths):
        """
        paths: (path, ...)
        """
        segments = {}
        while any(len(p)>1 for p in paths):
            # print(f'OrdPaths: paths: {paths}')
            p,c = self.getMostSharedPair(paths)
            if p.left.isadjoint and p.right.isadjoint:
                p = p.adjoint()
            segments[p.flatten()] = (p, c)
            # print(f'OrdPaths: mostShared c: {c} p: {p}')
            dx = p.deltalr()
            # print(f'dx: {dx}')
            paths = self.groupByPair(paths, p)
        paths = tuple([p[0] for p in paths])  # len(p)==1 from while above
        paths = self.minimizeShifts(paths, segments)
        self.paths = paths
        self.segments = segments
        #print(f'OrdPaths.paths: {self.paths}')
        #print(f'OrdPaths,segments: {self.segments}')
    def __eq__(self, x):
        if isinstance(x, OrdPaths):
            return self.paths == x.paths
        else:
            return False
    def __hash__(self):
        return hash(tuple([tuple(p) for p in self.paths]))

    def optimal(self, path):
        """
        path: (±dim, ...)
        return: (optimal splitting tree in OrdPath*, isadjoint)
        """
        if path in self.segments:
            return self.segments[path][0], False
        else:
            apath = tuple([-d for d in reversed(path)])
            if apath in self.segments:
                return self.segments[apath][0], True
            else:
                raise ValueError(f'Do not know how to optimize path: {path}')

    def getMostSharedPair(self, paths):
        "return: an OrdPathPair and its occurance among all the paths."
        pairs = {}
        while True:
            pair = None
            apair = None
            count = 0
            for ps in paths:
                # print(f'getMostSharedPair: ps: {ps}')
                if len(ps) < 2:
                    continue
                # for i in range(len(ps)-1):
                for i in reversed(range(len(ps)-1)):
                    t = OrdPathPair(ps[i],ps[i+1])
                    ta = t.adjoint()
                    # print(f'getMostSharedPair: t: {t}')
                    if t in pairs or ta in pairs:
                        continue
                    if pair is None:
                        pair = t
                        apair = ta
                        count = 1
                        # print(f'getMostSharedPair: pair: {pair}')
                    elif t == pair or t == apair:
                        # Now counting the pair
                        count += 1
            if pair is None:
                # no new pairs
                break
            else:
                pairs[pair] = count
        p = maxValueKey(pairs)
        return p,pairs[p]

    def groupByPair(self, paths, pair):
        apair = pair.adjoint()
        newpaths = []
        for ps in paths:
            p = []
            i = 0
            np = len(ps)
            while i < np:
                if i != np-1:
                    t = OrdPathPair(*ps[i:i+2])
                    if t == pair:
                        p.append(OrdPathSeg(pair,False))
                        i += 2
                    elif t == apair:
                        p.append(OrdPathSeg(pair,True))
                        i += 2
                    else:
                        p.append(ps[i])
                        i += 1
                else:
                    p.append(ps[i])
                    i += 1
            newpaths.append(p)
        return newpaths

    def minimizeShifts(self, paths, segments):
        # print(f'minimizeShifts: paths: {paths}')
        # print(f'minimizeShifts: segments: {segments}')
        # TODO
        return paths

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

    def prodList(self):
        return [self.prod(p.flatten()) for p in self.ordPaths.paths]
    def prod(self, pathRaw):
        path, isadjoint = self.ordPaths.optimal(pathRaw)
        z = self.prod_helper(path, isadjoint)
        p = path.position().x
        if any([x != 0 for x in p]):
            z = tf.roll(z, shift=[0]+[-i for i in p], axis=tuple(range(len(p)+1)))
        return z
    def prod_helper(self, path, isadjoint = False):
        if path in self.segments:
            z = self.segments[path]
            if isadjoint:
                z = self.g.adjoint(z)
        elif isinstance(path, OrdPathInt):
            z = self.x[:,abs(path.path)-1]
            if path.path < 0:
                z = self.g.adjoint(z)
            self.segments[path] = z
            if isadjoint:
                z = self.g.adjoint(z)
        elif isinstance(path, OrdPathPair):
            l = path.left
            r = path.right
            s = path.deltalr().x
            z = self.g.mul(self.prod_helper(l), tf.roll(self.prod_helper(r), shift=(0,)+s, axis=tuple(range(len(s)+1))))
            self.segments[path] = z
            if isadjoint:
                z = self.g.adjoint(z)
        elif isinstance(path, OrdPathSeg):
            z = self.prod_helper(path.path, path.isadjoint != isadjoint)
        else:
            raise ValueError(f'Do not know how to compute path: {path}')
        return z

if __name__ == '__main__':
    ps = OrdPaths([[1,1,2,2,-1,-1,-2,-2]])
    print(ps)
