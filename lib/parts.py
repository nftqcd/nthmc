class HypercubeParts:
    def __init__(self, parts, subset='all'):
        self.parts = parts
        self.subset = subset
    def __len__(self):
        return len(self.parts)
    def __getitem__(self, key):
        return self.parts[key]
    def __add__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([p+o for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([p+other for p in self.parts], subset=self.subset)
    def __radd__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([o+p for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([other+p for p in self.parts], subset=self.subset)
    def __sub__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([p-o for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([p-other for p in self.parts], subset=self.subset)
    def __rsub__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([o-p for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([other-p for p in self.parts], subset=self.subset)
    def __mul__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([p*o for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([p*other for p in self.parts], subset=self.subset)
    def __rmul__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([o*p for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([other*p for p in self.parts], subset=self.subset)
    def __truediv__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([p/o for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([p/other for p in self.parts], subset=self.subset)
    def __rtruediv__(self, other):
        if isinstance(other, HypercubeParts):
            if len(self)!=len(other):
                raise ValueError(f'size differ {len(self)} vs {len(other)}')
            if self.subset!=other.subset:
                raise ValueError(f'subset differ {self.subset} vs {other.subset}')
            return HypercubeParts([o/p for p,o in zip(self.parts,other.parts)], subset=self.subset)
        else:
            return HypercubeParts([other/p for p in self.parts], subset=self.subset)
    def __neg__(self):
        return HypercubeParts([-p for p in self.parts], subset=self.subset)
    def __pos__(self):
        return HypercubeParts(self.parts, subset=self.subset)
