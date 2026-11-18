from dataclasses import dataclass
import itertools, math
@dataclass(frozen=True)
class View:
    shape:tuple
    strides:tuple
    offset:int=0
    def addresses(self,capacity,limit=100000):
        if len(self.shape)!=len(self.strides) or len(self.shape)>12: raise ValueError("invalid dimensions")
        if not isinstance(capacity,int) or capacity<0: raise ValueError("invalid capacity")
        if not isinstance(self.offset,int) or any(not isinstance(x,int) or x<0 for x in self.shape) or any(not isinstance(x,int) for x in self.strides): raise ValueError("integer shape and strides required")
        if math.prod(self.shape)>limit: raise ValueError("analysis budget exceeded")
        out=[]
        for idx in itertools.product(*(range(n) for n in self.shape)):
            addr=self.offset+sum(i*s for i,s in zip(idx,self.strides))
            if not 0<=addr<capacity: raise ValueError("view outside allocation")
            out.append(addr)
        return out
    def analyze(self,capacity):
        a=self.addresses(capacity)
        return {"size":len(a),"minimum":min(a) if a else None,"maximum":max(a) if a else None,"writable":len(set(a))==len(a)}
