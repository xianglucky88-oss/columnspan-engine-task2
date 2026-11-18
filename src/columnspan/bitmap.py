class Bitmap:
    def __init__(self,values=()):
        self.blocks={}
        for v in values:
            if not isinstance(v,int) or not 0<=v<2**32: raise ValueError("uint32 required")
            hi,lo=divmod(v,65536);self.blocks[hi]=self.blocks.get(hi,0)|(1<<lo)
    def values(self):
        for hi,bits in sorted(self.blocks.items()):
            while bits:
                low=bits & -bits;yield hi*65536+low.bit_length()-1;bits^=low
    def operate(self,other,op):
        if op not in {"union","intersection","difference"}: raise ValueError("unknown set operation")
        result=Bitmap()
        for hi in self.blocks.keys()|other.blocks.keys():
            a,b=self.blocks.get(hi,0),other.blocks.get(hi,0)
            val=a|b if op=="union" else a&b if op=="intersection" else a&~b
            if val:result.blocks[hi]=val
        return result
    def rank(self,value):
        if not isinstance(value,int):raise ValueError("integer required")
        if value<0:return 0
        if value>=2**32:return sum(v.bit_count() for v in self.blocks.values())
        hi,lo=divmod(value,65536)
        return sum(v.bit_count() for k,v in self.blocks.items() if k<hi)+(self.blocks.get(hi,0)&((1<<(lo+1))-1)).bit_count()
    def select(self,k):
        if not isinstance(k,int) or k<0:raise ValueError("invalid rank")
        for hi,bits in sorted(self.blocks.items()):
            n=bits.bit_count()
            if k>=n:k-=n;continue
            while k:bits^=bits&-bits;k-=1
            return hi*65536+(bits&-bits).bit_length()-1
        raise IndexError("rank outside set")
