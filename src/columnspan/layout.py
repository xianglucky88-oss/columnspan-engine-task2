TYPES={"u8":(1,1),"u16":(2,2),"u32":(4,4),"u64":(8,8),"i32":(4,4),"f64":(8,8)}
def compile_layout(fields,byteorder="little",max_size=1_000_000):
    if byteorder not in {"little","big"}: raise ValueError("invalid byte order")
    if not isinstance(fields,list) or len(fields)>256: raise ValueError("field count limit")
    out=[];offset=0;alignment=1;names=set()
    for field in fields:
        name=field["name"];kind=field["type"];count=field.get("count",1)
        if not isinstance(name,str) or not name or name in names:raise ValueError("duplicate or empty field")
        if kind not in TYPES or not isinstance(count,int) or not 0<=count<=100000:raise ValueError("unsupported field")
        size,align=TYPES[kind];names.add(name);alignment=max(alignment,align)
        offset=(offset+align-1)//align*align
        out.append({"name":name,"offset":offset,"size":size*count,"alignment":align})
        offset+=size*count
        if offset>max_size:raise ValueError("layout size limit")
    return {"size":(offset+alignment-1)//alignment*alignment,"alignment":alignment,"byteorder":byteorder,"fields":out}
