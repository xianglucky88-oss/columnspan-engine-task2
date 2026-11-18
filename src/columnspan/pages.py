import io, struct
import pyarrow as pa
import pyarrow.parquet as pq

def encode_plain(values):
    if len(values)>100000: raise ValueError("value limit exceeded")
    return b"".join(struct.pack("<q",value) for value in values)

def decode_plain(data,count):
    if not isinstance(count,int) or count<0 or count>100000: raise ValueError("invalid count")
    if len(data)!=count*8: raise ValueError("truncated or trailing PLAIN data")
    return [x[0] for x in struct.iter_unpack("<q",data)]

def write_column(values):
    if len(values)>100000: raise ValueError("row limit exceeded")
    array=pa.array(values,type=pa.int64())
    stream=io.BytesIO()
    pq.write_table(pa.table({"value":array}),stream,use_dictionary=False,compression="NONE",write_statistics=True)
    return stream.getvalue()

def read_column(data,max_rows=100000,max_bytes=8_000_000):
    if len(data)>max_bytes: raise ValueError("input byte limit exceeded")
    pf=pq.ParquetFile(io.BytesIO(data))
    if pf.metadata.num_rows>max_rows: raise ValueError("row limit exceeded")
    schema=pf.schema_arrow
    if schema.names != ["value"] or schema.field(0).type!=pa.int64(): raise ValueError("expected one int64 column")
    return pf.read().column(0).to_pylist()
