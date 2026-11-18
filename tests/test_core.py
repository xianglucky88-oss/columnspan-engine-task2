import gc,itertools,io,random
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from columnspan import pages,ownership,views,bitmap,layout

def test_plain_fixed_vectors():
    raw=b'\x01'+b'\x00'*7+b'\xff'*8
    assert pages.decode_plain(raw,2)==[1,-1]
    assert pages.encode_plain([1,-1])==raw
    with pytest.raises(ValueError):pages.decode_plain(raw[:-1],2)
    with pytest.raises(ValueError):pages.decode_plain(raw,1)
def test_parquet_external_reader_and_writer():
    values=[None,-2,0,2**60,None,7]
    raw=pages.write_column(values)
    assert raw[:4]==raw[-4:]==b'PAR1'
    assert pq.read_table(io.BytesIO(raw)).to_pydict()=={'value':values}
    buf=io.BytesIO();pq.write_table(pa.table({'value':pa.array(values,type=pa.int64())}),buf,use_dictionary=True)
    assert pages.read_column(buf.getvalue())==values
    with pytest.raises(ValueError):pages.read_column(raw,max_rows=2)
    with pytest.raises(Exception):pages.read_column(raw[:-9])
def test_native_owner_survives_parent_release():
    lib=ownership.library();before=lib.release_count()
    a=ownership.import_native(4);v=a.slice(1,2);del a;gc.collect()
    assert v.to_pylist()==[1.5,2.5]
    assert lib.release_count()==before
    del v;gc.collect();assert lib.release_count()==before+1
@pytest.mark.parametrize('shape,strides,offset,capacity,writable',[
 ((2,3),(3,-1),2,6,True),((2,3),(0,1),0,3,False),((0,3),(3,1),0,0,True), ((),(),0,1,True)])
def test_views(shape,strides,offset,capacity,writable):
    v=views.View(shape,strides,offset);assert v.analyze(capacity)['writable']==writable
def test_view_bounds():
    with pytest.raises(ValueError):views.View((2,),(2,),0).analyze(2)
    with pytest.raises(ValueError):views.View((100001,),(1,)).analyze(100001)
def test_bitmap_against_sets():
    rng=random.Random(9)
    for _ in range(25):
        a=set(rng.sample(range(130000),45));b=set(rng.sample(range(130000),40));aa,bb=bitmap.Bitmap(a),bitmap.Bitmap(b)
        for op,expected in [('union',a|b),('intersection',a&b),('difference',a-b)]:assert set(aa.operate(bb,op).values())==expected
        ordered=sorted(a)
        for k,v in enumerate(ordered):assert aa.select(k)==v and aa.rank(v)==k+1
    with pytest.raises(ValueError):bitmap.Bitmap([-1])
def test_layout_alignment_and_validation():
    r=layout.compile_layout([{'name':'a','type':'u8'},{'name':'b','type':'u32'},{'name':'c','type':'u16'}])
    assert [x['offset'] for x in r['fields']]==[0,4,8] and r['size']==12
    with pytest.raises(ValueError):layout.compile_layout([{'name':'a','type':'u8'},{'name':'a','type':'u16'}])
    with pytest.raises(ValueError):layout.compile_layout([{'name':'a','type':'u64','count':100000}],max_size=10)
