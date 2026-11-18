from . import pages,ownership,views,bitmap,layout

def page_call(d):
    if d.get('action')=='read':return {'values':pages.read_column(bytes.fromhex(d['hex']))}
    data=pages.write_column(d['values']);return {'hex':data.hex(),'values':pages.read_column(data)}
def bitmap_call(d):
    a,b=bitmap.Bitmap(d['left']),bitmap.Bitmap(d.get('right',[]));out=a.operate(b,d.get('op','union'))
    return {'values':list(out.values()),'rank':out.rank(d.get('rank_at',65536))}
OPERATIONS={
 'pages':page_call,
 'ownership':lambda d:ownership.inspect_exchange(d.get('length',4)),
 'views':lambda d:views.View(tuple(d['shape']),tuple(d['strides']),d.get('offset',0)).analyze(d['capacity']),
 'bitmap':bitmap_call,
 'layout':lambda d:layout.compile_layout(d['fields'],d.get('byteorder','little')),
}
EXAMPLES={
 'pages':{'values':[1,None,-7,4]},'ownership':{'length':4},
 'views':{'shape':[2,3],'strides':[3,-1],'offset':2,'capacity':6},
 'bitmap':{'left':[1,3,65536],'right':[3,7],'op':'union'},
 'layout':{'fields':[{'name':'flag','type':'u8'},{'name':'value','type':'u32'}]},
}
