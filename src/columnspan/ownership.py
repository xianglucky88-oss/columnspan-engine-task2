import ctypes, gc, hashlib, subprocess, sys, tempfile, threading
from pathlib import Path
import pyarrow as pa
_LOCK=threading.Lock()
_LIB=None
_EXCHANGE_LOCK=threading.Lock()
class ArrowArray(ctypes.Structure): pass
class ArrowSchema(ctypes.Structure): pass
ArrowArray._fields_=[(name,ctypes.c_int64) for name in ("length","null_count","offset","n_buffers","n_children")]+[(name,ctypes.c_void_p) for name in ("buffers","children","dictionary","release","private_data")]
ArrowSchema._fields_=[(name,ctypes.c_void_p) for name in ("format","name","metadata")]+[("flags",ctypes.c_int64),("n_children",ctypes.c_int64)]+[(name,ctypes.c_void_p) for name in ("children","dictionary","release","private_data")]

def library():
    global _LIB
    with _LOCK:
        if _LIB is None:
            source=Path(__file__).with_name("native")/"producer.c"
            tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            folder=Path(tempfile.gettempdir())/("columnspan-native-"+tag)
            folder.mkdir(exist_ok=True)
            dest=folder/("producer.dylib" if sys.platform=="darwin" else "producer.so")
            if not dest.exists():
                temporary=folder/("producer-"+str(__import__('os').getpid())+".tmp")
                subprocess.run(["cc","-shared","-fPIC","-O2",str(source),"-o",str(temporary)],check=True,capture_output=True)
                temporary.replace(dest)
            _LIB=ctypes.CDLL(str(dest))
            _LIB.make_array.argtypes=[ctypes.POINTER(ArrowArray),ctypes.POINTER(ArrowSchema),ctypes.c_int64]
            _LIB.make_array.restype=ctypes.c_int
            _LIB.release_count.restype=ctypes.c_int
        return _LIB

def import_native(length=4):
    if not isinstance(length,int) or not 0<=length<=100000: raise ValueError("invalid array length")
    lib=library();a=ArrowArray();s=ArrowSchema()
    if lib.make_array(ctypes.byref(a),ctypes.byref(s),length): raise MemoryError("native allocation failed")
    try: return pa.Array._import_from_c(ctypes.addressof(a),ctypes.addressof(s))
    except BaseException:
        if a.release: ctypes.CFUNCTYPE(None,ctypes.POINTER(ArrowArray))(a.release)(ctypes.byref(a))
        raise

def _inspect_exchange(length=4):
    # Count delta is meaningful for this serialized demonstration only.
    lib=library();before=lib.release_count()
    a=import_native(length);values=a.to_pylist();view=a.slice(1);del a;gc.collect()
    retained=view.to_pylist();during=lib.release_count()-before
    del view;gc.collect()
    return {"values":values,"view":retained,"released_while_view_alive":during,"released_after":lib.release_count()-before}

def inspect_exchange(length=4):
    with _EXCHANGE_LOCK:
        return _inspect_exchange(length)
