from . import pages,ownership,views,bitmap,layout,cooccurrence

def page_call(d):
    if d.get('action')=='read':return {'values':pages.read_column(bytes.fromhex(d['hex']))}
    data=pages.write_column(d['values']);return {'hex':data.hex(),'values':pages.read_column(data)}
def bitmap_call(d):
    a,b=bitmap.Bitmap(d['left']),bitmap.Bitmap(d.get('right',[]));out=a.operate(b,d.get('op','union'))
    return {'values':list(out.values()),'rank':out.rank(d.get('rank_at',65536))}
_COOCCURRENCE_STUDIES={}
def cooccurrence_call(d):
    """Stateless-fronted, stateful-backed revisable co-occurrence operation."""
    action=d.get('action','create')
    if action=='create':
        study=cooccurrence.LabelStudy(
            universe=d['universe'],labels=d['labels'],
            method=d.get('method','holm'),alpha=d.get('alpha',0.05),
            min_joint_observations=d.get('min_joint_observations',2))
        study_id=str(len(_COOCCURRENCE_STUDIES))
        _COOCCURRENCE_STUDIES[study_id]=study
        return {'study_id':study_id,
                'identity':study.identity_token(),
                'family':study.family,
                'version':study.result(0)}
    if action=='restore':
        restored=cooccurrence.LabelStudy.from_dict(d['study'])
        new_id=str(len(_COOCCURRENCE_STUDIES))
        _COOCCURRENCE_STUDIES[new_id]=restored
        return {'study_id':new_id,'identity':restored.identity_token()}
    study_id=d.get('study_id')
    if study_id not in _COOCCURRENCE_STUDIES:
        raise cooccurrence.CooccurrenceError('unknown study_id')
    study=_COOCCURRENCE_STUDIES[study_id]
    if action=='scope':
        return study.affected_scope(d['changes'])
    if action=='correct':
        version=study.apply_corrections(d['batch'],
                                        identity_token=d.get('identity_token'))
        return {'version':study.result(version)}
    if action=='result':
        return {'version':study.result(d.get('version'))}
    if action=='verify':
        return study.verify_version(d.get('version'))
    if action=='diff':
        return study.diff_versions(d['base_version'],d.get('new_version'))
    if action=='export':
        return study.to_dict()
    raise cooccurrence.CooccurrenceError('unknown cooccurrence action')
OPERATIONS={
 'pages':page_call,
 'ownership':lambda d:ownership.inspect_exchange(d.get('length',4)),
 'views':lambda d:views.View(tuple(d['shape']),tuple(d['strides']),d.get('offset',0)).analyze(d['capacity']),
 'bitmap':bitmap_call,
 'layout':lambda d:layout.compile_layout(d['fields'],d.get('byteorder','little')),
 'cooccurrence':cooccurrence_call,
}
EXAMPLES={
 'pages':{'values':[1,None,-7,4]},'ownership':{'length':4},
 'views':{'shape':[2,3],'strides':[3,-1],'offset':2,'capacity':6},
 'bitmap':{'left':[1,3,65536],'right':[3,7],'op':'union'},
 'layout':{'fields':[{'name':'flag','type':'u8'},{'name':'value','type':'u32'}]},
 'cooccurrence':{'action':'create','universe':[0,1,2,3,4,5],
                 'labels':[{'name':'A','eligible':[0,1,2,3,4],'positive':[0,1,2]},
                           {'name':'B','eligible':[0,1,2,3,4],'positive':[0,1,3]},
                           {'name':'C','eligible':[0,1,2,3,4],'positive':[2,3,4]}],
                 'method':'holm','alpha':0.05},
}
