from copy import deepcopy
import asyncio
from time import perf_counter
import pytest
from app.curatorial_copy_review import (
    COPY_REVIEW_VERSION, copy_review_payload, copy_decisions_to_patches,
    parse_copy_review_batch, recover_auxiliary_copy_batch,
)
from app.generator import ExhibitionGenerator
from .test_copy_decisions import objects


def fixture(path='/epilogue/text'):
    frame = {'title':'Keep this exhibition', 'epilogue':{'text':'Unsupported claim'},
             'curatorialBrief':{'keyMessages':[{'text':'Unsupported claim',
                 'evidenceIds':['a:description'],'confidence':'supported'}]}}
    payload=copy_review_payload(frame,objects(),question='Look at places')
    payload['publicCopyFields']=[row for row in payload['publicCopyFields']
                               if row['path'] in ('/title',path)]
    payload['expectedFieldCount']=len(payload['publicCopyFields'])
    raw={'schemaVersion':'curatorial-copy-decisions-v2','decisions':[
        {'path':row['path'],'action':'keep'} if row['path'] != path else
        {'path':path,'action':'replace','replacement':row['original'],
         'changeKind':'neutralize','evidenceIds':[],'supportingQuotes':[],
         'reason':'Remove unsupported claim'} for row in payload['publicCopyFields']]}
    return frame,payload,copy_decisions_to_patches(raw,payload)


@pytest.mark.parametrize('path',['/epilogue/text','/curatorialBrief/keyMessages/0/text'])
def test_failed_auxiliary_field_does_not_discard_verified_title(path):
    frame,payload,initial=fixture(path)
    original=deepcopy(frame)
    result=recover_auxiliary_copy_batch({'schemaVersion':' '},initial,frame,payload)
    assert result is not None
    output,paths=result
    checked=parse_copy_review_batch(output,frame,payload)
    assert checked.review_passed and checked.frame['title']==frame['title']
    assert paths==(path,) and len(checked.changes)==1
    assert '尚未完成来源核对' in checked.changes[0]['replacement']
    assert checked.changes[0]['evidenceIds']==[] and frame==original


@pytest.mark.parametrize('fault',['title','unknown','duplicate','unresolved','missing_count','both_bad_schema'])
def test_recovery_cannot_bypass_core_or_envelope_contract(fault):
    frame,payload,initial=fixture()
    final={'schemaVersion':' '}
    if fault=='title':
        initial['changes'][0].update(path='/title',original=frame['title'],replacement=frame['title'])
    if fault=='unknown': initial['changes'][0]['path']='/unknown'
    if fault=='duplicate': initial['changes']*=2
    if fault=='unresolved': final={**initial,'outcome':'unresolved'}
    if fault=='missing_count': initial.pop('reviewedFieldCount')
    if fault=='both_bad_schema': initial['schemaVersion']=' '
    assert recover_auxiliary_copy_batch(final,initial,frame,payload) is None


def test_generator_marks_local_message_uncertain_without_mutating_original(monkeypatch):
    frame,payload,initial=fixture('/curatorialBrief/keyMessages/0/text')
    frame.pop('epilogue')
    payload=copy_review_payload(frame,objects(),question='Look at places')
    initial['reviewedFieldCount']=len(payload['publicCopyFields'])
    generator=ExhibitionGenerator.__new__(ExhibitionGenerator)
    async def model(prompt,batch,**kw):
        return {'schemaVersion':' '} if 'repair' in kw['stage'] else deepcopy(initial)
    monkeypatch.setattr(generator,'_generate_model_json',model)
    result=asyncio.run(generator._review_frame_copy(frame,payload,language='zh',deadline=perf_counter()+5))
    assert result.review_passed and result.status=='bounded_local_fallback'
    assert result.locally_neutralized_fields==1
    assert result.frame['curatorialBrief']['keyMessages'][0]['confidence']=='uncertain'
    assert frame['curatorialBrief']['keyMessages'][0]['confidence']=='supported'


def test_malformed_optional_only_batch_is_discarded_not_accepted():
    frame,payload,_=fixture('/curatorialBrief/keyMessages/0/text')
    payload['publicCopyFields']=[row for row in payload['publicCopyFields'] if row['path']!='/title']
    payload['expectedFieldCount']=1
    output,paths=recover_auxiliary_copy_batch({}, {},frame,payload)
    checked=parse_copy_review_batch(output,frame,payload)
    assert checked.review_passed and len(paths)==1
    assert checked.frame['title']==frame['title']
    assert checked.frame['curatorialBrief']['keyMessages'][0]['text']!='Unsupported claim'
