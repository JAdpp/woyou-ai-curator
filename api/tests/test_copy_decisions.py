from copy import deepcopy
import asyncio
from time import perf_counter
import pytest
from app.curatorial_copy_review import copy_decisions_to_patches, copy_review_payload, parse_copy_review
from app.models import MuseumObject, EvidenceChunk

def objects():
    return [MuseumObject(id='a', title='Harbor', rights='CC0',
        imageUrl='https://example.org/image', objectUrl='https://example.org/object',
        evidence=[EvidenceChunk(id='a:description',text='This scene is in Harbor A.',
            sourceUrl='https://example.org/object',sourceTitle='Record',sourceKind='institution_description')])]

def setup():
    frame = {'title':'Two places', 'subtitle':'Harbor C'}
    payload = copy_review_payload(frame, objects(), question='Compare places')
    return frame, payload

def test_keep_is_not_a_noop_patch():
    frame, payload = setup()
    output = {'schemaVersion':'curatorial-copy-decisions-v2', 'decisions':[
        {'path':row['path'],'action':'keep'} for row in payload['publicCopyFields']]}
    compiled = copy_decisions_to_patches(output, payload)
    assert compiled['changes'] == []
    assert parse_copy_review(compiled,frame,payload).review_passed

@pytest.mark.parametrize('fault',['missing','duplicate','unknown','extra','bad_action','noop','unbound'])
def test_invalid_decisions_still_fail_canonical_checks(fault):
    frame, payload = setup()
    rows = [{'path': '/title','action':'keep'}, {'path':'/subtitle','action':'replace',
             'replacement':'Harbor A','changeKind':'correct_fact','evidenceIds':['a:description'],
             'supportingQuotes':[{'evidenceId':'a:description','quote':'Harbor A'}], 'reason':'Fix place'}]
    output={'schemaVersion':'curatorial-copy-decisions-v2','decisions':rows}
    if fault=='missing': rows.pop()
    if fault=='duplicate': rows.append(deepcopy(rows[0]))
    if fault=='unknown': rows[0]['path']='/secret'
    if fault=='extra': rows[0]['replacement']='Unused'
    if fault=='bad_action': rows[0]['action']=[]
    if fault=='noop': rows[1]['replacement']='Harbor C'
    if fault=='unbound': rows[1]['supportingQuotes'][0]['quote']='Made up'
    result=parse_copy_review(copy_decisions_to_patches(output,payload),frame,payload)
    assert not result.review_passed
    assert result.frame == frame

def test_real_change_retains_exact_original_and_source_binding():
    frame,payload=setup()
    output={'schemaVersion':'curatorial-copy-decisions-v2','decisions':[
        {'path':'/title','action':'keep'}, {'path':'/subtitle','action':'replace',
         'replacement':'Harbor A','changeKind':'correct_fact','evidenceIds':['a:description'],
         'supportingQuotes':[{'evidenceId':'a:description','quote':'Harbor A'}],'reason':'Correct place'}]}
    result=parse_copy_review(copy_decisions_to_patches(output,payload),frame,payload)
    assert result.review_passed and result.frame['subtitle']=='Harbor A'
    assert result.changes[0]['original']=='Harbor C'

def test_unresolved_cannot_be_turned_into_pass():
    frame,payload=setup()
    output={'schemaVersion':'curatorial-copy-decisions-v2','decisions':[
        {'path':row['path'],'action':'unresolved'} for row in payload['publicCopyFields']]}
    assert not parse_copy_review(copy_decisions_to_patches(output,payload),frame,payload).review_passed


@pytest.mark.parametrize('repair_fault', [None, 'noop', 'schema', 'quote'])
def test_v2_repair_receives_raw_v2_not_internal_v1(monkeypatch, repair_fault):
    from app.generator import ExhibitionGenerator
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    frame, payload = setup()
    raw = {'schemaVersion':'curatorial-copy-decisions-v2', 'decisions':[
        {'path':'/title','action':'keep'}, {'path':'/subtitle','action':'replace',
         'replacement':'Harbor C','changeKind':'correct_fact','evidenceIds':['a:description'],
         'supportingQuotes':[{'evidenceId':'a:description','quote':'Harbor A'}],
         'reason':'Correct the unsupported place'}]}
    calls = []

    async def model(prompt, batch, *, stage, timeout_seconds):
        calls.append(stage)
        assert batch['schemaVersion'] == 'curatorial-copy-decisions-v2'
        if 'repair' not in stage:
            return deepcopy(raw)
        assert batch['previousReview'] == raw
        assert batch['contractErrors'] == ['invalid_noop_change']
        assert 'actually remove/correct' in prompt
        fixed = deepcopy(raw)
        fixed['decisions'][1]['replacement'] = 'Harbor A'
        if repair_fault == 'noop': fixed['decisions'][1]['replacement'] = 'Harbor C'
        if repair_fault == 'schema': fixed['schemaVersion'] = ' '
        if repair_fault == 'quote': fixed['decisions'][1]['supportingQuotes'][0]['quote'] = 'invented'
        return fixed

    monkeypatch.setattr(generator, '_generate_model_json', model)
    result = asyncio.run(generator._review_frame_copy(frame, payload, language='zh', deadline=perf_counter()+5))
    assert calls == ['frame_review:batch0', 'frame_review_repair:batch0']
    assert result.review_passed == (repair_fault is None)
    assert result.frame['subtitle'] == ('Harbor A' if repair_fault is None else 'Harbor C')
    assert payload['schemaVersion'] != 'curatorial-copy-decisions-v2'
