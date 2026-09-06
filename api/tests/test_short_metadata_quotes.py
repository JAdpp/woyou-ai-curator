import pytest
from app.retrieval_agent import _quote_is_visible, parse_audit, CONDITION_CONTRACT_VERSION
from app.collections import SearchResult
from app.models import MuseumObject, EvidenceChunk

@pytest.mark.parametrize('quote,source,metadata,expected',[
    ('Japan','Japan；Folding Fan；Fans',True,True),
    ('Japan','Japan；Folding Fan；Fans',False,False),
    ('Japan','Japanese；Folding Fan；Fans',True,False),
    ('gold','silver; gold; textile',True,True),
    ('gold','gold coloured decoration',True,False),
    ('war','The scene depicts war and peace.',False,False),
    ('China','China | Vessel',True,True),
    ('Fan','Fan',False,True),
])
def test_only_exact_complete_metadata_segments_allow_short_quotes(quote,source,metadata,expected):
    assert _quote_is_visible(quote,source,metadata=metadata) is expected

@pytest.mark.parametrize('kind,expected',[('institution_metadata',True),('institution_curatorial_text',False)])
def test_production_condition_parser_uses_source_kind_not_model_claim(kind,expected):
    obj=MuseumObject(id='x',title='Fan',rights='CC0',imageUrl='https://example.org/i',objectUrl='https://example.org/o',
        evidence=[EvidenceChunk(id='x:classification',text='Japan；Folding Fan；Fans',sourceKind=kind,
                               sourceUrl='https://example.org/o',sourceTitle='Record')])
    result=SearchResult(obj=obj,score=1)
    output={'conditionContractVersion':CONDITION_CONTRACT_VERSION,'answerability':'supported','accepted':[
        {'objectId':'x','evidenceIds':['x:classification'],'relevanceScore':1,'conditionEvidence':[
            {'conditionId':'p1','status':'supported','evidenceId':'x:classification','supportingQuote':'Japan','relation':'exact'}]}],
        'searchQueries':[],'expansionReason':'none'}
    audit=parse_audit(output,[result],question='Japan objects',max_expansion_queries=3,strict_conditions=True,
                      mandatory_predicates=('The object is from Japan',))
    assert bool(audit.accepted) is expected
