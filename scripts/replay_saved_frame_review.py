"""Replay captured frame-review outputs locally. No model, network or new questions."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'api'))
from app.models import Exhibition
from app.generator import ExhibitionGenerator
from app.curatorial_copy_review import copy_review_payload
from app.curation import apply_frame


async def replay(file):
    raw=json.loads(file.read_text(encoding='utf8'))
    exhibition=Exhibition.model_validate(raw['exhibition'])
    frame=deepcopy(next(row['output'] for row in raw['modelStages'] if row['stage']=='frame'))
    for chapter in frame['chapters']: chapter.pop('leadIn',None)
    structure=[{'index':i,'itemCount':len(ch.item_ids),
                'objectIds':[item.object.id for item in exhibition.items if item.id in ch.item_ids]}
               for i,ch in enumerate(exhibition.chapters)]
    payload=copy_review_payload(frame,[item.object for item in exhibition.items],
        question=raw['question'],chapter_structure=structure)
    outputs={row['stage']:row.get('output') for row in raw['modelStages'] if row['stage'].startswith('frame_review')}
    generator=ExhibitionGenerator.__new__(ExhibitionGenerator)
    async def recorded(prompt,payload,*,stage,**kwargs):
        if stage not in outputs: raise RuntimeError('missing_recorded_stage')
        return deepcopy(outputs[stage])
    generator._generate_model_json=recorded
    result=await generator._review_frame_copy(frame,payload,language='zh',deadline=perf_counter()+40)
    out={'case':file.parent.name,'version':file.parent.parent.name,'reviewPassed':result.review_passed,
         'status':result.status,'errors':result.errors,'localFields':result.locally_neutralized_fields}
    if result.review_passed:
        try: apply_frame(exhibition,result.frame); out['applyPassed']=True
        except (ValueError,TypeError,KeyError) as error: out['applyPassed']=False; out['applyError']=str(error)[:200]
    return out


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    args=parser.parse_args()
    for file in args.inputs:
        print(json.dumps(await replay(file),ensure_ascii=False),flush=True)


if __name__=='__main__': asyncio.run(main())
