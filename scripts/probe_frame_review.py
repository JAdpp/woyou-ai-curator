"""Review an existing saved frame only; execute explicitly in a new output directory."""
import argparse, asyncio, json, logging, sys
from pathlib import Path
from copy import deepcopy
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'api'))
from app.models import Exhibition
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.providers.deepseek import DeepSeekProvider
from app.curatorial_copy_review import copy_review_payload
from app.curation import apply_frame

async def run(args):
    logging.disable(logging.CRITICAL)
    args.output.mkdir(parents=True,exist_ok=False)
    settings=Settings.from_env()
    generator=ExhibitionGenerator(settings,None,DeepSeekProvider(settings))
    results=[]
    for file in args.inputs:
        raw=json.loads(file.read_text(encoding='utf-8'))
        exhibit=Exhibition.model_validate(raw['exhibition'])
        frame=deepcopy(next(row['output'] for row in raw['modelStages'] if row['stage']=='frame'))
        for chapter in frame['chapters']: chapter.pop('leadIn',None)
        structure=[{'index':i,'itemCount':len(ch.item_ids),
                    'objectIds':[item.object.id for item in exhibit.items if item.id in ch.item_ids]}
                   for i,ch in enumerate(exhibit.chapters)]
        payload=copy_review_payload(frame,[item.object for item in exhibit.items],question=raw['question'],chapter_structure=structure)
        stages=[]
        original=generator._generate_model_json
        async def observed(prompt,payload,**kwargs):
            start=perf_counter()
            output=await original(prompt,payload,**kwargs)
            stages.append({'stage':kwargs['stage'],'elapsedSeconds':perf_counter()-start,'output':output})
            return output
        generator._generate_model_json=observed
        start=perf_counter()
        row={'case':file.parent.name,'question':raw['question']}
        try:
            reviewed=await generator._review_frame_copy(frame,payload,language='zh',deadline=start+40)
            row.update(reviewed.to_diagnostics())
            row['reviewedFrame']=reviewed.frame
            if reviewed.review_passed:
                try:
                    apply_frame(exhibit,reviewed.frame)
                    row['applyPassed']=True
                except (ValueError,TypeError,KeyError) as error:
                    row['applyPassed']=False
                    row['applyError']=str(error)[:160]
        except Exception as error:
            row['errorType']=type(error).__name__
        finally:
            generator._generate_model_json=original
        row['elapsedSeconds']=perf_counter()-start
        row['modelStages']=stages
        results.append(row)
        (args.output/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({key:row.get(key) for key in ('case','status','reviewPassed','applyPassed','errors','errorType','elapsedSeconds')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute: asyncio.run(run(args))
    else: print(json.dumps({'cases':len(args.inputs),'paidCalls':0,'execute':False}))
