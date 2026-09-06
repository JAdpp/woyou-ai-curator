"""One explicit public visitor smoke, with persistent receipt and no POST retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]
BASE='http://47.89.246.208:8081'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'artifacts/qa/production-rc11/public-generation.json')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'execute':False,'paidRequests':0})); return 0
    marker=args.report.with_suffix('.once.json')
    if marker.exists() or args.report.exists(): raise RuntimeError('already_started_do_not_repost')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with marker.open('x') as handle: json.dump({'startedAt':datetime.now(timezone.utc).isoformat()},handle)
    old=json.loads((ROOT/'artifacts/qa/server-candidates/inquiry-v6-20260906-rc11/release-03-market-day/result.json').read_text(encoding='utf8'))
    result={'base':BASE,'request':old['request'],'newQuestion':False,'startedAt':datetime.now(timezone.utc).isoformat()}
    started=time.monotonic()
    def save(): args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    try:
        with httpx.Client(base_url=BASE,timeout=120,trust_env=False) as client:
            response=client.post('/api/exhibitions/generate',json=old['request'])
            result['createHttpStatus']=response.status_code
            response.raise_for_status()
            job=response.json(); result['jobId']=job['id']; save()
            while time.monotonic()-started<240:
                response=client.get('/api/jobs/'+job['id']); response.raise_for_status()
                job=response.json()
                if job['status'] in ('completed','failed'): break
                time.sleep(2)
            result['jobStatus']=job['status']; result['generationSeconds']=time.monotonic()-started
            if job['status']!='completed':
                result['error']=job.get('error'); result['passed']=False; save(); return 1
            eid=job['exhibitionId']; result['exhibitionId']=eid; result['exhibitionUrl']=BASE+'/exhibitions/'+eid
            response=client.get('/api/exhibitions/'+eid); response.raise_for_status(); exhibition=response.json()
            until=time.monotonic()+60
            while (exhibition.get('poster') or {}).get('status') not in ('ready','failed') and time.monotonic()<until:
                time.sleep(2)
                response=client.get('/api/exhibitions/'+eid); response.raise_for_status(); exhibition=response.json()
            result.update(title=exhibition['title'],itemCount=len(exhibition['items']),
                frameProvider=exhibition['versions']['provider'],promptVersion=exhibition['versions']['prompt'],
                validatorPassed=(exhibition.get('validation') or {}).get('passed'),
                coverageLimits=exhibition.get('coverageLimits',[]))
            poster=exhibition.get('poster') or {}
            result['posterStatus']=poster.get('status')
            url=poster.get('backgroundUrl','')
            if url.startswith('/generated/posters/'):
                response=client.get(url)
                result['poster']={'status':response.status_code,'bytes':len(response.content),'contentType':response.headers.get('content-type')}
            response=client.get('/api/exhibitions/'+eid+'/audio-guide',params={'kind':'lobby'})
            result['audio']={'status':response.status_code,'bytes':len(response.content),
                'contentType':response.headers.get('content-type'),'mp3Header':response.content[:3]==b'ID3'
                or (len(response.content)>1 and response.content[0]==255 and response.content[1]&224==224)}
            result['passed']=(result['validatorPassed'] is True and result['frameProvider']=='deepseek'
                and result.get('poster',{}).get('status')==200 and result['audio']['status']==200 and result['audio']['mp3Header'])
    except Exception as error:
        result['errorType']=type(error).__name__; result['passed']=False
    finally:
        result['finishedAt']=datetime.now(timezone.utc).isoformat(); save()
        print(json.dumps(result,ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
