"""Inspect or explicitly deploy the verified Inquiry Curator release. Never prints secrets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
PROD = Path('/opt/demos/inquiry-curator')
CANDIDATE = Path('/opt/demos/inquiry-curator-candidates/inquiry-v6-20260906-rc11')
MANIFEST = '3a7f53422a16b4ef68dd66190e7f2fc5fdd3b28a12faf5fcbe07fb92808c5449'
SERVICES = ('inquiry-api.service', 'inquiry-web.service')
UPDATES = {
    'RAG_MODE':'hybrid','DEFAULT_COLLECTION_ID':'global_open',
    'RAG_EMBEDDING_PROVIDER':'aliyun','RAG_EMBEDDING_MODEL':'qwen3.7-text-embedding',
    'RAG_EMBEDDING_DIMENSION':'768','RAG_RERANK_MODEL':'qwen3-rerank',
    'RAG_RERANK_ENABLED':'true','RAG_LLM_AUDIT_ENABLED':'true','RAG_STRUCTURED_FILTERS_ENABLED':'true',
    'RAG_INDEX_DIR':'api/runtime/cache/rag','RAG_FILTER_INDEX_DIR':'api/runtime/cache/filters',
    'RAG_TRACE_DIR':'api/runtime/traces/retrieval',
    'ALIYUN_TEXT_API_HOST':'https://llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com',
    'DEEPSEEK_LABELS_MODEL':'deepseek-v4-flash-vision-exp','DEEPSEEK_QUERY_REVIEW_THINKING':'false',
    'RAG_PLANNING_TIMEOUT_SECONDS':'16','RAG_RETRIEVAL_TIMEOUT_SECONDS':'70',
    'DEEPSEEK_TIMEOUT_SECONDS':'90','DEEPSEEK_FRAME_TIMEOUT_SECONDS':'55',
    'DEEPSEEK_LABELS_TIMEOUT_SECONDS':'40','GENERATION_JOB_TIMEOUT_SECONDS':'180',
    'GENERATION_POSTER_WAIT_SECONDS':'2',
}
SAFE_ENV = ('APP_ENV','STORE_MODE','STORE_PATH','COLLECTIONS_DIR','DEFAULT_COLLECTION_ID',
    'IMAGE_CACHE_DIR','ALIYUN_IMAGE_OUTPUT_DIR','ALIYUN_TTS_OUTPUT_DIR','RAG_MODE','RAG_INDEX_DIR',
    'RAG_FILTER_INDEX_DIR','RAG_TRACE_DIR','RAG_EMBEDDING_PROVIDER','RAG_EMBEDDING_MODEL',
    'RAG_EMBEDDING_DIMENSION','RAG_RERANK_MODEL','ALIYUN_TEXT_API_HOST','DEEPSEEK_MODEL',
    'DEEPSEEK_LABELS_MODEL','DEEPSEEK_QUERY_REVIEW_THINKING','ALIYUN_IMAGE_MODEL','ALIYUN_TTS_MODEL',
    'RAG_RETRIEVAL_TIMEOUT_SECONDS','RAG_PLANNING_TIMEOUT_SECONDS','DEEPSEEK_FRAME_TIMEOUT_SECONDS',
    'DEEPSEEK_LABELS_TIMEOUT_SECONDS','GENERATION_JOB_TIMEOUT_SECONDS')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def command(args, timeout=30):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,text=True)
    return result.returncode,result.stdout

def get(url,limit=4*1024*1024):
    with build_opener(ProxyHandler({})).open(url,timeout=15) as response:
        return response.status,response.read(limit)

def inspect():
    from dotenv import dotenv_values
    env=dotenv_values(PROD/'.env')
    result={'mode':'inspect','at':datetime.now(timezone.utc).isoformat(),'rootIsSymlink':PROD.is_symlink(),
            'environment':{key:env.get(key) for key in SAFE_ENV if key in env},
            'environmentSha256':sha(PROD/'.env'),'units':{},'paths':{}}
    for service in SERVICES:
        _,unit=command(['systemctl','cat',service])
        result['units'][service]=[line for line in unit.splitlines() if
            line.startswith(('WorkingDirectory=','ExecStart=','User=','Group=','EnvironmentFile=','FragmentPath='))
            and not re.search(r'password|api.?key|token=',line,re.I)]
    result['services']=command(['systemctl','show',*SERVICES,'-p','ActiveState','-p','SubState','-p','MainPID'])[1]
    result['otherServices']=command(['systemctl','list-units','--type=service','--no-legend','--plain'])[1]
    result['otherServices']='\n'.join(line for line in result['otherServices'].splitlines() if
        any(word in line for word in ('myth','trip','streamlit')))
    result['listening']=command(['ss','-tlnp'])[1]
    result['listening']='\n'.join(line for line in result['listening'].splitlines() if
        re.search(r':(?:808[123]|3000|900[123])\b',line))
    result['memory']=command(['free','-m'])[1]
    result['disk']=command(['df','-h','/opt/demos'])[1]
    for rel in ('.venv','api','api/runtime','data','data/collections','.next','public',
                'public/generated','public/generated/posters','.next/standalone/public/generated',
                '.next/standalone/public/generated/posters'):
        path=PROD/rel
        result['paths'][rel]={'exists':path.exists(),'symlink':path.is_symlink(),
            'resolved':str(path.resolve()),'entries':sorted(p.name for p in path.iterdir())[:30] if path.is_dir() else []}
    store=Path(env.get('STORE_PATH') or 'api/runtime/store.json')
    if not store.is_absolute(): store=PROD/store
    result['store']={'path':str(store),'exists':store.is_file(),'sha256':sha(store) if store.is_file() else None}
    result['nginx']={str(path):sha(path) for path in
        (Path('/etc/nginx/sites-available/demo-inquiry.conf'),Path('/etc/nginx/conf.d/demo-ratelimit.conf')) if path.is_file()}
    result['unitEnvironment']={}
    for service in SERVICES:
        _,body=command(['systemctl','show',service,'-p','Environment','--value'])
        result['unitEnvironment'][service]={pair.split('=',1)[0]:pair.split('=',1)[1]
            for pair in shlex.split(body) if '=' in pair and pair.split('=',1)[0] in (*SAFE_ENV,'PORT','HOSTNAME','API_INTERNAL_URL')}
    try:
        status,body=get('http://127.0.0.1:9001/health')
        health=json.loads(body)
        result['health']={'status':status,'retrieval':health.get('retrieval'),'configured':{
            key:value for key,value in health.items() if key.endswith('Configured')}}
    except Exception as error: result['health']={'errorType':type(error).__name__}
    return result

def checked(args, timeout=60):
    code,body=command(args,timeout)
    if code: raise RuntimeError('command_failed_'+Path(args[0]).name)
    return body

def verify_live():
    result={'mode':'verify','at':datetime.now(timezone.utc).isoformat(),'checks':[]}
    for service in SERVICES:
        state=checked(['systemctl','is-active',service]).strip()
        result['checks'].append({'id':service,'passed':state=='active'})
    deadline=time.monotonic()+60
    while True:
        try:
            status,body=get('http://127.0.0.1:9001/health')
            health=json.loads(body)
            if status==200: break
        except Exception:
            if time.monotonic()>=deadline: raise RuntimeError('api_health_timeout')
            time.sleep(1)
    retrieval=health.get('retrieval',{})
    result['retrieval']=retrieval
    result['checks'].append({'id':'hybrid_health','passed':retrieval.get('mode')=='hybrid'
        and retrieval.get('available') is True and retrieval.get('fingerprint')==
        '9174e55a68665ff83a5c94c7f722f7d5f2765081f5b4cba7dc5fe81f5d4ad978'})
    status,html=get('http://127.0.0.1:8081/')
    result['checks'].append({'id':'nginx_home','passed':status==200,'httpStatus':status})
    assets=list(dict.fromkeys(re.findall(r'(?:src|href)="([^" ]+\.(?:js|css)(?:\?[^" ]*)?)"',html.decode('utf8'))))
    if not assets: raise RuntimeError('no_static_assets')
    for asset in assets:
        if not asset.startswith('/_next/static/'): continue
        status,body=get('http://127.0.0.1:8081'+asset)
        result['checks'].append({'id':'static','path':asset,'passed':status==200 and len(body)>0,'bytes':len(body)})
    from urllib.error import HTTPError
    for route in ('/api/admin','/dev/admin/qrels'):
        try: status,_=get('http://127.0.0.1:8081'+route)
        except HTTPError as error: status=error.code
        result['checks'].append({'id':'admin_closed','path':route,'passed':status==404})
    result['passed']=all(row['passed'] for row in result['checks'])
    return result

def deploy(commit):
    """Preserve shared state, swap three runtime trees, rollback on failed checks."""
    from dotenv import dotenv_values
    if not re.fullmatch(r'[0-9a-f]{40}',commit): raise ValueError('exact_commit_required')
    if PROD.resolve()!=PROD or CANDIDATE.resolve()!=CANDIDATE: raise ValueError('unexpected_root_link')
    manifest=CANDIDATE/'release-manifest.json'
    if sha(manifest)!=MANIFEST: raise ValueError('candidate_manifest_changed')
    rows=json.loads(manifest.read_text())['files']
    for row in rows:
        path=CANDIDATE/row['path']
        if not path.resolve().is_relative_to(CANDIDATE) or not path.is_file() or sha(path)!=row['sha256']:
            raise ValueError('candidate_payload_changed')
    for case in ('release-03-market-day','release-02-fans','default-browse'):
        receipt=json.loads((CANDIDATE/'candidate-state/visitor-smokes'/f'{case}.receipt.json').read_text())
        if not (receipt['manifestSha256']==MANIFEST and receipt['processFinalized']
                and receipt['summary']['outcome']=='completed'
                and not receipt['summary']['publicFrameDeterministicFallback']):
            raise ValueError('candidate_visitor_gate_not_passed')
    label='rc11-'+commit[:12]
    stage=PROD/('.deploy-'+label)
    backup=PROD/('.rollback-'+label)
    receipt_path=PROD/('deployment-'+label+'.json')
    if any(path.exists() for path in (stage,backup,receipt_path)): raise ValueError('release_already_started')
    if shutil.disk_usage(PROD).free<900*1024*1024: raise ValueError('insufficient_staging_space')
    initial=inspect()
    # Only runtime trees change. User store, Python environment and media stay.
    groups=('api/app','.next/standalone','data/collections/global_open')
    for rel in groups:
        if (PROD/rel).is_symlink(): raise ValueError('unexpected_live_tree_link')
    stage.mkdir(mode=0o700); backup.mkdir(mode=0o700)
    for rel in groups:
        shutil.copytree(CANDIDATE/rel,stage/rel,symlinks=True)
    # Completed immutable index directories can coexist with the old indexes.
    for family in ('rag','filters'):
        index_root=CANDIDATE/'api/runtime/cache'/family
        sources=[path for path in index_root.rglob('*') if path.is_dir() and re.fullmatch(r'[0-9a-f]{64}',path.name)]
        if len(sources)!=1: raise ValueError('unexpected_index_count')
        for source in sources:
            target=PROD/'api/runtime/cache'/family/source.relative_to(index_root)
            if target.exists():
                for path in source.rglob('*'):
                    if path.is_file() and (not (target/path.relative_to(source)).is_file()
                        or sha(path)!=sha(target/path.relative_to(source))): raise ValueError('existing_index_differs')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                pending=target.with_name(target.name+'.pending-'+label)
                shutil.copytree(source,pending)
                pending.rename(target)
    # Preserve every original line except the explicitly approved non-secret
    # retrieval/budget settings; keys/authentication/media paths remain intact.
    original_env=(PROD/'.env').read_text()
    env_lines=[line for line in original_env.splitlines() if
        line.split('=',1)[0].strip() not in UPDATES]
    env_lines+=['', '# RC11 release '+commit]+[key+'='+value for key,value in UPDATES.items()]
    (stage/'.env').write_text('\n'.join(env_lines)+'\n'); (stage/'.env').chmod(0o600)
    before_env=dotenv_values(PROD/'.env'); next_env=dotenv_values(stage/'.env')
    if any(before_env.get(key)!=next_env.get(key) for key in before_env if key not in UPDATES):
        raise ValueError('unrelated_environment_changed')
    # Paths are exact and remain in this project's runtime, including rollback.
    shared=PROD/'api/runtime/media/posters'
    old_posters=PROD/'.next/standalone/public/generated/posters'
    if shared.exists() or not old_posters.is_dir() or old_posters.is_symlink():
        raise ValueError('unexpected_poster_storage')
    staged_posters=stage/'.next/standalone/public/generated/posters'
    if staged_posters.exists(): raise ValueError('candidate_contains_generated_media')
    staged_posters.parent.mkdir(parents=True,exist_ok=True)
    staged_posters.symlink_to(shared,target_is_directory=True)
    code,_=command(['/usr/bin/node','-e',f"require({json.dumps(str(stage/'.next/standalone/node_modules/sharp'))});"],30)
    if code: raise RuntimeError('staged_native_failed')
    changed=[]; stopped=False; poster_moved=False
    def interrupted(_number,_frame):
        raise InterruptedError('release_interrupted')
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(number,interrupted)
    result={'release':label,'commit':commit,'manifestSha256':MANIFEST,'startedAt':initial['at'],
            'mode':'deploy','backup':str(backup),'passed':False,'rolledBack':False,
            'environmentUpdates':UPDATES,'productionStorePreserved':False}
    try:
        checked(['systemctl','stop',*SERVICES]); stopped=True
        shutil.copy2(PROD/'.env',backup/'.env'); (backup/'.env').chmod(0o600)
        shutil.copy2(PROD/'api/runtime/store.json',backup/'store.snapshot.json')
        shared.parent.mkdir(parents=True,exist_ok=True)
        old_posters.rename(shared)
        poster_moved=True
        old_posters.symlink_to(shared,target_is_directory=True)
        for rel in groups:
            old=PROD/rel; saved=backup/rel; new=stage/rel
            saved.parent.mkdir(parents=True,exist_ok=True)
            old.rename(saved)
            changed.append(rel)
            new.rename(old)
        os.replace(stage/'.env',PROD/'.env')
        checked(['systemctl','start',*SERVICES]); stopped=False
        validation=verify_live()
        if not validation['passed']: raise RuntimeError('live_check_failed')
        result['validation']=validation
        result['productionStorePreserved']=sha(PROD/'api/runtime/store.json')==sha(backup/'store.snapshot.json')
        result['nginxUnchanged']=all(sha(Path(path))==digest for path,digest in initial['nginx'].items())
        if not result['productionStorePreserved'] or not result['nginxUnchanged']:
            raise RuntimeError('preserved_state_check_failed')
        result['passed']=True
    except BaseException as error:
        result['errorType']=type(error).__name__
        checked(['systemctl','stop',*SERVICES]); stopped=True
        for rel in reversed(changed):
            current=PROD/rel; failed=stage/rel; saved=backup/rel
            failed.parent.mkdir(parents=True,exist_ok=True)
            if current.exists(): current.rename(failed)
            saved.rename(current)
        if poster_moved and not old_posters.exists():
            # Repair the narrow failure window between moving media and
            # creating its shared link. The exact source was created by us.
            if old_posters.is_symlink(): raise RuntimeError('unexpected_broken_poster_link')
            shared.rename(old_posters)
        if (backup/'.env').is_file(): shutil.copy2(backup/'.env',PROD/'.env')
        checked(['systemctl','start',*SERVICES]); stopped=False
        result['rolledBack']=True
    finally:
        if stopped: command(['systemctl','start',*SERVICES])
        result['finishedAt']=datetime.now(timezone.utc).isoformat()
        result['liveApiSha256']=sha(PROD/'api/app/generator.py')
        with receipt_path.open('x') as handle: json.dump(result,handle,indent=2)
    return result

def client(args):
    sys.path.insert(0,str(ROOT))
    from scripts.candidate_visitor_smoke import _connect
    c=_connect(ROOT.parent/'服务器信息.txt',Path.home()/'.ssh/known_hosts')
    try:
        local=Path(__file__)
        remote='/tmp/inquiry-production-release-'+sha(local)[:16]+'.py'
        with c.open_sftp() as sftp:
            try: sftp.stat(remote)
            except FileNotFoundError:
                with sftp.open(remote,'wx') as handle: handle.write(local.read_bytes())
        argv=[(PROD/'.venv/bin/python').as_posix(),'-B',remote,'--remote','--mode',args.mode]
        if args.commit: argv+=['--commit',args.commit]
        _,stdout,stderr=c.exec_command(' '.join(shlex.quote(value) for value in argv),timeout=600)
        body=stdout.read(2*1024*1024)
        error_body=stderr.read(65536)
        code=stdout.channel.recv_exit_status()
        if not body.strip():
            # This branch is for process startup, before any .env read.
            print(json.dumps({'passed':False,'remoteExitCode':code,'stdoutBytes':len(body),
                              'stderrBytes':len(error_body)}))
            return 1
        result=json.loads(body)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x',encoding='utf8') as handle: json.dump(result,handle,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False))
        return code
    finally: c.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',action='store_true')
    parser.add_argument('--mode',choices=['inspect','deploy','verify'],default='inspect')
    parser.add_argument('--commit')
    parser.add_argument('--report',type=Path,default=ROOT/'artifacts/qa/production-rc11/before.json')
    args=parser.parse_args()
    try:
        if args.remote:
            result=inspect() if args.mode=='inspect' else verify_live() if args.mode=='verify' else deploy(args.commit or '')
            print(json.dumps(result,ensure_ascii=False))
            return 1 if result.get('passed') is False else 0
        return client(args)
    except Exception as error:
        print(json.dumps({'passed':False,'errorType':type(error).__name__}))
        return 1

if __name__=='__main__': raise SystemExit(main())
