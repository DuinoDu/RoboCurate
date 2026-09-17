#!/usr/bin/env python3
"""Score selected local recordings with an installed model, without editing reviews."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlencode, urlsplit

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from studio import StudioLibrary
from warp_progress.features import atomic_json


def score(args):
    workspace = Path(args.workspace).resolve()
    if urlsplit(args.server).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Scoring API must be a loopback server')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(route, body=None):
        request = urllib.request.Request(args.server+route, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=15) as response:
            return json.load(response)
    lib = StudioLibrary(workspace)
    report = dict(state='running', model_id=args.model_id, started_at=time.time(), episodes=[], review_writes_performed=False)
    output = Path(args.output).resolve()
    try:
        for root in lib.store.setting('roots', []):
            lib.add_root(Path(root))
        records = [lib.ref(ep) for ep in lib.order if lib.ref(ep).episode_id in args.episodes]
        if len(records) != len(set(args.episodes)):
            raise ValueError('Some requested recordings are missing or ambiguous')
        remote = not lib.progress_service.owns_jobs
        if remote and api('/api/health').get('workspace') != str(workspace):
            raise ValueError('Local server uses a different workspace')
        for ref in records:
            print('Scoring '+ref.episode_id, flush=True)
            if remote:
                api('/api/progress/jobs', dict(ep=ref.id, model_id=args.model_id))
            else:
                lib.progress_service.start(ref.id, args.model_id)
            prior = None
            while True:
                current = api('/api/progress?'+urlencode(dict(ep=ref.id, model_id=args.model_id))) if remote else lib.progress_service.current(ref.id, args.model_id)
                if current['state'] == 'ready':
                    break
                if current['state'] not in ('queued', 'running'):
                    raise RuntimeError(current.get('job', {}).get('message', current['state']))
                message = current.get('job', {}).get('message')
                if message != prior:
                    print(message, flush=True); prior = message
                time.sleep(3)
            result = current['result']
            report['episodes'].append(dict(name=ref.episode_id, episode_id=ref.id, signature=result['signature'],
                validation_status=result['validation_status'], summary=result['summary'], elapsed_s=result['elapsed_s'],
                review_url=args.server+'/#review/'+ref.id))
            atomic_json(output, report)
        report.update(state='complete', completed_at=time.time(),
                      limitation='These scores measure model estimates; no local task-success or policy-improvement claim.')
    except BaseException as exc:
        report.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        atomic_json(output, report)
        lib.progress_service.close(); lib.pool.shutdown(); lib.export_pool.shutdown()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=str(APP/'workspace'))
    parser.add_argument('--server', default='http://127.0.0.1:8421')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--output', default=str(APP/'workspace/warp/local_score_report.json'))
    score(parser.parse_args())
