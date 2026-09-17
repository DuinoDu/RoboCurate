"""Run reviewed, hash-pinned upstream functions with in-memory data adapters.

Only the named AST definitions are loaded, not package initialization, network
clients or hosted checks. Function bodies are unchanged for the comparison.
Sources remain under workspace/references/snapshots; Apache notices are retained.
"""
import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'workspace/references/snapshots'

class Severity(Enum):
    INFO='INFO'; PASS='PASS'; WARN='WARN'; FAIL='FAIL'; ERROR='ERROR'

class Tier(Enum):
    INTEGRITY='integrity'; QUALITY='quality'

class Result(SimpleNamespace):
    def __init__(self, **kw):
        super().__init__(tier=kw.pop('tier',Tier.INTEGRITY),**kw)

def definitions(path, expected, names, extra=None):
    raw=path.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected
    tree=ast.parse(raw);body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:body.append(node)
    assert len(body)==len(names)+1
    module=ModuleType('reference_'+expected[:12]);sys.modules[module.__name__]=module
    module.__dict__.update(np=np,dataclass=dataclass,CheckResult=Result,Severity=Severity,Tier=Tier)
    module.__dict__.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),module.__dict__)
    return module

class Episode:
    def __init__(self,t,v):self.t=t;self.v=v
    def channel(self,topic):return SimpleNamespace(timestamps=self.t,to_numpy=lambda field:self.v)

def motion_reference():
    mod=definitions(ROOT/'hflow/src/hflow/checks.py','296c68ed52aaf186d1ce6d2f69b4e44eb1b11a8b',
                    {'_TrajectoryProfile','_trajectory_profile','trajectory_metrics'})
    def run(t,v):
        r=mod.trajectory_metrics(Episode(t,v),topic='q')
        return {k.split('/',1)[1]:v for k,v in r.measurements.items()}
    return run

class Cache:
    def __init__(self,columns):pass
    def get_episode_data(self,ds,ep):return {'timestamp':ds.t}

class Dataset:
    def __init__(self,t,fps):self.t=t;self.fps=fps
    def __iter__(self):yield SimpleNamespace(length=len(self.t),episode_index=0)

def spacing_reference():
    mod=definitions(ROOT/'trajlens/src/trajlens/checks/temporal.py','f14ce1c7a56a490ecb1597436f1f6bf1107fcafb',
                    {'_TimestampSpacingCheck'},dict(ShardColumnCache=Cache,_DECODER_TOLERANCE_S=1e-4,_FAIL_MULTIPLIER=1.0))
    return lambda t,fps:mod._TimestampSpacingCheck().run(Dataset(t,fps),None).severity.value

def score_reference():
    mod=definitions(ROOT/'trajlens/src/trajlens/report/trust_score.py','4c7180405b4f4661a0105c7417b390d17a121c2a',
                    {'integrity_only','compute_trust_score'},dict(_FAIL_PENALTY=30,_WARN_PENALTY=5,_ERROR_PENALTY=10,_FAIL_CAP=60,_WARN_CAP=20))
    def run(checks):
        return mod.compute_trust_score([Result(severity=Severity[c['severity'] if c['severity']!='PASS' else 'INFO'],
                                               tier=Tier.QUALITY if c.get('tier')=='quality' else Tier.INTEGRITY) for c in checks])
    return run
