import test from 'node:test';
import assert from 'node:assert/strict';
import {scoreAt,curvePath,candidateClip} from '../static/progress-data.mjs';
test('null scores remain gaps, including at exact timestamps',()=>{
  assert.equal(scoreAt([0,1,2],[1,null,3],1),null);
  assert.equal(scoreAt([0,1,2],[1,null,3],1.9),null);
  assert.equal(scoreAt([0,1,2],[1,null,3],.9),1);
  assert.equal(scoreAt([0,1,2],[1,null,3],2.01),null);
  const result=curvePath([0,1,2],[1,null,-1],3);
  assert.equal((result.path.match(/M/g)||[]).length,2);
  assert.equal(result.path.includes('L'),false);
});
test('candidate only fills a bounded range, never assigns review conclusions',()=>{
  assert.deepEqual(candidateClip({kind:'forward',start_s:-2,end_s:12},10),{start:0,end:10,label:'进展正向候选'});
  assert.equal(candidateClip({kind:'forward',start_s:2,end_s:2.02},10),null);
  assert.equal(candidateClip({kind:'unknown',start_s:0,end_s:2},10),null);
});
