import test from 'node:test';
import assert from 'node:assert/strict';
import {stageAt,stageCurve,stageInputSummary,reviewReasons,stageEvidenceAt,preferredProgressModel,stageReviewIntervals,reviewTarget} from '../static/stage-data.mjs';
const r={times_s:[0,.5,1,1.5,2],stage:[0,1,-1,3,4],confidence:[.9,.7,null,.85,.99],needs_review:[false,true,true,false,false],progress:[0,.2,null,.8,1],stage_names:['接近','抓取','搬运','放置','释放后观察']};
test('default selects only installed ready stage models and preserves explicit choice',()=>{
  const models=[{id:'warp',ready:true},{id:'v1',kind:'stage_progress',stage_version:1,ready:true},{id:'v3',kind:'stage_progress',stage_version:3,ready:true},{id:'v4',kind:'stage_progress',stage_version:4,ready:false}];
  assert.equal(preferredProgressModel(models),'v3');assert.equal(preferredProgressModel(models,'warp'),'warp');
  assert.equal(preferredProgressModel(models.slice(0,1)),'warp');assert.equal(preferredProgressModel([]),'');
});
test('display evidence uses latest past sample and does not fill gaps or future data',()=>{
  const data={...r,evidence:{probability:[[.9,.1],null,null,[.2,.8],[.01,.99]],camera_valid:[[true,true],[true,false],[false,false],[true,true],[true,true]],state_valid:[true,false,false,true,true]}};
  assert.deepEqual(stageEvidenceAt(data,.49).probability,[.9,.1]);assert.equal(stageEvidenceAt(data,.6).state,false);
  assert.deepEqual(stageEvidenceAt(data,1.2).cameras,[false,false]);assert.equal(stageAt(data,1.2),null);
  assert.equal(stageEvidenceAt(data,3),null);assert.equal(stageEvidenceAt(data,-.1),null);
  assert.equal(stageEvidenceAt(r,0).probability,null);assert.equal(stageAt(data,NaN),null);
});
test('review queue merges only contiguous flagged cells and clips the last cell',()=>{
  const data={times_s:[0,.5,1,2,2.5,3],needs_review:[false,true,true,true,false,true],review_reasons:[0,1,2,8,0,32]};
  const ranges=stageReviewIntervals(data,3.2);
  assert.deepEqual(ranges.map(x=>[x.start_s,x.end_s,x.bits]),[[.5,1.5,3],[2,2.5,8],[3,3.2,32]]);
  assert.equal(reviewTarget(ranges,0,1).start_s,.5);assert.equal(reviewTarget(ranges,.6,1).start_s,2);
  assert.equal(reviewTarget(ranges,2,-1).start_s,.5);assert.equal(reviewTarget(ranges,.5,-1),null);
  assert.equal(reviewTarget(ranges,3.1,1),null);assert.equal(stageReviewIntervals(data,0).length,0);
});
test('stage reads are causal and missing camera samples stay unknown',()=>{
  assert.equal(stageAt(r,.49).stage,0);
  assert.equal(stageAt(r,.6).review,true);
  assert.equal(stageAt(r,1.2),null);
  assert.equal(stageAt(r,3),null);
  assert.equal(stageAt(r,2).name,'释放后观察');
});
test('stage plot breaks gaps without a velocity or completion label',()=>{
  const path=stageCurve(r,2);
  assert.equal((path.match(/M/g)||[]).length,2);
  assert(!path.includes('NaN'));
});
test('missing modality is shown separately from model confidence',()=>{
  const v2={...r,degraded:[true,false,false,false,false],summary:{camera_coverage:{head:0,right_wrist:1},state_coverage:.9}};
  assert.equal(stageAt(v2,0).degraded,true);
  assert.equal(stageAt(r,0).degraded,false);
  assert.equal(stageInputSummary(v2),'输入覆盖：头部 0.0% · 右腕 100.0% · 关节状态 90.0%');
  assert.equal(stageInputSummary(r),'');
});
test('review explanations distinguish transitions from missing data',()=>{
  assert.deepEqual(reviewReasons(16),['阶段切换附近']);
  assert.deepEqual(reviewReasons(4|16),['输入不完整','阶段切换附近']);
  assert.deepEqual(reviewReasons(32),['阶段证据不足']);
  assert.deepEqual(reviewReasons(64),['阶段过渡待核查']);
  assert.deepEqual(stageAt({...r,confidence:[.98,.7,null,.85,.99],needs_review:[true,true,true,false,false],review_reasons:[32,1,8,0,0]},0).reasons,['阶段证据不足']);
  const v3={...r,review_reasons:[0,16,8,0,0]};
  assert.deepEqual(stageAt(v3,.5).reasons,['阶段切换附近']);
  assert.deepEqual(stageAt(r,.5).reasons,[]);
});
