import test from 'node:test';
import assert from 'node:assert/strict';
import {latestPoint,trailRange,trailVisible} from '../static/trajectory-data.mjs';
const track={source:'state.q',side:'left',part:'foot',segments:[{times_s:[0,.03,.06],positions:[[0,0,0],[1,0,0],[2,0,0]]},{times_s:[1,1.03],positions:[[3,0,0],[4,0,0]]}]};
test('markers are causal and disappear across missing intervals',()=>{
 assert.equal(latestPoint(track,-.001),null);
 assert.deepEqual(latestPoint(track,.04).position,[1,0,0]);
 assert.equal(latestPoint(track,.5),null);
 assert.equal(latestPoint(track,1).time,1);
});
test('history contains no future; full episode requires explicit zero window',()=>{
 assert.deepEqual(trailRange([0,1,2,3,4],2.5,1),[2,3]);
 assert.deepEqual(trailRange([0,1,2],0,5),[0,1]);
 assert.deepEqual(trailRange([0,1,2],0,0),[0,3]);
});
test('humanoid body and side filtering applies to both target and state',()=>{
 const options={showTrails:true,showTargets:false,trailPart:'foot',trailSide:'both'};
 assert.equal(trailVisible(track,options),true);
 assert.equal(trailVisible({...track,source:'action.q'},options),false);
 assert.equal(trailVisible(track,{...options,trailPart:'wrist'}),false);
 assert.equal(trailVisible(track,{...options,trailSide:'right'}),false);
});

import {recordedGhosts,poseLinkVisible} from '../static/trajectory-data.mjs';
const poses={segments:[{times_s:[0,.5,1,1.5,2],q:[[0],[1],[2],[3],[4]]},{times_s:[4,4.5,5],q:[[8],[9],[10]]}]};
const poseOptions={showPoseTrail:true,showPast:true,showFuture:false,pastWindow:1,futureWindow:1,poseCount:2};
test('ghost history never leaks future and future records require explicit opt-in',()=>{
 const past=recordedGhosts(poses,1,poseOptions);
 assert.deepEqual(past.map(p=>p.time),[0,.5]);
 assert.ok(past.every(p=>p.direction===-1));
 const both=recordedGhosts(poses,1,{...poseOptions,showFuture:true});
 assert.deepEqual(both.filter(p=>p.direction===1).map(p=>p.time),[2,1.5]);
 assert.deepEqual(recordedGhosts(poses,3,poseOptions).map(p=>p.time),[2]);
 assert.deepEqual(recordedGhosts(poses,0,poseOptions),[]);
});
test('ghost pool is bounded and respects humanoid regions',()=>{
 assert.ok(recordedGhosts(poses,2,{...poseOptions,poseCount:100,showFuture:true}).length<=6);
 assert.equal(poseLinkVisible('left_knee_link','upper'),false);
 assert.equal(poseLinkVisible('right_wrist_yaw_link','upper'),true);
 assert.equal(poseLinkVisible('left_ankle_roll_link','lower'),true);
 assert.equal(poseLinkVisible('pelvis','all'),true);
});
