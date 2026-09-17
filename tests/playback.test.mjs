import test from 'node:test';
import assert from 'node:assert/strict';
import {nearestFrame,adjacentFrame,clampWindow,zoomWindow,parseFrameBatch} from '../static/playback.mjs';
test('frame stepping uses irregular source timestamps',()=>{
  const t=[.014,.050,.071,.118];
  assert.equal(nearestFrame(t,.073),2);
  assert.equal(adjacentFrame(t,.050,1),2);
  assert.equal(adjacentFrame(t,.014,-1),0);
  assert.equal(adjacentFrame([],0,1),-1);
});
test('duplicate timestamps retain distinct frame indexes',()=>{
  const t=[.01,.01,.01,.04];
  assert.equal(adjacentFrame(t,.01,1,1),2);
  assert.equal(adjacentFrame(t,.01,-1,2),1);
});
test('time window clamps to episode and preserves a usable span',()=>{
  assert.deepEqual(clampWindow(-2,2,10),[0,4]);
  assert.deepEqual(clampWindow(8,12,10),[6,10]);
  assert.deepEqual(clampWindow(4,2,10),[2,4]);
  assert.deepEqual(clampWindow(0,4,2),[0,2]);
});
test('zoom retains anchor and cannot exceed source bounds',()=>{
  assert.deepEqual(zoomWindow([0,10],5,.5,10),[2.5,7.5]);
  assert.deepEqual(zoomWindow([0,10],5,2,10),[0,10]);
});
test('batch parser preserves independent original JPEG bytes',async()=>{
  const meta=new TextEncoder().encode(JSON.stringify({token:'1:2',frames:[{index:4,time_s:.05,size:3},{index:5,time_s:.08,size:2}]}));
  const data=new Uint8Array(4+meta.length+5);new DataView(data.buffer).setUint32(0,meta.length,true);data.set(meta,4);data.set([1,2,3,4,5],4+meta.length);
  const parsed=parseFrameBatch(data.buffer);
  assert.equal(parsed.frames[1].index,5);
  assert.deepEqual([...new Uint8Array(await parsed.frames[0].blob.arrayBuffer())],[1,2,3]);
});
test('truncated batches are rejected',()=>{
  assert.throws(()=>parseFrameBatch(new ArrayBuffer(2)));
  const data=new ArrayBuffer(4);new DataView(data).setUint32(0,100,true);assert.throws(()=>parseFrameBatch(data));
});
