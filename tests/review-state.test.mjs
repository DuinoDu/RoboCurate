import test from 'node:test';
import assert from 'node:assert/strict';
import {sortEvents, nextPending, applyPendingBatch, undoBatch} from '../static/review-state.mjs';

const a={id:'a',start_s:1,end_s:2,kind:'tracking',severity:'review',threshold:.5,value:1};
const b={id:'b',start_s:2,end_s:3,kind:'gap',severity:'warning',threshold:100,value:250};
const c={id:'c',start_s:3,end_s:4,kind:'invalid',severity:'error'};
test('priority places missing numeric data and gaps before excursions',()=>{
  assert.deepEqual(sortEvents([a,c,b],'priority').map(e=>e.id),['c','b','a']);
  assert.deepEqual(sortEvents([c,a,b]).map(e=>e.id),['a','b','c']);
});
test('reverse navigation from no selection starts at the last pending event',()=>{
  assert.equal(nextPending([a,b,c],null,{},-1).event.id,'c');
  assert.equal(nextPending([a,b,c],null,{c:'accepted'},-1).event.id,'b');
});
test('navigation skips decided events and wraps predictably',()=>{
  assert.equal(nextPending([a,b,c],'c',{b:'accepted'},1).event.id,'a');
  assert.equal(nextPending([a,b],null,{a:'accepted',b:'confirmed'}),null);
});
test('batch only modifies pending events in the requested scope',()=>{
  const original={a:'confirmed'};
  const batch=applyPendingBatch(original,[a,b],'accepted');
  assert.deepEqual(batch.next,{a:'confirmed',b:'accepted'});
  assert.deepEqual(original,{a:'confirmed'});
  assert.equal(batch.changes.length,1);
});
test('undo preserves later individual decisions',()=>{
  const batch=applyPendingBatch({},[a,b],'accepted');
  const undone=undoBatch({...batch.next,b:'confirmed'},batch.changes);
  assert.deepEqual(undone.next,{b:'confirmed'});
  assert.equal(undone.restored,1);
});
test('invalid batch values cannot silently write arbitrary decisions',()=>{
  assert.throws(()=>applyPendingBatch({},[a],'approve-everything'));
});
