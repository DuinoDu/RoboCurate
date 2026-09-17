import test from 'node:test';
import assert from 'node:assert/strict';
import {compareDataGrades,ratingPoints,ratingText,ratingPercent} from '../static/grading-view.mjs';
const row=(name,grade,coverage=1)=>({name,quality:{grading:{grade,metrics:{coverage_fraction:coverage}}}});
test('quality sorting keeps unassessed data last in either direction',()=>{
  const rows=[row('pending',null),row('bad','D'),row('best','A'),row('fair','B')];
  assert.deepEqual([...rows].sort(compareDataGrades).map(r=>r.name),['best','fair','bad','pending']);
  assert.deepEqual([...rows].sort((a,b)=>compareDataGrades(a,b,false)).map(r=>r.name),['bad','fair','best','pending']);
});
test('coverage breaks ties within the same quality band',()=>{
  const rows=[row('lower','B',.95),row('higher','B',.98)];
  assert.equal([...rows].sort(compareDataGrades)[0].name,'higher');
});
test('continuous technical rating orders peers without overriding mandatory grade gates',()=>{
  const a=row('lower','B',.999),b=row('higher','B',.995),c=row('blocked','D',1);
  a.quality.rating={score:92};b.quality.rating={score:94};c.quality.rating={score:100};
  assert.deepEqual([c,a,b].sort(compareDataGrades).map(x=>x.name),['higher','lower','blocked']);
});

test('ten-point display preserves scale, bar length and missing values',()=>{
  const current={version:3,max_score:10,score:9.23};
  assert.equal(ratingText(current),'9.23');assert.equal(ratingPoints(current),9.23);
  assert.equal(ratingText({version:2,score:92.3}),'9.23');
  assert.ok(Math.abs(ratingPercent(current)-92.3)<1e-9);
  assert.equal(ratingText({max_score:10,score:null}),'—');
  assert.equal(ratingText(undefined),'—');
  assert.equal(ratingText({max_score:0,score:1}),'—');
});
test('mixed legacy and new scores sort on the same scale',()=>{
  const a=row('lower','B'),b=row('higher','B');
  a.quality.rating={score:92.3,version:2};b.quality.rating={score:9.4,max_score:10};
  assert.deepEqual([a,b].sort(compareDataGrades).map(x=>x.name),['higher','lower']);
});
