import test from 'node:test';
import assert from 'node:assert/strict';
import {workflowState,nextReviewStep,hasInstruction,reviewLabels} from '../static/workflow.mjs';
const base=()=>({cache:'ready',quality:{grading:{grade:'B'}},review:{grade:''},review_stale:false,readiness:{ready:false,blockers:['待人工分级']},source_instruction:'Validate v1.4.1 locomotion'});
test('quality and review remain separate and queues follow server prerequisites',()=>{
 const r=base();assert.equal(workflowState(r).queue,'review');
 r.review.grade='A';assert.equal(workflowState(r).queue,'complete');
 r.readiness.ready=true;assert.equal(workflowState(r).queue,'preflight');
 r.review_stale=true;assert.equal(workflowState(r).queue,'review');
 r.quality.grading.grade='D';assert.equal(workflowState(r).queue,'excluded');
 r.cache='absent';assert.equal(workflowState(r).action,'inspect');
});
test('unanalysed records need analysis and building state does not offer duplicate work',()=>{
 const r=base();r.cache='building';assert.equal(workflowState(r).label,'质检中');
 assert.equal(nextReviewStep(r,r.review,false).disabled,true);
});
test('next step never jumps over missing instruction or unsaved changes',()=>{
 const r=base();const draft={grade:'B',instruction:''};
 assert.equal(nextReviewStep(r,draft,true).action,'instruction');
 draft.instruction='拿起杯子并放入容器';assert.equal(nextReviewStep(r,draft,true).action,'save');
 r.readiness.ready=true;assert.equal(nextReviewStep(r,draft,false).action,'preflight');
 assert.equal(nextReviewStep(r,draft,true).action,'save');
});
test('review deferral and exclusion do not become approval',()=>{
 const r=base();
 assert.equal(nextReviewStep(r,{grade:'C'},true).action,'save');
 assert.equal(nextReviewStep(r,{grade:'C'},false).action,'decision');
 assert.equal(nextReviewStep(r,{grade:'D'},true).action,'save');
 assert.equal(nextReviewStep(r,{grade:'D'},false).action,'inspect');
});
test('manual labels are semantic while storage codes are retained',()=>{
 assert.deepEqual(Object.keys(reviewLabels),['A','B','C','D']);
 assert.equal(reviewLabels.B,'处理后保留');
 assert.equal(hasInstruction('validate v1.4.1'),false);
 assert.equal(hasInstruction('把物体放入盒子'),true);
});
