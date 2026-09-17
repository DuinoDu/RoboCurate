// Stage progress is a label-aligned proxy; it is never a WARP velocity or success score.
function sampleIndex(result,time){
  const t=result?.times_s||[];
  let lo=0,hi=t.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(t[mid]<=time)lo=mid+1;else hi=mid;}
  const i=lo-1;
  return i<0||!Number.isFinite(time)||time-t[i]>.501?-1:i;
}

export function stageEvidenceAt(result,time){
  const index=sampleIndex(result,time);
  if(index<0)return null;
  const evidence=result.evidence;
  return {index,time:result.times_s[index],probability:evidence?.probability?.[index]||null,
    cameras:evidence?.camera_valid?.[index]||null,state:evidence?.state_valid?.[index]??null,
    reasons:reviewReasons(result.review_reasons?.[index]||0),review:!!result.needs_review?.[index]};
}

export function stageAt(result,time){
  const i=sampleIndex(result,time);
  if(i<0||result.stage[i]<0||!Number.isFinite(result.confidence[i]))return null;
  return {index:i,stage:result.stage[i],name:result.stage_names[result.stage[i]],confidence:result.confidence[i],review:!!result.needs_review[i],degraded:!!result.degraded?.[i],reasons:reviewReasons(result.review_reasons?.[i]||0)};
}

export const stageColors=['#6486cf','#b28a28','#3c9d8a','#a877c2','#89939e'];

export function preferredProgressModel(models,saved=''){
  if(models.some(m=>m.id===saved))return saved;
  const stages=models.filter(m=>m.ready&&m.kind==='stage_progress').sort((a,b)=>(b.stage_version||1)-(a.stage_version||1));
  return stages[0]?.id||models.find(m=>m.ready)?.id||models[0]?.id||'';
}

// Intervals cover only flagged 2 Hz cells. Never bridge an unsampled gap.
export function stageReviewIntervals(result,duration){
  const ranges=[];let current=null;
  for(let i=0;i<(result?.times_s?.length||0);i++){
    const start=Math.max(0,result.times_s[i]),end=Math.min(duration,start+.5,result.times_s[i+1]??duration);
    if(!result.needs_review?.[i]||!Number.isFinite(start)||!(end>start)){current=null;continue;}
    if(!current||Math.abs(current.end_s-start)>1e-6){
      current={id:`review-${i}`,start_s:start,end_s:end,bits:0,samples:0,indices:[]};ranges.push(current);
    }
    current.end_s=end;current.samples++;current.indices.push(i);current.bits|=result.review_reasons?.[i]||0;
  }
  return ranges.map(r=>({...r,reasons:reviewReasons(r.bits),duration_s:r.end_s-r.start_s}));
}

export function reviewTarget(ranges,time,direction){
  const current=ranges.findIndex(r=>r.start_s<=time+1e-6&&time<r.end_s-1e-6);
  if(current>=0)return ranges[current+(direction>0?1:-1)]||null;
  return direction>0?ranges.find(r=>r.start_s>time+1e-6)||null:[...ranges].reverse().find(r=>r.end_s<=time+1e-6)||null;
}

export function reviewReasons(bits){
  return [[4,'输入不完整'],[64,'阶段过渡待核查'],[16,'阶段切换附近'],[2,'模型意见分歧'],[1,'把握度较低'],[32,'阶段证据不足'],[8,'无有效图像']].filter(([flag])=>bits&flag).map(([,label])=>label);
}

export function stageInputSummary(result){
  const s=result?.summary;
  if(!s?.camera_coverage)return '';
  const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'未知';
  return `输入覆盖：头部 ${pct(s.camera_coverage.head)} · 右腕 ${pct(s.camera_coverage.right_wrist)} · 关节状态 ${pct(s.state_coverage)}`;
}

export function stageCurve(result,duration,width=600,height=130){
  const times=result.times_s||[],values=result.progress||[];
  let path='',active=false,last=-Infinity;
  for(let i=0;i<times.length;i++){
    const t=times[i],v=values[i];
    if(!Number.isFinite(t)||!Number.isFinite(v)||result.stage[i]<0){active=false;continue;}
    if(t-last>.751)active=false;
    const x=Math.max(0,Math.min(width,t/Math.max(.001,duration)*width));
    const y=height-8-Math.max(0,Math.min(1,v))*(height-16);
    path+=`${active?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;active=true;last=t;
  }
  return path.trim();
}
