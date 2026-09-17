/** Missing scores are gaps, never zeros or interpolated completion. */
export function scoreAt(times, values, time) {
  if (!times.length || time < times[0] || time > times.at(-1)) return null;
  let lo=0, hi=times.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<=time)lo=mid+1;else hi=mid;}
  const value=values[lo-1];
  return typeof value==='number' && Number.isFinite(value) ? value : null;
}

export function curvePath(times, values, duration, width=600, height=130) {
  const finite=values.filter(v=>typeof v==='number' && Number.isFinite(v));
  const bound=Math.max(2, ...finite.map(Math.abs));
  const y=v=>height/2-v/bound*(height/2-8);
  let path='', connected=false;
  for(let i=0;i<times.length;i++){
    const v=values[i];
    if(typeof v!=='number'||!Number.isFinite(v)||!Number.isFinite(times[i])){connected=false;continue;}
    const x=times[i]/Math.max(duration, .001)*width;
    path+=`${connected?'L':'M'}${x.toFixed(2)},${y(v).toFixed(2)} `;
    connected=true;
  }
  return {path, bound, zero:height/2, standard:y(1)};
}

export function candidateClip(suggestion, duration) {
  if(!suggestion || !['forward','stall','regression'].includes(suggestion.kind)) return null;
  const start=Math.max(0,Number(suggestion.start_s)),end=Math.min(duration,Number(suggestion.end_s));
  if(!Number.isFinite(start+end)||end-start<.1)return null;
  return {start,end,label:{forward:'进展正向候选',stall:'进展停滞候选',regression:'进展回退候选'}[suggestion.kind]};
}
