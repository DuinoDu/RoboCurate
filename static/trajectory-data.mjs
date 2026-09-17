// All comparisons are episode-relative seconds, never float epoch timestamps.
export function upperBound(times, value) {
  let lo=0, hi=times.length;
  while(lo<hi) { const mid=(lo+hi)>>>1; if(times[mid]<=value) lo=mid+1; else hi=mid; }
  return lo;
}
export function trailRange(times, time, windowSeconds) {
  if(windowSeconds===0) return [0,times.length]; // explicitly requested full episode
  const end=upperBound(times,time), start=upperBound(times,time-windowSeconds-1e-9);
  return [start,Math.max(start,end)];
}
export function latestPoint(track, time, maxAge=.1) {
  let point=null;
  for(const segment of track.segments) {
    const i=upperBound(segment.times_s,time)-1;
    if(i>=0 && time-segment.times_s[i]<=maxAge && (!point || segment.times_s[i]>point.time))
      point={position:segment.positions[i],time:segment.times_s[i]};
  }
  return point;
}
export const trailVisible=(track,opts)=>opts.showTrails &&
  (opts.trailPart==='all'||(track.part||'wrist')===(opts.trailPart||'wrist')) &&
  (opts.trailSide==='both'||track.side===opts.trailSide) &&
  (track.source==='state.q'||opts.showTargets);

// Uniformly spaced samples from the recorded timeline. No interpolation or
// prediction, and an invalid/gapped neighborhood produces no ghost pose.
export function recordedGhosts(poses,time,opts) {
  if(!poses?.segments || !opts.showPoseTrail)return [];
  const results=[],seen=new Set(),count=Math.max(1,Math.min(3,Math.round(opts.poseCount||3)));
  for(const direction of [-1,1]) {
    if(direction<0?!opts.showPast:!opts.showFuture)continue;
    const window=direction<0?opts.pastWindow:opts.futureWindow;
    for(let i=count;i>=1;i--) {
      const target=time+direction*window*i/count;
      let point=null;
      for(const segment of poses.segments) {
        const index=upperBound(segment.times_s,target)-1;
        if(index<0)continue;
        const stamp=segment.times_s[index];
        if(target-stamp>.1 || (stamp-time)*direction<=.015)continue;
        if(!point || stamp>point.time)point={time:stamp,q:segment.q[index],direction};
      }
      if(point && !seen.has(point.time)) {results.push(point);seen.add(point.time)}
    }
  }
  return results;
}

export function poseLinkVisible(name,scope) {
  if(scope==='all')return true;
  const upper=/shoulder|elbow|wrist|hand/.test(name),lower=/hip|knee|ankle|foot/.test(name);
  return scope==='upper'?upper:scope==='lower'?lower:true;
}
