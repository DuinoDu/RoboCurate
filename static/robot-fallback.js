// CPU projection of the same URDF geometry and native-timestamp wrist trails.
import * as THREE from 'three';
import {trailRange,latestPoint,trailVisible,recordedGhosts,poseLinkVisible} from './trajectory-data.mjs';

export class SkeletonView {
  constructor(host,data,desc) {
    Object.assign(this,{host,data,desc,visible:true,solid:true,yaw:-.7,elevation:.18,sample:0,time:0,zoom:1});
    this.opts={source:'state.q',showGhost:false,ghost:'action.q',applyImu:false,showTrails:true,showTargets:true,trailSide:'both',trailWindow:5,showPoseTrail:true,showPast:true,showFuture:false,pastWindow:2,futureWindow:2,poseCount:3,poseOpacity:.18,poseScope:'all',showGrid:true};
    this.links=new Map(desc.links.map(l=>[l.name,new THREE.Group()]));this.joints=new Map();
    for(const j of desc.joints) {
      const origin=new THREE.Group();origin.position.fromArray(j.xyz);origin.rotation.set(...j.rpy,'ZYX');
      const child=this.links.get(j.child);this.links.get(j.parent).add(origin);origin.add(child);
      this.joints.set(j.name,{node:child,axis:new THREE.Vector3(...j.axis).normalize(),joint:j});
    }
    this.root=this.links.get(desc.root);this.root.position.z=.8;
  }
  async mount() {
    this.host.innerHTML='';this.canvas=document.createElement('canvas');this.canvas.className='robot-canvas';
    this.canvas.dataset.renderer='skeleton';this.canvas.setAttribute('aria-label','G1 三维骨架与双腕轨迹，兼容渲染；拖动旋转，点击轨迹点定位');
    this.host.append(this.canvas);this.ctx=this.canvas.getContext('2d');this.ro=new ResizeObserver(()=>this.resize());this.ro.observe(this.host);
    let prev=null,start=null;
    this.canvas.onpointerdown=e=>{start=prev=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId)};
    this.canvas.onpointermove=e=>{if(!prev)return;this.yaw+=(e.clientX-prev[0])*.008;this.elevation=Math.max(-1.45,Math.min(Math.PI/2,this.elevation+(e.clientY-prev[1])*.004));prev=[e.clientX,e.clientY];this.draw()};
    this.canvas.onpointerup=e=>{if(start&&Math.hypot(e.clientX-start[0],e.clientY-start[1])<4)this.pick(e);prev=start=null};
    this.canvas.onpointercancel=()=>{prev=start=null};this.resize();
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.4,Math.min(4,this.zoom*Math.exp(-e.deltaY*.001)));this.draw()},{passive:false});
  }
  resize() { if(!this.canvas)return;const r=this.host.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);this.canvas.width=r.width*dpr;this.canvas.height=r.height*dpr;this.w=r.width;this.h=r.height;this.ctx.setTransform(dpr,0,0,dpr,0,0);this.draw() }
  pose(sample) {this.sample=sample;this.time=sample/this.data.meta.timeline.rate_hz;this.draw()}
  setTime(time) {this.time=time;this.draw()}
  setTrajectories(payload) {this.trajectories=payload;this.draw()}
  setView(preset='iso') {const views={iso:[-.7,.18],front:[Math.PI/2,0],side:[0,0],top:[0,Math.PI/2]};[this.yaw,this.elevation]=views[preset]||views.iso;this.zoom=1;this.draw()}
  fitView() {
    this.zoom=1;
    const points=[...this.positions(this.opts.source).values()];
    for(const track of this.trajectories?.tracks||[])if(trailVisible(track,this.opts))for(const segment of track.segments) {
      const [a,b]=trailRange(segment.times_s,this.time,this.opts.trailWindow);
      for(let i=a;i<b;i++){const p=segment.positions[i];points.push(new THREE.Vector3(p[0],p[1],p[2]+this.root.position.z));}
    }
    const projected=points.map(p=>this.project(p));
    const extentX=Math.max(...projected.map(p=>Math.abs(p[0]-this.w/2))),extentY=Math.max(...projected.map(p=>Math.abs(p[1]-this.h*.49)));
    this.zoom=Math.max(.4,Math.min(3,(this.w/2-24)/Math.max(extentX,1),(this.h/2-55)/Math.max(extentY,1)));this.draw();
  }
  positions(key) {
    for(let i=0;i<this.data.meta.joints.names.length;i++) {const j=this.joints.get(this.data.meta.joints.names[i]+'_joint');if(j){const q=Array.isArray(key)?key[i]:this.data.value(key,i,this.sample);j.node.quaternion.setFromAxisAngle(j.axis,Number.isFinite(q)?q:0)}}
    this.root.quaternion.identity();
    if(this.opts.applyImu&&this.data.has('imu.quat')) {const q=[0,1,2,3].map(i=>this.data.value('imu.quat',i,this.sample));if(q.every(Number.isFinite))this.root.quaternion.set(q[1],q[2],q[3],q[0]).normalize()}
    this.root.updateMatrixWorld(true);return new Map([...this.links].map(([name,node])=>[name,node.getWorldPosition(new THREE.Vector3())]));
  }
  project(p) {const x=Math.cos(this.yaw)*p.x-Math.sin(this.yaw)*p.y,y=Math.sin(this.yaw)*p.x+Math.cos(this.yaw)*p.y,z=p.z-.72,scale=Math.min(this.w/2.1,this.h/1.7)*this.zoom;return [this.w/2+x*scale,this.h*.49-(z*Math.cos(this.elevation)-y*Math.sin(this.elevation))*scale]}
  line(a,b,color,width=1) {const c=this.ctx;c.beginPath();c.moveTo(...this.project(a));c.lineTo(...this.project(b));c.strokeStyle=color;c.lineWidth=width;c.lineCap='round';c.stroke()}
  body(key,ghost) {
    const p=this.positions(key),c=this.ctx;
    for(const j of this.desc.joints) {const a=p.get(j.parent),b=p.get(j.child);if(a.distanceTo(b)<.015)continue;this.line(a,b,ghost?'#aa94ee66':'#8a8c94',ghost?3:9);if(!ghost)this.line(a,b,'#d7d8df',3)}
    for(const [name,v] of p) {if(!name.includes('link'))continue;const [x,y]=this.project(v);c.beginPath();c.arc(x,y,ghost?2:3,0,Math.PI*2);c.fillStyle=ghost?'#b29eef':'#b6b7c5';c.fill()}
  }
  trails() {
    const c=this.ctx;
    for(const track of this.trajectories?.tracks||[]) {
      if(!trailVisible(track,this.opts))continue;
      const target=track.source==='action.q',color=track.side==='left'?'#d4ed75':'#71d5e1';
      c.strokeStyle=color;c.fillStyle=color;c.lineWidth=target?1:1.7;c.globalAlpha=target?.4:.9;c.setLineDash(target?[4,4]:[]);
      for(const s of track.segments) {
        const [start,end]=trailRange(s.times_s,this.time,this.opts.trailWindow);c.beginPath();
        for(let i=start;i<end;i++) {const [x,y]=this.project(new THREE.Vector3(s.positions[i][0],s.positions[i][1],s.positions[i][2]+this.root.position.z));if(i===start)c.moveTo(x,y);else c.lineTo(x,y);this.pickPoints.push({x,y,time:s.times_s[i]})}
        c.stroke();
        for(let i=start;i<end;i+=3) {const [x,y]=this.project(new THREE.Vector3(s.positions[i][0],s.positions[i][1],s.positions[i][2]+this.root.position.z));c.beginPath();c.arc(x,y,1.6,0,Math.PI*2);c.fill()}
      }
      c.setLineDash([]);c.globalAlpha=1;
      const p=latestPoint(track,this.time,this.trajectories.max_gap_s);
      if(p) {const [x,y]=this.project(new THREE.Vector3(p.position[0],p.position[1],p.position[2]+this.root.position.z));c.beginPath();c.arc(x,y,target?4:5,0,Math.PI*2);if(target)c.stroke();else c.fill()}
    }
    c.setLineDash([]);c.globalAlpha=1;
  }
  recordedTrail() {
    const c=this.ctx;
    for(const pose of recordedGhosts(this.trajectories?.poses,this.time,this.opts)) {
      const points=this.positions(pose.q),color=pose.direction<0?'#a18ae0':'#e0bd77';c.globalAlpha=this.opts.poseOpacity;
      for(const j of this.desc.joints) {
        if(!poseLinkVisible(j.child,this.opts.poseScope))continue;
        const a=points.get(j.parent),b=points.get(j.child);
        if(a.distanceTo(b)<.015)continue;
        this.line(a,b,color,6);
        const [x,y]=this.project(b);this.pickPoints.push({x,y,time:pose.time});
      }
    }
    c.globalAlpha=1;
  }
  jointFrames() {
    if(!this.opts.showJointFrames)return;
    for(const [name,node] of this.links) {
      if(!/wrist_yaw_link|ankle_roll_link/.test(name))continue;
      if(this.opts.trailPart!=='all'&&(name.includes('wrist')?'wrist':'foot')!==this.opts.trailPart)continue;
      const start=node.getWorldPosition(new THREE.Vector3()),q=node.getWorldQuaternion(new THREE.Quaternion());
      [[1,0,0,'#e9908a'],[0,1,0,'#b9d779'],[0,0,1,'#83b9ec']].forEach(([x,y,z,color])=>this.line(start,new THREE.Vector3(x,y,z).multiplyScalar(.1).applyQuaternion(q).add(start),color,2));
    }
  }
  pick(e) {const r=this.canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,dist=10;for(const p of this.pickPoints||[]) {const d=Math.hypot(p.x-x,p.y-y);if(d<dist){best=p;dist=d}}if(best)this.onSeek?.(best.time)}
  draw() {
    if(!this.ctx||!this.w)return;const c=this.ctx,css=getComputedStyle(document.documentElement),tone=k=>css.getPropertyValue(k).trim();
    c.clearRect(0,0,this.w,this.h);c.fillStyle=tone('--surface-1')||'#17181d';c.fillRect(0,0,this.w,this.h);
    if(this.opts.showGrid)for(let i=-4;i<=4;i++) {this.line(new THREE.Vector3(i*.25,-1,0),new THREE.Vector3(i*.25,1,0),tone('--grid'));this.line(new THREE.Vector3(-1,i*.25,0),new THREE.Vector3(1,i*.25,0),tone('--grid'))}
    this.pickPoints=[];this.recordedTrail();
    if(this.opts.showGhost)this.body(this.opts.ghost,true);this.body(this.opts.source,false);this.jointFrames();this.trails();
  }
  applyTheme() {this.draw()}
  destroy() {this.ro?.disconnect();this.canvas?.remove()}
}
