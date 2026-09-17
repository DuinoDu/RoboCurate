/* Canvas renderer of the supplied G1 STL geometry, for browsers without WebGL.
   Display-only vertex clustering is precomputed in g1-preview.json. Kinematics
   still use the original URDF and measured joint values. No posed illustration. */
import * as THREE from 'three';
import {SkeletonView} from './robot-fallback.js';
import {recordedGhosts,poseLinkVisible} from './trajectory-data.mjs';

import {prepareMesh,surfaceTone,focusLink} from './viewer/mesh-preview.mjs';
import {RasterSurface} from './viewer/software-raster.mjs';
import {trailRange,latestPoint,trailVisible} from './trajectory-data.mjs';

export class SolidCanvasView extends SkeletonView {
  async mount() {
    this.host.innerHTML='<div class="empty"><span class="spinner"></span>加载 G1 实体模型…</div>';
    const response=await fetch('/viewer/g1-preview.json');if(!response.ok)throw Error('G1 实体预览资源不可读取');
    const preview=await response.json();if(this.disposed)return;
    this.meshes=new Map(Object.entries(preview.meshes).map(([name,mesh])=>[name,{detail:prepareMesh(mesh.detail),ghost:prepareMesh(mesh.ghost)}]));
    this.visuals=[];this.colorCache=new Map();
    for(const link of this.desc.links)for(const visual of link.visuals) {
      if(visual.kind!=='mesh')continue;
      if(!this.meshes.has(visual.file))throw Error('实体预览缺少 '+visual.file);
      const origin=new THREE.Matrix4().compose(new THREE.Vector3(...visual.xyz),new THREE.Quaternion().setFromEuler(new THREE.Euler(...visual.rpy,'ZYX')),new THREE.Vector3(...(visual.scale||[1,1,1])));
      const color=surfaceTone(visual.color);
      this.visuals.push({name:link.name,link:this.links.get(link.name),origin,mesh:this.meshes.get(visual.file),color});
    }
    // Fix the visual ground offset once using the first recorded pose.
    this.root.position.z=0;this.positions('state.q');let min=Infinity,max=-Infinity;
    for(const visual of this.visuals) {
      const m=new THREE.Matrix4().multiplyMatrices(visual.link.matrixWorld,visual.origin).elements,v=visual.mesh.detail.vertices;
      for(let i=0;i<v.length;i+=3){const z=m[2]*v[i]+m[6]*v[i+1]+m[10]*v[i+2]+m[14];min=Math.min(min,z);max=Math.max(max,z);}
    }
    this.root.position.z=Number.isFinite(min)?-min:.8;
    this.centerZ=Number.isFinite(max)?(max+min)/2+this.root.position.z:.72;
    this.modelHeight=Number.isFinite(max)?(max-min)*1.22:1.7;
    this.world=new THREE.Matrix4();this.normal=new THREE.Matrix3();
    this.target=new THREE.Vector3(0,0,this.centerZ);this.focusScope='all';
    this.surface=new RasterSurface();this.meshCanvas=document.createElement('canvas');this.meshContext=this.meshCanvas.getContext('2d');
    this.renderCount=0;this.renderedFaces=0;
    await super.mount();
    this.canvas.dataset.renderer='solid-canvas';
    this.canvas.setAttribute('aria-label','G1 实体模型，兼容渲染；拖动旋转，滚轮缩放，点击轨迹或残影定位');
  }
  draw() {
    if(this.disposed||this.drawRequest||!this.visible)return;
    this.drawRequest=requestAnimationFrame(()=>{this.drawRequest=null;if(this.visible)this.renderMesh();});
  }
  resize() {
    super.resize();
    if(!this.surface||!this.w||!this.h)return;
    const scale=Math.min(1.65,Math.sqrt(1_100_000/(this.w*this.h)));
    this.surface.resize(this.w*scale,this.h*scale);this.rasterScale=this.surface.width/this.w;
    this.meshCanvas.width=this.surface.width;this.meshCanvas.height=this.surface.height;
    this.imageData=new ImageData(this.surface.pixels,this.surface.width,this.surface.height);
  }
  project(p) {
    const target=this.target||{x:0,y:0,z:this.centerZ??.72},dx=p.x-target.x,dy=p.y-target.y;
    const x=Math.cos(this.yaw)*dx-Math.sin(this.yaw)*dy,y=Math.sin(this.yaw)*dx+Math.cos(this.yaw)*dy;
    const z=p.z-target.z,scale=Math.min(this.w/2.1,this.h/(this.modelHeight||1.7))*this.zoom;
    return [this.w/2+x*scale,this.h*.51-(z*Math.cos(this.elevation)-y*Math.sin(this.elevation))*scale];
  }
  setFocus(scope='all') {this.focusScope=scope;this.fitView();}
  setView(preset='iso') {super.setView(preset);if(this.visuals)this.fitView();}
  fitView() {
    if(!this.visuals||!this.w)return;
    this.positions(this.opts.source);
    const box=new THREE.Box3(),p=new THREE.Vector3();
    for(const visual of this.visuals) {
      if(!focusLink(visual.name,this.focusScope))continue;
      const matrix=new THREE.Matrix4().multiplyMatrices(visual.link.matrixWorld,visual.origin),v=visual.mesh.detail.vertices;
      for(let i=0;i<v.length;i+=3)box.expandByPoint(p.set(v[i],v[i+1],v[i+2]).applyMatrix4(matrix));
    }
    for(const track of this.trajectories?.tracks||[]) {
      if(!trailVisible(track,this.opts)||(this.focusScope==='upper'&&track.part==='foot')||(this.focusScope==='lower'&&track.part==='wrist'))continue;
      for(const segment of track.segments){const [a,b]=trailRange(segment.times_s,this.time,this.opts.trailWindow);for(let i=a;i<b;i++){const v=segment.positions[i];box.expandByPoint(p.set(v[0],v[1],v[2]+this.root.position.z));}}
    }
    if(box.isEmpty())return;
    box.getCenter(this.target);this.zoom=1;
    let ex=1,ey=1;
    for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){
      const [px,py]=this.project(p.set(x,y,z));ex=Math.max(ex,Math.abs(px-this.w/2));ey=Math.max(ey,Math.abs(py-this.h*.51));
    }
    this.zoom=Math.max(.35,Math.min(4,(this.w/2-40)/ex,(this.h/2-60)/ey));this.draw();
  }
  // Preserve hard material edges, smooth curved shell lighting.
  cornerLight(n,x,y,z,sy,cy,ce,se) {
    const nx=n[0]*x+n[3]*y+n[6]*z,ny=n[1]*x+n[4]*y+n[7]*z,nz=n[2]*x+n[5]*y+n[8]*z;
    const key=Math.max(0,nx*.38-ny*.5+nz*.77),fill=Math.max(0,-nx*.64+ny*.43+nz*.64);
    const facing=Math.max(0,-((sy*nx+cy*ny)*ce-nz*se));
    return .42+.57*key+.18*fill+.07*(1-facing)**3;
  }
  rasterBody(key,{ghost=false,color=null,alpha=1,scope='all',owner=-1}={}) {
    this.positions(key);
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),ce=Math.cos(this.elevation),se=Math.sin(this.elevation),rs=this.rasterScale;
    const scale=Math.min(this.w/2.1,this.h/this.modelHeight)*this.zoom,target=this.target;
    for(const visual of this.visuals) {
      if(!poseLinkVisible(visual.name,scope))continue;
      const mesh=ghost?visual.mesh.ghost:visual.mesh.detail;
      this.world.multiplyMatrices(visual.link.matrixWorld,visual.origin);this.normal.getNormalMatrix(this.world);
      const m=this.world.elements,n=this.normal.elements,v=mesh.vertices,projected=mesh.projected;
      for(let i=0;i<v.length;i+=3) {
        const x=m[0]*v[i]+m[4]*v[i+1]+m[8]*v[i+2]+m[12]-target.x,y=m[1]*v[i]+m[5]*v[i+1]+m[9]*v[i+2]+m[13]-target.y;
        const z=m[2]*v[i]+m[6]*v[i+1]+m[10]*v[i+2]+m[14]-target.z,xr=cy*x-sy*y,yr=sy*x+cy*y;
        projected[i]=(this.w/2+xr*scale)*rs;projected[i+1]=(this.h*.51+(yr*se-z*ce)*scale)*rs;projected[i+2]=yr*ce-z*se;
      }
      const f=mesh.faces,ns=mesh.normals,shade=mesh.shading,rgb=color||visual.color;
      for(let i=0;i<f.length;i+=3) {
        const nx=n[0]*ns[i]+n[3]*ns[i+1]+n[6]*ns[i+2],ny=n[1]*ns[i]+n[4]*ns[i+1]+n[7]*ns[i+2],nz=n[2]*ns[i]+n[5]*ns[i+1]+n[8]*ns[i+2];
        if((sy*nx+cy*ny)*ce-nz*se>=0)continue;
        const a=f[i]*3,b=f[i+1]*3,c=f[i+2]*3,j=i*3;
        const la=ghost?1:this.cornerLight(n,shade[j],shade[j+1],shade[j+2],sy,cy,ce,se);
        const lb=ghost?1:this.cornerLight(n,shade[j+3],shade[j+4],shade[j+5],sy,cy,ce,se);
        const lc=ghost?1:this.cornerLight(n,shade[j+6],shade[j+7],shade[j+8],sy,cy,ce,se);
        this.surface.triangle(projected[a],projected[a+1],projected[a+2],projected[b],projected[b+1],projected[b+2],projected[c],projected[c+1],projected[c+2],...rgb,la,lb,lc,alpha,owner);
        this.renderedFaces++;
      }
    }
  }
  trails() {
    const c=this.ctx;
    for(const track of this.trajectories?.tracks||[]) {
      if(!trailVisible(track,this.opts))continue;
      const target=track.source==='action.q',color=track.side==='left'?'#cce578':'#81c6d6';
      c.strokeStyle=color;c.fillStyle=color;c.lineWidth=target?.9:1.25;c.globalAlpha=target?.25:.65;c.setLineDash(target?[4,5]:[]);
      for(const segment of track.segments) {
        const [start,end]=trailRange(segment.times_s,this.time,this.opts.trailWindow);c.beginPath();let previousDot=null;
        for(let i=start;i<end;i++) {
          const p=segment.positions[i],world=new THREE.Vector3(p[0],p[1],p[2]+this.root.position.z),[x,y]=this.project(world);
          if(i===start)c.moveTo(x,y);else c.lineTo(x,y);this.pickPoints.push({x,y,time:segment.times_s[i]});
        }
        c.stroke();c.setLineDash([]);
        for(let i=start;i<end;i++) {
          const p=segment.positions[i],[x,y]=this.project(new THREE.Vector3(p[0],p[1],p[2]+this.root.position.z));
          if(previousDot&&(Math.hypot(x-previousDot.x,y-previousDot.y)<7||segment.times_s[i]-previousDot.time<.12))continue;
          c.beginPath();c.arc(x,y,target?1:1.65,0,Math.PI*2);c.fill();previousDot={x,y,time:segment.times_s[i]};
        }
        c.setLineDash(target?[4,5]:[]);
      }
      c.setLineDash([]);c.globalAlpha=1;
      const point=latestPoint(track,this.time,this.trajectories.max_gap_s);
      if(point) {
        const p=point.position,[x,y]=this.project(new THREE.Vector3(p[0],p[1],p[2]+this.root.position.z));
        c.beginPath();c.arc(x,y,target?4:5,0,Math.PI*2);c.lineWidth=1.5;if(target)c.stroke();else{c.fill();c.strokeStyle='#f2f5e7';c.lineWidth=1;c.stroke();}
      }
    }
    c.globalAlpha=1;c.setLineDash([]);
  }
  pick(e) {
    const rect=this.canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
    const px=Math.floor(x*this.rasterScale),py=Math.floor(y*this.rasterScale),owner=this.surface.owner[py*this.surface.width+px];
    if(owner>=0&&this.ghostTimes[owner]!=null){this.onSeek?.(this.ghostTimes[owner]);return;}
    super.pick(e);
  }
  renderMesh() {
    if(!this.ctx||!this.w||!this.visuals||!this.imageData||this.disposed||!this.visible)return;
    const started=performance.now(),c=this.ctx,css=getComputedStyle(document.documentElement),tone=k=>css.getPropertyValue(k).trim();
    c.clearRect(0,0,this.w,this.h);c.fillStyle=tone('--surface-1');c.fillRect(0,0,this.w,this.h);
    const dark=document.documentElement.dataset.theme==='dark';
    const ambient=c.createRadialGradient(this.w*.5,this.h*.4,20,this.w*.5,this.h*.5,this.w*.65);
    ambient.addColorStop(0,dark?'#ffffff05':'#ffffff40');ambient.addColorStop(1,'#00000000');c.fillStyle=ambient;c.fillRect(0,0,this.w,this.h);
    if(this.opts.showGrid) {
      c.globalAlpha=.65;for(let i=-5;i<=5;i++){this.line(new THREE.Vector3(i*.25,-1.25,0),new THREE.Vector3(i*.25,1.25,0),tone('--grid'));this.line(new THREE.Vector3(-1.25,i*.25,0),new THREE.Vector3(1.25,i*.25,0),tone('--grid'));}c.globalAlpha=1;
      const [x,y]=this.project(new THREE.Vector3(0,0,0)),radius=Math.min(this.w,this.h)*.2*this.zoom;
      c.save();c.translate(x,y);c.scale(1,.32);const shadow=c.createRadialGradient(0,0,0,0,0,radius);shadow.addColorStop(0,dark?'#00000055':'#42465033');shadow.addColorStop(1,'#00000000');c.fillStyle=shadow;c.beginPath();c.arc(0,0,radius,0,Math.PI*2);c.fill();c.restore();
    }
    this.surface.clear();this.pickPoints=[];this.renderedFaces=0;this.ghostTimes=[];
    this.rasterBody(this.opts.source);
    const poses=recordedGhosts(this.trajectories?.poses,this.time,this.opts);
    for(const pose of poses) {
      const window=pose.direction<0?this.opts.pastWindow:this.opts.futureWindow;
      const fade=.6+.4*(1-Math.min(1,Math.abs(this.time-pose.time)/window));
      const owner=this.ghostTimes.push(pose.time)-1;
      this.rasterBody(pose.q,{ghost:true,color:pose.direction<0?[163,144,214]:[219,189,126],alpha:this.opts.poseOpacity*fade,scope:this.opts.poseScope,owner});
    }
    if(this.opts.showGhost)this.rasterBody(this.opts.ghost,{ghost:true,color:[112,190,205],alpha:.23,owner:this.ghostTimes.length});
    this.meshContext.putImageData(this.imageData,0,0);c.imageSmoothingEnabled=true;c.drawImage(this.meshCanvas,0,0,this.w,this.h);
    this.positions(this.opts.source);this.jointFrames();this.trails();
    this.canvas.dataset.renderMs=(performance.now()-started).toFixed(2);this.canvas.dataset.renderedFaces=String(this.renderedFaces);this.canvas.dataset.renderCount=String(++this.renderCount);
  }
  destroy() {this.disposed=true;cancelAnimationFrame(this.drawRequest);super.destroy();}
}
