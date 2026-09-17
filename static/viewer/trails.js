import * as THREE from 'three';
import {trailRange,latestPoint,trailVisible} from '../trajectory-data.mjs';

export class TrailLayer {
  constructor(scene,payload,height) {
    this.group=new THREE.Group();this.group.position.z=height;scene.add(this.group);
    this.payload=payload;this.entries=[];
    for(const track of payload.tracks) {
      const target=track.source==='action.q', color=track.side==='left'?0xd4ed75:0x71d5e1;
      const material=target?new THREE.LineDashedMaterial({color,dashSize:.02,gapSize:.018,transparent:true,opacity:.28}):new THREE.LineBasicMaterial({color,transparent:true,opacity:.72});
      const protectCurrent={stencilWrite:true,stencilRef:1,stencilWriteMask:0,stencilFunc:THREE.NotEqualStencilFunc,stencilFail:THREE.KeepStencilOp,stencilZFail:THREE.KeepStencilOp,stencilZPass:THREE.KeepStencilOp};
      Object.assign(material,protectCurrent);
      const segments=track.segments.map(segment=>{
        const geometry=new THREE.BufferGeometry().setFromPoints(segment.positions.map(p=>new THREE.Vector3(...p)));
        const line=new THREE.Line(geometry,material);line.computeLineDistances();this.group.add(line);
        const indices=[];
        for(let i=0;i<segment.times_s.length;i++) {
          const prev=indices.at(-1);
          if(prev!=null && i!==segment.times_s.length-1 &&
            (segment.times_s[i]-segment.times_s[prev]<.12 || Math.hypot(...segment.positions[i].map((v,j)=>v-segment.positions[prev][j]))<.006))continue;
          indices.push(i);
        }
        const dotGeometry=new THREE.BufferGeometry().setFromPoints(indices.map(i=>new THREE.Vector3(...segment.positions[i])));
        const dots=new THREE.Points(dotGeometry,new THREE.PointsMaterial({color,size:target?.005:.009,transparent:true,opacity:target?.25:.75}));this.group.add(dots);
        Object.assign(dots.material,protectCurrent);
        return {segment,line,dots,indices,dotTimes:indices.map(i=>segment.times_s[i])};
      });
      const marker=new THREE.Mesh(new THREE.SphereGeometry(target?.014:.021,12,8),new THREE.MeshBasicMaterial({color,wireframe:target}));
      this.group.add(marker);this.entries.push({track,segments,marker,material});
    }
  }
  update(time,opts) {
    for(const {track,segments,marker} of this.entries) {
      const visible=trailVisible(track,opts);
      for(const {segment,line,dots,dotTimes} of segments) {
        const [start,end]=trailRange(segment.times_s,time,opts.trailWindow);
        line.geometry.setDrawRange(start,end-start);line.visible=visible && end-start>1;
        const [da,db]=trailRange(dotTimes,time,opts.trailWindow);dots.geometry.setDrawRange(da,db-da);
        dots.visible=visible && db>da;
      }
      const point=visible && latestPoint(track,time,this.payload.max_gap_s);
      marker.visible=!!point;if(point) marker.position.fromArray(point.position);
    }
  }
  pick(camera,ndc,maxDistance=Infinity) {
    const ray=new THREE.Raycaster();ray.params.Points.threshold=.018;ray.setFromCamera(ndc,camera);
    const segments=this.entries.flatMap(e=>e.segments).filter(s=>s.dots.visible);
    for(const hit of ray.intersectObjects(segments.map(s=>s.dots))) {
      if(hit.distance>maxDistance+.015)continue;
      const s=segments.find(s=>s.dots===hit.object),r=s.dots.geometry.drawRange;
      if(hit.index>=r.start && hit.index<r.start+r.count)return s.segment.times_s[s.indices[hit.index]];
    }
    return null;
  }
  destroy() {
    const geometries=new Set(),materials=new Set();
    this.group.traverse(node=>{if(node.geometry)geometries.add(node.geometry);if(node.material)materials.add(node.material)});
    for(const g of geometries)g.dispose();for(const m of materials)m.dispose();
    this.group.removeFromParent();
  }
}
