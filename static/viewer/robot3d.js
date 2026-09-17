/* 3D view of the G1, posed from the episode.
 *
 * The scene is deliberately *joint-space*: the pelvis stays at the origin and
 * the shared trajectory view keeps IMU rotation disabled. The raw episode carries
 * no odometry — there is no measured base position anywhere in it — so any
 * translation drawn here would be invented. `ref.root_pos` does exist, but it
 * lives in the retarget task frame rather than the measured robot's frame, so
 * overlaying the two would imply an alignment the data does not support. It
 * stays on its own chart in Overview instead.
 *
 * That makes the comparison this view *can* make an honest one: two poses in
 * the same root frame, so the difference you see is joint tracking and nothing
 * else.
 */

import * as THREE from "three";
import { STLLoader } from "./vendor/STLLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { TrailLayer } from './trails.js';
import {recordedGhosts,poseLinkVisible} from '../trajectory-data.mjs';
import {surfaceTone,focusLink} from './mesh-preview.mjs';

/** Pose sources, in the order they appear in the picker. */
export const POSE_SOURCES = [
  { id: "state.q", label: "Measured", note: "what the robot did" },
  { id: "action.q", label: "Control target", note: "recorded target joints" },
  { id: "ref.dof", label: "Retarget ref", note: "what the operator drove" },
];

/* Eye positions in the URDF's own frame: +X is the direction the robot faces,
 * +Y is its left, +Z is up. So the *front* view sits out along +X looking back
 * at the chest, and the *side* view sits off to -Y looking at the profile.
 * (These two were swapped once; the labels are only meaningful against the
 * axis convention, so it is spelled out here rather than left to memory.) */
const CAMERA_PRESETS = {
  iso: [2.0, -2.0, 1.5],
  front: [3.0, 0.0, 1.0],
  side: [0.0, -3.0, 1.0],
  top: [0.01, -0.01, 3.6],
};

export class RobotView {
  constructor(host, data, store, desc) {
    this.host = host;
    this.data = data;
    this.store = store;
    this.desc = desc;
    this.dirty = true;
    // The view now lives in the always-on split beside the camera, so it
    // starts visible. The flag stays because the render loop still needs a way
    // to be told the canvas is off-screen and skip the GPU work.
    this.visible = true;
    this.jointNodes = new Map();
    this.linkMaterials = [];
    this.ghost = null;
    this.opts = {
      source: "state.q",
      ghost: "action.q",
      showGhost: false,
      applyImu: false,
      colorByError: false,
      showTrails:true, showTargets:false, trailSide:'both', trailWindow:2,
      showPoseTrail:false,showPast:true,showFuture:false,pastWindow:2,futureWindow:2,poseCount:3,poseOpacity:.18,poseScope:'all',showGrid:true,showJointFrames:false,
    };
    this.errorScale = 0.1;
    this.focusScope='all';
  }

  /** Fetch the description; a 404 means "no 3D tab", not a failure. */
  static async describe() {
    const res = await fetch("/api/robot");
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      return { ok: false, reason: detail.detail || `HTTP ${res.status}` };
    }
    return { ok: true, desc: await res.json() };
  }

  async mount(onProgress = () => {}) {
    this._buildScene();
    const geoms = await this._loadMeshes(onProgress);
    if(this.destroyed) { for(const g of geoms.values())g?.dispose(); return; }
    this.meshGeometries=geoms;
    this.solid = this._buildRobot(geoms, false);
    this.scene.add(this.solid.root);
    this.jointFrames=[];
    for(const [name,entry] of this.solid.links)if(/wrist_yaw_link|ankle_roll_link/.test(name)) {
      const axes=new THREE.AxesHelper(.10);axes.visible=false;entry.group.add(axes);this.jointFrames.push({name,axes});
    }
    this.ghost = this._buildRobot(geoms, true);
    this.ghost.root.visible = false;
    this.scene.add(this.ghost.root);

    this._groundOffset();
    this.errorScale = this._errorScale();
    this.host.dataset.meshFiles=String(geoms.size);
    this.host.dataset.meshTriangles=String([...geoms.values()].reduce((n,g)=>n+(g?.attributes.position.count||0)/3,0));
    this._loop();
  }

  /* ------------------------------------------------------------- scene -- */

  _buildScene() {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "robot-canvas";
    this.canvas.dataset.renderer='webgl';
    this.canvas.setAttribute('aria-label','机器人与双腕轨迹；拖动旋转，滚轮缩放');
    this.host.appendChild(this.canvas);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      stencil: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.05;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.05, 100);
    // URDF is Z-up; three.js defaults to Y-up. Telling the camera and the
    // controls about it is enough — the model itself is never rotated, so
    // published joint axes keep meaning what the URDF says they mean.
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...CAMERA_PRESETS.iso);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0, 0, 0.75);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance=.3;this.controls.maxDistance=12;
    this.controls.addEventListener("change", () => { this.dirty = true; });
    let start=null;
    this.canvas.addEventListener('pointerdown',e=>{start=[e.clientX,e.clientY]});
    this.canvas.addEventListener('pointerup',e=>{
      if(start&&Math.hypot(e.clientX-start[0],e.clientY-start[1])<4) {
        const r=this.canvas.getBoundingClientRect();
        const ndc=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
        const ray=new THREE.Raycaster();ray.setFromCamera(ndc,this.camera);
        const solidHit=ray.intersectObject(this.solid.root,true).find(h=>h.object.isMesh&&h.object.visible);
        if(solidHit){start=null;return;}
        let time=this.trails?.pick(this.camera,ndc,solidHit?.distance??Infinity);
        // Recorded references are stencil-masked under the current body.
        // Clicking that opaque body must not select an invisible old pose.
        if(time==null && !solidHit) {
          const roots=(this.poseGhosts||[]).filter(g=>g.root.visible).map(g=>g.root);
          const hit=ray.intersectObjects(roots,true).find(h=>h.object.visible);
          if(hit){let node=hit.object;while(node && node.userData.recordedTime==null)node=node.parent;time=node?.userData.recordedTime;}
        }
        if(time!=null)this.onSeek?.(time);
      }
      start=null;
    });

    this.scene.add(new THREE.HemisphereLight(0xf3f5ff, 0x353b49, 1.6));
    const key = new THREE.DirectionalLight(0xfffaf2, 2.4);
    key.position.set(3, -4, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xd5e2ff, 1.1);
    fill.position.set(-3, 2, 3);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(6, 24);
    this.grid.rotation.x = Math.PI / 2; // GridHelper is XZ; the floor is XY
    this.scene.add(this.grid);
    this.grid.material.transparent=true;this.grid.material.opacity=.35;
    const shadowCanvas=document.createElement('canvas');shadowCanvas.width=shadowCanvas.height=128;
    const shadowContext=shadowCanvas.getContext('2d'),gradient=shadowContext.createRadialGradient(64,64,2,64,64,64);
    gradient.addColorStop(0,'rgba(0,0,0,.4)');gradient.addColorStop(1,'rgba(0,0,0,0)');shadowContext.fillStyle=gradient;shadowContext.fillRect(0,0,128,128);
    this.floorShadow=new THREE.Mesh(new THREE.PlaneGeometry(1.15,.85),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(shadowCanvas),transparent:true,depthWrite:false}));
    this.floorShadow.position.z=.002;this.scene.add(this.floorShadow);

    this.applyTheme();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.host);
  }

  /** Canvas colours are baked at draw time, so a theme switch must repaint. */
  applyTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (k) => cs.getPropertyValue(k).trim();
    this.renderer.setClearColor(new THREE.Color(v("--surface-1")), 1);
    const gridColor = new THREE.Color(v("--grid"));
    this.grid.material.color = gridColor;
    this.grid.material.needsUpdate = true;
    this.errorHigh = new THREE.Color(v("--series-2"));
    this.ghostColor = new THREE.Color(v("--blue"));
    if (this.ghost) {
      for (const m of this.ghost.materials) m.color.copy(this.ghostColor);
    }
    this.dirty = true;
  }

  resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /* ------------------------------------------------------------ meshes -- */

  async _loadMeshes(onProgress) {
    const loader = new STLLoader();
    const files = this.desc.mesh_files;
    const out = new Map();
    let done = 0;
    // Sequential rather than parallel: ~19 MB over a local socket saturates
    // instantly either way, and one at a time gives a truthful progress count.
    for (const name of files) {
      if(this.destroyed) break;
      try {
        const geom = await loader.loadAsync(
          `/api/mesh?f=${encodeURIComponent(name)}`,
        );
        geom.computeVertexNormals();
        out.set(name, geom);
      } catch {
        // A single unreadable mesh costs one link, not the whole robot.
        out.set(name, null);
      }
      done += 1;
      onProgress(`Loading robot meshes… ${done}/${files.length}`);
    }
    if(!this.destroyed && [...out.values()].some(g=>!g)) {
      for(const g of out.values())g?.dispose();
      throw Error('部分原始 STL 未能读取，完整模型尚未加载完成。');
    }
    return out;
  }

  /* ----------------------------------------------------------- kinematics -- */

  _buildRobot(geoms, isGhost) {
    const links = new Map();
    const materials = [];
    for (const link of this.desc.links) {
      const group = new THREE.Group();
      group.name = link.name;
      for (const vis of link.visuals) {
        const geom = vis.kind === "mesh" ? geoms.get(vis.file) : this._prim(vis);
        if (!geom) continue;
        const [r,g,b]=surfaceTone(vis.color);
        const base = new THREE.Color().setRGB(r/255,g/255,b/255,THREE.SRGBColorSpace);
        const mat = new THREE.MeshStandardMaterial({
          color: isGhost ? this.ghostColor : base,
          roughness:.7,metalness:.08,
          transparent: isGhost,
          opacity: isGhost ? 0.22 : 1,
          depthWrite: !isGhost,
          stencilWrite:true,stencilRef:1,
          stencilWriteMask:isGhost?0:0xff,
          stencilFunc:isGhost?THREE.NotEqualStencilFunc:THREE.AlwaysStencilFunc,
          stencilFail:THREE.KeepStencilOp,stencilZFail:THREE.KeepStencilOp,
          stencilZPass:isGhost?THREE.KeepStencilOp:THREE.ReplaceStencilOp,
        });
        mat.userData.base = base.clone();
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.fromArray(vis.xyz);
        mesh.rotation.set(vis.rpy[0], vis.rpy[1], vis.rpy[2], "ZYX");
        if (vis.kind === "mesh" && vis.scale) mesh.scale.fromArray(vis.scale);
        group.add(mesh);
        materials.push(mat);
      }
      links.set(link.name, { group, materials: group.children.map((c) => c.material) });
    }

    const joints = new Map();
    for (const j of this.desc.joints) {
      const parent = links.get(j.parent);
      const child = links.get(j.child);
      if (!parent || !child) continue;
      // Static origin and variable joint value live on separate nodes, so a
      // pose update only ever writes one quaternion per joint.
      const origin = new THREE.Group();
      origin.position.fromArray(j.xyz);
      origin.rotation.set(j.rpy[0], j.rpy[1], j.rpy[2], "ZYX");
      parent.group.add(origin);
      origin.add(child.group);
      if (j.type === "revolute" || j.type === "continuous") {
        joints.set(j.name, {
          node: child.group,
          axis: new THREE.Vector3().fromArray(j.axis).normalize(),
          link: j.child,
        });
      }
    }

    return {
      root: links.get(this.desc.root).group,
      links,
      joints,
      materials,
    };
  }

  _prim(vis) {
    if (vis.kind === "box") return new THREE.BoxGeometry(...vis.size);
    if (vis.kind === "sphere") return new THREE.SphereGeometry(vis.radius, 20, 14);
    if (vis.kind === "cylinder") {
      const g = new THREE.CylinderGeometry(vis.radius, vis.radius, vis.length, 24);
      g.rotateX(Math.PI / 2); // URDF cylinders run along Z, three.js along Y
      return g;
    }
    return null;
  }

  /* -------------------------------------------------------------- posing -- */

  /** Standing height for a root-pinned robot: put the lowest vertex on the
   *  floor once, at the first sample, and hold it. Recomputing per frame would
   *  invent the vertical motion the episode does not record. */
  _groundOffset() {
    this._applyJoints(this.solid, this.opts.source, 0);
    this.solid.root.position.set(0, 0, 0);
    this.solid.root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.solid.root);
    this.standHeight = isFinite(box.min.z) ? -box.min.z : 0.79;
  }

  /** Tracking-error scale: the 99th percentile over the episode, so the ramp
   *  is set by this episode's own spread rather than a guessed constant. */
  _errorScale() {
    const d = this.data;
    if (!d.has("action.q") || !d.has("state.q")) return 0.1;
    const vals = [];
    const step = Math.max(1, Math.floor(d.samples / 400));
    for (let j = 0; j < 29; j++) {
      const a = d.series("action.q", j);
      const m = d.series("state.q", j);
      for (let i = 0; i < d.samples; i += step) {
        const e = Math.abs(a[i] - m[i]);
        if (isFinite(e)) vals.push(e);
      }
    }
    if (!vals.length) return 0.1;
    vals.sort((x, y) => x - y);
    return Math.max(vals[Math.floor(vals.length * 0.99)], 0.02);
  }

  _applyJoints(robot, sourceKey, sample) {
    const names = this.data.meta.joints.names;
    for (let i = 0; i < names.length; i++) {
      const node = robot.joints.get(`${names[i]}_joint`);
      if (!node) continue;
      const q = this.data.value(sourceKey, i, sample);
      node.node.quaternion.setFromAxisAngle(node.axis, isFinite(q) ? q : 0);
    }
  }

  _applyRoot(robot, sample) {
    robot.root.position.set(0, 0, this.standHeight ?? 0.79);
    if (this.opts.applyImu && this.data.has("imu.quat")) {
      const w = this.data.value("imu.quat", 0, sample);
      const x = this.data.value("imu.quat", 1, sample);
      const y = this.data.value("imu.quat", 2, sample);
      const z = this.data.value("imu.quat", 3, sample);
      if ([w, x, y, z].every(isFinite)) {
        robot.root.quaternion.set(x, y, z, w).normalize();
        return;
      }
    }
    robot.root.quaternion.identity();
  }

  /** Tint each link toward the ramp's high end by its own joint's error. */
  _applyErrorTint(sample) {
    const on = this.opts.colorByError &&
      this.data.has("action.q") && this.data.has("state.q");
    const names = this.data.meta.joints.names;
    const errOf = new Map();
    if (on) {
      for (let i = 0; i < names.length; i++) {
        const node = this.solid.joints.get(`${names[i]}_joint`);
        if (!node) continue;
        const e = Math.abs(
          this.data.value("action.q", i, sample) -
            this.data.value("state.q", i, sample),
        );
        errOf.set(node.link, isFinite(e) ? e : 0);
      }
    }
    for (const [name, entry] of this.solid.links) {
      const t = on ? Math.min((errOf.get(name) ?? 0) / this.errorScale, 1) : 0;
      for (const mat of entry.materials) {
        mat.color.copy(mat.userData.base);
        if (t > 0) mat.color.lerp(this.errorHigh, t);
      }
    }
  }

  pose(sample) {
    if(this.destroyed || !this.solid)return;
    this.sample=sample;
    this._applyJoints(this.solid, this.opts.source, sample);
    this._applyRoot(this.solid, sample);
    this._applyErrorTint(sample);
    if (this.opts.showGhost) {
      this._applyJoints(this.ghost, this.opts.ghost, sample);
      this._applyRoot(this.ghost, sample);
    }
    this.ghost.root.visible = this.opts.showGhost;
    this.trails?.update(sample/this.data.meta.timeline.rate_hz,this.opts);
    this.dirty = true;
  }

  setTrajectories(payload) {
    this.trails?.destroy();
    this.recordedPoses=payload.poses;
    this.trails=new TrailLayer(this.scene,payload,this.standHeight??.79);
    this.trails.update((this.sample||0)/this.data.meta.timeline.rate_hz,this.opts);
    this.dirty=true;
  }

  setTime(seconds) {
    this.trails?.update(seconds,this.opts);
    this._updateRecordedGhosts(seconds);
    if(this.grid)this.grid.visible=this.opts.showGrid;
    if(this.floorShadow)this.floorShadow.visible=this.opts.showGrid;
    for(const {name,axes} of this.jointFrames||[])axes.visible=this.opts.showJointFrames &&
      (this.opts.trailPart==='all'||(name.includes('wrist')?'wrist':'foot')===this.opts.trailPart);
    this.dirty=true;
  }

  _updateRecordedGhosts(seconds) {
    if(!this.meshGeometries)return;
    const poses=recordedGhosts(this.recordedPoses,seconds,this.opts);
    const key=poses.map(p=>p.time).join(',')+'|'+this.opts.poseScope+'|'+this.opts.poseOpacity;
    if(key===this.recordedKey)return;
    this.recordedKey=key;this.poseGhosts ||= [];
    while(this.poseGhosts.length<poses.length) {
      const ghost=this._buildRobot(this.meshGeometries,true);
      this.scene.add(ghost.root);this.poseGhosts.push(ghost);
    }
    this.poseGhosts.forEach((ghost,i)=>{
      const pose=poses[i];ghost.root.visible=!!pose;if(!pose)return;
      ghost.root.position.set(0,0,this.standHeight??.79);ghost.root.quaternion.identity();
      ghost.root.userData.recordedTime=pose.time;
      this.recordedPoses.names.forEach((name,j)=>{const joint=ghost.joints.get(name+'_joint');if(joint)joint.node.quaternion.setFromAxisAngle(joint.axis,pose.q[j]);});
      for(const [name,entry] of ghost.links)for(const mesh of entry.group.children) {
        if(!mesh.isMesh)continue;
        mesh.visible=poseLinkVisible(name,this.opts.poseScope);
        mesh.material.color.set(pose.direction<0?0xa18ae0:0xe0bd77);
        const window=pose.direction<0?this.opts.pastWindow:this.opts.futureWindow;
        mesh.material.opacity=this.opts.poseOpacity*(.6+.4*(1-Math.min(1,Math.abs(seconds-pose.time)/window)));
      }
    });
  }

  setView(preset) {
    const p = CAMERA_PRESETS[preset] || CAMERA_PRESETS.iso;
    this.camera.position.set(...p);
    this.controls.target.set(0, 0, 0.75);
    this.controls.update();
    this.fitView();
    this.dirty = true;
  }

  setFocus(scope='all') {this.focusScope=scope;this.fitView();}

  fitView() {
    if(!this.solid)return;
    this.solid.root.updateMatrixWorld(true);
    this.scene.updateMatrixWorld(true);
    const box=new THREE.Box3();
    for(const [name,entry] of this.solid.links) {
      if(!focusLink(name,this.focusScope))continue;
      for(const mesh of entry.group.children)if(mesh.isMesh)box.union(new THREE.Box3().setFromObject(mesh));
    }
    for(const {track,segments} of this.trails?.entries||[])for(const {line,dots} of segments) {
      if(!dots.visible)continue;
      if((this.focusScope==='upper'&&track.part==='foot')||(this.focusScope==='lower'&&track.part==='wrist'))continue;
      const a=line.geometry.attributes.position,r=line.geometry.drawRange;
      for(let i=r.start;i<Math.min(a.count,r.start+r.count);i++)box.expandByPoint(new THREE.Vector3().fromBufferAttribute(a,i).applyMatrix4(line.matrixWorld));
    }
    if(box.isEmpty())return;
    const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const fov=this.camera.fov*Math.PI/180;
    const distance=Math.max(size.z/2/Math.tan(fov/2),Math.hypot(size.x,size.y)/2/(Math.tan(fov/2)*this.camera.aspect))*1.28;
    const direction=this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(center);this.camera.position.copy(center).addScaledVector(direction,Math.max(distance,.45));
    this.controls.update();this.dirty=true;
  }

  _loop() {
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      if (!this.visible) return;
      const moved = this.controls.update();
      if (!this.dirty && !moved) return;
      this.dirty = false;
      const started=performance.now();
      this.renderer.render(this.scene, this.camera);
      this.canvas.dataset.renderMs=(performance.now()-started).toFixed(2);
      this.canvas.dataset.renderedFaces=String(this.renderer.info.render.triangles);
    };
    tick();
  }

  destroy() {
    this.destroyed=true;
    cancelAnimationFrame(this._raf);
    this.ro?.disconnect();
    this.controls?.dispose();
    this.trails?.destroy();
    const geometries=new Set(),materials=new Set(),textures=new Set();
    this.scene?.traverse(node=>{if(node.geometry)geometries.add(node.geometry);if(node.material)for(const m of [node.material].flat())materials.add(m)});
    for(const g of geometries)g.dispose();for(const m of materials){if(m.map)textures.add(m.map);m.dispose();}for(const t of textures)t.dispose();
    this.renderer?.dispose();
    this.renderer?.forceContextLoss();
    this.canvas?.remove();
  }
}
