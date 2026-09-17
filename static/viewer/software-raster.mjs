// Orthographic, depth-tested display rasterizer. It does not change kinematics.
// Opaque current geometry owns its pixels; overlapping time references have
// one coverage value rather than accumulating alpha once per triangle.
export class RasterSurface {
  constructor(width=1,height=1) {this.resize(width,height);}
  resize(width,height) {
    width=Math.max(1,Math.ceil(width));height=Math.max(1,Math.ceil(height));
    if(width===this.width&&height===this.height)return;
    this.width=width;this.height=height;
    const count=width*height;
    this.pixels=new Uint8ClampedArray(count*4);
    this.depth=new Float32Array(count);this.ghostDepth=new Float32Array(count);
    this.owner=new Int16Array(count);this.clear();
  }
  clear() {this.pixels.fill(0);this.depth.fill(Infinity);this.ghostDepth.fill(Infinity);this.owner.fill(-1);}
  triangle(ax,ay,az,bx,by,bz,cx,cy,cz,r,g,b,la=1,lb=1,lc=1,opacity=1,owner=-1) {
    if(!Number.isFinite(az+bz+cz))return;
    const area=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    if(!Number.isFinite(area)||Math.abs(area)<1e-9)return;
    const inv=1/area,w=this.width;
    const x0=Math.max(0,Math.ceil(Math.min(ax,bx,cx)-.5)),x1=Math.min(w-1,Math.floor(Math.max(ax,bx,cx)-.5));
    const y0=Math.max(0,Math.ceil(Math.min(ay,by,cy)-.5)),y1=Math.min(this.height-1,Math.floor(Math.max(ay,by,cy)-.5));
    if(x0>x1||y0>y1)return;
    const da=(by-cy)*inv,db=(cy-ay)*inv,dc=(ay-by)*inv;
    const ea=(cx-bx)*inv,eb=(ax-cx)*inv,ec=(bx-ax)*inv;
    let rowA=((bx-x0-.5)*(cy-y0-.5)-(by-y0-.5)*(cx-x0-.5))*inv;
    let rowB=((cx-x0-.5)*(ay-y0-.5)-(cy-y0-.5)*(ax-x0-.5))*inv;
    let rowC=1-rowA-rowB;
    const ghost=owner>=0,depth=ghost?this.ghostDepth:this.depth,pixels=this.pixels;
    const alpha=Math.round(Math.max(0,Math.min(1,opacity))*255);
    for(let y=y0;y<=y1;y++) {
      let a=rowA,bb=rowB,c=rowC,index=y*w+x0;
      for(let x=x0;x<=x1;x++,index++,a+=da,bb+=db,c+=dc) {
        if(a< -1e-7||bb< -1e-7||c< -1e-7)continue;
        if(ghost&&this.depth[index]!==Infinity)continue;
        const z=a*az+bb*bz+c*cz;
        if(z>=depth[index])continue;
        depth[index]=z;
        const light=Math.max(0,Math.min(1.18,a*la+bb*lb+c*lc)),offset=index*4;
        pixels[offset]=r*light;pixels[offset+1]=g*light;pixels[offset+2]=b*light;pixels[offset+3]=alpha;
        this.owner[index]=owner;
      }
      rowA+=ea;rowB+=eb;rowC+=ec;
    }
  }
}
