export function surfaceTone(color) {
  // Preserve the supplied material split while separating the light shell
  // from dark joints, head and feet against a dark inspection background.
  return color.slice(0,3).map((v,i)=>Math.max(28,Math.min(238,(52+(v-.2)*336)*[.97,.99,1.02][i])));
}

export function prepareMesh(mesh,creaseAngle=55) {
  if(!Array.isArray(mesh.vertices)||!Array.isArray(mesh.faces)||mesh.vertices.length%3||mesh.faces.length%3||mesh.vertices.some(v=>!Number.isFinite(v))||
     mesh.faces.some(i=>!Number.isInteger(i)||i<0||i>=mesh.vertices.length/3))throw Error('Invalid preview mesh indices');
  const vertices=new Float32Array(mesh.vertices),faces=new Uint32Array(mesh.faces),normals=new Float32Array(faces.length);
  const areas=new Float32Array(faces.length/3),adjacent=Array.from({length:vertices.length/3},()=>[]);
  for(let i=0;i<faces.length;i+=3) {
    const a=faces[i]*3,b=faces[i+1]*3,c=faces[i+2]*3;
    const x1=vertices[b]-vertices[a],y1=vertices[b+1]-vertices[a+1],z1=vertices[b+2]-vertices[a+2];
    const x2=vertices[c]-vertices[a],y2=vertices[c+1]-vertices[a+1],z2=vertices[c+2]-vertices[a+2];
    const nx=y1*z2-z1*y2,ny=z1*x2-x1*z2,nz=x1*y2-y1*x2,n=Math.hypot(nx,ny,nz)||1;
    normals.set([nx/n,ny/n,nz/n],i);areas[i/3]=n;
    for(let j=0;j<3;j++)adjacent[faces[i+j]].push(i);
  }
  const shading=new Float32Array(faces.length*3),cos=Math.cos(creaseAngle*Math.PI/180);
  for(let i=0;i<faces.length;i+=3)for(let corner=0;corner<3;corner++) {
    let x=0,y=0,z=0;
    for(const j of adjacent[faces[i+corner]]) {
      if(normals[i]*normals[j]+normals[i+1]*normals[j+1]+normals[i+2]*normals[j+2]<cos)continue;
      const area=areas[j/3];x+=normals[j]*area;y+=normals[j+1]*area;z+=normals[j+2]*area;
    }
    const length=Math.hypot(x,y,z)||1;shading.set([x/length,y/length,z/length],i*3+corner*3);
  }
  return {vertices,faces,normals,shading,projected:new Float32Array(vertices.length)};
}

export function focusLink(name,scope) {
  if(scope==='upper')return /shoulder|elbow|wrist|hand|torso|head/.test(name);
  if(scope==='lower')return /pelvis|hip|knee|ankle|foot/.test(name);
  return true;
}
