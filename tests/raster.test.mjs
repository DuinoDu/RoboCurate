import test from 'node:test';
import assert from 'node:assert/strict';
import {RasterSurface} from '../static/viewer/software-raster.mjs';
import {prepareMesh,surfaceTone} from '../static/viewer/mesh-preview.mjs';
const pixel=(r,x,y)=>[...r.pixels.slice((y*r.width+x)*4,(y*r.width+x)*4+4)];
function tri(r,z,color,opacity=1,owner=-1){r.triangle(0,0,z,8,0,z,0,8,z,...color,1,1,1,opacity,owner);}
test('nearest opaque surface wins independently of triangle order',()=>{
 const r=new RasterSurface(8,8);tri(r,1,[200,200,200]);tri(r,2,[255,0,0]);assert.deepEqual(pixel(r,1,1),[200,200,200,255]);
 r.clear();tri(r,2,[255,0,0]);tri(r,1,[200,200,200]);assert.deepEqual(pixel(r,1,1),[200,200,200,255]);
});
test('shared edges cover a quad without transparent seams',()=>{
 const r=new RasterSurface(8,8);tri(r,1,[240,240,240]);r.triangle(8,0,1,8,8,1,0,8,1,240,240,240);
 assert.ok([...r.pixels].filter((_,i)=>i%4===3).every(a=>a===255));
});
test('recorded ghosts do not tint the current robot and do not accumulate per-face opacity',()=>{
 const r=new RasterSurface(8,8);tri(r,1,[230,230,230]);tri(r,-2,[100,0,255],.2,2);assert.deepEqual(pixel(r,1,1),[230,230,230,255]);
 r.clear();tri(r,1,[100,0,255],.2,2);tri(r,0,[100,0,255],.2,3);assert.equal(pixel(r,1,1)[3],51);assert.equal(r.owner[9],3);
});
test('clipped and degenerate triangles stay inside the buffer',()=>{
 const r=new RasterSurface(8,8);r.triangle(-10,-10,1,20,-10,1,-10,20,1,50,70,90);tri(r,NaN,[1,2,3]);assert.equal(r.pixels.length,256);assert.deepEqual(pixel(r,0,0),[50,70,90,255]);
});
test('lighting interpolates over the face',()=>{
 const r=new RasterSurface(8,8);r.triangle(0,0,1,8,0,1,0,8,1,200,200,200,.3,1,.3);assert.ok(pixel(r,5,1)[0]>pixel(r,1,1)[0]);
});
test('preview validation catches nested NumPy indices and normal smoothing preserves planar normals',()=>{
 assert.throws(()=>prepareMesh({vertices:[0,0,0],faces:[[0,0,0]]}));
 const m=prepareMesh({vertices:[0,0,0,1,0,0,0,1,0],faces:[0,1,2]});
 assert.deepEqual([...m.shading],[0,0,1,0,0,1,0,0,1]);assert.ok(surfaceTone([.7,.7,.7])[0]>surfaceTone([.2,.2,.2])[0]+100);
});
