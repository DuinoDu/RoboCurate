"""Build low-detail display meshes from the supplied G1 STL files.

Vertex clustering keeps averaged source vertices and original face winding.
These meshes are ONLY a CPU display fallback; URDF kinematics and source data
are not altered. Run from the project root after changing the robot assets.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from urdf import parse_urdf

ROOT=Path(__file__).resolve().parents[1]

def simplify(path,cell):
    dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    count=int.from_bytes(path.read_bytes()[80:84],'little')
    records=np.fromfile(path,dtype=dtype,count=count,offset=84)
    vertices=records['v'].reshape(-1,3).astype(np.float64)
    _,inverse=np.unique(np.rint(vertices/cell).astype(np.int64),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    v=np.column_stack([np.bincount(inverse,weights=vertices[:,i])/counts for i in range(3)])
    f=inverse.reshape(-1,3)
    keep=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])
    f=f[keep]
    _,unique=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(unique)]
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    f=f[np.linalg.norm(cross,axis=1)>1e-10]
    used,inv=np.unique(f,return_inverse=True)
    return dict(vertices=v[used].round(6).reshape(-1).tolist(),faces=inv.reshape(-1).astype(int).tolist(),original_faces=count)

def build():
    desc=parse_urdf(ROOT/'robot/g1_29dof_rev_1_0.urdf');meshes={}
    for name in desc['mesh_files']:
        path=Path(desc['mesh_dir'])/name
        meshes[name]=dict(detail=simplify(path,.012),ghost=simplify(path,.045),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(version=2,method='source_vertex_clustering',units='m',detail_cell_m=.012,ghost_cell_m=.045,meshes=meshes)
    output=ROOT/'static/viewer/g1-preview.json';output.write_text(json.dumps(result,separators=(',',':')))
    print(dict(path=str(output),bytes=output.stat().st_size,detail_faces=sum(len(m['detail']['faces'])//3 for m in meshes.values()),ghost_faces=sum(len(m['ghost']['faces'])//3 for m in meshes.values())))

if __name__=='__main__':build()
