// Diagnostic geometry from the ACTUAL loaded Gaussian parameters.
// Q diag(s^2) Q^T has a k-sigma contour with semi-axis lengths k*s.
// These opaque triangle ellipsoids are a visualization, not the splat renderer.
import * as THREE from 'three';

export class GaussianInspector {
  constructor(group,onSelect) {
    this.group=group; this.onSelect=onSelect; this.source=null;
    this.sample=[]; this.shown=[]; this.selected=null;
    this.sphere=new THREE.SphereGeometry(1,12,8);
    this.material=new THREE.MeshStandardMaterial({roughness:.85,metalness:0});
    this.highlight=new THREE.Mesh(new THREE.SphereGeometry(1,20,12),
      new THREE.MeshBasicMaterial({color:0xffec9c,wireframe:true,depthTest:false}));
    this.highlight.renderOrder=10; this.highlight.visible=false; group.add(this.highlight);
    this.instanceMesh=null; this.points=null;
  }

  setSource(splatMesh) {
    this.source=splatMesh.packedSplats; this.sample=[]; this.selected=null;
    const total=this.source.numSplats;
    // Fixed deterministic sample, spread through the file, for bounded inspection.
    // Large captures can have >1M Gaussians. Rendering still uses all of them.
    const stride=Math.max(1,Math.ceil(total/50000));
    for(let index=0;index<total;index+=stride) {
      const p=this.source.getSplat(index);
      if(p.opacity<.03||!Number.isFinite(p.center.lengthSq())||!Number.isFinite(p.scales.lengthSq())) continue;
      // Spark reuses scratch vectors between getSplat calls. Retain owned copies.
      this.sample.push({index,center:p.center.clone(),scales:p.scales.clone(),
        quaternion:p.quaternion.clone(),color:p.color.clone(),opacity:p.opacity});
    }
    this.highlight.visible=false;
  }

  rebuild({sigma,budget,localOnly,radius,focus,isolate,mode}) {
    if(!this.source) return 0;
    this.sigma=sigma; this.mode=mode;
    const localFocus=this.group.worldToLocal(focus.clone());
    let candidates=this.sample;
    if(isolate&&this.selected) candidates=[this.selected];
    else if(localOnly) candidates=candidates.filter(p=>p.center.distanceTo(localFocus)<=radius);
    const count=Math.min(budget,candidates.length);
    this.shown=Array.from({length:count},(_,i)=>candidates[Math.floor(i*candidates.length/count)]);
    if(this.instanceMesh) {this.group.remove(this.instanceMesh);this.instanceMesh.dispose();}
    if(this.points) {this.group.remove(this.points);this.points.geometry.dispose();this.points.material.dispose();}
    this.instanceMesh=new THREE.InstancedMesh(this.sphere,this.material,count);
    this.instanceMesh.frustumCulled=false;
    const matrix=new THREE.Matrix4(),scale=new THREE.Vector3();
    const positions=new Float32Array(count*3),colors=new Float32Array(count*3);
    for(let i=0;i<count;i++) {
      const p=this.shown[i];
      scale.copy(p.scales).multiplyScalar(sigma);
      matrix.compose(p.center,p.quaternion,scale);
      this.instanceMesh.setMatrixAt(i,matrix);
      this.instanceMesh.setColorAt(i,p.color);
      p.center.toArray(positions,i*3);p.color.toArray(colors,i*3);
    }
    this.instanceMesh.instanceMatrix.needsUpdate=true;
    if(this.instanceMesh.instanceColor) this.instanceMesh.instanceColor.needsUpdate=true;
    this.instanceMesh.computeBoundingSphere();this.group.add(this.instanceMesh);
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    this.points=new THREE.Points(geometry,new THREE.PointsMaterial({size:3,sizeAttenuation:false,vertexColors:true}));
    this.group.add(this.points);
    this.instanceMesh.visible=mode==='ellipsoids';this.points.visible=mode==='points';
    this.showSelection();
    return count;
  }

  pick(raycaster) {
    if(this.mode!=='ellipsoids'||!this.instanceMesh) return;
    const hit=raycaster.intersectObject(this.instanceMesh,false)[0];
    if(!hit) return;
    this.selected=this.shown[hit.instanceId];this.showSelection();this.onSelect(this.selected);
  }

  showSelection() {
    this.highlight.visible=!!this.selected&&this.mode==='ellipsoids';
    if(!this.selected) return;
    this.highlight.position.copy(this.selected.center);
    this.highlight.quaternion.copy(this.selected.quaternion);
    this.highlight.scale.copy(this.selected.scales).multiplyScalar(this.sigma*1.03);
  }
}
