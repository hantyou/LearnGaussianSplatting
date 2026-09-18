// The renderer and the inspector share one camera, so changing mode preserves pose.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Navigation {
  constructor(camera, canvas) {
    this.camera=camera; this.canvas=canvas; this.mode='orbit'; this.speed=1; this.sceneScale=1;
    this.keys=new Set(); this.dragging=false;
    this.orbit=new OrbitControls(camera,canvas);
    this.orbit.enableDamping=true; this.orbit.dampingFactor=0.12;
    this.orbit.minDistance=0.015; this.orbit.maxDistance=300;
    canvas.tabIndex=0;
    canvas.setAttribute('aria-label','3D scene. Drag to orbit, or use fly controls after choosing Fly through.');
    canvas.addEventListener('pointerdown',event=>{
      canvas.focus();
      if(this.mode==='fly' && event.button===0) {
        this.dragging=true; canvas.setPointerCapture(event.pointerId);
      }
    });
    canvas.addEventListener('pointerup',()=>{this.dragging=false;});
    canvas.addEventListener('pointermove',event=>{
      if(this.mode!=='fly'||!this.dragging) return;
      const euler=new THREE.Euler().setFromQuaternion(camera.quaternion,'YXZ');
      euler.y-=event.movementX*.004; euler.x-=event.movementY*.004;
      euler.x=THREE.MathUtils.clamp(euler.x,-Math.PI*.49,Math.PI*.49);
      camera.quaternion.setFromEuler(euler);
    });
    canvas.addEventListener('keydown',event=>{
      if(this.mode==='fly'&&['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(event.code)) {
        event.preventDefault(); this.keys.add(event.code);
      }
    });
    window.addEventListener('keyup',event=>this.keys.delete(event.code));
    window.addEventListener('blur',()=>this.keys.clear());
    canvas.addEventListener('blur',()=>{this.keys.clear();this.dragging=false;});
  }

  setMode(mode) {
    this.keys.clear(); this.dragging=false;
    if(mode==='orbit'&&this.mode==='fly') {
      const forward=this.camera.getWorldDirection(new THREE.Vector3());
      this.orbit.target.copy(this.camera.position).addScaledVector(forward,this.sceneScale);
    }
    this.mode=mode; this.orbit.enabled=mode==='orbit';
  }

  reset(position, target) {
    this.camera.position.fromArray(position); this.camera.up.set(0,1,0);
    this.orbit.target.fromArray(target); this.camera.lookAt(this.orbit.target);
    this.orbit.update();
  }

  focus(point,distance) {
    const direction=this.camera.position.clone().sub(point).normalize();
    this.camera.position.copy(point).addScaledVector(direction,distance);
    this.orbit.target.copy(point); this.camera.lookAt(point); this.orbit.update();
  }

  update(delta) {
    if(this.mode==='orbit') {this.orbit.update(); return;}
    const forward=this.camera.getWorldDirection(new THREE.Vector3());
    const right=new THREE.Vector3().crossVectors(forward,this.camera.up).normalize();
    const amount=Math.min(delta,.05)*this.speed*this.sceneScale*(this.keys.has('ShiftLeft')||this.keys.has('ShiftRight')?3:1);
    if(this.keys.has('KeyW')) this.camera.position.addScaledVector(forward,amount);
    if(this.keys.has('KeyS')) this.camera.position.addScaledVector(forward,-amount);
    if(this.keys.has('KeyD')) this.camera.position.addScaledVector(right,amount);
    if(this.keys.has('KeyA')) this.camera.position.addScaledVector(right,-amount);
    if(this.keys.has('KeyE')) this.camera.position.y+=amount;
    if(this.keys.has('KeyQ')) this.camera.position.y-=amount;
  }
}
