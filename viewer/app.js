// Read this orchestration after the Python core. Spark supplies the fast renderer;
// inspector.js displays the same data as explicit 3D covariance ellipsoids.
import * as THREE from 'three';
import { SparkRenderer,SplatMesh } from '@sparkjsdev/spark';
import { Navigation } from './navigation.js';
import { GaussianInspector } from './inspector.js';

const $=selector=>document.querySelector(selector);
const viewport=$('#viewport'),loading=$('#loading');
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0e1621);
const camera=new THREE.PerspectiveCamera(55,1,.01,500);
const renderer=new THREE.WebGLRenderer({antialias:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));
renderer.outputColorSpace=THREE.SRGBColorSpace;
viewport.append(renderer.domElement);
const spark=new SparkRenderer({renderer,sortRadial:false,preBlurAmount:.3,blurAmount:0});
scene.add(spark);
scene.add(new THREE.HemisphereLight(0xe6f0ff,0x3b4656,2));
const light=new THREE.DirectionalLight(0xffedcf,2);light.position.set(3,7,5);scene.add(light);
const navigation=new Navigation(camera,renderer.domElement);
const group=new THREE.Group();scene.add(group);
const diagnosticGroup=new THREE.Group();group.add(diagnosticGroup);
const inspector=new GaussianInspector(diagnosticGroup,showSelection);
const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
const config={
  street:{kind:'synthetic',position:[3.8,4.3,-5],target:[0,.25,0],scale:2,
    note:'Synthetic miniature street · 755 Gaussians. Target is procedural; Learned fits six rendered images from perturbed initialization. Change the camera and compare Before / Learned / Target.'},
  nine:{kind:'synthetic',position:[0,.5,3.2],target:[0,0,0],scale:1,
    note:'The original nine-Gaussian exercise, now in 3D. These are the parameters produced by your complete Python implementation. Compare stages while keeping the same viewpoint.'},
  room:{kind:'real',position:[0,0,0],target:[0,0,-2],scale:.5,
    note:'Real capture · pretrained Room sample from cakewalk/splat-data. This lab renders the downloaded reconstruction; it did not train it. The compact .splat file stores constant RGB, without view-dependent spherical harmonics.'},
  train:{kind:'real',position:[-3.01,.11,3.75],target:[-.15,.45,-1.5],scale:1.5,
    note:'Real capture · pretrained Train sample from cakewalk/splat-data. Explore the locomotive and surrounding scene. Sparse views and unobserved areas can leave floaters and holes; the model is not a watertight mesh.'}
};
let currentMesh=null,currentScene='street',stage='after',mode='render',busy=false;
let total=0,shown=0,sceneVersion=0;

function message(text,error=false) {
  loading.hidden=false;loading.textContent=text;loading.setAttribute('role',error?'alert':'status');
}

function refreshDiagnostics() {
  if(!currentMesh) return;
  const sigma=+$('#sigma').value;
  shown=inspector.rebuild({sigma,budget:+$('#budget').value,
    localOnly:$('#local-only').checked,radius:+$('#radius').value,
    focus:navigation.orbit.target,isolate:$('#isolate').checked,mode});
  currentMesh.visible=mode==='render';spark.visible=mode==='render';
  diagnosticGroup.visible=mode!=='render';
  $('#status').textContent=mode==='render'?`${total.toLocaleString()} Gaussians · rendered appearance`:
    `${shown.toLocaleString()} of ${total.toLocaleString()} Gaussians · ${mode==='points'?'centers':`${sigma}σ contours`}`;
  $('#sample-note').textContent=total>50000?
    'Diagnostic modes use a deterministic pool of up to 50,000 primitives, then apply the sample limit and local filter. Rendering uses the full model.':
    'Solid ellipsoids show actual spatial shape. The sample limit affects diagnostics only; it does not change the reconstructed image.';
  $('#diagnostics').style.opacity=mode==='render'?'.65':'1';
}

function setMode(value) {
  mode=value;
  for(const button of document.querySelectorAll('[data-mode]')) button.setAttribute('aria-pressed',String(button.dataset.mode===value));
  refreshDiagnostics();
}

function showSelection(p) {
  $('#selection').hidden=false;$('#selection-hint').hidden=true;
  $('#sel-index').textContent=p.index.toLocaleString();
  // Coordinates are in source/local scene units; the whole scene may be rotated for display.
  $('#sel-center').textContent=p.center.toArray().map(x=>x.toFixed(4)).join(', ');
  $('#sel-scales').textContent=p.scales.toArray().map(x=>x.toPrecision(4)).join(', ');
  const q=p.quaternion;
  $('#sel-quat').textContent=[q.w,q.x,q.y,q.z].map(x=>x.toFixed(4)).join(', ');
  $('#sel-color').textContent=p.color.toArray().map(x=>x.toFixed(3)).join(', ');
  $('#sel-opacity').textContent=p.opacity.toFixed(4);
  $('#sel-frame').textContent=group.rotation.x===0?
    'Parameters use the source Y-up frame.':
    'Parameters use the source frame. Display rotates it 180° around X: displayed Y and Z have opposite signs.';
}

async function fromJSON(url) {
  const response=await fetch(url);
  if(!response.ok) throw new Error(`Missing ${url}. Run the preparation or training command in the viewer guide.`);
  const data=await response.json();
  const mesh=new SplatMesh({constructSplats:packed=>{
    const center=new THREE.Vector3(),scales=new THREE.Vector3(),q=new THREE.Quaternion(),color=new THREE.Color();
    for(let i=0;i<data.means.length;i++) {
      const [w,x,y,z]=data.quaternions[i]; // Python wxyz -> Three.js xyzw.
      packed.pushSplat(center.fromArray(data.means[i]),scales.fromArray(data.scales[i]),
        q.set(x,y,z,w),data.opacities[i],color.setRGB(...data.colors[i]));
    }
  }});
  await mesh.initialized;
  return {mesh,flip:!data.yUp};
}

async function loadScene(name,nextStage='after',keepCamera=false) {
  if(busy) return;
  busy=true;sceneVersion++;
  $('#scene').disabled=true;
  document.querySelectorAll('[data-stage]').forEach(e=>e.disabled=true);
  message(`Loading ${name==='street'?'the miniature street':name}…`);
  try {
    const cfg=config[name];let mesh,flip;
    if(cfg.kind==='real') {
      mesh=new SplatMesh({url:`data/${name}.splat`,lod:false,
        onProgress:event=>{if(event.total) message(`Loading ${name}… ${Math.round(event.loaded/event.total*100)}%`);}});
      await mesh.initialized;flip=true;
    } else {
      ({mesh,flip}=await fromJSON(`data/${name}-${nextStage}.json`));
    }
    if(currentMesh) {group.remove(currentMesh);currentMesh.dispose();}
    currentMesh=mesh;currentScene=name;stage=nextStage;
    group.rotation.set(flip?Math.PI:0,0,0);group.add(mesh);group.updateMatrixWorld(true);
    total=mesh.packedSplats.numSplats;
    inspector.setSource(mesh);
    navigation.sceneScale=cfg.scale;
    if(!keepCamera) navigation.reset(cfg.position,cfg.target);
    $('#learning-controls').hidden=cfg.kind==='real';
    $('#scene-note').textContent=cfg.note;
    $('#selection').hidden=true;$('#selection-hint').hidden=false;$('#isolate').checked=false;
    $('#local-only').checked=false;
    for(const button of document.querySelectorAll('[data-stage]')) button.setAttribute('aria-pressed',String(button.dataset.stage===stage));
    $('#scene').value=name;
    refreshDiagnostics();loading.hidden=true;
  } catch(error) {
    console.error(error);
    $('#scene').value=currentScene;
    message(`Could not load this scene. ${error.message} See docs/07-viewer.md for setup commands.`,true);
  } finally {
    busy=false;$('#scene').disabled=false;
    document.querySelectorAll('[data-stage]').forEach(e=>e.disabled=false);
  }
}

$('#scene').addEventListener('change',event=>loadScene(event.target.value,'after'));
for(const button of document.querySelectorAll('[data-stage]')) button.addEventListener('click',()=>loadScene(currentScene,button.dataset.stage,true));
for(const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click',()=>setMode(button.dataset.mode));
$('#reset').addEventListener('click',()=>{const c=config[currentScene];navigation.reset(c.position,c.target);refreshDiagnostics();});
$('#navigation').addEventListener('change',event=>{
  navigation.setMode(event.target.value);
  $('#navigation-help').textContent=event.target.value==='fly'?
    'Click the scene, then W/A/S/D to move, Q/E down/up, Shift to speed up. Left-drag to look around. No collision detection.':
    'Drag to orbit · right-drag to pan · scroll to zoom. Double-click a point on the scene to focus.';
});
$('#speed').addEventListener('input',event=>{navigation.speed=+event.target.value;$('#speed-value').textContent=`${navigation.speed.toFixed(1)}×`;});
$('#radius').addEventListener('input',event=>{$('#radius-value').textContent=(+event.target.value).toFixed(1);});
for(const id of ['sigma','budget','local-only','radius','isolate']) $('#'+id).addEventListener('change',refreshDiagnostics);
$('#focus-selected').addEventListener('click',()=>{
  const p=inspector.selected;if(!p) return;
  const point=diagnosticGroup.localToWorld(p.center.clone());
  navigation.focus(point,Math.max(p.scales.length()*6,.12));refreshDiagnostics();
});

function ray(event) {
  const r=renderer.domElement.getBoundingClientRect();
  pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);
  raycaster.setFromCamera(pointer,camera);
  return raycaster;
}
let down=null;
renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
renderer.domElement.addEventListener('pointerup',event=>{
  if(event.button===0&&down&&Math.hypot(event.clientX-down[0],event.clientY-down[1])<4) inspector.pick(ray(event));
  down=null;
});
renderer.domElement.addEventListener('dblclick',event=>{
  if(!currentMesh||navigation.mode!=='orbit') return;
  let hits;
  if(mode==='render') hits=ray(event).intersectObject(currentMesh,false);
  else if(mode==='ellipsoids') hits=ray(event).intersectObject(inspector.instanceMesh,false);
  else hits=ray(event).intersectObject(inspector.points,false);
  if(hits[0]) {navigation.orbit.target.copy(hits[0].point);refreshDiagnostics();}
});

new ResizeObserver(()=>{
  const width=viewport.clientWidth,height=viewport.clientHeight;
  renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();
}).observe(viewport);
let last=performance.now(),frameCount=0,frameStart=last;
renderer.setAnimationLoop(now=>{
  navigation.update((now-last)/1000);last=now;
  renderer.render(scene,camera);frameCount++;
  if(now-frameStart>1000) {$('#fps').textContent=`${Math.round(frameCount*1000/(now-frameStart))} fps`;frameCount=0;frameStart=now;}
});
// Small read-only diagnostics for local automated verification.
window.splatLab={getState:()=>({scene:currentScene,stage,mode,total,shown,busy,sceneVersion,
  camera:camera.position.toArray(),selected:inspector.selected?.index??null})};
window.addEventListener('unhandledrejection',event=>message(`Viewer error: ${event.reason?.message||event.reason}`,true));
const requestedScene=new URLSearchParams(location.search).get('scene');
loadScene(Object.hasOwn(config,requestedScene)?requestedScene:'street','after');
