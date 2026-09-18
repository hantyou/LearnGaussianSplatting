"""Fit a miniature 3D street from multiple images using the readable CPU renderer.

This controlled synthetic experiment uses a known number of primitives and
perturbed teacher initialization (much easier than geometry estimated from photos).
It is NOT training from real street photographs. All five parameter groups learn.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .device import CHOICES, describe_device, format_report, select_device, synchronize
from .street import make_street, street_camera
from .scene import GaussianScene
from .render import render
from .export import export_scene,load_scene_json
from .demo import save_comparison, psnr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=350)
    parser.add_argument("--size",type=int,default=128)
    parser.add_argument("--device",default="auto",choices=CHOICES,
                        help="auto uses a GPU when one is usable, else the CPU")
    parser.add_argument("--student",action="store_true",help="Use your completed exercises/core.py")
    args = parser.parse_args()
    if args.steps < 1 or args.size < 8:
        parser.error("Use positive steps and size >= 8")
    from . import core
    if args.student:
        from exercises import core
    device=select_device(args.device)
    device_info=describe_device(device,args.device)
    print(format_report(device_info),flush=True)
    torch.set_num_threads(4); torch.manual_seed(17)
    root=Path(__file__).resolve().parents[1]
    out=root/"outputs"/"street"; out.mkdir(parents=True,exist_ok=True)
    data=root/"viewer"/"data"
    # Build and perturb on the CPU so a GPU run starts from identical bits.
    teacher=make_street().requires_grad_(False)
    with torch.no_grad():
        student=GaussianScene(teacher.means+0.075*torch.randn_like(teacher.means),
                              (teacher.log_scales+0.3*torch.randn_like(teacher.log_scales)).exp(),
                              teacher.quaternions+0.12*torch.randn_like(teacher.quaternions),
                              (teacher.color_logits+1.0*torch.randn_like(teacher.color_logits)).sigmoid(),
                              torch.full_like(teacher.opacities,0.52))
    export_scene(teacher,data/"street-target.json","Miniature street: procedural target")
    export_scene(student,data/"street-before.json","Miniature street: before fitting")
    teacher=teacher.to(device); student=student.to(device)
    train_angles=[0,60,120,180,240,300]
    heldout_angles=[30,90,150,210,270,330]
    cameras=[street_camera(a,args.size,device) for a in train_angles+heldout_angles]
    with torch.no_grad():
        targets=[render(teacher,c)[0] for c in cameras]
        before=[render(student,c,core)[0] for c in cameras]
    def score(images):
        errors=[(im-target).square().mean() for im,target in zip(images,targets)]
        return [psnr(torch.stack(part).mean()) for part in [errors[:6],errors[6:]]]
    initial=score(before)
    optimizer=torch.optim.Adam([
        {"params":[student.means],"lr":.004},
        {"params":[student.log_scales,student.quaternions],"lr":.007},
        {"params":[student.color_logits,student.opacity_logits],"lr":.025}])
    synchronize(device)  # GPU work is queued, so time it between two barriers.
    start=time.perf_counter()
    print(f"Training {len(student.means)} Gaussians from six {args.size}x{args.size} images",flush=True)
    for step in range(args.steps):
        i=step%6
        optimizer.zero_grad()
        image,_=render(student,cameras[i],core)
        loss=(image-targets[i]).square().mean()
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
        loss.backward(); optimizer.step()
        if (step+1)%50==0:
            print(f"{step+1}/{args.steps}: current training view MSE={loss.item():.6f}",flush=True)
    synchronize(device)
    seconds=time.perf_counter()-start
    with torch.no_grad(): after=[render(student,c,core)[0] for c in cameras]
    final=score(after)
    export_scene(student,data/"street-after.json","Miniature street: fitted reconstruction")
    # NumPy checkpoints live on the host; .cpu() is a no-op for a CPU run.
    np.savez(out/"learned_scene.npz",**{k:v.detach().cpu().numpy() for k,v in student.state_dict().items()})
    save_street_preview(teacher,load_scene_json(data/"street-before.json",device),student,out/"comparison.png",core)
    report={**device_info,
            "gaussians":len(student.means),"steps":args.steps,"size":args.size,
            "initial_train_psnr":initial[0],"initial_heldout_psnr":initial[1],
            "final_train_psnr":final[0],"final_heldout_psnr":final[1],
            "train_angles":train_angles,"heldout_angles":heldout_angles,
            "seconds":seconds,
            "student_core":args.student,
            "experiment":"Synthetic Gaussian teacher, perturbed initialization, fixed count, RGB MSE"}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


@torch.no_grad()
def save_street_preview(teacher,before,after,path,core=None):
    if core is None:
        from . import core
    cameras=[street_camera(a,96,teacher.means.device) for a in [0,30]]
    targets=[render(teacher,c)[0] for c in cameras]
    views=[targets]+[[render(model,c,core)[0] for c in cameras] for model in [before,after]]
    save_comparison(path,*views,1,title="755 learnable 3D Gaussians · miniature street")


if __name__ == "__main__": main()
