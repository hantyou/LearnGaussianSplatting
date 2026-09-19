"""Export the learning scenes and optionally download two public real-scene models.

python -m splatlab.prepare_viewer --download
Only downloads .splat model data from the named public source, never executable code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from .export import export_scene, load_checkpoint
from .scene import make_teacher, make_student
from .street import make_street

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "viewer" / "data"
SAMPLES = {"room": 50988032, "train": 32848256, "garden": 186713088}


def download(name, expected_bytes):
    url = f"https://huggingface.co/cakewalk/splat-data/resolve/main/{name}.splat"
    path = DATA / f"{name}.splat"
    if not path.exists() or path.stat().st_size != expected_bytes:
        partial = path.with_suffix(".partial")
        print(f"Downloading {name}: {expected_bytes/1e6:.1f} MB", flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as dest:
            while chunk := response.read(1024*1024):
                dest.write(chunk)
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(f"Unexpected size for {name}; keeping partial file for diagnosis")
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Ready: {name}, {expected_bytes//32:,} Gaussians", flush=True)
    return {"name":name,"source":url,"bytes":expected_bytes,"sha256":digest,
            "provenance":"Pretrained public sample distributed by cakewalk/splat-data; not trained by this lab",
            "license_note":"Source repository says assets have differing licenses; no per-file license is declared there. Do not assume the viewer's MIT license covers the model."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    teacher = make_teacher()
    export_scene(teacher,DATA/"nine-target.json","Nine Gaussians: target",y_up=False)
    export_scene(make_student(teacher),DATA/"nine-before.json","Nine Gaussians: initialization",y_up=False)
    trained = ROOT / "outputs" / "reference" / "learned_scene.npz"
    if trained.exists():
        export_scene(load_checkpoint(trained),DATA/"nine-after.json","Nine Gaussians: learned",y_up=False)
    export_scene(make_street(),DATA/"street-target.json","Miniature street: procedural target")
    if args.download:
        with ThreadPoolExecutor(max_workers=2) as pool:
            manifest = list(pool.map(lambda pair:download(*pair),SAMPLES.items()))
        (DATA/"sources.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Viewer data prepared.")


if __name__ == "__main__":
    main()
