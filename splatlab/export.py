"""Readable interchange for the browser. Quaternions stay in Python w,x,y,z order."""
import json
from pathlib import Path
import numpy as np
import torch
from .scene import GaussianScene


def export_scene(scene, path, name, y_up=True):
    with torch.no_grad():
        record = {
            "name": name, "yUp": y_up,
            "means": scene.means.tolist(), "scales": scene.log_scales.exp().tolist(),
            "quaternions": torch.nn.functional.normalize(scene.quaternions, dim=-1).tolist(),
            "colors": scene.colors.tolist(), "opacities": scene.opacities.tolist(),
        }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")


def load_checkpoint(path):
    with np.load(path) as data:
        state = {key: torch.from_numpy(data[key]) for key in data.files}
    scene = GaussianScene(state["means"], state["log_scales"].exp(), state["quaternions"],
                          state["color_logits"].sigmoid(), state["opacity_logits"].sigmoid())
    scene.load_state_dict(state)
    return scene


def load_scene_json(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    return GaussianScene(*(torch.tensor(data[key],dtype=torch.float32) for key in
                           ("means","scales","quaternions","colors","opacities")))
