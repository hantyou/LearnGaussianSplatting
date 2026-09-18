"""Guard the geometry handoff from the learning code to the 3D viewer."""
import json
import torch
from splatlab.core import covariance_3d
from splatlab.export import export_scene, load_scene_json
from splatlab.scene import make_teacher, make_student
from splatlab.street import street_camera


def test_export_preserves_rotated_covariance_and_appearance(tmp_path):
    scene = make_student(make_teacher(), seed=7)
    path = tmp_path / 'scene.json'
    export_scene(scene, path, 'roundtrip', y_up=False)
    data = json.loads(path.read_text())
    assert data['yUp'] is False
    restored = load_scene_json(path)
    torch.testing.assert_close(covariance_3d(scene.log_scales, scene.quaternions),
                               covariance_3d(restored.log_scales, restored.quaternions))
    torch.testing.assert_close(scene.means, restored.means)
    torch.testing.assert_close(scene.colors, restored.colors)
    torch.testing.assert_close(scene.opacities, restored.opacities)


def test_y_up_street_camera_has_downward_image_axis_and_positive_depth():
    for angle in [0, 60, 150, 270]:
        camera = street_camera(angle)
        torch.testing.assert_close(camera.R @ camera.R.T, torch.eye(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.det(camera.R), torch.tensor(1.))
        focus = camera.R @ torch.tensor([0., .35, 0.]) + camera.t
        torch.testing.assert_close(focus[:2], torch.zeros(2), atol=1e-6, rtol=0)
        assert focus[2] > 0
        # Raising a point in world Y moves it up (smaller camera/image y).
        assert camera.R[1, 1] < 0
