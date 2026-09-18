import torch
from splatlab.camera import orbit_camera
from splatlab.render import render
from splatlab.scene import make_teacher, make_student


def test_camera_rotation_is_orthonormal_and_origin_is_centered():
    camera = orbit_camera(35)
    torch.testing.assert_close(camera.R @ camera.R.T, torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.det(camera.R), torch.tensor(1.))
    assert abs(camera.t[0]) < 1e-6 and abs(camera.t[1]) < 1e-6 and camera.t[2] > 0


def test_behind_camera_is_culled_without_nan(core):
    camera = orbit_camera(0, size=12)
    scene = make_teacher()
    with torch.no_grad():
        scene.means[:, 2] = -10.
    image, alpha = render(scene, camera, core)
    assert torch.isfinite(image).all()
    assert torch.count_nonzero(alpha) == 0
    torch.testing.assert_close(image, image.new_tensor([0.035, 0.045, 0.065]).expand(12, 12, 3))


def test_training_improves_an_unseen_view(core):
    torch.set_num_threads(2)
    teacher = make_teacher().requires_grad_(False)
    student = make_student(teacher)
    cameras = [orbit_camera(a, size=20) for a in [-35, 0, 35, 17]]
    with torch.no_grad():
        targets = [render(teacher, c)[0] for c in cameras]
        initial = (render(student, cameras[-1], core)[0] - targets[-1]).square().mean()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    for step in range(60):
        index = step % 3
        optimizer.zero_grad()
        loss = (render(student, cameras[index], core)[0] - targets[index]).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (render(student, cameras[-1], core)[0] - targets[-1]).square().mean()
    assert final < initial * 0.4
