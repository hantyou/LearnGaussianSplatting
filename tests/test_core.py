"""Numerical truths and finite differences, not just code-against-code checks."""

import math
import torch


def test_covariance_axis_lengths_and_rotation(core):
    scales = torch.tensor([[1., 2., 3.]], dtype=torch.double)
    # 90 degrees around z swaps x/y variances.
    q = torch.tensor([[math.sqrt(0.5), 0., 0., math.sqrt(0.5)]], dtype=torch.double)
    cov = core.covariance_3d(scales.log(), q)
    torch.testing.assert_close(cov[0], torch.diag(torch.tensor([4., 1., 9.], dtype=torch.double)))
    assert torch.linalg.eigvalsh(cov).min() > 0


def test_covariance_quaternion_sign_and_magnitude(core):
    logs = torch.tensor([[-1., -2., -0.5]], dtype=torch.double)
    q = torch.tensor([[1., 0.2, -0.4, 0.3]], dtype=torch.double)
    torch.testing.assert_close(core.covariance_3d(logs, q), core.covariance_3d(logs, -3*q))


def test_projection_known_values_and_camera_translation(core):
    mean = torch.tensor([[1., 0., 4.]], dtype=torch.double)
    cov = torch.diag(torch.tensor([0.04, 0.01, 0.09], dtype=torch.double))[None]
    R = torch.eye(3, dtype=torch.double)
    K = torch.tensor([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]], dtype=torch.double)
    uv, cov2, z = core.project_gaussians(mean, cov, R, torch.zeros(3, dtype=torch.double), K)
    torch.testing.assert_close(uv, torch.tensor([[57., 24.]], dtype=torch.double))
    torch.testing.assert_close(cov2[0], torch.diag(torch.tensor([28.815625, 6.55], dtype=torch.double)))
    torch.testing.assert_close(z, torch.tensor([4.], dtype=torch.double))
    # Move camera center to x=1: t=(-1,0,0), projection lands on principal point.
    uv2, _, _ = core.project_gaussians(mean, cov, R, torch.tensor([-1., 0., 0.], dtype=torch.double), K)
    torch.testing.assert_close(uv2, torch.tensor([[32., 24.]], dtype=torch.double))


def test_projection_rotation_and_depth_scaling(core):
    # Rotate world covariance and center by 90 degrees about camera z.
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.double)
    K = torch.diag(torch.tensor([100., 100., 1.], dtype=torch.double))
    cov = torch.diag(torch.tensor([0.04, 0.01, 0.09], dtype=torch.double))[None]
    mean = torch.tensor([[0., 0., 4.]], dtype=torch.double)
    t = torch.zeros(3, dtype=torch.double)
    _, near_cov, _ = core.project_gaussians(mean, cov, R, t, K, pixel_variance=0.)
    _, far_cov, _ = core.project_gaussians(2*mean, cov, R, t, K, pixel_variance=0.)
    torch.testing.assert_close(near_cov[0], torch.diag(torch.tensor([6.25, 25.], dtype=torch.double)))
    torch.testing.assert_close(far_cov, near_cov / 4)


def test_footprint_peak_and_one_sigma(core):
    uv = torch.tensor([[2., 2.]])
    cov = torch.diag(torch.tensor([4., 1.]))[None]
    alpha = core.gaussian_alpha(uv, cov, torch.tensor([0.8]), 5, 5)
    torch.testing.assert_close(alpha[0, 2, 2], torch.tensor(0.8))
    torch.testing.assert_close(alpha[0, 2, 4], torch.tensor(0.8*math.exp(-0.5)))
    torch.testing.assert_close(alpha[0, 3, 2], alpha[0, 2, 4])


def test_footprint_rotated_ellipse(core):
    cov = torch.tensor([[[2.5, 1.5], [1.5, 2.5]]])
    alpha = core.gaussian_alpha(torch.tensor([[2., 2.]]), cov, torch.tensor([0.8]), 5, 5)
    # Broad axis points diagonally down-right, not up-right.
    assert alpha[0, 3, 3] > alpha[0, 1, 3]


def test_composite_known_colors_and_order(core):
    alpha = torch.tensor([[[0.5]], [[0.5]]])
    colors = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
    background = torch.ones(3)
    image, opacity = core.composite(alpha, colors, background)
    torch.testing.assert_close(image[0, 0], torch.tensor([0.75, 0.25, 0.5]))
    torch.testing.assert_close(opacity[0, 0], torch.tensor(0.75))
    reversed_image, _ = core.composite(alpha, colors.flip(0), background)
    torch.testing.assert_close(reversed_image[0, 0], torch.tensor([0.5, 0.25, 0.75]))


def test_composite_empty_scene(core):
    background = torch.tensor([0.1, 0.2, 0.3])
    image, opacity = core.composite(torch.empty(0, 3, 4), torch.empty(0, 3), background)
    torch.testing.assert_close(image, background.expand(3, 4, 3))
    assert torch.count_nonzero(opacity) == 0


def test_composite_opacity_gradients_include_hiding_suffix(core):
    # A foreground opacity both adds its own color AND hides all layers behind it.
    # This catches a detached transmittance, which a one-splat/black-bg test misses.
    alpha = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double, requires_grad=True)
    colors = torch.tensor([[0.8, 0.1, 0.3], [0.2, 0.7, 0.4]], dtype=torch.double)
    bg = torch.tensor([0.3, 0.2, 0.9], dtype=torch.double)
    image, _ = core.composite(alpha, colors, bg)
    suffix = alpha[1, 0, 0].detach()*colors[1] + (1-alpha[1, 0, 0].detach())*bg
    expected_front = colors[0] - suffix
    expected_back = (1-alpha[0, 0, 0].detach())*(colors[1]-bg)
    for channel in range(3):
        derivative, = torch.autograd.grad(image[0, 0, channel], alpha, retain_graph=True)
        torch.testing.assert_close(derivative[:, 0, 0],
                                   torch.stack((expected_front[channel], expected_back[channel])))


def test_end_to_end_gradients_by_finite_difference(core):
    # Double precision, no depth ties, no alpha clamp boundary or culling boundary.
    mean = torch.tensor([[0.03, -0.05, 3.]], dtype=torch.double, requires_grad=True)
    logs = torch.tensor([[-1.2, -1.8, -1.5]], dtype=torch.double, requires_grad=True)
    q = torch.tensor([[1., 0.1, 0.2, -0.1]], dtype=torch.double, requires_grad=True)
    opacity_logit = torch.tensor([0.4], dtype=torch.double, requires_grad=True)
    colors = torch.tensor([[0.6, 0.2, 0.9]], dtype=torch.double, requires_grad=True)
    K = torch.tensor([[8., 0., 2.], [0., 8., 2.], [0., 0., 1.]], dtype=torch.double)
    def forward(mean, logs, q, opacity_logit, colors):
        cov = core.covariance_3d(logs, q)
        uv, cov2, _ = core.project_gaussians(mean, cov, torch.eye(3, dtype=torch.double),
                                           torch.zeros(3, dtype=torch.double), K)
        alpha = core.gaussian_alpha(uv, cov2, opacity_logit.sigmoid(), 4, 4)
        return core.composite(alpha, colors, torch.zeros(3, dtype=torch.double))[0]
    assert torch.autograd.gradcheck(forward, (mean, logs, q, opacity_logit, colors), atol=1e-5)
