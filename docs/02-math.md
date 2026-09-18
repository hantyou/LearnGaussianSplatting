# 2. From a Gaussian in space to a pixel loss

This chapter derives the equations implemented in `splatlab/core.py`. We use
column vectors in equations and store batches as rows in code. All cameras are
fixed, calibrated pinhole cameras; lens distortion is absent from this lab.

## 2.1 Parameters and units

For Gaussian i, learn

\[
\theta_i=(\mu_i,\ell_i,q_i,a_i,b_i).
\]

| Symbol | Shape | Meaning |
|---|---|---|
| \(\mu_i\) | 3 | Center in world units |
| \(\ell_i\) | 3 | Log standard deviations |
| \(q_i\) | 4 | Quaternion in w,x,y,z order; normalized before use |
| \(a_i\) | 1 | Opacity logit |
| \(b_i\) | 3 | RGB logits in our simplified model |

Use \(s=\exp(\ell)\), \(o=\operatorname{sigmoid}(a)\), and
\(c=\operatorname{sigmoid}(b)\). Then scales stay positive and opacity/color
stay in (0,1). Logits are unconstrained optimizer variables; they are not
probabilities themselves. The original implementation also uses exponential
scales, sigmoid opacity, and normalized rotation parameters.
[Parameter implementation](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py).

## 2.2 Covariance is geometry

Let Q be the rotation matrix of the normalized quaternion. Then

\[
\boxed{\Sigma=Q\,\operatorname{diag}(s_x^2,s_y^2,s_z^2)Q^T.}
\]

The eigenvectors are the ellipsoid axes; square roots of the eigenvalues are
their standard deviations. The contour
\((x-\mu)^T\Sigma^{-1}(x-\mu)=1\) has semi-axis lengths s.
It is not the boundary of the Gaussian, which has infinite support.

For nonzero v,
\[
v^T\Sigma v=\sum_k s_k^2(Q^Tv)_k^2>0.
\]

This explains positive definiteness instead of merely asserting it. Learning
nine arbitrary matrix entries would not preserve symmetry or positive
definiteness. Our parameterization does, for finite positive scales and a valid
rotation. Zero quaternions are invalid rotation parameters; initialize nonzero.

**Code:** `covariance_3d`. Notice `exp(2 * log_scales)`, not `exp(log_scales)`.

## 2.3 Put it in the camera coordinate system

Our camera axes are +x right, +y down, +z forward. Let R,t map world to camera:

\[
\mu_c=R\mu+t,\qquad \Sigma_c=R\Sigma R^T.
\]

Translation disappears from covariance because it cancels in deviations from
the mean. If the camera's world position is C, then \(t=-RC\), not t=C.
This is one of the most common implementation mistakes.

Keep the **world frame**, **camera frame**, and **viewer display frame** separate:

| Frame | Convention |
|---|---|
| Nine-Gaussian world | +y down |
| Miniature-street world | +y up |
| Both Python cameras | +x right, +y down, +z forward |
| Three.js viewer | +y up; camera looks along its local −z axis |

Each camera's R maps its particular world into the same pinhole convention.
For example, the street camera's downward row has a negative world-Y component.
The browser rotates Y-down source scenes 180° about X, flipping both Y and Z;
this is a proper rotation, not a reflection. The inspector reports parameters in
the source frame and labels that distinction. See [the scene guide](07-viewer.md).

The intrinsic matrix is
\[
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
\]
Focal lengths and principal point are in pixels; depth is in world units.
We sample pixels at integer (u,v) coordinates, so a centered principal point is
\(((W-1)/2,(H-1)/2)\). Other libraries may use different pixel-center conventions.

## 2.4 Project the center, then propagate the shape

For \(\mu_c=(x,y,z)^T\), z>0:
\[
\pi(x,y,z)=
\begin{bmatrix}f_x x/z+c_x\\f_y y/z+c_y\end{bmatrix}=m.
\]

Division by z makes perspective nonlinear. A projected 3D Gaussian is therefore
**not exactly Gaussian**. Linearize the camera near the center:
\[
\pi(\mu_c+\delta)\approx\pi(\mu_c)+J\delta,
\quad
J=\begin{bmatrix}
f_x/z&0&-f_xx/z^2\\
0&f_y/z&-f_yy/z^2
\end{bmatrix}.
\]

The ambiguous-looking \(f_xx\) means \(f_x\cdot x\), and similarly for y.
Each Jacobian row is the derivative of one output coordinate with respect to
x,y,z. You can derive it directly using the quotient rule.

Now apply familiar linear covariance propagation:
\[
E[(J\delta)(J\delta)^T]=J E[\delta\delta^T]J^T=J\Sigma_cJ^T.
\]
We therefore use
\[
\boxed{\Sigma_{2D}=J R\Sigma R^T J^T+\epsilon I_2.}
\]

The added \(\epsilon=0.3\) is a **pixel variance** in our lab, matching a common
classic rasterizer setting. It gives a footprint floor and improves invertibility;
it is not camera noise or a complete antialiasing solution. Covariance projection
and its parameter conventions are documented in
[gsplat](https://docs.gsplat.studio/main/apis/rasterization.html), with the classic
filter visible in the [original rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu).

**Worked example.** Let R=I, t=0, \(\mu=(1,0,4)\),
\(\Sigma=\operatorname{diag}(0.04,0.01,0.09)\),
\(f_x=f_y=100\), \((c_x,c_y)=(32,24)\). Then
\[
m=(57,24),\quad
J=\begin{bmatrix}25&0&-6.25\\0&25&0\end{bmatrix},
\quad \Sigma_{2D}=\begin{bmatrix}28.815625&0\\0&6.55\end{bmatrix}.
\]
The z variance contributes to the horizontal footprint because the center is
off the optical axis. Merely dropping the third row/column of the 3D covariance
would miss this effect. The tests check these exact values.

**Approximation limits:** broad splats near a camera, large viewing angles, and
camera distortion make local linearization less reliable. Our renderer culls
centers at z<=near before division; it does not integrate the portion of a
Gaussian that intersects the near plane. Keep the demo splats away from it.

## 2.5 Evaluate an image footprint

For pixel p=(u,v), define \(d=p-m\):
\[
G(p)=\exp(-\tfrac12 d^T\Sigma_{2D}^{-1}d),\qquad
\boxed{\alpha(p)=\min(0.99,oG(p)).}
\]

The quadratic form is squared Mahalanobis distance. One standard deviation from
the center gives G=exp(-1/2), along any principal axis. The peak is G=1 regardless
of ellipse area. Do **not** include \(1/(2\pi\sqrt{\det\Sigma_{2D}})\).
These are opacity footprints, not normalized probability densities.

**Code:** `gaussian_alpha`. Tensor shapes expand from [N,2] to [N,H,W,2]; the
quadratic form reduces the final vector dimensions, leaving [N,H,W].
The dense evaluation is intentionally readable and expensive.

## 2.6 Visibility turns footprints into an image

Sort splats by increasing camera-space center depth. At a particular pixel,
\[
T_1=1,\qquad T_i=\prod_{j<i}(1-\alpha_j),\qquad w_i=T_i\alpha_i.
\]
Then
\[
\boxed{C=\sum_i w_i c_i+T_{N+1}c_{bg}.}
\]

T is remaining transmittance. The weights plus background transmittance sum to
one: \(\sum_i w_i+T_{N+1}=1\). This follows by telescoping
\(w_i=T_i-T_{i+1}\). Normalizing only the foreground weights would erase the
effect of overall transparency.

With a half-opaque red splat in front of a half-opaque blue one and a white
background, C=0.5 red+0.25 blue+0.25 white=(0.75,0.25,0.50). Reversing depth order
gives (0.50,0.25,0.75). This is why a plain Gaussian sum is insufficient.

Sorting by center is an approximation to ordering extended overlapping volumes.
The sort and visibility decisions are discrete; gradients flow through the
selected values for the current ordering, not through permutation changes.

The expression resembles sampled volume rendering, where opacity over a ray
segment is \(1-\exp(-\sigma\Delta)\). Our footprint opacity should not be
mistaken for an exact ray integral of a normalized 3D Gaussian density.

## 2.7 What backpropagation actually computes

Our loss is
\[
L=\frac{1}{3HW}\sum_{p,k}(C_k(p)-I_k(p))^2.
\]
Thus \(\partial L/\partial C_k=2(C_k-I_k)/(3HW)\). For a single footprint,
ignoring the alpha clamp and the covariance's own dependence on position:
\[
\frac{\partial G}{\partial m}=G\Sigma_{2D}^{-1}(p-m).
\]
This gives a useful intuition: pixels with residuals can pull the center.
The complete position derivative also includes changes in J and therefore
\(\Sigma_{2D}\); PyTorch handles both paths in our implementation.

For compositing, \(\partial C/\partial c_i=T_i\alpha_i\). Let B_i be the
color obtained by compositing everything behind i with initial transmittance 1,
including the background. Then
\[
\frac{\partial C}{\partial\alpha_i}=T_i(c_i-B_i).
\]
Increasing opacity both adds the splat's own color **and hides the suffix**.
Ignoring the second effect gives the wrong gradient. Heavy occlusion also
suppresses gradients to hidden splats because T is small.

Finally, chain through \(\partial o/\partial a=o(1-o)\) and
\(\partial s/\partial\ell=s\). A nearly saturated sigmoid has a small gradient.
`zero_grad()`, `backward()`, and `step()` implement this process; they do not
discover a global optimum. Our finite-difference test checks center, scale,
rotation, opacity, and color derivatives together.

## 2.8 The appearance and loss used in the original system

View dependence can be represented by spherical harmonics (SH):
\[
c_i(d)=\sum_{l=0}^{L}\sum_{m=-l}^{l}\beta_{i,lm}Y_{lm}(d).
\]
Each Y is a fixed directional basis function; each beta is a learned RGB vector.
Degree L has \((L+1)^2\) coefficients per channel. Degree zero is constant;
higher degrees allow smooth directional changes. Specify the direction convention
and real-SH normalization when implementing it. SH appearance is not a physical
BRDF or a guarantee that the scene can be relit.

The original training objective combines an L1 photometric term with a structural
image similarity term, conventionally written
\(L=(1-\lambda)L_1+\lambda(1-\operatorname{SSIM})\), with lambda=0.2.
We use MSE to keep the first experiment transparent.
[Original paper, optimization](https://arxiv.org/html/2308.04079v1#S5).

For local image patches x,y, a common SSIM expression is
\[
\operatorname{SSIM}(x,y)=
\frac{(2\bar x\bar y+C_1)(2\sigma_{xy}+C_2)}
{(\bar x^2+\bar y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
\]
It compares local brightness, contrast, and correlation. Windowing, constants,
and averaging matter when reproducing an implementation. L1 alone corresponds
to a Laplace residual model under standard assumptions; MSE to a Gaussian one.
Adding SSIM is a perceptual objective choice, not automatically a likelihood.

## 2.9 Learn where to spend primitives

A fixed set of Gaussians may lack capacity. The original code accumulates
projected-position gradient statistics. High-gradient small splats can be
cloned; high-gradient large splats can be split into smaller ones. Low-opacity
and some oversized splats can be pruned, and opacity resets are used during
training. These are scheduled structural updates outside ordinary gradient
descent. Read `densify_and_clone`, `densify_and_split`, and `densify_and_prune`
in the [model implementation](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py).

The student interpretation: gradients say “the current representation struggles
here”; scale helps decide whether to add local capacity or subdivide a broad
primitive. This is a heuristic signal, not a posterior uncertainty estimate.
Cloning changes opacity accumulation, and replacing parameters requires updating
Adam's optimizer state. These details make faithful densification a separate
lesson, rather than a few unexplained lines in our first demo.

## 2.10 Why the original renderer is fast

It groups projected splats into screen tiles, sorts the relevant contributions,
and processes pixels on the GPU, with thresholds and early termination for very
small remaining transmittance. The CUDA implementation avoids our dense N×H×W
allocation. The representation makes this feasible, but efficient kernels are
an essential part of the performance result.
[Rasterizer source](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu).

**Checkpoint:** derive the projection Jacobian and compute the two-splat example
without looking. Then explain why a beautiful render does not establish either
metric geometry accuracy or a calibrated uncertainty estimate.
