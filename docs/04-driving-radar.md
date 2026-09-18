# 4. Autonomous driving, perception, and radar

The word “Gaussian” is used in several different roles. Keeping the following
tasks separate prevents a lot of confusion when reading driving papers.

| Task | What is estimated or produced? | A representative method |
|---|---|---|
| Reconstruct a recorded drive | A renderable scene with background and moving actors | [Street Gaussians](https://arxiv.org/abs/2401.01339) |
| Simulate observations | Images or sensor measurements at specified poses/times | [SplatAD](https://arxiv.org/abs/2411.16816), [RadarSplat](https://arxiv.org/abs/2506.01379) |
| Estimate a map and ego pose | A persistent representation and trajectory | [SplaTAM](https://spla-tam.github.io/) is an RGB-D research example, not a complete driving stack |
| Perceive the current surroundings | Occupancy, semantics, or BEV features | [GaussianFormer](https://arxiv.org/abs/2405.17429), [GaussianCaR](https://arxiv.org/abs/2602.08784) |

## 4.1 Turning a static reconstruction into a driving scene

A useful scene graph is a static background plus object-local Gaussian sets.
For actor k with pose \((R_k(t),t_k(t))\):
\[
\mu_{ki}^{world}(t)=R_k(t)\mu_{ki}^{local}+t_k(t),\qquad
\Sigma_{ki}^{world}(t)=R_k(t)\Sigma_{ki}^{local}R_k(t)^T.
\]
Then apply the sensor's pose at its acquisition time. This is the same covariance
algebra you implemented in the camera renderer, used twice: object to world,
then world to sensor. Actor motion and ego motion must not be conflated.

Street Gaussians separates foreground vehicles and background, optimizes tracked
object poses, and models changing appearance. It demonstrates how an explicit
representation can support scene composition and editing.
[Paper](https://arxiv.org/abs/2401.01339).

For a real sequence you must also handle calibration, synchronization, sparse
view coverage, sky, moving pedestrians, and changes in illumination. A car viewed
mostly from behind has little evidence about its front. Moving an actor to an
unobserved location may expose holes, shadows baked into appearance, or incorrect
occlusion. These are limitations to measure, not problems solved automatically
by choosing Gaussians.

SplatAD jointly renders camera and LiDAR data with sensor-specific projection,
timing, and return modeling, including intensity and ray-drop probability.
This is more than rendering an RGB depth image and treating it as a LiDAR scan.
[Paper](https://arxiv.org/html/2411.16816v2).

## 4.2 How this can help perception

There are two distinct routes:

**Render data, then train or test a perception model.** Reconstruct a drive,
vary a supported viewpoint or actor configuration, and generate observations.
Evaluate whether a detector/occupancy model benefits on **held-out real scenes**.
Image PSNR alone does not establish better detection, better radar statistics,
or more reliable closed-loop driving.

**Use Gaussian geometry inside the perception model.** Predict locations,
spatial extents, and features from current sensors; project or aggregate those
features into 3D/BEV; train with semantic/occupancy objectives. GaussianFormer
illustrates semantic Gaussian occupancy representations. GaussianCaR maps image
and radar information into a shared BEV representation for segmentation; its
arXiv record identifies acceptance to ICRA 2026.
[GaussianFormer](https://arxiv.org/abs/2405.17429), [GaussianCaR](https://arxiv.org/abs/2602.08784).

For your YOLO background, the difference is the target: a view renderer minimizes
image error for a particular scene; a detector learns categories and boxes across
scenes. Sharing splatting machinery does not make those losses interchangeable.

## 4.3 Radar: change the measurement model

Radar reflection strength is not RGB color. A radar observation may encode range,
azimuth, elevation, radial velocity, and power—or only some of these. Models may
need antenna response, range attenuation, multipath, saturation, spectral leakage,
and the processing that creates detections. Raw complex signals, power images,
and thresholded point detections are different learning targets.

RadarSplat (ICCV 2025) studies radar synthesis and reconstruction using scanning
radar data from Boreas, with noise/multipath and spectral-leakage modeling.
Read it as a specific radar image-formation system, not proof that ordinary RGB
splatting reproduces arbitrary automotive radar or Doppler data.
[Paper](https://arxiv.org/html/2506.01379v1), [official implementation](https://github.com/umautobots/radarsplat).

As a geometry sketch, in radar coordinates let \(p=(x,y,z)\),
\(r=\|p\|\), and \(\hat r=p/r\). A possible measurement function is
\[
h(p,v)=\left[
r,\ \operatorname{atan2}(y,x),\
\operatorname{atan2}(z,\sqrt{x^2+y^2}),\
\hat r^T(v_{object}-v_{sensor})
\right]^T.
\]
Here positive radial velocity means receding; actual datasets may use different
conventions or provide compensated velocities. Transform velocities into a common
frame and account for sensor motion. This equation is an illustrative geometric
model, **not RadarSplat's full renderer**.

You can propagate a spatial Gaussian through h with a Jacobian, as for the
camera, but that only gives an approximate footprint in measurement coordinates.
It does not supply scattering physics or a noise likelihood. If velocities are
random variables too, propagate a joint position/velocity covariance, including
cross-covariances; a 3×3 shape matrix alone is insufficient for that uncertainty.

## 4.4 A Bayesian connection you can use immediately

For a 2D radar detection \((r,\theta)\), convert to BEV coordinates:
\[
x=r\cos\theta,\quad y=r\sin\theta,\quad
H=\begin{bmatrix}\cos\theta&-r\sin\theta\\
\sin\theta&r\cos\theta\end{bmatrix}.
\]
If measurement covariance in polar coordinates is \(R_{polar}\), first-order
Cartesian measurement covariance is
\[
R_{BEV}\approx H R_{polar} H^T.
\]
Angular standard deviation of 1 degree at 50 m gives roughly 0.87 m tangential
standard deviation. This motivates an elongated spatial kernel when aggregating
radar evidence into BEV. It is an uncertainty-propagation argument, not evidence
that every learned Gaussian extent is a calibrated uncertainty estimate.

For a fusion research project, keep separate variables for **object extent**,
**measurement noise**, and **uncertainty in estimated state**. Under a suitable
independent additive-error model, covariances can add; they cannot be substituted
for each other simply because all are Gaussian-shaped.

## 4.5 Choose the experiment to fit the data you already know

| Dataset | Relevant representation | A sensible first extension |
|---|---|---|
| [nuScenes](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) | Calibrated cameras and radar point detections, with radar velocity-related fields | Radar-informed BEV feature aggregation or sparse geometric constraints |
| [RADIATE](https://pro.hw.ac.uk/radiate/doc/dataset/) | Scanning range–azimuth radar images; its documentation states no Doppler output | Fit/synthesize radar power imagery with a suitable forward model |
| [View of Delft](https://tudelft-iv.github.io/view-of-delft-dataset/) | Calibrated camera/LiDAR and 3+1D radar data | Explore spatial/velocity evidence and camera–radar fusion |

These are **suggested projects**, not implemented integrations. A point-detection
dataset cannot generally recover the discarded raw radar signal. RADIATE is closer
in data type to scanning-radar image synthesis, but still requires adapting sensor
parameters and preprocessing; it is not a drop-in Boreas replacement.

Start with a short static, well-calibrated sequence before dynamic traffic. Split
by scene for cross-scene perception claims, and by held-out pose/time for
scene-specific rendering claims. Check time offsets before increasing model size.

**A practical project for you:** splat radar detections into BEV with propagated
measurement covariance, combine them with camera features, and compare with
fixed-size kernels at equal compute. Report BEV IoU or detection metrics, plus
performance versus range and weather. This tests a focused hypothesis without
first building a full radar simulator.
