# A robust shape model for blood vessels analysis

**Pau Romero, Abel Pedrós, Rafael Sebastian, Miguel Lozano, Ignacio García‑Fernández***  
*CoMMLab – Computational Multiscale Simulation Lab, University of Valencia, Spain*

---

## ABSTRACT

The availability of *digital twins* for the cardiovascular system will enable insightful computational tools for both research and clinical practice. This, however, demands robust and well‑defined models and methods for the different steps involved in the process. We present a **vessel coordinate system (VCS)** that enables the unambiguous definition of locations in a vessel section by adapting the idea of cylindrical coordinates to the vessel geometry. Using the VCS model, point correspondence can be defined among different samples of a cohort, allowing data transfer, quantitative comparison, shape coregistration, or population analysis. Furthermore, the VCS model allows for the generation of specific meshes (e.g., cylindrical grids) necessary for an accurate reconstruction of the geometries used in fluid simulations. We provide the technical details for coordinates computation and discuss the assumptions taken to guarantee that they are well defined. The VCS model is tested in a series of applications: we present a robust, low‑dimensional, patient‑specific vascular model and use it to study phenotype variability of the thoracic aorta within a patient cohort. Point correspondence is exploited to build a **haemodynamics atlas** of the same cohort, where fluid simulations (Navier–Stokes, finite‑volume method) are conducted using **OpenFOAM v10**. We finally discuss the impact of the VCS in areas such as shape modeling and computational fluid dynamics (CFD).

---

## 1. Introduction

Encoding and representation of anatomy is one of the grounding pieces for the development of the so‑called *digital twin*. When building a digital patient, data from different sources has to be put together on a common anatomical substrate. The combination of mechanistic models with data‑science models has proven to be a powerful tandem for interpreting simulation results, generating data for model training, or synthesizing large virtual cohorts for in‑silico trials. These computational tools require suitable representations of the anatomy to encode patient information, which means representations that are independent of the patient from a computational point of view.

In the case of modeling vascular anatomy, we find different technical solutions and strategies that can be classified in four groups. Some authors use a set of anatomical traits that are relevant for their clinical problem. In order to study characteristics associated to ascending aorta aneurysm, for instance, reduced sets of anatomical features have been defined and related to biomechanical descriptors or biomarkers. This approach is not intended to provide an encoding of the anatomy beyond the goal of the study, and it does not allow, for instance, the reconstruction of an aorta from its characteristics.

A common approach for a computational description of vessels is to extract their centerline and analyze the morphology of cross sections, or even to build models by bifurcations using their lumen as the union of spheres located along the centerlines. Other works build atlases of vessel bifurcations by studying centerline trees or by modeling cross sections at regular distances. In order to have an analytic description of the wall of a vessel, some authors use closed curves (e.g., ellipses or polynomials) along the centerline. A more general approach, in the sense that it is not specific for vessels, is the use of **large deformation diffeomorphic metric mapping (LDDMM)** to encode the geometry. This technique characterizes geometries through the integral of vector fields on their surface, and has been successfully applied to study variability of aorta shape and other organs. When machine learning models are to be involved, it is often a requirement that all the samples use a common encoding; the same dimensionality, with common meaning for all features. With this aim, some works use template meshes for the wall of the thoracic aorta and project points onto the wall of each sample; others estimate functional biomarkers using neural networks trained on a template.

This diversity of methods enables descriptions tailored to the requirements of each problem. However, not all methods are equally adequate for inter‑patient comparison or for population‑based statistical studies, and some hamper reproducibility and comparability. For these reasons, some authors stress the importance of having robust and standardized methods for the phases of the development and analysis of multiscale biophysical models and simulation results.

In this paper, we propose a new representation of the anatomy of blood vessels. We present a **Vessel Coordinate System (VCS)** that serves as a reference to describe a location on the vessel wall or in its lumen. The VCS enables quantitative comparison of data fields defined on a set of vessels corresponding to the same anatomical section, and serves as a tool for the analysis and visualization of fields defined on a vessel. We provide a detailed description of the different steps to build the reference system and justify the particular decisions, with the aim of reproducibility, anatomy comparability, and robustness. We show that, on the proposed VCS, a robust, low‑dimensional, differentiable representation of the vessel surface can be built, enabling point‑wise comparison of anatomies and of data fields defined on them, without ambiguity and at a very reasonable computational cost. After defining the VCS, we show use cases: a patient‑specific vascular model for the thoracic aorta is built from imaging data; a statistical shape model of the aorta is developed from a cohort; and a haemodynamic atlas is computed for that cohort.

---

## 2. Material and methods

We develop a coordinate system that enables a unique description of a point inside the vessel lumen or on the vessel wall. Our scheme is similar in spirit to cylindrical coordinates, but adapted to tubular vessels and equipped with a robust definition of the longitudinal axis and local reference frames.

### 2.1. Material

We use a dataset of **30 thoracic aortas** to test the methodology on a wide range of anatomies with variable ratios of curvature and diameter. The cohort contains patients aged 78–89 years old, suffering from aortic valve stenosis; some present ascending aorta aneurysms. Expert radiologists manually segmented the aortas from CT images acquired at the mesosystolic phase of the cardiac cycle. The VCS is restricted to single‑branch vessels; supra‑aortic branches and coronary arteries are removed in all cases. The final input is a set of **30 anonymized triangular surface meshes**. The acquisition procedures met the requirements of the Declaration of Helsinki and were approved by the institutional ethics committee.

### 2.2. Vessel coordinate system

The wall of a vessel segment **with no bifurcations** can be considered homeomorphic to a cylinder. The proposed VCS is a generalization of the cylindrical coordinate system specialized to vessels, aligning a smoothly varying **centerline** with the longitudinal direction. A longitudinal axis must be defined, however, and the local frames must be robustly transported along it.

#### 2.2.1. Definition of the vessel coordinate system

Consider a vessel segment limited by two cross sections. Let the wall (or lumen) be segmented, and let the wall be represented as a triangle mesh or volumetric surface. Assume the two ends are defined by two planes \(A\) and \(B\) approximately orthogonal to the vessel direction, and that we have two points \(p_A \in A\) and \(p_B \in B\) inside the vessel lumen.

We build a regular parametric centerline curve \(c:I\to\mathbb{R}^3\) that defines the longitudinal direction, and a **local orthonormal reference frame** along \(c\), \(\{t(\tau), v_1(\tau), v_2(\tau)\}\), with unit tangent \(t\). Given a point \(x\in\mathbb{R}^3\) on the wall or inside the lumen, its cylindrical‑like VCS coordinates \((\tau,\theta,\rho)\) are defined as

$$
\tau \;=\; \underset{t\in I}{\arg\min}\; \| x - c(t)\|, \tag{1}
$$

$$
\theta \;=\; \operatorname{angle}\!\left(v_1(\tau),\; x - c(\tau)\right), \tag{2}
$$

$$
\rho \;=\; \|x - c(\tau)\|. \tag{3}
$$

Given \$((\tau,\theta,\rho)\)$, the Cartesian point is recovered by
$$
x(\tau,\theta,\rho) \;=\; c(\tau) \;+\; \rho\big(v_1(\tau)\cos\theta + v_2(\tau)\sin\theta\big). \tag{4}
$$

For this coordinate system to be well defined, (i) the **longitudinal curve** \(c\) and (ii) the **local reference frame** must be robustly computed.

#### 2.2.2. Computation of the longitudinal curve

We compute the curve that best represents the medial axis of the geometry. On a volumetric discretization of the vessel lumen (derived from imaging or from the wall), we compute the **distance field** to the wall points. We then build a discrete path that traverses lumen points maximizing the distance to the wall along the curve. We use an **A\***‑graph search on a neighborhood defined by distance, taking the inverse of the distance scalar field as the cost function. The discrete path is then regularized to a differentiable curve by fitting a **cubic B‑spline** through the points, with approximately constant speed and parameter domain \(I = [0,1]\).

#### 2.2.3. Local reference frame

The angular coordinate \(\theta\) requires a stable angular origin in the normal plane along the centerline. To ensure stability among different patients and anatomies, we transport a **reference frame** along \(c\) using **parallel transport** in the normal plane. Formally, given a parametric curve \(c:[0,1]\!\to\!\mathbb{R}^3\) and a vector \(v^0 \perp t(0)\), the parallel transport \(v(s)\) of \(v^0\) can be expressed in terms of the tangent vector \(t\) via rotations around \(t\) (see standard references). Starting from an initial frame \(\{v_1^0, v_2^0\}\) at \(c(0)\), we forward‑Euler integrate the transport, and when necessary rotate by an angle \(\alpha=\arccos\big(t(s)\!\cdot\!t(s+h)\big)\) around the direction \(r = t(s)\times t(s+h)\). The transported local frame at \(c(\tau)\) is \(\{v_1(\tau),v_2(\tau)\}\), dropping the parameter \(s\) unless it leads to ambiguity.

#### 2.2.4. Computing coordinates of a point

To compute coordinates \((\tau,\theta,\rho)\) of a given point \(x\), we assume the minimizer in Eq. (1) is unique (reasonable for typical vessel segments).

- **Longitudinal coordinate** \(\tau\). Solve Eq. (1) for \(\tau\). This is a 1‑D minimization, and the optimality condition implies orthogonality of \(c'(\tau)\) to the vector \(c(\tau)-x\). We exploit that \(c\) is a spline to compute derivatives analytically.
- **Radial distance** \(\rho\). Once \(\tau\) is known, set \(\rho= \|c(\tau)-x\|\).
- **Angular coordinate** \(\theta\). Solve the parallel‑transport reference frame as above and compute \(\theta\) from the dot product of \(v_1(\tau)\) with the vector \(x-c(\tau)\); \(\theta \in [0,2\pi)\).

#### 2.3. Encoding vessel anatomy

We next present contexts where the VCS can be applied. First, we show how to use the coordinates to build a representation of the vessel itself, under the assumption that the wall is **star‑convex** with respect to the centerline (a mild condition satisfied widely in practice). With this assumption, the vessel wall can be described as a differentiable surface parametrized by \((\tau,\theta)\) with a radius function \(\rho_w(\tau,\theta)\). Thus,

$$
x(\tau,\theta) \;=\; c(\tau) \;+\; \rho_w(\tau,\theta)\,\big(v_1(\tau)\cos\theta + v_2(\tau)\sin\theta\big). \tag{5}
$$

##### 2.3.1. Patient‑specific vascular model

We approximate the centerline \(c\) using a **B‑spline** of order 3,
$$
c(\tau) \approx \sum_{i=0}^{m} \mathbf{c}_i\, B_{L_i}(\tau), \tag{6}
$$
where \(\mathbf{c}_i\in\mathbb{R}^3\) are control points and \(\{B_{L_i}\}\) are basis functions over a uniform knot vector on \([0,1]\).

For the wall radius we fit a **uniform bivariate spline**,
$$
\rho_w(\tau,\theta) \approx \sum_{i=0}^{n_\tau}\sum_{j=0}^{n_\theta} b_{ij}\, B_i(\tau)\, B_j(\theta), \qquad (\tau,\theta)\in[0,1]\times[0,2\pi). \tag{7}
$$

Combining (5)–(7), the aorta wall is represented as
$$
x(\tau,\theta) \;=\; \sum_{i=0}^{m}\mathbf{c}_i\, B_{L_i}(\tau) \;+\; 
\left(\sum_{i=0}^{n_\tau}\sum_{j=0}^{n_\theta} b_{ij}\, B_i(\tau)\, B_j(\theta)\right)\!
\big(v_1(\tau)\cos\theta + v_2(\tau)\sin\theta\big). \tag{8}
$$

This yields a differentiable parameterization with a **reduced number of degrees of freedom**. We gather the spline coefficients into a feature vector
$$
\mathbf{a} = (\mathbf{c}_0,\ldots,\mathbf{c}_m,\, b_{00},\ldots,b_{n_\tau n_\theta}) = (a_1,\ldots,a_q). \tag{9}
$$

Given a wall mesh with Cartesian points \(p\), we compute VCS coordinates \((\tau(p),\theta(p),\rho(p))\) and the **approximation residual**
$$
r(p) \;=\; \big\|\, p - x(\tau(p),\theta(p))\,\big\|. \tag{10}
$$

##### 2.3.2. Coresgistration of a cohort of vessels

A common preprocessing step when dealing with several samples of the same anatomy is geometry alignment. Using the VCS, each vessel yields a **regular grid** in the \((\tau,\theta)\) domain, giving an automatic one‑to‑one point correspondence **by definition**. During alignment, the centerline coefficients serve as control points so that rigid transformations operate consistently in the VCS.

##### 2.3.3. Population representation and analysis

Let \(\{\mathbf{a}^k\}_{k=1}^M\) denote the feature vectors of the \(M\) aortas in the cohort. We perform **Principal Component Analysis (PCA)** on the coefficient space,
$$
\mathbf{a} \;=\; \boldsymbol{\mu} \;+\; \sum_{\nu=1}^{m'} \alpha_\nu\, \mathbf{u}_\nu, \tag{11}
$$
where \(\boldsymbol{\mu}\) is the mean vector and \(\mathbf{u}_\nu\) are uncorrelated principal directions (“deformation modes”). This enables statistical shape modeling and population analysis.

### 2.4. Application to inter‑patient variability analysis

The VCS defines a one‑to‑one point correspondence between vessels, enabling quantitative comparison of any field defined on the surface or in the lumen. We illustrate this by building a **haemodynamic atlas** from CFD simulations on the cohort.

#### 2.4.1. CFD simulation setup

We follow a standard setup for pulsatile aortic flow. Rigid walls and **steady‑state** simulations with constant inflow boundary conditions in peak systole are considered. Blood is modeled as **incompressible Newtonian fluid** with **kinematic viscosity** \(3.37\times10^{-6}\, \text{m}^2/\text{s}\). A constant **Dirichlet** condition for the inlet velocity (flat profile, mean \(1.2\,\text{m/s}\)) is imposed on an effective inlet area of **1 cm²** (representative of critical aortic stenosis). No‑slip at the wall; **zero‑pressure (Neumann)** at the outlet. We discretize the Navier–Stokes equations with a **finite‑volume method** using **OpenFOAM v10** and the SIMPLE algorithm, using hexa‑dominant meshes generated with `snappyHexMesh`. Convergence is declared when residuals fall below \(10^{-5}\). *These simulations are used for methodological illustration; no biophysical conclusions are drawn.*

#### 2.4.2. Haemodynamic atlas on the aorta

For each aorta we compute the fields of **velocity**, **pressure**, and **wall shear stress** (the latter only on the wall). Each field is sampled on the same regular VCS grid, turning each subject into a comparable vector of measurements. PCA is performed per field to study the most significant modes of flow variability across the cohort.

---

## 3. Results

We present results following the structure of Section 2: (i) analysis of the patient‑specific model as a tool to approximate the vessel wall, (ii) results of the statistical shape analysis of the cohort, and (iii) construction of a haemodynamic atlas.

### 3.1. Patient‑specific model

The parametric model depends on the size of the knot vectors. Let \(L\) be the number of knots for the **centerline** parameter \(\tau\) in (6), and \(K\) and \(R\) the numbers of knots for \(\tau\) and \(\theta\) in the **radius surface** (7), respectively. We select representative values \(L=9\), \(R=15\), \(K=19\) for the cohort.

Quality is assessed via the residual \(r\) in (10). Across the 30 aortas the B‑spline surface reproduces the input wall tightly, with residuals typically **below 1 mm**, except for localized regions (e.g., the sinuses of Valsalva and near the aortic arch) where curvature is extreme or the centerline slightly wiggles when \(L\) is too high.

A **sensitivity analysis** varying \((L,R,K)\in\{5,\dots,19\}^3\) reveals: (i) increasing \(K\) consistently improves approximation (smoother surfaces), (ii) very low or very high \(L\) increases error—too few degrees of freedom underfit, too many induce centerline oscillations, and (iii) low \(R\) degrades the upper‑tail of the residual distribution (75‑th percentile), especially for large \(L\).

### 3.2. Population analysis

Patient‑specific models were computed for all 30 aortas. PCA on the feature vectors yielded the **mean aorta** and principal deformation modes. The average geometry is smooth and captures the sinuses of Valsalva, indicating the VCS yields a consistent point correspondence. The first deformation mode predominantly changes the **radius** (narrowing/widening), while the second mode counter‑balances ascending vs. descending sections. Other modes show **local effects** around the sinuses and the sinotubular junction. Centerline warping is measured as the Euclidean distance between points with equal \(\tau\) on mean and deformed curves; the third mode mainly affects bending of the distal arch, while a fifth mode affects the overall roundness without much centerline change.

### 3.3. Haemodynamic atlas

CFD was performed on each aorta. On the shared VCS grid we computed averages and PCA per field. Velocity magnitudes, pressures (interior and wall), and wall shear stress exhibit cohort‑characteristic regions of high variability, aligning with anatomical features highlighted by the shape modes (e.g., sinuses and arch). The VCS facilitates **field visualization** by clipping the lumen with constant \(\theta\) planes and by mapping wall‑only quantities without re‑meshing.

---

## 4. Discussion

We define a coordinate system that **adapts to vessel anatomy** and allows the unambiguous identification of a point both in the lumen and on the wall. This enables the definition of a correspondence between the anatomy of different patients, with applications to inter‑patient comparison, quantitative cohort analysis, and patient‑specific data aggregation. The computation only requires identification of the vessel segment and a reference frame for one of the cross sections; from there on, all steps are deterministic and reproducible.

The **patient‑specific model** based on spline approximations yields a compact, differentiable representation suitable for statistical modeling and simulation. The **population model** describes principal modes of anatomical variation, while the **haemodynamic atlas** demonstrates how the same VCS supports quantitative comparison of CFD fields across individuals.

We anticipate the VCS to be useful in digital‑twin pipelines, where it can standardize data exchange between imaging, geometry processing, simulation, and machine‑learning components. Future work includes extending the method to bifurcating vessels, refining robustness under extreme anatomies, and exploring clinical endpoints driven by VCS‑based biomarkers.

---

*Notes*: Figures and figure captions referenced in the original article have been intentionally omitted as requested. Typesetting uses Markdown with LaTeX math for clarity.
