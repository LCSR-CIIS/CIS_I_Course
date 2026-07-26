# Author: X.M. Christine Zhu
# Date: 03/28/2026

# Kinematic Error Propagation — Pseudo Code & System Overview

A framework for **first-order uncertainty propagation** through geometric networks (kinematic chains)
following the **CIS I right-multiplicative perturbation** convention for SE(3) transforms.

---

## 0. High-Level System Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GeometricNetwork                                 │
│                                                                         │
│   Frames: {W, Trk, Cam, Robot, Tool, ...}                               │
│                                                                         │
│        W ──── Trk ──── Cam                                              │
│        │                 │                                              │
│       Robot             Marker                                          │
│        │                                                                │
│       Tool                                                              │
│                                                                         │
│   Each edge stores: UncertainTransform { F_nom (4×4), C (6×6) }         │
│                                                                         │
│   Queries:                                                              │
│     query(A, B)                → PathResult       (single BFS path)     │
│     query_frame(A, B)          → FusedQueryResult (all paths, fused)    │
│     query_point(pt, frame)     → (p, Cp)          (point + covariance)  │
│     find_independent_loops()   → loop pairs                             │
│     condition_on_loop(...)     → LoopPosterior (tighter covariance)     │
│     query_auto_loop_posterior  → MultiLoopPosterior (fully automatic)   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Perturbation Model (Right-Multiplicative, CIS I)

The true transform is modeled as a small perturbation around a nominal:

```
T_true = T_nom ∘ Exp(η)

where η = [α; ε]  ∈ R^6    (CIS I ordering)
              α = rotation perturbation  (3×1)
              ε = translation perturbation  (3×1)

Gaussian noise: η ~ N(0, C),   C is 6×6 covariance matrix
```

---

## 2. SE(3) Math Primitives  (`se3.py`)

```
────────────────────────────────────────────────────────────────
FUNCTION exp_se3(ξ ∈ R^6) → T ∈ SE(3)          [Rodrigues formula]
  φ = ξ[0:3],  ρ = ξ[3:6]                       (rotation, translation)
  θ = ||φ||
  IF θ < ε THEN use series expansion
  R = I + sin(θ)/θ · [φ]× + (1-cos(θ))/θ² · [φ]×²
  J = I + (1-cos(θ))/θ² · [φ]× + (θ-sin(θ))/θ³ · [φ]×²
  t = J · ρ
  RETURN [[R, t], [0,0,0,1]]
────────────────────────────────────────────────────────────────
FUNCTION log_se3(T ∈ SE(3)) → ξ ∈ R^6
  R = T[0:3,0:3],  t = T[0:3,3]
  θ = arccos(clip((trace(R)-1)/2, -1, 1))
  IF θ < ε THEN φ = [R32-R23, R13-R31, R21-R12] / 2
  ELSE       φ = θ/(2·sin(θ)) · [R32-R23, R13-R31, R21-R12]
  J⁻¹ = I - φ/2 × + (1/θ² - (1+cos(θ))/(2θ·sin(θ))) · [φ]×²
  ρ = J⁻¹ · t
  RETURN [φ; ρ]
────────────────────────────────────────────────────────────────
FUNCTION adjoint_se3(T) → Ad_T ∈ R^{6×6}
  R = T[0:3,0:3],  t = T[0:3,3]
  RETURN [[R,        0   ],
          [[t]×·R,   R   ]]      ← maps perturbations between frames
────────────────────────────────────────────────────────────────
FUNCTION skew(w ∈ R^3) → W ∈ R^{3×3}       [skew-symmetric matrix]
  RETURN [[ 0,  -w2,  w1],
          [ w2,   0, -w0],
          [-w1,  w0,   0]]
────────────────────────────────────────────────────────────────
```

---

## 3. UncertainTransform  (`uncertain_geometry.py`)

A rigid transform with attached Gaussian uncertainty.

```
CLASS UncertainTransform:
    F_nom      : 4×4          ← nominal homogeneous transform
    C          : 6×6          ← covariance of perturbation η (CIS I: [α; ε])
    convention : Convention   ← RIGHT (default) or LEFT
```

### 3.0 Convention enum

Two perturbation conventions are supported:

```
Convention.RIGHT  (body-frame, CIS I default):
    T_true = F_nom @ Exp(η)     η ~ N(0, C)

Convention.LEFT  (world-frame):
    T_true = Exp(η) @ F_nom     η ~ N(0, C)

Bridge formulas:
    C_left  = Ad(F_nom)    @ C_right @ Ad(F_nom)^T
    C_right = Ad(F_nom^-1) @ C_left  @ Ad(F_nom^-1)^T
```

```
FUNCTION as_right_convention(U):
    IF U.convention == RIGHT: RETURN copy of U
    Ad_inv   = adjoint_se3(inv(U.F_nom))
    C_right  = Ad_inv @ U.C @ Ad_inv^T
    RETURN UncertainTransform(U.F_nom, symmetrise(C_right), RIGHT)

FUNCTION as_left_convention(U):
    IF U.convention == LEFT: RETURN copy of U
    Ad       = adjoint_se3(U.F_nom)
    C_left   = Ad @ U.C @ Ad^T
    RETURN UncertainTransform(U.F_nom, symmetrise(C_left), LEFT)

FUNCTION from_left_covariance(F_nom, C_left):
    Ad_inv   = adjoint_se3(inv(F_nom))
    C_right  = Ad_inv @ C_left @ Ad_inv^T
    RETURN UncertainTransform(F_nom, C_right, RIGHT)
```

### 3a. Composition  (`A @ B` or `A.compose(B)`)

`compose` dispatches to one of four subroutines based on the convention
of each operand. The result convention always follows the left operand.

```
Dispatch table:
    self \ other | RIGHT        | LEFT
    -------------|--------------|----------------
    RIGHT        | _compose_rr  | _compose_rl
    LEFT         | _compose_lr  | _compose_ll
```

```
─── RIGHT @ RIGHT → RIGHT ─────────────────────────────────────
    T_ab = F_ab @ Exp(η_ab),   T_bc = F_bc @ Exp(η_bc)
    T_ac = F_ac @ Exp(Ad(F_bc^{-1}) η_ab + η_bc)

FUNCTION _compose_rr(F_ab, C_ab, F_bc, C_bc):
    F_ac    = F_ab @ F_bc
    Ad_inv  = adjoint_se3(inv(F_bc))
    C_ac    = Ad_inv @ C_ab @ Ad_inv^T + C_bc
    RETURN UncertainTransform(F_ac, symmetrise(C_ac), RIGHT)

─── LEFT @ LEFT → LEFT ─────────────────────────────────────────
    T_ab = Exp(η_ab) @ F_ab,   T_bc = Exp(η_bc) @ F_bc
    T_ac = Exp(η_ab + Ad(F_ab) η_bc) @ F_ac

FUNCTION _compose_ll(F_ab, C_ab, F_bc, C_bc):
    F_ac    = F_ab @ F_bc
    Ad_ab   = adjoint_se3(F_ab)
    C_ac    = C_ab + Ad_ab @ C_bc @ Ad_ab^T
    RETURN UncertainTransform(F_ac, symmetrise(C_ac), LEFT)

─── RIGHT @ LEFT → RIGHT ───────────────────────────────────────
    T_ab = F_ab @ Exp(η_R),   T_bc = Exp(η_L) @ F_bc
    T_ac ≈ F_ac @ Exp(Ad(F_bc^{-1})(η_R + η_L))

FUNCTION _compose_rl(F_ab, C_ab, F_bc, C_bc):
    F_ac    = F_ab @ F_bc
    Ad_inv  = adjoint_se3(inv(F_bc))
    C_ac    = Ad_inv @ (C_ab + C_bc) @ Ad_inv^T
    RETURN UncertainTransform(F_ac, symmetrise(C_ac), RIGHT)

─── LEFT @ RIGHT → LEFT ────────────────────────────────────────
    T_ab = Exp(η_L) @ F_ab,   T_bc = F_bc @ Exp(η_R)
    T_ac = Exp(η_L + Ad(F_ac) η_R) @ F_ac

FUNCTION _compose_lr(F_ab, C_ab, F_bc, C_bc):
    F_ac    = F_ab @ F_bc
    Ad_ac   = adjoint_se3(F_ac)
    C_ac    = C_ab + Ad_ac @ C_bc @ Ad_ac^T
    RETURN UncertainTransform(F_ac, symmetrise(C_ac), LEFT)
```

Note: `symmetrise(C) = 0.5 * (C + C^T)` enforces exact symmetry after
floating-point arithmetic.

### 3b. Inversion

```
─── RIGHT convention ───────────────────────────────────────────
    T_true = F_nom @ Exp(η_R)
    T_true^{-1} = F_nom^{-1} @ Exp(-η_R)   (re-expressed as right)
    C_inv = Ad(F_nom) @ C @ Ad(F_nom)^T

─── LEFT convention ────────────────────────────────────────────
    T_true = Exp(η_L) @ F_nom
    T_true^{-1} = Exp(-Ad(F_nom^{-1}) η_L) @ F_nom^{-1}
    C_inv = Ad(F_nom^{-1}) @ C @ Ad(F_nom^{-1})^T

FUNCTION inv(U):
    F_inv = inv_se3(U.F_nom)
    IF U.convention == RIGHT:
        Ad = adjoint_se3(U.F_nom)
    ELSE:
        Ad = adjoint_se3(F_inv)
    C_inv = Ad @ U.C @ Ad^T
    RETURN UncertainTransform(F_inv, C_inv, U.convention)
```

### 3c. Point Transformation (`apply_to_point`)

```
FUNCTION apply_to_point(U, p_local ∈ R^3, Cp_local ∈ R^{3×3} or None):

    p_nom = R · p_local + t            ← nominal transformed point

    CIS I Jacobian w.r.t. η = [α; ε]  (right-perturbation convention):
    J_η = [-R · [p_local]×  |  R]     ← shape 3×6

    C_point =  J_η · C · J_ηᵀ                    ← from pose uncertainty
    IF Cp_local is not None:
        C_point += R · Cp_local · Rᵀ              ← from point's own uncertainty

    RETURN (p_nom, C_point)
```

---

## 3d. Uncertain Cartesian Types  (`uncertain_types.py`)

Higher-level wrappers that carry covariance through ordinary arithmetic.
All first-order Gaussian error propagation, matching the CIS I convention.

```
CLASS uScalar:
    val : float    ← nominal scalar value
    var : float    ← variance  (σ²)

Operations (first-order Gaussian):
    us1 + us2  →  val = v1+v2,          var = σ1²+σ2²
    us1 - us2  →  val = v1-v2,          var = σ1²+σ2²
    us1 * us2  →  val = v1*v2,          var = (v2·σ1)²+(v1·σ2)²
    us1 / us2  →  val = v1/v2,          var = (σ1/v2)²+(v1·σ2/v2²)²
    us1 ** n   →  val = v1^n,           var = (n·v1^(n-1)·σ1)²
    a   * us1  →  val = a·v1,           var = (a·σ1)²
    -us1       →  val = -v1,            var = σ1²
```

```
CLASS uVector:
    val : ndarray (n,)    ← nominal vector
    cov : ndarray (n,n)   ← covariance matrix

Operations:
    uv1 + uv2         →  val = v1+v2,   cov = C1+C2
    uv1 - uv2         →  val = v1-v2,   cov = C1+C2
    a   * uv1         →  val = a·v1,    cov = a²·C1
    A   @ uv1         →  val = A·v1,    cov = A·C1·Aᵀ     (A is 2-D matrix)
    w   @ uv1         →  uScalar:  val = w·v1, var = w·C1·wᵀ  (w is 1-D)
    uv1 @ A           →  val = v1·A,    cov = A^T·C1·A
    uv1.dot(w)        →  uScalar:  same as w @ uv1
    uv1.norm()        →  uScalar:  val = ||v1||,
                                   var = (v1/||v1||)·C1·(v1/||v1||)ᵀ
```

```
CLASS uvct3:
    p : vct3           ← nominal 3-D point
    C : 3×3            ← position covariance

Constructors:
    uvct3(p)           p is vct3 or array-like, zero covariance
    uvct3(p, C)        p + 3×3 covariance

Properties:
    .x, .y, .z         coordinate access
    .p                 underlying vct3

Operations:
    uvct3 + uvct3  →  uvct3:  p=p1+p2,   C=C1+C2
    uvct3 + vct3   →  uvct3:  p=p1+p2,   C=C1      (p2 certain)
    vct3  + uvct3  →  uvct3:  same
```

```
CLASS uRot:
    R          : Rot    ← nominal rotation
    C          : 3×3    ← rotation covariance (right-perturbation, α-space)
    convention : Convention   ← always RIGHT after construction

Constructors:
    uRot(R)                  Rot, zero covariance
    uRot(R, Calpha)          Rot + 3×3 covariance (right-convention)
    uRot('x'/'y'/'z', uAngle)
        C = (ax ⊗ ax) · var(uAngle)           (axis is certain)
    uRot(uAxis, angle)       axis uncertain, angle certain
        C = angle² · C_axis
    uRot(uAxis, uAngle)      both uncertain (first-order)
        C = angle² · C_axis + (ax ⊗ ax) · var(uAngle)

Multiplication (right-perturbation convention throughout):
    uRot * uvct3  →  uvct3:  p = R·p_in,  C = J_α·C_R·J_αᵀ + R·C_p·Rᵀ
                             where J_α = -R·[p_in]×
    uRot * vct3   →  uvct3:  same without C_p term
    uRot * uRot   →  uRot:   R = R1·R2,  C = R2ᵀ·C1·R2 + C2
    uRot * Rot    →  uRot:   R = R1·R2,  C = R2ᵀ·C1·R2
    Rot  * uRot   →  uRot:   R = R1·R2,  C = C2        (R1 certain)

Convention bridge:
    uRot.from_left_covariance(R, C_left)
        C_right = Rᵀ · C_left · R
        RETURN uRot(R, C_right)
    uRot.as_right_convention()   → no-op if already RIGHT
    uRot.as_left_convention()    → C_left = R · C_right · Rᵀ
```

```
CLASS uFrame:
    Wraps UncertainTransform; provides Frame/Rot/vct3-typed interface.

    F_nom  : 4×4   ← nominal homogeneous matrix
    C      : 6×6   ← pose covariance [α; ε] (right-convention)

Constructors:
    uFrame(F)            Frame, zero covariance
    uFrame(F, covEta)    Frame + 6×6 pose covariance
    uFrame(uR, up)       uncertain rotation + uncertain point
                         C = block_diag(uR.C, up.C)
    uFrame(R,  up)       certain rotation  + uncertain point
    uFrame(uR, p)        uncertain rotation + certain  point

Properties:
    .F         → Frame   (nominal)
    .F_nom     → 4×4 numpy array
    .C         → 6×6 covariance
    .convention → Convention

Multiplication:
    uFrame * uFrame  →  uFrame  (delegates to UncertainTransform.compose)
    uFrame * Frame   →  uFrame
    uFrame * uvct3   →  uvct3   (apply_to_point with point covariance)
    uFrame * vct3    →  uvct3   (apply_to_point, no input covariance)
    Frame  * uFrame  →  uFrame  (left frame is certain)

Methods:
    .inv()                    → uFrame (delegates to UncertainTransform.inv)
    .to_uncertain_transform() → UncertainTransform

Convention bridge (same semantics as UncertainTransform):
    uFrame.from_left_covariance(F_or_nom, C_left) → uFrame
    uFrame.as_right_convention() → uFrame
    uFrame.as_left_convention()  → uFrame
```

---

## 4. GeometricNetwork  (`network.py`)

```
CLASS GeometricNetwork:
    _adj     : dict[frame → dict[neighbor → Edge]]
    _points  : dict[point_name → PointNode]

    ┌── Edge ──────────────────────────────────────────┐
    │  transform  : UncertainTransform                 │
    │  is_certain : bool   (perfectly known edge)      │
    │  is_forward : bool   (canonical direction)       │
    │  edge_type  : "se3" | "rot_only" | "trans_only"  │
    │  edge_id    : int    (shared with inverse edge)  │
    └──────────────────────────────────────────────────┘

    ┌── PointNode ─────────────────────────────────────┐
    │  frame    : str       (attached frame)           │
    │  p_local  : (3,)      (local coordinates)        │
    │  Cp       : 3×3       (local point covariance)   │
    └──────────────────────────────────────────────────┘
```

### 4a. Adding Edges

```
FUNCTION add_edge(src, dst, U_transform, add_inverse=True, edge_type="se3"):
    Store U_transform in _adj[src][dst]
    IF add_inverse:
        Store U_transform.inv() in _adj[dst][src]
        Both directions share the SAME edge_id   ← key for correlation tracking
```

### Edge Types Explained

```
   se3:        full 6-DOF uncertain transform  (default)
   rot_only:   covariance has zeros in translation block
               ┌──────────────┐
               │ C_rot │  0   │
               │───────┼──────│
               │   0   │  0   │
               └──────────────┘
   trans_only: covariance has zeros in rotation block
               ┌──────────────┐
               │   0   │  0   │
               │───────┼──────│
               │   0   │C_trans│
               └──────────────┘
   vector:     rotation forced to identity, pure displacement
```

---

## 5. Path Finding & Composition  (`network.py`)

### 5a. Single Shortest Path (BFS)

```
FUNCTION find_path(start, goal):
    queue = [[start]]
    visited = {start}
    WHILE queue not empty:
        path = queue.pop_front()
        node = path[-1]
        IF node == goal: RETURN path
        FOR each neighbor of node:
            IF neighbor not visited:
                visited.add(neighbor)
                queue.push(path + [neighbor])
    RETURN None   (no path found)
```

### 5b. All Simple Paths (DFS)

```
FUNCTION find_all_paths(start, goal, max_depth=∞):
    results = []
    stack   = [([start], {start})]
    WHILE stack not empty:
        (path, visited) = stack.pop()
        node = path[-1]
        IF node == goal:
            results.append(path)
            CONTINUE
        IF len(path) >= max_depth: CONTINUE
        FOR each neighbor of node:
            IF neighbor not visited:
                stack.push((path+[neighbor], visited∪{neighbor}))
    RETURN results
```

### 5c. Path Composition  (accumulate covariance)

```
FUNCTION compose_along_path(path: list[frame]):
    T = identity UncertainTransform
    FOR each consecutive pair (src, dst) in path:
        edge = _adj[src][dst]
        T = T @ edge.transform       ← calls compose() in §3a
    RETURN T       (full propagated pose + covariance from start to end)
```

### 5d. Main Query

```
FUNCTION query(start, goal):
    path      = find_path(start, goal)
    transform = compose_along_path(path)
    RETURN PathResult(path, transform, edge_ids, certain_mask, ...)
```

### 5e. Multi-Path Query with Bayesian Fusion  (`query_frame`)

**The situation:** Two (or more) simple paths exist between the same pair of frames.
`query()` picks only the BFS shortest path and ignores the rest.
`query_frame()` uses ALL paths and fuses them optimally via Bayes' rule.

```
           path 1: A ──── B ──── C
                    \           /
           path 2:   ──── D ────

    Each path independently estimates the transform perturbation at C:
        path 1:   eta_C ~ N(0, C_1)
        path 2:   eta_C ~ N(0, C_2)

    Since both estimate the SAME quantity, the constraint is:
        eta_1 - eta_2  ≈  0
```

**Perturbation model per path:**

Each path k composes its edges to get an accumulated uncertain transform.
Using the right-multiplicative CIS I convention, the perturbation at the goal
frame accumulates as:

```
    T_k  =  F_nom_k  ∘  Exp(eta_k)

    where  eta_k ~ N(0, C_k)

    C_k  =  Ad_{(F_e2·...·F_em)^{-1}}  *  C_e1  *  Ad_{...}^T
          + Ad_{(F_e3·...·F_em)^{-1}}  *  C_e2  *  Ad_{...}^T
          + ...
          + C_em
         (accumulated via compose() along the path)
```

**Bayes' rule — information form:**

The key insight is that two independent Gaussian estimates of the same quantity
can be fused by summing their precision matrices (inverses of covariances).

Start from Bayes' rule:

```
    p(eta | path 1, path 2)  ∝  p(path 1 | eta)  *  p(path 2 | eta)  *  p(eta)

    Since both likelihoods are Gaussian, their product is also Gaussian.
    In information (precision) form this becomes a simple sum:

    precision_posterior  =  precision_prior  +  precision_1  +  precision_2
```

With a flat (uninformative) prior, this simplifies to:

```
    C_fused^{-1}  =  C_1^{-1}  +  C_2^{-1}  +  ...  +  C_N^{-1}

    C_fused       =  inv(  C_1^{-1}  +  C_2^{-1}  +  ...  +  C_N^{-1}  )
```

The fused mean (when means are non-zero):

```
    mu_fused  =  C_fused  *  ( C_1^{-1} * mu_1  +  C_2^{-1} * mu_2  +  ...  +  C_N^{-1} * mu_N )
```

Key property:

```
    trace(C_fused)  <  trace(C_k)   for all k

    i.e. C_fused is strictly MORE certain than any individual path.
```

In the special case of two equal-uncertainty paths (C_1 = C_2 = C):

```
    C_fused^{-1}  =  C^{-1}  +  C^{-1}  =  2 * C^{-1}

    C_fused       =  (1/2) * C

    trace(C_fused)  =  (1/2) * trace(C)    ← exactly half the uncertainty
```

**Why the nominal transform is not fused:**

In a nominally consistent network all paths give the same nominal transform:

```
    F_nom_1  =  F_nom_2  =  ...  =  F_nom_N
```

So we keep F_nom from path 1 and only fuse the covariances.

**Result type:**

```
CLASS FusedQueryResult:
    transform    : UncertainTransform   ← fused (F_nom from path 1, C = C_fused)
    n_paths      : int                  ← how many paths were found and fused
    path_results : list[PathResult]     ← per-path results before fusion
```

**Pseudocode:**

```
FUNCTION query_frame(start, goal, max_depth=inf):

    ── Step 1: find all simple paths ─────────────────────────────────────
    path_results = query_all_paths(start, goal, max_depth)
    IF path_results is empty:
        RAISE "No path found"

    ── Step 2: single path — no fusion needed ────────────────────────────
    IF len(path_results) == 1:
        RETURN FusedQueryResult(
            transform    = path_results[0].transform,
            n_paths      = 1,
            path_results = path_results
        )

    ── Step 3: multiple paths — apply Bayes' rule ────────────────────────
    F_nom_ref = path_results[0].transform.F_nom    ← nominal from first path

    info_sum = zeros(6×6)
    FOR each path_result in path_results:
        C_k      = path_result.transform.C
        info_sum = info_sum  +  inv(C_k)           ← accumulate precisions

    C_fused = inv( info_sum )                       ← Bayes' rule

    RETURN FusedQueryResult(
        transform    = UncertainTransform(F_nom_ref, C_fused),
        n_paths      = len(path_results),
        path_results = path_results
    )
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Example: two equal-uncertainty paths, sigma = 0.1 per edge     │
│                                                                 │
│      A ──── B ──── C      path 1: trace(C_1) = 0.14             │
│      │             │                                            │
│      └──── D ──────┘      path 2: trace(C_2) = 0.14             │
│                                                                 │
│  info_sum = C_1^{-1} + C_2^{-1}  =  2 * C_1^{-1}                │
│  C_fused  = inv(info_sum)         =  (1/2) * C_1                │
│                                                                 │
│  trace(C_fused) = 0.065   ← roughly half                        │
│                                                                 │
│  Intuition: two independent measurements of the same thing      │
│  give you twice the information → half the uncertainty.         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Loop Discovery  (`network.py`)

**Idea:** Use a BFS spanning tree. Every edge NOT in the tree creates one independent loop.

```
FUNCTION find_independent_loops(start=None, end=None):
    ── Build BFS spanning tree (undirected) ──────────────────────
    tree_edges = {}
    queue      = [root]
    visited    = {root}
    WHILE queue not empty:
        u = queue.pop()
        FOR each neighbor v of u (undirected):
            IF v not visited:
                tree_edges.add((u,v))
                visited.add(v)
                queue.push(v)

    ── Identify back edges (not in tree) ─────────────────────────
    loops = []
    FOR each edge (u, v) in graph:
        IF (u,v) NOT in tree_edges AND (v,u) NOT in tree_edges:
            path_direct = [u, v]                   ← uses back edge
            path_tree   = tree_path(u, v)           ← routes thru tree
            loops.append((path_direct, path_tree))

    IF start/end specified: filter and orient loops
    RETURN loops
```

```
┌──────────────────────────────────────────────────────────────┐
│  Example: Square network with diagonal                       │
│                                                              │
│      A ──── B                                                │
│      │    ╲ │                                                │
│      │     ╲│                                                │
│      C ──── D                                                │
│                                                              │
│  Spanning tree: A-B, A-C, C-D                                │
│  Back edges:    B-D, A-D  →  2 independent loops             │
│                                                              │
│  Loop 1: path_direct=[B,D]  vs  path_tree=[B,A,C,D]          │
│  Loop 2: path_direct=[A,D]  vs  path_tree=[A,C,D]            │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Closed-Loop Constraint Inference  (`closed_loop.py`)

### 7a. The Idea

When two paths share the same start/end frames, the loop they form should be **consistent** (close to identity). This is a constraint we can use to **reduce uncertainty**.

```
         path_res (residual)
    A ──────────────────────── B
    │                          │
    └──────── path_k ──────────┘
         (alternative path)

Constraint:  path_res⁻¹ ∘ path_k  ≈  I      ← loop closes to identity
```

### 7b. Loop Residual Linearization

```
FUNCTION linearize_loop_residual(T_res, T_k, delta=1e-7):
    r₀ = log_se3(T_res.F_nom⁻¹ · T_k.F_nom)   ← 6D residual at zero perturb.

    FOR j = 0..5:
        e_j = one-hot vector at j
        Perturb T_res by +delta in direction j → r_plus
        Perturb T_res by -delta in direction j → r_minus
        J_res[:,j] = (r_plus - r_minus) / (2·delta)   ← finite difference

        Same for T_k → J_k[:,j]

    RETURN LoopLinearization(r₀, J_res, J_k)
```

The linear model is:
```
r(η_res, η_k) ≈ r₀  +  J_res · η_res  +  J_k · η_k
```

### 7c. Single Loop Conditioning  (Bayesian update)

```
FUNCTION condition_on_loop(T_res, T_k, C_nu):
    lin  = linearize_loop_residual(T_res, T_k)

    Prior:
        x = [η_res; η_k] ~ N(0, C₀)
        C₀ = block_diag(T_res.C, T_k.C)      ← 12×12

    Observation model:
        r ≈ 0  →  y = 0 = H·x + b + ν
        H = [J_res | J_k]                     ← 6×12
        b = r₀                                 ← 6×1 nominal residual
        ν ~ N(0, C_nu)                         ← measurement noise

    ┌────────────────────────────────────────────────────┐
    │  Information Filter Update:                        │
    │                                                    │
    │  Λ₀    = C₀⁻¹                   ← prior info       │
    │  Λ_obs = Hᵀ · C_nu⁻¹ · H       ← obs. info         │
    │  Λ_post = Λ₀ + Λ_obs                               │
    │  C_post = Λ_post⁻¹                                 │
    └────────────────────────────────────────────────────┘

    Extract blocks from C_post:
        C_res_post  = C_post[0:6,  0:6 ]
        C_k_post    = C_post[6:12, 6:12]
        C_cross     = C_post[0:6,  6:12]

    RETURN LoopPosterior(C_res_post, C_k_post, C_cross, C_post)
```

**Effect:** Both paths get tighter covariance because the constraint links them.

### 7d. Multi-Loop Conditioning  (N loops simultaneously)

```
FUNCTION condition_on_multiple_loops(T_res, [(T_k1, C_ν1), ...]):
    N = number of loops

    State x = [η_res; η_k1; ...; η_kN]   ← dim 6(1+N)

    C₀ = block_diag(C_res, C_k1, ..., C_kN)

    Stack observation rows:
        H_full  = [H_1; H_2; ...; H_N]          ← 6N × 6(1+N)
        C_nu_full = block_diag(C_ν1, ..., C_νN)  ← 6N × 6N

    Information update (same formula, larger matrices):
        C_post = (C₀⁻¹ + H_fullᵀ C_nu_full⁻¹ H_full)⁻¹

    RETURN MultiLoopPosterior(C_res_post, [C_k_list], [C_cross_list], C_post)
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Why simultaneous?                                              │
│                                                                 │
│  Sequential conditioning (one loop at a time) loses the         │
│  cross-information between loops.                               │
│  Joint conditioning uses ALL constraints at once → tighter.     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Gaussian Covariance Fusion  (`closed_loop.py`)

When multiple **independent** paths connect the same two frames, fuse them optimally.

```
FUNCTION fuse_gaussian_covs(covs: list of 6×6 matrices):
    info_sum = Σ_k  C_k⁻¹               ← sum of information matrices
    C_fused  = info_sum⁻¹
    RETURN C_fused

FUNCTION fuse_gaussians(means: list of R^6, covs: list of 6×6):
    info_sum = Σ_k C_k⁻¹
    C_fused  = info_sum⁻¹
    μ_fused  = C_fused · Σ_k (C_k⁻¹ · μ_k)
    RETURN (μ_fused, C_fused)
```

Used in:
```
FUNCTION query_fused_paths(start, goal):
    all_paths = find_all_paths(start, goal)
    transforms = [compose_along_path(p) for p in all_paths]
    means  = [log_se3(T.F_nom) for T in transforms]
    covs   = [T.C for T in transforms]
    (μ_f, C_f) = fuse_gaussians(means, covs)
    RETURN UncertainTransform(exp_se3(μ_f), C_f)
```

---

## 9. Point-to-Point Queries  (`network.py`)

### 9a. Independent estimate (no correlation)

```
FUNCTION query_relative_vector_independent(pt_src, pt_dst, frame):
    p_src, Cp_src = query_point(pt_src, frame)
    p_dst, Cp_dst = query_point(pt_dst, frame)

    delta   = p_dst - p_src
    C_delta = Cp_dst + Cp_src      ← assumes independence (upper bound)
    RETURN (delta, C_delta)
```

### 9b. Correlation-aware estimate

When two points share edges (same kinematic chain), their uncertainties are **correlated** — the naive sum overestimates uncertainty.

```
FUNCTION query_relative_vector(pt_src, pt_dst, frame):
    (p_src, Cp_src, J_src_per_edge) = query_point_with_edge_jacobians(pt_src, frame)
    (p_dst, Cp_dst, J_dst_per_edge) = query_point_with_edge_jacobians(pt_dst, frame)

    shared_edge_ids = common edge_ids in both paths

    C_cross = Σ_{e ∈ shared}  J_dst,e · C_e · J_src,eᵀ

    delta   = p_dst - p_src
    C_delta = Cp_dst + Cp_src - C_cross - C_crossᵀ   ← accounts for correlation

    RETURN (delta, C_delta)
```

```
┌──────────────────────────────────────────────────────────────────┐
│  Why cross-covariance matters:                                   │
│                                                                  │
│        W ──── A ──── src_point                                   │
│        │                                                         │
│        └──── A ──── dst_point                                    │
│                                                                  │
│  Both points share edge W→A.  If that edge has large             │
│  uncertainty, BOTH points move together → their relative         │
│  distance is more certain than the naive sum would suggest.      │
└──────────────────────────────────────────────────────────────────┘
```

### 9c. Distance and variance

```
FUNCTION query_distance(pt_src, pt_dst, frame):
    (delta, C_delta) = query_relative_vector(pt_src, pt_dst, frame)

    d = ||delta||
    IF d > ε:
        ∂d/∂delta = delta / d
    ELSE:
        ∂d/∂delta = [1, 0, 0]     ← regularize near-zero case

    Var(d) ≈ (∂d/∂delta)ᵀ · C_delta · (∂d/∂delta)
    RETURN (d, Var(d))
```

---

## 10. Automatic Loop Posterior  (`network.py`)

```
FUNCTION query_auto_loop_posterior(start, goal, C_nu_scale=1e-6):
    ── Find all simple paths ─────────────────────────────────────
    all_paths = find_all_paths(start, goal)
    IF len(all_paths) < 2: return None   (no loops possible)

    ── Compose all paths ─────────────────────────────────────────
    T_list = [compose_along_path(p) for p in all_paths]

    T_res = T_list[0]                   ← "residual" path (reference)
    alternates = T_list[1:]             ← all other paths

    ── Build per-loop noise covariance ───────────────────────────
    FOR each T_k in alternates:
        C_nu_k = C_nu_scale · I_{6×6}

    ── Simultaneously condition on all N loops ───────────────────
    posterior = condition_on_multiple_loops(T_res,
                    [(T_k, C_nu_k) for T_k in alternates])

    RETURN posterior
```

---

## 11. End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TYPICAL USAGE FLOW                               │
│                                                                     │
│  1. BUILD NETWORK                                                   │
│     net = GeometricNetwork()                                        │
│     net.add_edge("W",   "Trk",  U_trk,  add_inverse=True)          │
│     net.add_edge("Trk", "Cam",  U_cam,  add_inverse=True)          │
│     net.add_edge("W",   "Robot",U_robot,add_inverse=True)          │
│     net.add_point("tool_tip", "Tool", p_local, Cp_local)           │
│                                                                     │
│  2. QUERY FRAME TRANSFORM (single path)                             │
│     result = net.query("Cam", "Robot")                              │
│     # result.transform.F_nom  ← best-estimate pose                 │
│     # result.transform.C      ← propagated 6×6 covariance          │
│                                                                     │
│  2b. QUERY FRAME TRANSFORM (multi-path, Bayes-fused)               │
│     result = net.query_frame("Cam", "Robot")                        │
│     # result.transform.C      ← fused covariance (tighter)         │
│     # result.n_paths          ← how many paths were fused           │
│     # result.path_results     ← individual path results             │
│                                                                     │
│  3. QUERY POINT LOCATION                                            │
│     (p, Cp) = net.query_point("tool_tip", "Cam")                   │
│                                                                     │
│  4. QUERY DISTANCE                                                  │
│     (d, Var_d) = net.query_distance("pt_A", "pt_B", "World")       │
│                                                                     │
│  5. CLOSE LOOPS (reduce uncertainty)                                │
│     posterior = net.query_auto_loop_posterior("W", "W")            │
│     # posterior.C_res  ← tighter covariance after conditioning     │
│                                                                     │
│  6. FUSE MULTIPLE PATHS                                             │
│     fused = net.query_fused_paths("Cam", "Tool")                   │
│     # Combines all path estimates information-optimally            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Module Organization

```
src/uncertainty_networks/
├── se3.py                   SE(3) math primitives
│                            exp_se3, log_se3, adjoint_se3, skew, inv_se3
│
├── nominal_types.py         Nominal (non-uncertain) geometric types
│                            vct3 (3-D point), Rot (rotation matrix),
│                            Frame (rigid transform = Rot + vct3)
│
├── uncertain_geometry.py    Core uncertain transform type
│                            Convention (RIGHT / LEFT)
│                            UncertainTransform (compose, inv, apply_to_point,
│                              as_right_convention, as_left_convention,
│                              from_left_covariance, from_nominal_frame)
│
├── uncertain_types.py       User-facing uncertain Cartesian types
│                            uScalar, uVector
│                            uvct3  (uncertain 3-D point)
│                            uRot   (uncertain rotation, convention-aware)
│                            uFrame (uncertain rigid frame, wraps UncertainTransform)
│
├── observations.py          General observation / factor interface
│                            Observation (abstract base)
│                            LoopObservation      (6-D loop-closure constraint)
│                            PointObservation     (3-D point measurement)
│                            DistanceObservation  (1-D distance measurement)
│                            ConditioningResult
│                            condition_on_observations  (joint info-filter update)
│
├── network.py               Main graph structure + all queries
│                            GeometricNetwork, Edge, PathResult, PointNode
│
└── closed_loop.py           Loop constraints & Bayesian inference
                             LoopLinearization, LoopPosterior,
                             condition_on_loop, condition_on_multiple_loops,
                             fuse_gaussian_covs, fuse_gaussians
```

---

## 13. Covariance Flow Diagram

```
   Edge 1: (C₁)         Edge 2: (C₂)         Edge 3: (C₃)
  ─────────────────────────────────────────────────────────►
  A ─────────────── B ─────────────── C ─────────────── D

  Composed covariance from A to D (right-perturbation convention):
  C_AD = Ad_{(F₂·F₃)^{-1}} · C₁ · Ad_{(F₂·F₃)^{-1}}ᵀ
       + Ad_{F₃^{-1}}       · C₂ · Ad_{F₃^{-1}}ᵀ
       + C₃

  The last edge contributes its covariance directly; earlier edges
  are mapped through the suffix-adjoint-inverse of all subsequent transforms.
```

---

## 15. General Observation Interface  (`observations.py`)

Generalises the hard-coded loop constraint in `closed_loop.py` to any sensor
measurement.  All concrete types share one linear-Gaussian form and can be
conditioned upon jointly via `condition_on_observations`.

### 15a. Observation (abstract base)

```
CLASS Observation  (abstract):
    Represents a measurement  r(eta) ≈ r0 + Σ_k  J_k · eta_k  ≈  0 + ν
    where  ν ~ N(0, C_nu).

    Abstract methods:
        .residual()   → (m,)      nominal residual r0 at zero perturbation
        .jacobians()  → {key: (m,6)}  ∂r/∂eta_k for each named state key
        .noise_cov()  → (m,m)     C_nu

    Derived properties:
        .dim          → int        residual dimension m
        .state_keys   → list[str]  which state variables this touches
```

### 15b. Concrete Observation Types

```
─── LoopObservation ──────────────────────────────────────────────────
    Constraint: two paths with the same endpoints should compose to I.

    Residual (dim=6):
        r = log_se3( F_res^{-1} · F_k )  ≈  0

    Jacobians via finite differences (same as closed_loop.linearize_loop_residual):
        J_res  (6×6),   J_k  (6×6)

    Constructor:
        LoopObservation(F_res, F_k, key_res, key_k, C_nu=1e-9·I)

─── PointObservation ─────────────────────────────────────────────────
    Constraint: observed 3D point z in a coordinate frame.

    Residual (dim=3):
        r = p_nom - z

    Jacobian (CIS I right-perturbation):
        J_eta = [-R·[p_in]×  |  R]        shape (3,6)
        where p_in = R^T (p_nom - t)  is the point in the source frame

    If F_nom not given, falls back to:
        J_eta = [-[p_nom]×  |  I_3]       (identity approximation)

    Constructor:
        PointObservation(p_nom, J_eta, key, z, C_nu)   ← manual
        PointObservation.build(p_nom, key, z, C_nu, F_nom=None)  ← auto Jacobian

─── DistanceObservation ──────────────────────────────────────────────
    Constraint: observed scalar distance z between two points p1, p2.

    Residual (dim=1):
        r = d_nom - z,    d_nom = ||p1 - p2||

    Jacobians (unit vector u = (p1-p2)/d_nom):
        J_1 = u^T  J_eta_1     shape (1,6)
        J_2 = -u^T J_eta_2     shape (1,6)
        If key_1 == key_2: merge into single entry J_1 + J_2.

    Constructor:
        DistanceObservation(p1, p2, J1, J2, key_1, key_2, z, sigma)
        DistanceObservation.build(p1, p2, key_1, key_2, z, sigma,
                                  F_nom_1=None, F_nom_2=None)
```

### 15c. Joint Conditioning

```
FUNCTION condition_on_observations(priors, observations):
    priors       : {key: (6,6)}     prior covariance per state variable
    observations : list[Observation]  any mix of types

    ── Assemble state ─────────────────────────────────────────────
    keys  = sorted(priors.keys())         K variables
    C0    = block_diag(C_k for k in keys)  shape (6K, 6K)

    ── Assemble H and C_nu ────────────────────────────────────────
    FOR each observation i (residual dim m_i):
        H_i      = zeros(m_i, 6K)
        FOR each (key, J_k) in observation.jacobians():
            col_offset = 6 * keys.index(key)
            H_i[:, col_offset : col_offset+6] = J_k

    H      = vstack([H_i])                  shape (M, 6K)
    C_nu   = block_diag(obs.noise_cov())    shape (M, M)

    ── Information filter update ───────────────────────────────────
    C_post = (C0^{-1} + H^T · C_nu^{-1} · H)^{-1}    shape (6K, 6K)

    ── Extract per-key posteriors ──────────────────────────────────
    posteriors[key_k] = C_post[6k:6k+6, 6k:6k+6]

    RETURN ConditioningResult(posteriors, C_post, keys, r0_stacked)
```

```
CLASS ConditioningResult:
    posteriors : {key: (6,6)}   per-variable posterior covariance
    C_full     : (6K, 6K)       full joint posterior (enables cross-terms)
    keys       : list[str]      sorted key ordering matching C_full blocks
    r0         : (M,)           stacked nominal residuals

    .cross_cov(key_a, key_b) → (6,6)   off-diagonal block of C_full
```

```
┌──────────────────────────────────────────────────────────────────────┐
│  Why use observations.py instead of closed_loop.py directly?         │
│                                                                      │
│  closed_loop.condition_on_loop handles ONE loop at a time and is     │
│  hard-coded to the 6D loop residual.                                 │
│                                                                      │
│  condition_on_observations handles:                                  │
│    • any number of loop constraints                                  │
│    • point-position measurements (3D)                                │
│    • distance measurements (1D)                                      │
│    • future sensor types (implement Observation subclass)            │
│    … all jointly, in one information-filter update.                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 14. Loop Conditioning Diagram

```
Before conditioning:
  path_res: A ──────────────────── B    covariance C_res (large)
  path_k:   A ──────────────────── B    covariance C_k   (large)

              constraint: path_res⁻¹ ∘ path_k ≈ I

After conditioning:
  path_res: A ──────────────────── B    covariance C_res_post (SMALLER)
  path_k:   A ──────────────────── B    covariance C_k_post   (SMALLER)
  cross-covariance: C_cross (non-zero — paths are now correlated)

  Intuition: knowing both paths agree gives us more confidence
  in BOTH individual estimates.
```

