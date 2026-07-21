# Author: X.M. Christine Zhu
# Date: 04/04/2026

"""
GeometricNetwork: unified Gaussian linear system for uncertain geometric frames.

Mathematical framework (Doc 2 — Structural Identification)
----------------------------------------------------------
Each unique physical edge e has ONE latent variable

    η_{e,canonical} ~ N(0, C_e)

shared by every path that traverses it.  This is the structural identity index
α(i, e) from Doc 2: two paths with the same edge_id at the same position share
the same draw, not independent copies.

For m simple paths from start to goal, each path i gives one linear estimate
of the unknown transform perturbation η_0:

    b_i = Σ_e  A_e^i  η_{e,canonical}

where A_e^i maps the canonical edge variable to the path perturbation.

The full stacked covariance S (6m × 6m) follows Doc 2 Section 1:

    S_{iℓ} = Σ_e  A_e^i  C_e  (A_e^ℓ)^T  · 1{α(i,e) = α(ℓ,e)}

The indicator is 1 when paths i and ℓ share the same physical edge (same
edge_id); the off-diagonal blocks S_{iℓ} are generally non-zero.

With A_0 = [I; I; ...; I] (all paths estimate the same η_0), the information
form (Doc 2 Section 2.2) gives:

    C_0 = (A_0^T S^{-1} A_0)^{-1} = (Σ_{i,ℓ} [S^{-1}]_{iℓ})^{-1}

This single formula handles every topology automatically:
  - Open chain      (1 path)  → standard covariance propagation
  - Independent multipaths    → reduces to information-form fusion
  - Multipaths with shared edges → off-diagonal S_{iℓ} prevent double-counting
  - Closed loops              → same as multipath, no special treatment needed

Point queries
-------------
Cross-covariance between two points in the same frame is computed via
shared-edge Jacobians (same structural identification: same edge_id → same
physical uncertainty contributes to both points).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .uncertain_geometry import UncertainTransform
from .uncertain_types import uFrame, uvct3
from .nominal_types import vct3, Frame
from .se3 import adjoint_se3, inv_se3, skew


Array = np.ndarray


def _sym(A: Array) -> Array:
    """Symmetrise a square matrix."""
    return 0.5 * (A + A.T)


# ------------------------------------------------------------------ #
#  Data containers                                                     #
# ------------------------------------------------------------------ #

@dataclass
class PathResult:
    """
    Result of a single-path frame-to-frame query.

    Attributes
    ----------
    path : list[str]
        Frame names [start, ..., goal].
    transform : uFrame
        Composed uncertain frame along this path.
        Access .F_nom for the 4×4 nominal matrix, .C for the 6×6 covariance,
        and .F for the nominal Frame object.
    edge_ids : list[str]
        edge_id for each hop (length = len(path) - 1).
    certain_mask : list[bool]
        is_certain flag per hop.
    forward_mask : list[bool]
        is_forward flag per hop (True = canonical direction).
    edge_types : list[str]
        edge_type per hop: "se3", "rot_only", "trans_only", or "vector".
    """
    path: List[str]
    transform: uFrame
    edge_ids: List[str] = field(default_factory=list)
    certain_mask: List[bool] = field(default_factory=list)
    forward_mask: List[bool] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)


@dataclass
class FusedQueryResult:
    """
    Result of the unified multi-path query (query_frame).

    The covariance C_0 is computed from the full stacked covariance S using
    the information form from Doc 2.  Off-diagonal blocks of S correctly
    account for shared physical edges between paths.

    Attributes
    ----------
    transform : uFrame
        Best-estimate uncertain frame.  Nominal from first path (all paths
        agree nominally in a consistent network).  C is the unified C_0.
        Access .F for the nominal Frame, .F_nom for the 4×4 matrix.
    n_paths : int
        Number of simple paths found and used.
    path_results : list[PathResult]
        Per-path results before fusion (for diagnostics).
    """
    transform: uFrame
    n_paths: int
    path_results: List[PathResult]


@dataclass
class PointNode:
    """A 3-D point rigidly attached to a coordinate frame."""
    frame: str
    point: uvct3   # nominal local coordinates + local covariance


@dataclass
class Edge:
    """
    Directed edge storing an uncertain transform.

    edge_id : str
        Unique identifier shared by the forward and inverse edge.
        This IS the structural identity index α(i, j) from Doc 2:
        all paths that use this physical edge share one latent variable.
    is_certain : bool
        True if this edge is perfectly known (C ≈ 0).
    is_forward : bool
        True if this is the canonical direction the edge was added.
        The inverse edge has the same edge_id with is_forward=False.
    edge_type : str
        "se3", "rot_only", "trans_only", or "vector".
    """
    edge_id: str
    transform: UncertainTransform
    is_certain: bool = False
    is_forward: bool = True
    edge_type: str = "se3"


_VALID_EDGE_TYPES = frozenset({"se3", "rot_only", "trans_only", "vector"})


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _project_covariance(C: Array, edge_type: str) -> Array:
    """Zero out covariance blocks that are inactive for the given edge type."""
    C = np.array(C, dtype=np.float64)
    if edge_type == "rot_only":
        C[3:, :] = 0.0
        C[:, 3:] = 0.0
    elif edge_type in ("trans_only", "vector"):
        C[:3, :] = 0.0
        C[:, :3] = 0.0
    return C


# ------------------------------------------------------------------ #
#  GeometricNetwork                                                    #
# ------------------------------------------------------------------ #

class GeometricNetwork:
    """
    Directed graph of coordinate frames connected by uncertain SE(3) transforms.

    Implements the unified Gaussian linear system framework from Doc 2.
    The main query entry point is query_frame(), which finds all simple paths
    and returns the correct fused covariance for any network topology.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, Dict[str, Edge]] = {}
        self._points: Dict[str, PointNode] = {}
        self._edge_counter: int = 0

    # ---------------------------------------------------------------- #
    #  Frame / edge construction                                        #
    # ---------------------------------------------------------------- #

    def add_frame(self, name: str) -> None:
        if name not in self._adj:
            self._adj[name] = {}

    def add_edge(
        self,
        src: str,
        dst: str,
        T_src_dst: UncertainTransform | uFrame | Frame,
        add_inverse: bool = True,
        is_certain: bool = False,
        edge_type: str = "se3",
    ) -> str:
        """
        Add a directed edge src -> dst.

        The forward and inverse edges share the same edge_id, which is the
        structural identity index α from Doc 2: all paths that traverse this
        physical relationship share the same latent variable η_{e,canonical}.

        Parameters
        ----------
        src, dst : str
        T_src_dst : UncertainTransform | uFrame | Frame
            Passing a Frame (certain transform) automatically sets is_certain=True
            and uses zero covariance.
        add_inverse : bool
            Also add the reverse edge (same edge_id, is_forward=False).
        is_certain : bool
            Mark edge as perfectly known.
        edge_type : {"se3", "rot_only", "trans_only", "vector"}

        Returns
        -------
        edge_id : str
        """
        if isinstance(T_src_dst, uFrame):
            T_src_dst = T_src_dst.to_uncertain_transform()
        elif isinstance(T_src_dst, Frame):
            T_src_dst = UncertainTransform.from_nominal_frame(T_src_dst)
            is_certain = True

        if edge_type not in _VALID_EDGE_TYPES:
            raise ValueError(
                f"edge_type must be one of {sorted(_VALID_EDGE_TYPES)}, got '{edge_type}'"
            )

        self.add_frame(src)
        self.add_frame(dst)

        edge_id = f"e{self._edge_counter:06d}"
        self._edge_counter += 1

        C_proj = _project_covariance(T_src_dst.C, edge_type)

        F_nom = T_src_dst.F_nom
        if edge_type == "vector":
            F_nom = np.array(F_nom, dtype=np.float64)
            F_nom[:3, :3] = np.eye(3)

        T_fwd = UncertainTransform(F_nom, C_proj)

        self._adj[src][dst] = Edge(
            edge_id=edge_id,
            transform=T_fwd,
            is_certain=is_certain,
            is_forward=True,
            edge_type=edge_type,
        )

        if add_inverse:
            self._adj[dst][src] = Edge(
                edge_id=edge_id,
                transform=T_fwd.inv(),
                is_certain=is_certain,
                is_forward=False,
                edge_type=edge_type,
            )

        return edge_id

    # ---------------------------------------------------------------- #
    #  Point management                                                 #
    # ---------------------------------------------------------------- #

    def add_point(
        self,
        name: str,
        frame: str,
        point: uvct3,
    ) -> None:
        if frame not in self._adj:
            raise KeyError(f"Unknown frame '{frame}'")
        self._points[name] = PointNode(frame=frame, point=point)

    def has_point(self, name: str) -> bool:
        return name in self._points

    # ---------------------------------------------------------------- #
    #  Basic utilities                                                  #
    # ---------------------------------------------------------------- #

    def has_frame(self, name: str) -> bool:
        return name in self._adj

    def frames(self) -> List[str]:
        """All frame names in insertion order."""
        return list(self._adj.keys())

    def forward_edges(self) -> List[Tuple[str, str]]:
        """All edges in their canonical (forward) direction as (src, dst) pairs."""
        result = []
        for src, neighbors in self._adj.items():
            for dst, edge in neighbors.items():
                if edge.is_forward:
                    result.append((src, dst))
        return result

    def neighbors(self, name: str) -> List[str]:
        return list(self._adj.get(name, {}).keys())

    def get_edge(self, src: str, dst: str) -> Optional[UncertainTransform]:
        e = self._adj.get(src, {}).get(dst, None)
        return None if e is None else e.transform

    def get_edge_obj(self, src: str, dst: str) -> Optional[Edge]:
        return self._adj.get(src, {}).get(dst, None)

    # ---------------------------------------------------------------- #
    #  Path finding                                                     #
    # ---------------------------------------------------------------- #

    def find_path(self, start: str, goal: str) -> List[str]:
        """BFS shortest path from start to goal."""
        if start not in self._adj or goal not in self._adj:
            return []
        if start == goal:
            return [start]

        q = deque([start])
        parent: Dict[str, Optional[str]] = {start: None}

        while q:
            cur = q.popleft()
            for nxt in self._adj[cur]:
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == goal:
                    q.clear()
                    break
                q.append(nxt)

        if goal not in parent:
            return []

        path: List[str] = []
        node: Optional[str] = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def _edges_along_path(self, path: List[str]) -> List[Edge]:
        return [self._adj[a][b] for a, b in zip(path[:-1], path[1:])]

    def find_all_paths(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Find all simple paths from start to goal via DFS.
        A simple path visits each frame at most once (no cycles).
        """
        if start not in self._adj or goal not in self._adj:
            return []
        if start == goal:
            return [[start]]

        all_paths: List[List[str]] = []

        def _dfs(current: str, path: List[str], visited: set) -> None:
            if current == goal:
                all_paths.append(list(path))
                return
            if max_depth is not None and len(path) > max_depth:
                return
            for neighbor in self._adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    _dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        _dfs(start, [start], {start})
        return all_paths

    def query_all_paths(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> List[PathResult]:
        """
        Compute a PathResult for every simple path from start to goal.

        The per-path covariance stored in each PathResult is the standard
        composed covariance (diagonal term S_{ii} from the S matrix).
        For the fully fused result use query_frame().
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: {start}")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: {goal}")

        paths = self.find_all_paths(start, goal, max_depth=max_depth)
        results: List[PathResult] = []

        for path in paths:
            edges = self._edges_along_path(path)
            T = UncertainTransform.identity()
            for e in edges:
                T = T @ e.transform
            results.append(PathResult(
                path=path,
                transform=uFrame(T),
                edge_ids=[e.edge_id for e in edges],
                certain_mask=[e.is_certain for e in edges],
                forward_mask=[e.is_forward for e in edges],
                edge_types=[e.edge_type for e in edges],
            ))

        return results

    # ---------------------------------------------------------------- #
    #  Single-path frame queries                                        #
    # ---------------------------------------------------------------- #

    def query(self, start: str, goal: str) -> PathResult:
        """
        Query via the BFS shortest path only (single path, no multi-path fusion).
        For the correct fused result over all paths use query_frame().
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: {start}")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: {goal}")

        path = self.find_path(start, goal)
        if not path:
            raise ValueError(f"No path found from '{start}' to '{goal}'")

        edges = self._edges_along_path(path)
        T = UncertainTransform.identity()
        for e in edges:
            T = T @ e.transform

        return PathResult(
            path=path,
            transform=uFrame(T),
            edge_ids=[e.edge_id for e in edges],
            certain_mask=[e.is_certain for e in edges],
            forward_mask=[e.is_forward for e in edges],
            edge_types=[e.edge_type for e in edges],
        )

    def query_transform_on_path(self, path: List[str]) -> PathResult:
        """Compute the uncertain transform along a user-specified path."""
        if not path:
            raise ValueError("Path is empty.")
        for v in path:
            if v not in self._adj:
                raise KeyError(f"Unknown frame in path: {v}")
        for a, b in zip(path[:-1], path[1:]):
            if b not in self._adj.get(a, {}):
                raise ValueError(f"No edge {a} -> {b} in network.")

        edges = self._edges_along_path(path)
        T = UncertainTransform.identity()
        for e in edges:
            T = T @ e.transform

        return PathResult(
            path=path,
            transform=uFrame(T),
            edge_ids=[e.edge_id for e in edges],
            certain_mask=[e.is_certain for e in edges],
            forward_mask=[e.is_forward for e in edges],
            edge_types=[e.edge_type for e in edges],
        )

    # ---------------------------------------------------------------- #
    #  Unified multi-path query — core of the framework                 #
    # ---------------------------------------------------------------- #

    def _compute_path_canonical_terms(
        self,
        path: List[str],
    ) -> Dict[str, Tuple[Array, Array]]:
        """
        For each edge in the path, compute the canonical linear map and
        canonical covariance, following Doc 2 (structural identification).

        The canonical variable η_{e,canonical} is always defined in the
        FORWARD direction of the edge.  For a path that traverses the edge
        in the inverse direction the effective linear map picks up an extra
        factor so that the cross-covariance formula remains correct.

        Returns
        -------
        dict  edge_id -> (A_canonical, C_canonical)

        A_canonical : (6,6)
            Linear map:  η_0 contribution from this edge = A_canonical @ η_{e,canonical}

            Forward traversal (is_forward=True):
                A_canonical = Ad_{T_suffix^{-1}}
                where T_suffix = product of all edge nominals AFTER this edge.

            Inverse traversal (is_forward=False):
                Under CIS I right-perturbation, inverting a transform gives
                    η_inv ≈ -Ad_{F_fwd} η_fwd    (first-order)
                where F_fwd = inv(e.transform.F_nom), so:
                    A_canonical = -Ad_{T_suffix^{-1}} @ Ad_{F_fwd_nom}

        C_canonical : (6,6)
            Covariance of the canonical (forward) variable.
            For forward edges this is just C_e.
            For inverse edges it is recovered as:
                C_canonical = Ad_{F_inv_nom} C_inv Ad_{F_inv_nom}^T
            where F_inv_nom = e.transform.F_nom (the stored inverse nominal).
        """
        edges = self._edges_along_path(path)
        m = len(edges)

        # Backward pass: T_suffixes[k] = F_{k+1} @ ... @ F_{m-1}
        T_suffix = np.eye(4, dtype=np.float64)
        T_suffixes: List[Array] = [None] * m  # type: ignore[list-item]
        for k in range(m - 1, -1, -1):
            T_suffixes[k] = T_suffix.copy()
            T_suffix = edges[k].transform.F_nom @ T_suffix

        result: Dict[str, Tuple[Array, Array]] = {}

        for k, e in enumerate(edges):
            Ad_suf_inv = adjoint_se3(inv_se3(T_suffixes[k]))

            if e.is_forward:
                # Canonical variable is η_fwd, contribution = Ad_{T_suf^{-1}} @ η_fwd
                A_can = Ad_suf_inv.copy()
                C_can = e.transform.C.copy()
            else:
                # η_inv ≈ -Ad_{F_fwd} η_fwd  (RIGHT convention)
                # F_fwd = inv(e.transform.F_nom)
                Ad_Ffwd = adjoint_se3(inv_se3(e.transform.F_nom))
                A_can = -Ad_suf_inv @ Ad_Ffwd

                # Recover C_canonical from stored C_inv:
                # Under RIGHT convention: C_inv = Ad_{F_fwd} C_fwd Ad_{F_fwd}^T
                # So: C_fwd = Ad_{F_inv_nom} C_inv Ad_{F_inv_nom}^T
                Ad_Finv = adjoint_se3(e.transform.F_nom)
                C_can = _sym(Ad_Finv @ e.transform.C @ Ad_Finv.T)

            result[e.edge_id] = (A_can, C_can)

        return result

    def query_frame(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> FusedQueryResult:
        """
        Query the best-estimate transform from start to goal using ALL simple
        paths and the unified Gaussian linear system framework (Doc 2).

        Algorithm
        ---------
        1. Find all m simple paths from start to goal.

        2. For each path i compute canonical terms (A_e^i, C_e) per edge.

        3. Build the full stacked covariance S  (6m × 6m):

               S_{iℓ} = Σ_{shared edges e}  A_e^i  C_e  (A_e^ℓ)^T

           "Shared" means same physical edge_id — the structural identity
           α(i, e) = α(ℓ, e) from Doc 2.

        4. Apply the information form (Doc 2 Section 2.2):

               C_0 = (A_0^T S^{-1} A_0)^{-1}

           With A_0 = [I; I; ...; I]:

               C_0 = ( Σ_{i,ℓ} [S^{-1}]_{iℓ} )^{-1}

        Special cases handled automatically
        ------------------------------------
        m = 1  (open chain)          : C_0 = S_{11} = path covariance.
        m > 1, no shared edges       : S block-diagonal → standard info fusion.
        m > 1, shared edges present  : off-diagonal S_{iℓ} prevent double-
                                       counting shared uncertainty.
        Closed loops                 : same as multipath — no special treatment.

        Parameters
        ----------
        start, goal : str
        max_depth : int, optional
            Limit path search depth (useful for large or dense networks).

        Returns
        -------
        FusedQueryResult
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: '{start}'")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: '{goal}'")

        # Step 1 — find all simple paths and per-path results.
        path_results = self.query_all_paths(start, goal, max_depth=max_depth)
        if not path_results:
            raise ValueError(f"No path found from '{start}' to '{goal}'")

        m = len(path_results)

        # Step 2 — single path: trivially correct, no matrix inversion needed.
        if m == 1:
            return FusedQueryResult(
                transform=path_results[0].transform,
                n_paths=1,
                path_results=path_results,
            )

        # Step 3 — canonical terms per path:  edge_id -> (A_e^i, C_canonical_e).
        canonical_terms = [
            self._compute_path_canonical_terms(pr.path)
            for pr in path_results
        ]

        # Step 4 — build full S matrix (6m × 6m).
        #
        # S_{iℓ} = Σ_{e: α(i,e)=α(ℓ,e)}  A_e^i @ C_canonical_e @ (A_e^ℓ)^T
        #
        # For the diagonal (i == ℓ) this equals the per-path composed covariance.
        # For off-diagonal blocks it captures shared-edge correlations.
        S = np.zeros((6 * m, 6 * m), dtype=np.float64)

        for i in range(m):
            for l in range(m):
                S_il = np.zeros((6, 6), dtype=np.float64)
                shared_eids = (
                    set(canonical_terms[i].keys()) & set(canonical_terms[l].keys())
                )
                for eid in shared_eids:
                    A_i, C_can = canonical_terms[i][eid]
                    A_l, _     = canonical_terms[l][eid]
                    S_il += A_i @ C_can @ A_l.T
                S[6 * i : 6 * i + 6, 6 * l : 6 * l + 6] = S_il

        S = _sym(S)

        # Step 5 — information form: C_0 = (A_0^T S^{-1} A_0)^{-1}.
        # A_0 = [I;I;...;I]  →  A_0^T S^{-1} A_0 = sum of all (6×6) blocks of S^{-1}.
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        info = np.zeros((6, 6), dtype=np.float64)
        for i in range(m):
            for l in range(m):
                info += S_inv[6 * i : 6 * i + 6, 6 * l : 6 * l + 6]

        info = _sym(info)

        try:
            C_0 = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            C_0 = np.linalg.pinv(info)

        C_0 = _sym(C_0)

        # Step 6 — nominal: all paths agree in a nominally consistent network.
        F_nom = path_results[0].transform.F_nom

        return FusedQueryResult(
            transform=uFrame(UncertainTransform(F_nom, C_0)),
            n_paths=m,
            path_results=path_results,
        )

    # ---------------------------------------------------------------- #
    #  Point propagation (with per-edge Jacobians)                      #
    # ---------------------------------------------------------------- #

    def _query_point_with_edge_jacobians(
        self,
        point_name: str,
        target_frame: str,
    ) -> Tuple[Array, Array, Dict[str, Tuple[Array, Array]]]:
        """
        Propagate a point into target_frame, returning per-edge Jacobians.

        Returns
        -------
        p_out : (3,)
        Cp_out : (3,3)
        edge_terms : dict  edge_id -> (J_e, C_e)
            J_e : (3,6)  Jacobian of point position w.r.t. edge perturbation η_e
            C_e : (6,6)  covariance of that edge (as stored)
        """
        if point_name not in self._points:
            raise KeyError(f"Unknown point '{point_name}'")
        if target_frame not in self._adj:
            raise KeyError(f"Unknown target frame '{target_frame}'")

        point_node = self._points[point_name]
        source_frame = point_node.frame
        p_local = np.array([point_node.point.x, point_node.point.y, point_node.point.z], dtype=np.float64)
        Cp_local = point_node.point.C

        if source_frame == target_frame:
            return p_local.copy(), Cp_local.copy(), {}

        path = self.find_path(source_frame, target_frame)
        if not path:
            raise ValueError(
                f"No path from '{source_frame}' to '{target_frame}'"
            )

        edges = self._edges_along_path(path)
        m = len(edges)

        # Backward pass: T_suffixes[k] = F_{k+1} @ ... @ F_{m-1}
        T_suffix = np.eye(4, dtype=np.float64)
        T_suffixes: List[Array] = [None] * m  # type: ignore[list-item]
        for k in range(m - 1, -1, -1):
            T_suffixes[k] = T_suffix.copy()
            T_suffix = edges[k].transform.F_nom @ T_suffix

        # After the backward pass T_suffix holds the full product F_0 @ ... @ F_{m-1}
        T_total = T_suffix
        R_total = T_total[:3, :3]
        t_total = T_total[:3, 3]

        p_nom = R_total @ p_local + t_total

        # Per-edge Jacobians under CIS I right-perturbation.
        # J_k = R_01k @ [-skew(p_suf_k), I]   (3×6)
        # where R_01k  = rotation of F_0 @ ... @ F_k
        #       p_suf_k = T_suffix_k @ p_in (point expressed after edge k)
        edge_terms: Dict[str, Tuple[Array, Array]] = {}
        Cp_edges = np.zeros((3, 3), dtype=np.float64)

        R_prefix = np.eye(3, dtype=np.float64)  # rotation before edge k
        p_in = p_local

        for k, e in enumerate(edges):
            R_k = e.transform.F_nom[:3, :3]
            R_01k = R_prefix @ R_k  # rotation up to and including edge k

            T_suf = T_suffixes[k]
            p_suf_k = T_suf[:3, :3] @ p_in + T_suf[:3, 3]

            J_e = R_01k @ np.hstack([-skew(p_suf_k), np.eye(3, dtype=np.float64)])
            C_e = e.transform.C
            Cp_edges += J_e @ C_e @ J_e.T
            edge_terms[e.edge_id] = (J_e, C_e)

            R_prefix = R_01k

        # Intrinsic point covariance mapped by R_total.
        Cp_point = R_total @ Cp_local @ R_total.T
        Cp_out = _sym(Cp_edges + Cp_point)

        return p_nom, Cp_out, edge_terms

    # ---------------------------------------------------------------- #
    #  Point queries                                                    #
    # ---------------------------------------------------------------- #

    def query_point(
        self,
        point_name: str,
        target_frame: str,
    ) -> uvct3:
        """Query a point into target_frame.  Returns uvct3."""
        p, Cp, _ = self._query_point_with_edge_jacobians(point_name, target_frame)
        return uvct3(vct3(float(p[0]), float(p[1]), float(p[2])), Cp)

    def query_point_to_point(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> Tuple[uvct3, uvct3]:
        """
        Query two points into a common frame.
        Returns (src_uvct3, dst_uvct3).
        """
        if query_frame not in self._adj:
            raise KeyError(f"Unknown query frame '{query_frame}'")
        src = self.query_point(src_point, query_frame)
        dst = self.query_point(dst_point, query_frame)
        return src, dst

    def query_relative_vector_independent(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> uvct3:
        """
        Relative vector with the independence approximation (upper bound):
            delta   = p_dst - p_src
            C_delta = C_dst + C_src
        """
        src, dst = self.query_point_to_point(src_point, dst_point, query_frame)
        p_src = np.array([src.x, src.y, src.z])
        p_dst = np.array([dst.x, dst.y, dst.z])
        delta = p_dst - p_src
        return uvct3(vct3(float(delta[0]), float(delta[1]), float(delta[2])), _sym(dst.C + src.C))

    def query_relative_vector(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> uvct3:
        """
        Correlation-aware relative vector (correct).

        Applies the structural identification model: edges with the same
        edge_id contribute a cross-covariance between the two point estimates.

            Cross   = Cov(p_dst, p_src) = Σ_{e shared} J_dst,e @ C_e @ J_src,e^T

            C_delta = C_dst + C_src - Cross - Cross^T

        Parameters
        ----------
        src_point, dst_point : str
            Point names registered with add_point().
        query_frame : str
            The frame in which the relative vector is expressed.

        Returns
        -------
        uvct3
            .p : vct3   — nominal delta (p_dst - p_src)
            .C : (3,3)  — correlation-aware covariance of the delta
        """
        if query_frame not in self._adj:
            raise KeyError(f"Unknown query frame '{query_frame}'")

        p_src, Cp_src, terms_src = self._query_point_with_edge_jacobians(
            src_point, query_frame
        )
        p_dst, Cp_dst, terms_dst = self._query_point_with_edge_jacobians(
            dst_point, query_frame
        )

        delta = p_dst - p_src

        # Cross-covariance from shared physical edges (same edge_id).
        cross = np.zeros((3, 3), dtype=np.float64)
        shared_eids = set(terms_src.keys()) & set(terms_dst.keys())
        for eid in shared_eids:
            J_src, C_e = terms_src[eid]
            J_dst, _   = terms_dst[eid]
            cross += J_dst @ C_e @ J_src.T   # Cov(p_dst, p_src)

        C_delta = _sym(Cp_dst + Cp_src - cross - cross.T)
        return uvct3(vct3(float(delta[0]), float(delta[1]), float(delta[2])), C_delta)

    def query_distance(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
        eps: float = 1e-12,
    ) -> Tuple[float, float]:
        """
        Euclidean distance and its variance using the correlation-aware C_delta.

        Returns
        -------
        d     : float   nominal distance
        var_d : float   first-order variance
        """
        result = self.query_relative_vector(src_point, dst_point, query_frame)
        delta   = np.array([result.x, result.y, result.z])
        C_delta = result.C
        d = float(np.linalg.norm(delta))

        if d < eps:
            return d, float(np.trace(C_delta))

        J = delta.reshape(1, 3) / d            # (1,3) unit-vector Jacobian
        var_d = float((J @ C_delta @ J.T).item())
        return d, var_d

    # ---------------------------------------------------------------- #
    #  Loop-closure queries                                            #
    # ---------------------------------------------------------------- #

    def query_closed_loop_posterior(
        self,
        path_res: List[str],
        path_k: List[str],
        C_nu: Optional[Array] = None,
    ) -> "LoopPosterior":
        """
        Condition on a single loop-closure constraint.

        Composes the uncertain transform along each path, linearises the loop
        residual r = Log(T_res^{-1} T_k), and applies a Gaussian information
        update to obtain posterior covariances for both paths.

        Parameters
        ----------
        path_res : list[str]
            Reference path (list of frame names, same start and end as path_k).
        path_k : list[str]
            Alternative path.
        C_nu : (6,6) ndarray, optional
            Loop noise covariance. Default: 1e-9 * I (tight constraint).

        Returns
        -------
        LoopPosterior
            .C_res   — posterior covariance of the reference path transform
            .C_k     — posterior covariance of the alternative path transform
            .C_cross — cross-covariance between the two
            .C_full  — full 12×12 joint posterior
        """
        from .closed_loop import linearize_loop_residual, condition_on_loop

        T_res = self.query_transform_on_path(path_res).transform
        T_k   = self.query_transform_on_path(path_k).transform
        lin   = linearize_loop_residual(T_res.F_nom, T_k.F_nom)
        return condition_on_loop(T_res.C, T_k.C, lin, C_nu=C_nu)

    def query_auto_loop_posterior(
        self,
        start: str,
        goal: str,
        C_nu: Optional[Array] = None,
        max_depth: Optional[int] = None,
    ) -> "LoopPosterior":
        """
        Automatically discover all simple paths from start to goal and apply
        every independent loop-closure constraint simultaneously.

        The first path found is used as the reference; every additional path
        forms one loop constraint against it. All constraints are applied in a
        single joint information update.

        Parameters
        ----------
        start, goal : str
        C_nu : (6,6) ndarray, optional
            Loop noise covariance applied to every constraint. Default: 1e-9 * I.
        max_depth : int, optional
            Limit DFS depth for large networks.

        Returns
        -------
        LoopPosterior  (reference = first path found)

        Raises
        ------
        ValueError
            If fewer than two paths exist between start and goal.
        """
        from .closed_loop import (
            linearize_loop_residual,
            condition_on_loop,
            condition_on_multiple_loops,
            LoopLinearization,
        )

        paths = self.find_all_paths(start, goal, max_depth=max_depth)
        if len(paths) < 2:
            raise ValueError(
                f"query_auto_loop_posterior requires at least 2 paths from "
                f"'{start}' to '{goal}', found {len(paths)}."
            )

        T_res = self.query_transform_on_path(paths[0]).transform

        if len(paths) == 2:
            T_k  = self.query_transform_on_path(paths[1]).transform
            lin  = linearize_loop_residual(T_res.F_nom, T_k.F_nom)
            return condition_on_loop(T_res.C, T_k.C, lin, C_nu=C_nu)

        # Multiple alternative paths — joint update.
        alt_transforms = [
            self.query_transform_on_path(p).transform for p in paths[1:]
        ]
        lins = [
            linearize_loop_residual(T_res.F_nom, T_k.F_nom)
            for T_k in alt_transforms
        ]
        C_ks    = [T_k.C for T_k in alt_transforms]
        C_nu_list = [C_nu] * len(lins) if C_nu is not None else None
        return condition_on_multiple_loops(T_res.C, C_ks, lins, C_nu_list=C_nu_list)

    # ---------------------------------------------------------------- #
    #  Batch utilities                                                  #
    # ---------------------------------------------------------------- #

    def evaluate_pairs(
        self,
        pairs: List[Tuple[str, str, str]],
        compute_distance: bool = False,
    ) -> dict:
        """
        Evaluate multiple point-to-point queries in one call.

        Parameters
        ----------
        pairs : list of (src_point, dst_point, query_frame)
        compute_distance : bool

        Returns
        -------
        dict  (src, dst, query_frame) -> {
            "ind"   : uvct3  — independence-approximation relative vector,
            "corr"  : uvct3  — correlation-aware relative vector,
            "d"     : float  (only if compute_distance=True),
            "var_d" : float  (only if compute_distance=True),
        }
        """
        results = {}
        for (src, dst, q) in pairs:
            entry: dict = {
                "ind":  self.query_relative_vector_independent(src, dst, q),
                "corr": self.query_relative_vector(src, dst, q),
            }
            if compute_distance:
                d, var_d = self.query_distance(src, dst, q)
                entry["d"]     = d
                entry["var_d"] = var_d
            results[(src, dst, q)] = entry
        return results
