# Author: X.M. Christine Zhu
# Date: 04/06/2026

"""
Visualization utilities for GeometricNetwork.

Two backends
------------
- matplotlib  : static figures for posters / papers  (plot_network_static)
- plotly      : interactive 3-D viewer for live demos (plot_network_interactive)

Interactive features
--------------------
- Dark background with subtle grid
- Gradient (Plasma colorscale) uncertainty ellipsoids
- Directional arrow cones on kinematic edges
- Auto-rotating camera with ▶ / ⏸ buttons
- Rich hover cards showing frame name and position

Note on ellipsoid_sigma
-----------------------
Physical uncertainties in surgical / robotics networks are often millimetres
while the network spans tens of centimetres.  Pass a large ``ellipsoid_sigma``
(e.g. 20-30) so the ellipsoids are visually prominent on a poster.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .network import GeometricNetwork


Array = np.ndarray


# ── internal helpers ──────────────────────────────────────────────────────────

def _ellipsoid_surface(
    center: Array,
    cov3: Array,
    n_sigma: float = 2.0,
    n_pts: int = 40,
) -> Tuple[Array, Array, Array]:
    """
    Parametric surface of the 3-D uncertainty ellipsoid.
    Returns X, Y, Z arrays of shape (n_pts, n_pts).
    """
    eigvals, eigvecs = np.linalg.eigh(cov3)
    eigvals = np.maximum(eigvals, 1e-30)
    radii   = n_sigma * np.sqrt(eigvals)

    u = np.linspace(0.0, 2.0 * np.pi, n_pts)
    v = np.linspace(0.0, np.pi,       n_pts)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    ell = pts @ (eigvecs * radii).T + center

    X = ell[:, 0].reshape(n_pts, n_pts)
    Y = ell[:, 1].reshape(n_pts, n_pts)
    Z = ell[:, 2].reshape(n_pts, n_pts)
    return X, Y, Z


def _gather_frame_data(
    net: GeometricNetwork,
    reference_frame: str,
    frames: List[str],
) -> Dict[str, Tuple[Array, Array, Array]]:
    """
    Query each frame relative to ``reference_frame`` via query_frame().
    Returns dict  frame_name -> (position (3,), rotation (3,3), cov3 (3,3))
    where cov3 is the translation sub-block C[3:6, 3:6].
    """
    data: Dict[str, Tuple[Array, Array, Array]] = {}
    for f in frames:
        if f == reference_frame:
            data[f] = (np.zeros(3), np.eye(3), np.zeros((3, 3)))
        else:
            r   = net.query_frame(reference_frame, f)
            F   = r.transform.F_nom
            C   = r.transform.C
            data[f] = (F[:3, 3].copy(), F[:3, :3].copy(), C[3:6, 3:6].copy())
    return data


def _auto_triad_scale(positions: Array) -> float:
    span = np.max(positions.max(axis=0) - positions.min(axis=0))
    return float(max(span * 0.08, 0.01))


# ── static backend (matplotlib) ───────────────────────────────────────────────

def plot_network_static(
    net: GeometricNetwork,
    reference_frame: str,
    title: str = "",
    frames: Optional[List[str]] = None,
    edges: Optional[List[Tuple[str, str]]] = None,
    ellipsoid_sigma: float = 2.0,
    triad_scale: Optional[float] = None,
    ellipsoid_alpha: float = 0.30,
    frame_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (8, 7),
    elev: float = 20.0,
    azim: float = -55.0,
    save_path: Optional[str] = None,
):
    """
    Static 3-D network plot (matplotlib) — for poster / report.

    Parameters
    ----------
    net : GeometricNetwork
    reference_frame : str
    title : str
    frames : list[str], optional     — defaults to all frames
    edges  : list[(str,str)], optional — defaults to all forward edges
    ellipsoid_sigma : float
    triad_scale : float, optional    — auto-computed if not given
    ellipsoid_alpha : float
    frame_colors : dict, optional
    figsize, elev, azim
    save_path : str, optional        — saves PNG if given
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.lines import Line2D

    if frames is None:
        frames = net.frames()
    if edges is None:
        edges = net.forward_edges()

    data      = _gather_frame_data(net, reference_frame, frames)
    positions = np.array([data[f][0] for f in frames])

    if triad_scale is None:
        triad_scale = _auto_triad_scale(positions)

    _default_colors = [
        "#FF7F0E", "#9467BD", "#8C564B", "#E377C2",
        "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
    ]
    _triad_colors = ["#d62728", "#2ca02c", "#1f77b4"]

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    for src, dst in edges:
        if src in data and dst in data:
            p0, p1 = data[src][0], data[dst][0]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="gray", lw=1.8, alpha=0.6, zorder=1,
            )

    for fi, fname in enumerate(frames):
        pos, rot, cov3 = data[fname]
        is_ref = fname == reference_frame
        color  = (frame_colors or {}).get(
            fname,
            "#2ca02c" if is_ref else _default_colors[fi % len(_default_colors)],
        )

        ax.scatter(*pos, color=color,
                   s=90 if is_ref else 55, zorder=5, depthshade=False)
        ax.text(pos[0], pos[1], pos[2] + triad_scale * 0.45,
                fname, fontsize=9, ha="center",
                fontweight="bold" if is_ref else "normal", color="black")

        for i, tc in enumerate(_triad_colors):
            d = rot[:, i] * triad_scale
            ax.quiver(*pos, *d, color=tc, linewidth=1.8,
                      arrow_length_ratio=0.25, normalize=False)

        if not np.allclose(cov3, 0.0) and np.all(np.isfinite(cov3)):
            try:
                X, Y, Z = _ellipsoid_surface(pos, cov3, n_sigma=ellipsoid_sigma)
                ax.plot_surface(X, Y, Z, alpha=ellipsoid_alpha, color=color,
                                linewidth=0, antialiased=True)
            except (np.linalg.LinAlgError, ValueError):
                pass

    center = positions.mean(axis=0)
    half   = max(
        float(np.max(np.abs(positions - center))) + triad_scale * 2.5, 0.05
    )
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_xlabel("X (m)", labelpad=6)
    ax.set_ylabel("Y (m)", labelpad=6)
    ax.set_zlabel("Z (m)", labelpad=6)
    ax.set_title(title, fontsize=10, pad=10)

    legend_elements = [
        Line2D([0], [0], color="#d62728", lw=2, label="x-axis"),
        Line2D([0], [0], color="#2ca02c", lw=2, label="y-axis"),
        Line2D([0], [0], color="#1f77b4", lw=2, label="z-axis"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [static]      saved → {save_path}")

    return fig, ax


# ── interactive backend (plotly) ──────────────────────────────────────────────

# Vibrant palette designed for dark backgrounds
_DARK_PALETTE = [
    "#FF6B6B",   # coral
    "#4ECDC4",   # teal
    "#45B7D1",   # sky blue
    "#96CEB4",   # sage
    "#FFEAA7",   # gold
    "#DDA0DD",   # plum
    "#98FB98",   # pale green
    "#FFB347",   # peach
]
_TRIAD_HEX  = ["#FF4444", "#44FF88", "#4488FF"]   # x y z  (bright on dark)
_AXIS_NAMES = ["x-axis", "y-axis", "z-axis"]


def _make_rotation_frames(n_frames: int = 120,
                           radius: float = 2.2,
                           height: float = 0.9):
    """120 camera-only animation frames for one full rotation."""
    import plotly.graph_objects as go
    frames = []
    for i, t in enumerate(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)):
        frames.append(go.Frame(
            layout=dict(scene_camera=dict(
                eye=dict(x=radius * np.cos(t),
                         y=radius * np.sin(t),
                         z=height),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            )),
            name=str(i),
        ))
    return frames


def plot_network_interactive(
    net: GeometricNetwork,
    reference_frame: str,
    title: str = "",
    frames: Optional[List[str]] = None,
    edges: Optional[List[Tuple[str, str]]] = None,
    ellipsoid_sigma: float = 2.0,
    triad_scale: Optional[float] = None,
    ellipsoid_opacity: float = 0.45,
    frame_colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    # ── style knobs ──────────────────────────────────────────────────────────
    dark_theme: bool = True,
    animate_rotation: bool = True,
    show_edge_arrows: bool = True,
    ellipsoid_colorscale: str = "Plasma",
):
    """
    Fancy interactive 3-D network plot using plotly.

    Features
    --------
    - Dark background with subtle grid (``dark_theme=True``)
    - Gradient (Plasma) uncertainty ellipsoids
    - Directional arrow cones on each kinematic edge
    - Auto-rotating camera with ▶ / ⏸ buttons (``animate_rotation=True``)
    - Rich hover cards

    Parameters
    ----------
    net : GeometricNetwork
    reference_frame : str
    title : str
    frames : list[str], optional
    edges  : list[(str,str)], optional
    ellipsoid_sigma : float
    triad_scale : float, optional
    ellipsoid_opacity : float
    frame_colors : dict, optional
    save_path : str, optional     — saves self-contained HTML
    dark_theme : bool
    animate_rotation : bool
    show_edge_arrows : bool
    ellipsoid_colorscale : str    — any plotly colorscale name

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if frames is None:
        frames = net.frames()
    if edges is None:
        edges = net.forward_edges()

    data      = _gather_frame_data(net, reference_frame, frames)
    positions = np.array([data[f][0] for f in frames])

    if triad_scale is None:
        triad_scale = _auto_triad_scale(positions)

    template = "plotly_dark" if dark_theme else "plotly_white"
    bg_color = "rgb(12, 12, 22)" if dark_theme else "rgb(250, 250, 252)"
    grid_col = "rgba(80,80,120,0.4)" if dark_theme else "rgba(200,200,220,0.6)"

    fig = go.Figure()

    # ── kinematic edges ───────────────────────────────────────────────────────
    for src, dst in edges:
        if src not in data or dst not in data:
            continue
        p0, p1 = data[src][0], data[dst][0]
        direction = p1 - p0
        length    = float(np.linalg.norm(direction))

        edge_color = "rgba(180,180,220,0.7)" if dark_theme else "rgba(100,100,140,0.7)"

        # Shaft — stop slightly before tip so the cone sits cleanly
        shaft_frac = 0.82 if (show_edge_arrows and length > 1e-9) else 1.0
        p_mid = p0 + direction * shaft_frac
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p_mid[0], None],
            y=[p0[1], p_mid[1], None],
            z=[p0[2], p_mid[2], None],
            mode="lines",
            line=dict(color=edge_color, width=5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Arrowhead cone at destination
        if show_edge_arrows and length > 1e-9:
            unit = direction / length
            cone_size = max(length * 0.18, triad_scale * 0.6)
            fig.add_trace(go.Cone(
                x=[p1[0]], y=[p1[1]], z=[p1[2]],
                u=[unit[0]], v=[unit[1]], w=[unit[2]],
                sizemode="absolute",
                sizeref=cone_size,
                anchor="tip",
                colorscale=[[0, edge_color], [1, edge_color]],
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── frames ────────────────────────────────────────────────────────────────
    for fi, fname in enumerate(frames):
        pos, rot, cov3 = data[fname]
        is_ref = fname == reference_frame
        color  = (frame_colors or {}).get(
            fname,
            "#00FF99" if is_ref else _DARK_PALETTE[fi % len(_DARK_PALETTE)],
        )

        # ── origin marker + label ─────────────────────────────────────────────
        hover_txt = (
            f"<b>{fname}</b>{'  ★ reference' if is_ref else ''}<br>"
            f"x = {pos[0]:.4f} m<br>"
            f"y = {pos[1]:.4f} m<br>"
            f"z = {pos[2]:.4f} m"
        )
        if not np.allclose(cov3, 0.0):
            hover_txt += f"<br>trace(C_trans) = {np.trace(cov3):.3e}"

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text",
            marker=dict(
                size=12 if is_ref else 8,
                color=color,
                symbol="diamond" if is_ref else "circle",
                line=dict(color="white" if dark_theme else "black", width=1.5),
                opacity=1.0,
            ),
            text=[f"<b>{fname}</b>"],
            textposition="top center",
            textfont=dict(
                size=13 if is_ref else 11,
                color="white" if dark_theme else "black",
                family="Arial Black" if is_ref else "Arial",
            ),
            name=fname,
            showlegend=True,
            hovertemplate=hover_txt + "<extra></extra>",
        ))

        # ── coordinate triad ──────────────────────────────────────────────────
        for i, (tc, aname) in enumerate(zip(_TRIAD_HEX, _AXIS_NAMES)):
            end = pos + rot[:, i] * triad_scale
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end[0], None],
                y=[pos[1], end[1], None],
                z=[pos[2], end[2], None],
                mode="lines",
                line=dict(color=tc, width=4),
                legendgroup=aname,
                name=aname,
                showlegend=(fi == 0),
                hoverinfo="skip",
            ))

        # ── uncertainty ellipsoid ─────────────────────────────────────────────
        if not np.allclose(cov3, 0.0) and np.all(np.isfinite(cov3)):
            try:
                X, Y, Z = _ellipsoid_surface(
                    pos, cov3, n_sigma=ellipsoid_sigma, n_pts=40
                )
                # Use height (Z) as the surface colour for a gradient effect
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=Z,
                    colorscale=ellipsoid_colorscale,
                    opacity=ellipsoid_opacity,
                    showscale=False,
                    name=f"{fname}  ({ellipsoid_sigma}σ ellipsoid)",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{fname}</b><br>"
                        f"{ellipsoid_sigma}σ positional uncertainty<br>"
                        f"trace(C_trans) = {np.trace(cov3):.3e}<extra></extra>"
                    ),
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.8,
                        roughness=0.5,
                        specular=0.6,
                        fresnel=0.4,
                    ),
                    lightposition=dict(x=1000, y=1000, z=500),
                ))
            except (np.linalg.LinAlgError, ValueError):
                pass

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template=template,
        paper_bgcolor=bg_color,
        title=dict(
            text=title,
            font=dict(size=15, color="white" if dark_theme else "black",
                      family="Arial"),
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=dict(
                title="X (m)",
                gridcolor=grid_col,
                showbackground=True,
                backgroundcolor="rgba(20,20,40,0.6)" if dark_theme else "rgba(240,240,255,0.6)",
                title_font=dict(color="white" if dark_theme else "black"),
            ),
            yaxis=dict(
                title="Y (m)",
                gridcolor=grid_col,
                showbackground=True,
                backgroundcolor="rgba(20,30,40,0.6)" if dark_theme else "rgba(240,248,255,0.6)",
                title_font=dict(color="white" if dark_theme else "black"),
            ),
            zaxis=dict(
                title="Z (m)",
                gridcolor=grid_col,
                showbackground=True,
                backgroundcolor="rgba(25,20,40,0.6)" if dark_theme else "rgba(248,240,255,0.6)",
                title_font=dict(color="white" if dark_theme else "black"),
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=2.0, y=1.4, z=0.9),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, b=40, t=60),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(30,30,50,0.85)" if dark_theme else "rgba(255,255,255,0.85)",
            bordercolor="rgba(150,150,200,0.5)",
            borderwidth=1,
            font=dict(color="white" if dark_theme else "black", size=11),
        ),
    )

    # ── auto-rotation animation ────────────────────────────────────────────────
    if animate_rotation:
        fig.frames = _make_rotation_frames(n_frames=120, radius=2.2, height=0.9)

        play_args = [
            None,
            {
                "frame": {"duration": 50, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
                "mode": "immediate",
            },
        ]
        pause_args = [
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ]

        btn_bg   = "rgba(40,40,70,0.9)"  if dark_theme else "rgba(220,220,240,0.9)"
        btn_font = dict(color="white" if dark_theme else "black", size=12)

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=0.06,
                xanchor="left",
                yanchor="bottom",
                bgcolor=btn_bg,
                bordercolor="rgba(150,150,200,0.6)",
                font=btn_font,
                buttons=[
                    dict(label="▶  Auto-rotate",
                         method="animate", args=play_args),
                    dict(label="⏸  Pause",
                         method="animate", args=pause_args),
                ],
            )],
            # Slider hidden but required to drive the animation frames
            sliders=[dict(
                steps=[
                    dict(method="animate",
                         args=[[str(i)],
                               {"frame": {"duration": 50, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
                         label="")
                    for i in range(len(fig.frames))
                ],
                transition=dict(duration=0),
                x=0.0, y=0.0, len=1.0,
                currentvalue=dict(visible=False),
                visible=False,
            )],
        )

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
        print(f"  [interactive] saved → {save_path}")

    return fig
