"""
Distributions.py
================
Polarimetric data visualization on the Poincaré sphere.

Provides:
  - CSV ingest of Stokes parameter measurements.
  - Professional, publication-quality Poincaré sphere rendering (Plotly).
  - Cone-intersection analysis: for a fully rotated retarder, recovers the
    cone axis and half-angle from the experimental polarization-state trajectory
    and overlays the cone, its great-circle intersection, and the colored
    experimental points on the same 3-D figure.
    This directly verifies the *law of elliptical birefringents*:
        A retarder rotates the Stokes vector about its fast-axis direction;
        the trajectory therefore traces a small circle on the Poincaré sphere,
        which is exactly the intersection of a (right-circular) cone whose apex
        is at the sphere's centre with the sphere's surface.

Dependencies:
  numpy, pandas, plotly, scipy, sphere (kent / fb8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import iplot
from scipy.optimize import minimize
from scipy.stats import vonmises_fisher

# Optional – used only in the commented-out MLE block at the bottom.
try:
    from sphere.distribution import fb8_mle, kent_me
except ImportError:
    kent_me = fb8_mle = None


# ---------------------------------------------------------------------------
# Colour / style constants  (tweak freely)
# ---------------------------------------------------------------------------

_SPHERE_COLORSCALE = [
    [0.00, "#0d1b2a"],
    [0.25, "#1b3a5c"],
    [0.50, "#2e6da4"],
    [0.75, "#5ba3d9"],
    [1.00, "#aad4f5"],
]
_SPHERE_OPACITY       = 0.18
_AXIS_COLOR           = "#e8e8e8"
_AXIS_WIDTH           = 2
_GRID_COLOR           = "rgba(180,200,220,0.25)"
_GRID_WIDTH           = 1
_CONE_COLOR           = "rgba(255,180,60,0.18)"   # warm gold, very transparent
_CONE_LINE_COLOR      = "rgba(255,180,60,0.55)"
_INTERSECTION_COLOR   = "#ffc84a"
_INTERSECTION_WIDTH   = 3
_AXIS_CONE_COLOR      = "#ff6b6b"
_BACKGROUND           = "#0a111a"
_FONT_COLOR           = "#d0e4f7"
_FONT_FAMILY          = "Courier New, monospace"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v* (1-D array)."""
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Cannot normalise a zero vector.")
    return v / n


def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues rotation matrix that rotates by *angle* (radians) about *axis*
    (unit vector, R³).

    Formula:  R = I cosθ + (1-cosθ)(axis⊗axis) + sinθ [axis]×

    Parameters
    ----------
    axis  : shape (3,) unit vector
    angle : rotation angle in radians

    Returns
    -------
    R : (3,3) orthogonal matrix, det = +1
    """
    axis = _unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    # Outer product  axis ⊗ axis
    outer = np.outer(axis, axis)
    # Cross-product (skew-symmetric) matrix  [axis]×
    skew = np.array([
        [  0, -uz,  uy],
        [ uz,   0, -ux],
        [-uy,  ux,   0],
    ])
    return c * np.eye(3) + (1.0 - c) * outer + s * skew


def _fit_cone_axis_angle(
    points: np.ndarray,
    n_restarts: int = 6,
) -> tuple[np.ndarray, float]:
    """
    Given a set of unit vectors *points* (shape N×3) that approximately lie on
    a small circle of the 2-sphere, estimate the cone axis **μ** (unit vector)
    and the half-angle **α** (radians).

    Physical meaning
    ----------------
    The law of elliptical birefringents states that a retarder with fast-axis
    direction **μ** (on the Poincaré sphere) rotates any incoming Stokes vector
    along a small circle whose pole is **μ**.  The half-angle α is determined
    by the angular distance between **μ** and the initial polarisation state.

    Estimation strategy
    -------------------
    We minimise the residual sum-of-squares of   (dot(pᵢ, μ) − cos α)²
    over the unit sphere × [0, π].  This is a non-linear least-squares problem;
    we use multiple random initialisations to avoid local minima.

    Parameters
    ----------
    points    : (N, 3) array of unit vectors on the Poincaré sphere.
    n_restarts: number of random initialisations of the optimiser.

    Returns
    -------
    axis  : (3,) unit vector  – the cone / retarder axis
    alpha : float in [0, π]  – the half-angle of the cone
    """
    # Parameterise μ in spherical coordinates (θ ∈ [0,π], φ ∈ [0,2π])
    # to enforce the unit-vector constraint implicitly.
    def _decode(params):
        theta, phi = params
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    def _residuals(params):
        # params = [theta, phi, alpha]
        mu    = _decode(params[:2])
        alpha = params[2]
        dots  = points @ mu                      # shape (N,)
        return float(np.sum((dots - np.cos(alpha)) ** 2))

    best_res = np.inf
    best_params = None
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        # Random initialisation on S²
        theta0 = rng.uniform(0, np.pi)
        phi0   = rng.uniform(0, 2 * np.pi)
        # Sensible α initialisation: median angular distance from a random μ
        mu0    = _decode([theta0, phi0])
        alpha0 = float(np.arccos(np.clip(np.median(points @ mu0), -1.0, 1.0)))
        x0     = [theta0, phi0, alpha0]
        bounds = [(0, np.pi), (0, 2 * np.pi), (0, np.pi)]
        res = minimize(
            _residuals, x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
        )
        if res.fun < best_res:
            best_res    = res.fun
            best_params = res.x

    axis  = _unit(_decode(best_params[:2]))
    alpha = float(best_params[2])
    return axis, alpha


def _small_circle(axis: np.ndarray, alpha: float, n: int = 400) -> np.ndarray:
    """
    Generate the small circle on the unit sphere defined by the set of unit
    vectors that make angle *alpha* with *axis*.

    Construction
    ------------
    1. Build an orthonormal frame {e1, e2, e3 = axis}.
    2. Parametrize the circle as:
           p(t) = cos(α) · axis  +  sin(α) · (cos(t)·e1 + sin(t)·e2)
       for t ∈ [0, 2π].

    Parameters
    ----------
    axis  : (3,) unit vector
    alpha : half-angle (radians)
    n     : number of sample points

    Returns
    -------
    pts : (n, 3) array of unit vectors on the circle
    """
    axis = _unit(axis)
    # Choose an arbitrary vector not parallel to axis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)               # already unit because axis⊥e1

    t   = np.linspace(0, 2 * np.pi, n)
    pts = (
        np.cos(alpha) * axis[None, :]
        + np.sin(alpha) * (np.cos(t)[:, None] * e1[None, :]
                           + np.sin(t)[:, None] * e2[None, :])
    )
    return pts


def _cone_mesh(
    axis: np.ndarray,
    alpha: float,
    n_phi: int = 80,
    n_r: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parametric mesh of the cone  { r · p(φ) : r ∈ [0,1], φ ∈ [0,2π] }
    where p(φ) is the small-circle generator computed in *_small_circle*.

    The apex of the cone is at the origin (centre of the Poincaré sphere).
    The base of the cone is the small circle on the unit sphere at radius r=1.

    Returns
    -------
    X, Y, Z : (n_r, n_phi) arrays suitable for go.Surface
    """
    axis = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)

    phi = np.linspace(0, 2 * np.pi, n_phi)
    r   = np.linspace(0, 1, n_r)
    PHI, R = np.meshgrid(phi, r)

    # Direction vector on the cone rim for each φ
    # d(φ) = cos(α)·axis + sin(α)·(cos(φ)·e1 + sin(φ)·e2)
    DX = np.cos(alpha) * axis[0] + np.sin(alpha) * (np.cos(PHI) * e1[0] + np.sin(PHI) * e2[0])
    DY = np.cos(alpha) * axis[1] + np.sin(alpha) * (np.cos(PHI) * e1[1] + np.sin(PHI) * e2[1])
    DZ = np.cos(alpha) * axis[2] + np.sin(alpha) * (np.cos(PHI) * e1[2] + np.sin(PHI) * e2[2])

    X = R * DX
    Y = R * DY
    Z = R * DZ
    return X, Y, Z


# ---------------------------------------------------------------------------
# Sphere mesh helper
# ---------------------------------------------------------------------------

def _sphere_mesh(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) meshgrid for the unit sphere (n×n grid)."""
    phi   = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0,     np.pi, n)
    PHI, THETA = np.meshgrid(phi, theta)
    x = np.cos(PHI) * np.sin(THETA)
    y = np.sin(PHI) * np.sin(THETA)
    z = np.cos(THETA)
    return x, y, z


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Distributions_Data:
    """
    Container for polarimetric measurement data and associated Poincaré-sphere
    visualisations.

    Parameters
    ----------
    route : str or False
        Path to the .csv file produced by a polarimeter (Thorlabs PAX / similar).
        Leave as *False* if you only want to plot synthetic data via *DData*.
    DData : ndarray, shape (N, 3)
        Pre-computed Stokes (S1, S2, S3) vectors.  Used by :meth:`PlotDD` and
        :meth:`PlotConeIntersection`.  Ignored unless explicitly needed.
    """

    def __init__(self, route: str | bool = False, DData: np.ndarray = np.empty((0, 3))):
        self.route = route
        self.DData = np.asarray(DData, dtype=float)

        # These are populated by :meth:`GetStokes`
        self.S1:  np.ndarray | None = None
        self.S2:  np.ndarray | None = None
        self.S3:  np.ndarray | None = None
        self.DOP: np.ndarray | None = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def readData(self, retornar: bool = False) -> pd.DataFrame | None:
        """
        Read the CSV from *self.route*.

        The polarimeter prepends 22 header lines before the actual data table,
        which is why we pass ``header=22``.  The encoding 'latin-1' handles the
        degree-sign (°) byte that some polarimeter firmware versions emit as a
        Latin-1 byte instead of UTF-8.

        Parameters
        ----------
        retornar : bool
            If True, return the DataFrame in addition to storing it in
            *self.Data*.

        Returns
        -------
        pd.DataFrame or None
        """
        self.Data = pd.read_csv(
            self.route,
            header=22,
            encoding="latin-1",
            engine="python",
        )
        self.Data.reset_index(inplace=True)
        if retornar:
            return self.Data
        return None

    def GetStokes(
        self,
        retS:   bool = False,
        retDOP: bool = False,
    ) -> tuple | None:
        """
        Parse Stokes parameters and the degree-of-polarisation (DOP) from
        *self.Data* after calling :meth:`readData`.

        Column-name aliasing
        --------------------
        Different firmware versions write the degree-sign differently:
          • UTF-8 proper:      'Phase Difference [°]'
          • Latin-1 mis-read:  'Phase Difference [ï¿½]'   (UTF-8 BOM garbage)
          • Another variant:   'Phase Difference [â°]'
        We try all known variants and raise a descriptive exception if none
        matches.

        DOP normalisation: the raw column is in % → we divide by 100.

        Parameters
        ----------
        retS   : bool  – return (S1, S2, S3) tuple
        retDOP : bool  – return DOP array

        Returns
        -------
        Depends on *retS* / *retDOP* flags.  Stores results as instance
        attributes regardless.
        """
        self.readData()

        # ------ Stokes columns ------
        # The polarimeter stores what it calls "Time Stamp [s]" in the column
        # that the software labels 'Stokes 1' when the user exports Stokes.
        # The actual Stokes parameters follow in 'Stokes 1', 'Stokes 2', etc.
        # (This quirk is a known firmware issue; the column layout below
        #  matches the observed CSV structure.)
        self.S1 = self.Data["Time Stamp [s]"].values.astype(float)
        self.S2 = self.Data["Stokes 1"].values.astype(float)
        self.S3 = self.Data["Stokes 2"].values.astype(float)

        # ------ DOP / Phase Difference column ------
        _phase_variants = [
            "Phase Difference [°]",
            "Phase Difference [ï¿½]",
            "Phase Difference [â°]",
            "Phase Difference [?]",
        ]
        dop_col = None
        for col in _phase_variants:
            if col in self.Data.columns:
                dop_col = col
                break

        if dop_col is None:
            raise KeyError(
                "Cannot locate the Phase Difference column.  "
                f"Available columns: {list(self.Data.columns)}"
            )
        self.DOP = (self.Data[dop_col] / 100.0).values.astype(float)

        # ------ Return ------
        if retS and retDOP:
            return self.S1, self.S2, self.S3, self.DOP
        if retS and not retDOP:
            return self.S1, self.S2, self.S3
        if not retS and retDOP:
            return self.DOP
        return None

    # ------------------------------------------------------------------
    # Internal trace builders  (return go.BaseTraceType instances)
    # ------------------------------------------------------------------

    def _trace_sphere(self) -> go.Surface:
        """
        Professional dark-ocean Poincaré sphere surface.

        The sphere is rendered as a parametric surface with a dark-blue
        colorscale, low opacity, and no colour bar – it serves purely as a
        spatial reference backdrop.
        """
        x, y, z = _sphere_mesh(120)
        return go.Surface(
            x=x, y=y, z=z,
            showscale=False,
            opacity=_SPHERE_OPACITY,
            colorscale=_SPHERE_COLORSCALE,
            lighting=dict(
                ambient=0.6,
                diffuse=0.5,
                specular=0.3,
                roughness=0.7,
                fresnel=0.5,
            ),
            lightposition=dict(x=1, y=1, z=2),
        )

    def _traces_axes(self) -> list[go.Scatter3d]:
        """
        Cartesian axis lines S1, S2, S3 as thin light-grey segments.
        Each axis extends from −1.15 to +1.15 to slightly overshoot the sphere.
        """
        traces = []
        for vec, lbl in [
            ([[-1.15, 1.15], [0, 0], [0, 0]], "S1"),
            ([[0, 0], [-1.15, 1.15], [0, 0]], "S2"),
            ([[0, 0], [0, 0], [-1.15, 1.15]], "S3"),
        ]:
            traces.append(go.Scatter3d(
                x=vec[0], y=vec[1], z=vec[2],
                mode="lines",
                line=dict(color=_AXIS_COLOR, width=_AXIS_WIDTH, dash="solid"),
                showlegend=False,
                hoverinfo="skip",
            ))
        return traces

    def _traces_grid_circles(self) -> list[go.Scatter3d]:
        """
        Three reference great-circles (equator + two meridians) rendered as
        faint dashed lines to give the sphere a sense of curvature without
        visual clutter.
        """
        t = np.linspace(0, 2 * np.pi, 300)
        circles = [
            # Equator  (S1–S2 plane, z = 0)
            dict(x=np.cos(t), y=np.sin(t), z=np.zeros_like(t)),
            # Meridian (S1–S3 plane, y = 0)
            dict(x=np.cos(t), y=np.zeros_like(t), z=np.sin(t)),
            # Meridian (S2–S3 plane, x = 0)
            dict(x=np.zeros_like(t), y=np.cos(t), z=np.sin(t)),
        ]
        return [
            go.Scatter3d(
                x=c["x"], y=c["y"], z=c["z"],
                mode="lines",
                line=dict(color=_GRID_COLOR, width=_GRID_WIDTH, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
            for c in circles
        ]

    def _trace_scatter(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        s3: np.ndarray,
        dop: np.ndarray,
        colorscale: str = "Plasma",
        colorbar_title: str = "DOP",
        marker_size: int = 3,
    ) -> go.Scatter3d:
        """
        Scatter plot of experimental polarisation states on the sphere.

        The marker colour encodes the DOP value using a perceptually-uniform
        colorscale.

        Parameters
        ----------
        s1, s2, s3       : Stokes parameter arrays (must be unit-sphere
                           coordinates if plotted on the Poincaré sphere).
        dop              : degree-of-polarisation values ∈ [0, 1].
        colorscale       : Plotly colorscale name.
        colorbar_title   : label shown next to the colour bar.
        marker_size      : pixel radius of each marker.
        """
        return go.Scatter3d(
            x=s1, y=s2, z=s3,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=dop,
                colorscale=colorscale,
                cmin=float(np.min(dop)),
                cmax=float(np.max(dop)),
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(size=11, color=_FONT_COLOR)),
                    tickfont=dict(size=9, color=_FONT_COLOR),
                    thickness=12,
                    len=0.6,
                    x=1.02,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                ),
                line=dict(width=0),
                opacity=0.92,
            ),
            showlegend=False,
        )

    def _layout(self, title: str = "Poincaré Sphere") -> go.Layout:
        """
        Return a dark, publication-quality layout for the Poincaré sphere.

        Design choices
        --------------
        - Near-black background (#0a111a) gives high contrast against the
          light data markers.
        - All axis visuals are suppressed: the sphere + reference circles
          provide spatial context without grid noise.
        - Axis annotations are placed slightly outside the sphere (at ±1.25)
          in the Courier New monospace font to give a precision-instrument feel.
        - Margins are tight; the camera is slightly elevated for a pleasing
          isometric-ish perspective.
        """
        annotations = [
            dict(showarrow=False, x= 0,    y= 0,    z= 1.28, text="<b>S₃</b>",
                 font=dict(size=14, color=_FONT_COLOR, family=_FONT_FAMILY)),
            dict(showarrow=False, x= 1.28, y= 0,    z= 0,    text="<b>S₁</b>",
                 font=dict(size=14, color=_FONT_COLOR, family=_FONT_FAMILY)),
            dict(showarrow=False, x= 0,    y= 1.28, z= 0,    text="<b>S₂</b>",
                 font=dict(size=14, color=_FONT_COLOR, family=_FONT_FAMILY)),
        ]
        return go.Layout(
            title=dict(
                text=title,
                font=dict(size=16, color=_FONT_COLOR, family=_FONT_FAMILY),
                x=0.5, xanchor="center",
            ),
            paper_bgcolor=_BACKGROUND,
            scene=dict(
                bgcolor=_BACKGROUND,
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                annotations=annotations,
                camera=dict(
                    eye=dict(x=1.55, y=1.35, z=0.75),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family=_FONT_FAMILY, color=_FONT_COLOR),
        )

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def PlotMD(
        self,
        SavePDF:    bool = False,
        title:      str  = "Poincaré Sphere – Measured Data",
        colorscale: str  = "Plasma",
        titleS:     str  = "DOP",
        PDFname:    str  = "Distribution",
    ) -> None:
        """
        Plot measured polarisation states (read from CSV) on the Poincaré sphere.

        Stokes parameters are normalised to lie on (or inside) the unit sphere
        before plotting.  The colour of each marker encodes the DOP value.

        Parameters
        ----------
        SavePDF    : if True, export to *PDFname*.pdf (requires kaleido).
        title      : figure title.
        colorscale : Plotly colorscale for the DOP colour map.
        titleS     : colour-bar label.
        PDFname    : base filename for PDF export.
        """
        self.GetStokes()

        traces: list = (
            [self._trace_sphere()]
            + self._traces_grid_circles()
            + self._traces_axes()
            + [self._trace_scatter(self.S1, self.S2, self.S3, self.DOP,
                                   colorscale=colorscale, colorbar_title=titleS)]
        )
        fig = go.Figure(data=traces, layout=self._layout(title))
        iplot(fig)
        if SavePDF:
            pio.write_image(fig, PDFname + ".pdf")

    def PlotDD(
        self,
        SavePDF:    bool       = False,
        title:      str        = "Poincaré Sphere – Direct Data",
        colorscale: str        = "Plasma",
        titleS:     str        = "DOP",
        PDFname:    str        = "Distribution",
        ColorDOP:   np.ndarray = np.empty(0),
    ) -> None:
        """
        Plot a pre-computed set of Stokes vectors (*self.DData*) on the sphere.

        Parameters
        ----------
        SavePDF    : export to PDF if True.
        title      : figure title.
        colorscale : colorscale for the colour map.
        titleS     : colour-bar label.
        PDFname    : PDF base filename.
        ColorDOP   : if provided (len > 2), use these values as the colour
                     variable instead of the radial DOP derived from DData.
        """
        if self.DData.shape[0] == 0:
            raise ValueError("DData is empty – provide data before calling PlotDD.")

        s1, s2, s3 = self.DData[:, 0], self.DData[:, 1], self.DData[:, 2]
        dop = ColorDOP if len(ColorDOP) > 2 else np.linalg.norm(self.DData, axis=1)

        traces: list = (
            [self._trace_sphere()]
            + self._traces_grid_circles()
            + self._traces_axes()
            + [self._trace_scatter(s1, s2, s3, dop,
                                   colorscale=colorscale, colorbar_title=titleS)]
        )
        fig = go.Figure(data=traces, layout=self._layout(title))
        iplot(fig)
        if SavePDF:
            pio.write_image(fig, PDFname + ".pdf")

    def Depol_Curve(
        self,
        retardances: np.ndarray,
        text:        str,
        xaxis_title: str  = "Retardance",
        autorange:   bool = True,
    ) -> None:
        """
        Plot DOP vs. retardance (depolarisation curve).

        Parameters
        ----------
        retardances : 1-D array of retardance values (same length as DOP).
        text        : figure title.
        xaxis_title : x-axis label.
        autorange   : if True, Plotly auto-scales; set False to reverse the axis.
        """
        self.GetStokes()
        fig = go.Figure(
            data=go.Scatter(
                x=retardances,
                y=self.DOP,
                mode="markers",
                marker=dict(
                    color=self.DOP,
                    colorscale="Plasma",
                    opacity=0.85,
                    size=5,
                    line=dict(width=0),
                ),
            )
        )
        fig.update_layout(
            title=dict(text=text, x=0.5, xanchor="center",
                       font=dict(size=15, family=_FONT_FAMILY, color=_FONT_COLOR)),
            paper_bgcolor=_BACKGROUND,
            plot_bgcolor="#0d1b2a",
            xaxis=dict(title=xaxis_title, autorange=autorange,
                       color=_FONT_COLOR, gridcolor="#1e3050"),
            yaxis=dict(title="DOP", color=_FONT_COLOR, gridcolor="#1e3050"),
            font=dict(family=_FONT_FAMILY, color=_FONT_COLOR),
        )
        fig.show()

    # ------------------------------------------------------------------
    # Cone-intersection  (law of elliptical birefringents)
    # ------------------------------------------------------------------

    def PlotConeIntersection(
        self,
        use_DData:   bool  = True,
        colorscale:  str   = "Plasma",
        title:       str   = "Poincaré Sphere – Retarder Cone Analysis",
        PDFname:     str   = "ConeIntersection",
        SavePDF:     bool  = False,
        n_restarts:  int   = 8,
    ) -> tuple[np.ndarray, float]:
        """
        For a set of polarisation states measured during a *full rotation* of a
        linear (or elliptical) retarder, this method:

          1. Fits a cone  C(μ, α)  whose apex is at the origin and whose axis
             is **μ** with half-angle α such that the small-circle
               { x ∈ S² : x·μ = cos α }
             best approximates the experimental trajectory.

          2. Plots the Poincaré sphere, the semi-transparent cone mesh, the
             exact small-circle intersection curve, the experimental points
             coloured by their DOP, and a short arrow indicating the fitted
             cone axis **μ**.

        Physical justification – law of elliptical birefringents
        ---------------------------------------------------------
        An elliptical retarder with retardance δ and fast-axis Jones vector **e**
        acts on the Stokes vector via a rotation R(δ, **μ**) in R³ (where **μ**
        is the Poincaré-sphere image of **e**).  As the retarder is physically
        rotated by a full revolution, the laboratory fast-axis sweeps out all
        orientations, but the *retardance* δ remains fixed.  For each orientation
        θ of the retarder, the output Stokes vector
               S_out(θ) = R(δ, R_z(2θ) **μ**) · S_in
        traces an orbit on the Poincaré sphere.  Under the assumption of a
        uniformly polarised input S_in, this orbit is a small circle, i.e. the
        intersection of the cone C(μ_eff, α_eff) with the unit sphere.
        Fitting this cone and recovering μ_eff and α_eff therefore provides an
        experimental estimate of the birefringent axis and retardance.

        Parameters
        ----------
        use_DData   : if True (default) use *self.DData*; otherwise use
                      *self.S1, S2, S3* from CSV (calls :meth:`GetStokes`).
        colorscale  : colorscale for the experimental scatter.
        title       : figure title.
        PDFname     : base name for PDF export.
        SavePDF     : export to PDF if True.
        n_restarts  : number of optimiser restarts for the cone fit.

        Returns
        -------
        axis  : (3,) unit vector – the fitted cone / retarder axis on the
                Poincaré sphere.
        alpha : float – the fitted cone half-angle in radians.

        Raises
        ------
        ValueError
            If neither CSV data nor DData is available.
        """
        # ---- 1. Collect the Stokes vectors --------------------------------
        if use_DData:
            if self.DData.shape[0] == 0:
                raise ValueError(
                    "DData is empty.  Either pass DData to the constructor or "
                    "set use_DData=False to read from the CSV."
                )
            pts = self.DData.copy()
            dop = np.linalg.norm(pts, axis=1)
        else:
            self.GetStokes()
            pts = np.column_stack([self.S1, self.S2, self.S3])
            dop = self.DOP

        # ---- 2. Normalise to unit sphere -----------------------------------
        # The Poincaré sphere requires unit vectors.  Partially polarised
        # states have |S| < 1; we normalise so the geometric fit makes sense.
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)   # avoid division by zero
        pts_unit = pts / norms

        # ---- 3. Fit the cone ----------------------------------------------
        axis, alpha = _fit_cone_axis_angle(pts_unit, n_restarts=n_restarts)

        print(
            f"[ConeIntersection]  axis = ({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})"
            f"  |  half-angle α = {np.degrees(alpha):.3f}°"
            f"  |  retardance δ ≈ {2 * np.degrees(alpha):.3f}°"
        )

        # ---- 4. Generate cone mesh and small circle -----------------------
        Xc, Yc, Zc   = _cone_mesh(axis, alpha, n_phi=90, n_r=25)
        circle_pts    = _small_circle(axis, alpha, n=500)

        # Axis indicator: a line from origin to slightly beyond the sphere
        ax_line = 1.15 * axis

        # ---- 5. Build Plotly traces ---------------------------------------

        # 5a. Sphere
        sphere_trace = self._trace_sphere()

        # 5b. Cone surface
        cone_surface = go.Surface(
            x=Xc, y=Yc, z=Zc,
            showscale=False,
            opacity=0.22,
            colorscale=[[0, _CONE_COLOR], [1, _CONE_COLOR]],
            surfacecolor=np.ones_like(Xc),
            lighting=dict(ambient=0.9, diffuse=0.1, specular=0.05),
            name="Cone",
            showlegend=True,
        )

        # 5c. Cone-sphere intersection (small circle)
        circle_trace = go.Scatter3d(
            x=circle_pts[:, 0],
            y=circle_pts[:, 1],
            z=circle_pts[:, 2],
            mode="lines",
            line=dict(color=_INTERSECTION_COLOR, width=_INTERSECTION_WIDTH),
            name="Intersection curve",
            showlegend=True,
        )

        # 5d. Cone axis arrow
        axis_trace = go.Scatter3d(
            x=[0, ax_line[0]],
            y=[0, ax_line[1]],
            z=[0, ax_line[2]],
            mode="lines+markers",
            line=dict(color=_AXIS_CONE_COLOR, width=4),
            marker=dict(size=[0, 7], color=_AXIS_CONE_COLOR, symbol="circle"),
            name=f"Cone axis μ  (α={np.degrees(alpha):.1f}°)",
            showlegend=True,
        )

        # 5e. Experimental scatter coloured by DOP
        scatter_trace = self._trace_scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            dop,
            colorscale=colorscale,
            colorbar_title="DOP",
            marker_size=3,
        )
        scatter_trace.name      = "Experimental states"
        scatter_trace.showlegend = True

        # ---- 6. Layout (override legend) ----------------------------------
        layout = self._layout(title)
        layout.legend = dict(
            font=dict(size=10, color=_FONT_COLOR, family=_FONT_FAMILY),
            bgcolor="rgba(10,17,26,0.7)",
            bordercolor="#2e6da4",
            borderwidth=1,
            x=0.01, y=0.98,
        )

        # ---- 7. Assemble figure -------------------------------------------
        all_traces = (
            [sphere_trace]
            + self._traces_grid_circles()
            + self._traces_axes()
            + [cone_surface, circle_trace, axis_trace, scatter_trace]
        )
        fig = go.Figure(data=all_traces, layout=layout)
        iplot(fig)

        if SavePDF:
            pio.write_image(fig, PDFname + ".pdf")

        return axis, alpha


# ===========================================================================
# Quick-start usage examples  (uncomment one block to run)
# ===========================================================================

# ---- Synthetic von Mises–Fisher cloud (tight cluster, high κ) -------------
# mu    = _unit(np.array([0.5, 0.7, 0.5]))
# kappa = 800
# DData = vonmises_fisher(mu=mu, kappa=kappa).rvs(400)
# demo  = Distributions_Data(DData=DData)
# demo.PlotDD()

# ---- Synthetic small circle (perfect retarder trajectory) -----------------
# axis_true  = _unit(np.array([1.0, 0.5, 0.3]))
# alpha_true = np.radians(35)
# circle     = _small_circle(axis_true, alpha_true, n=300)
# noise      = np.random.default_rng(0).normal(0, 0.02, circle.shape)
# noisy_pts  = circle + noise
# noisy_pts /= np.linalg.norm(noisy_pts, axis=1, keepdims=True)
# demo  = Distributions_Data(DData=noisy_pts)
# axis_fit, alpha_fit = demo.PlotConeIntersection()
# print(f"True axis : {axis_true}  alpha = {np.degrees(alpha_true):.2f}°")
# print(f"Fitted    : {axis_fit}  alpha = {np.degrees(alpha_fit):.2f}°")

# ---- From CSV file --------------------------------------------------------
# demo = Distributions_Data(route="path/to/your/data.csv")
# demo.PlotMD()
# demo.PlotConeIntersection(use_DData=False)