import re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
import sympy as spy
import mpmath
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# ---------------------------------------------------------------------------
# Module-level cached constants (created once at import time)
# ---------------------------------------------------------------------------

# Symbolic frequency variable shared by all Waveplate instances
_X = spy.symbols(r'2\pi/\lambda', real=True)

# Stokes–Jones transformation matrix and its inverse
_STOKES_T = spy.Matrix([
    [1,  0,    0,  1],
    [1,  0,    0, -1],
    [0,  1,    1,  0],
    [0, 1j, -1j,  0],
])
_STOKES_T_INV = _STOKES_T.inv()

# Pauli matrices (fixed, never change)
_PAULI = [
    spy.Matrix([[1,  0], [0,  1]]),
    spy.Matrix([[1,  0], [0, -1]]),
    spy.Matrix([[0,  1], [1,  0]]),
    spy.Matrix([[0, -1j], [1j, 0]]),
]


class Coherence:
    """
    This class creates a temporal complex degree of coherence function associated with a light source.
    You can use the Gaussian, Lorentz, GaussLorentz or Blackbody coherence functions to model
    a partially coherent light source.
    """

    coherences = ['Gaussian', 'Lorentzian', 'GaussLorentz', 'BlackBody']

    @staticmethod
    def gaussian(OptPathDiff, coh_length, wavelength=0.633):
        """
        Gaussian coherence function.

        Parameters
        ----------
        OptPathDiff : float
        coh_length : float
        wavelength : float, optional  (default 0.633 um)

        Returns
        -------
        complex
        """
        return np.exp(-(np.pi / 2) * (OptPathDiff / coh_length) ** 2
                      - (2j * OptPathDiff * np.pi / wavelength))

    @staticmethod
    def lorentzian(OptPathDiff, coh_length, wavelength=0.633):
        """
        Lorentzian coherence function.

        Parameters
        ----------
        OptPathDiff : float
        coh_length : float
        wavelength : float, optional  (default 0.633 um)

        Returns
        -------
        complex
        """
        return np.exp(-np.abs(OptPathDiff) / coh_length
                      - (2j * OptPathDiff * np.pi / wavelength))

    @staticmethod
    def gauss_lorentz(OptPathDiff, coh_length: tuple, wavelength=0.633):
        """
        Gaussian-Lorentzian coherence function.

        Parameters
        ----------
        OptPathDiff : float
        coh_length : tuple  (coh_length_G, coh_length_L)
        wavelength : float, optional  (default 0.633 um)

        Returns
        -------
        complex
        """
        coh_length_G, coh_length_L = coh_length
        return np.exp(-(np.pi / 2) * (OptPathDiff / coh_length_G) ** 2
                      - np.abs(OptPathDiff) / coh_length_L
                      - (2j * OptPathDiff * np.pi / wavelength))

    @staticmethod
    def blackbody(OptPathDiff, wavelength):
        return 90 / np.pi ** 4 * mpmath.zeta(4, 1 + 1j * 1.16e-19 * OptPathDiff / wavelength)

    def __init__(self, coherence_func='Gaussian', **kwargs):
        self.coherence_func = coherence_func
        if self.coherence_func not in self.coherences:
            raise ValueError(f'Coherence function not valid, please use one in {self.coherences}')
        self.wavelength = kwargs.get("wavelength")
        self.coh_length = kwargs.get("coh_length")   # cached once; not re-read on every eval()

    def eval(self, OptPathDiff):
        """
        Evaluate the coherence function at a given optical path difference.

        Parameters
        ----------
        OptPathDiff : float

        Returns
        -------
        complex
        """
        if self.coherence_func == 'Gaussian':
            return Coherence.gaussian(OptPathDiff, self.coh_length, self.wavelength)
        elif self.coherence_func == 'Lorentzian':
            return Coherence.lorentzian(OptPathDiff, self.coh_length, self.wavelength)
        elif self.coherence_func == 'GaussLorentz':
            return Coherence.gauss_lorentz(OptPathDiff, self.coh_length, self.wavelength)
        elif self.coherence_func == 'BlackBody':
            return Coherence.blackbody(OptPathDiff, self.wavelength)


class State(spy.Matrix):
    """
    Polarization state of a light source expressed as a coherence (Jones) matrix.
    Parameterised by orientation angle alpha and ellipticity angle chi (degrees).
    An optional basis transformation can be specified via the basis argument.
    """

    @staticmethod
    def pauli_matrices():
        """Return the four 2×2 Pauli matrices."""
        return _PAULI

    @staticmethod
    def basis_change(alpha_0, chi_0):
        """
        Transformation matrix from the linear basis to a new basis defined by
        orientation angle alpha_0 and ellipticity angle chi_0 (degrees).
        """
        alpha = np.deg2rad(alpha_0)
        chi   = np.deg2rad(chi_0)
        T11 = np.cos(alpha) * np.cos(chi) - 1j * np.sin(alpha) * np.sin(chi)
        T12 = np.cos(alpha) * np.sin(chi) + 1j * np.sin(alpha) * np.cos(chi)
        T21 = np.sin(alpha) * np.cos(chi) + 1j * np.cos(alpha) * np.sin(chi)
        T22 = np.sin(alpha) * np.sin(chi) - 1j * np.cos(alpha) * np.cos(chi)
        return spy.simplify(spy.Matrix([[T11, T12], [T21, T22]]))

    def __new__(cls, alpha=0, chi=0, basis=(0, 0)):
        # SymPy internally calls cls(matrix) when unifying types during multiplication.
        if isinstance(alpha, spy.MatrixBase):
            return super(State, cls).__new__(cls, alpha)

        cls.alpha = np.deg2rad(alpha)
        cls.chi   = np.deg2rad(chi)
        cls.basis = basis
        cls.Matrix_Basis_Change = State.basis_change(basis[0], basis[1])

        # Jones vector in the linear (lab) basis
        cls._state = spy.Matrix([
            np.cos(cls.alpha) * np.cos(cls.chi) - 1j * np.sin(cls.alpha) * np.sin(cls.chi),
            np.sin(cls.alpha) * np.cos(cls.chi) + 1j * np.cos(cls.alpha) * np.sin(cls.chi),
        ])

        # Components in the requested basis (identity transform when basis == (0,0))
        if basis == (0, 0) or basis == [0, 0]:
            components = cls._state
        else:
            components = cls.Matrix_Basis_Change.inv() * cls._state

        cls.E1 = spy.N(spy.nsimplify(spy.simplify(components[0]), rational=True, tolerance=1e-3), 4)
        cls.E2 = spy.N(spy.nsimplify(spy.simplify(components[1]), rational=True, tolerance=1e-3), 4)

        # Coherence (polarization) matrix J = |E><E|
        return super(State, cls).__new__(cls, [
            [spy.nsimplify(spy.simplify(cls.E1 * np.conjugate(cls.E1)), rational=True, tolerance=1e-3),
             spy.nsimplify(spy.simplify(cls.E1 * np.conjugate(cls.E2)), rational=True, tolerance=1e-3)],
            [spy.nsimplify(spy.simplify(cls.E2 * np.conjugate(cls.E1)), rational=True, tolerance=1e-3),
             spy.nsimplify(spy.simplify(cls.E2 * np.conjugate(cls.E2)), rational=True, tolerance=1e-3)],
        ])

    def __init__(self, alpha=0, chi=0, basis=(0, 0)):
        if isinstance(alpha, spy.MatrixBase):
            return
        self.alpha = alpha
        self.chi   = chi
        self.basis = basis
        self.dop   = 1
        # Reuse the matrix already computed in __new__ instead of recomputing
        self.Matrix_Basis_Change = State.basis_change(basis[0], basis[1])

    @property
    def jones(self):
        """Jones vector of the polarization state."""
        return spy.Matrix([[self.E1], [self.E2]])

    @property
    def stokes(self):
        """Stokes vector (4×1) of the polarization state."""
        X = spy.kronecker_product(self.jones, self.jones.conjugate())
        return _STOKES_T * X

    @property
    def s0(self):
        return self.stokes[0]

    @property
    def s1(self):
        return self.stokes[1]

    @property
    def s2(self):
        return self.stokes[2]

    @property
    def s3(self):
        return self.stokes[3]

    def ellipse(self):
        """Plot the polarization ellipse associated with this state."""
        fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

        tan_chi = np.tan(np.deg2rad(self.chi))
        denom   = np.sqrt(tan_chi ** 2 + 1)
        horizontal_axis = 1 / denom
        vertical_axis   = tan_chi / denom

        ellipse_patch = Ellipse(
            xy=(0, 0),
            width=horizontal_axis,
            height=vertical_axis,
            angle=self.alpha,
            facecolor="none",
            edgecolor="b",
        )
        ax.add_patch(ellipse_patch)

        vertices = ellipse_patch.get_co_vertices()
        t = Affine2D().rotate_deg(ellipse_patch.angle)
        ax.plot(vertices[0][0], vertices[0][1],
                color="b", marker=MarkerStyle(">", "full", t), markersize=10)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])
        plt.show()

    def Poincare_sphere(self):
        Graphic([self])

    def operate(self, operators, coherence):
        """
        Apply a list of operators (Composite_waveplate objects) to this state.

        Parameters
        ----------
        operators : list of Composite_waveplate
        coherence : Coherence

        Returns
        -------
        list of Partial_State
        """
        M = spy.Matrix(self)  # plain matrix avoids SymPy type-unification overhead

        # Pre-compute all coherence values needed across every operator
        all_opds = set()
        for op in operators:
            for opd in op.list_of_phases()[0]:
                all_opds.update({opd, -opd})
        coh_cache = {opd: coherence.eval(opd) for opd in all_opds}

        operated_state = []
        for operator in operators:
            op_M  = spy.Matrix(operator)
            state = spy.expand(op_M * M * op_M.adjoint()).rewrite(spy.cos).expand()

            list_of_OPD, list_of_sines, list_of_cosines = operator.list_of_phases()
            for OPD, sine, cosine in zip(list_of_OPD, list_of_sines, list_of_cosines):
                state = state.subs(sine,   np.imag(coh_cache[OPD]))
                state = state.subs(cosine, np.real(coh_cache[-OPD]))

            operated_state.append(Partial_State(state[0], state[1], state[2], state[3]))

        return operated_state


class Partial_State(spy.Matrix):
    """
    Partially polarized state described by the four elements J11, J12, J21, J22
    of the coherence (polarization) matrix.
    """

    def __new__(cls, J11, J12=None, J21=None, J22=None, basis=(0, 0)):
        # SymPy internally calls cls(matrix) when unifying types during multiplication.
        if isinstance(J11, spy.MatrixBase):
            return super(Partial_State, cls).__new__(cls, J11)
        return super(Partial_State, cls).__new__(cls, [[J11, J12], [J21, J22]])

    def __init__(self, J11, J12=None, J21=None, J22=None, basis=(0, 0)):
        if isinstance(J11, spy.MatrixBase):
            return
        self.J11  = J11
        self.J12  = J12
        self.J21  = J21
        self.J22  = J22
        self.base = basis

    @property
    def dop(self):
        """Degree of Polarization."""
        return float(spy.re(spy.sqrt(1 - 4 * self.det())))

    @property
    def stokes(self):
        """Stokes parameters [S0, S1, S2, S3] of the partial state."""
        return [spy.N(spy.nsimplify(spy.re(spy.trace(sigma * spy.Matrix(self))),
                                    tolerance=1e-3), 3)
                for sigma in _PAULI]

    @property
    def s1(self):
        return self.stokes[1]

    @property
    def s2(self):
        return self.stokes[2]

    @property
    def s3(self):
        return self.stokes[3]

    def Poincare_sphere(self):
        """Plot this state on the Poincaré sphere."""
        Graphic([self])


class Rotation(spy.Matrix):
    """Rotation matrix operator in Jones representation."""

    def __new__(cls, angle, basis=(0, 0)):
        cls.angle = np.deg2rad(angle)
        cls.Basis = basis
        rotation = spy.Matrix([
            [ spy.cos(cls.angle), spy.sin(cls.angle)],
            [-spy.sin(cls.angle), spy.cos(cls.angle)],
        ])
        if basis == (0, 0) or basis == [0, 0]:
            return spy.nsimplify(spy.simplify(rotation), tolerance=0.001)
        else:
            T = State.basis_change(basis[0], basis[1])
            return spy.nsimplify(spy.simplify(T.inv() * rotation * T), tolerance=0.001)

    @property
    def mueller(self):
        """Mueller matrix of this rotation."""
        M = spy.kronecker_product(self, self.conjugate())
        return spy.simplify(_STOKES_T * M * _STOKES_T_INV)


class Waveplate(spy.Matrix):
    """
    Jones matrix operator for a waveplate with a given optical path difference (OPD),
    orientation angle, eigenstate basis, and lab basis.
    """

    def __new__(cls, OptPathDiff, angle=0, eigenstate=(0, 0), basis=(0, 0)):
        # SymPy internally calls cls(matrix) when unifying types during multiplication.
        if isinstance(OptPathDiff, spy.MatrixBase):
            return super(Waveplate, cls).__new__(cls, OptPathDiff)

        cls.OptPathDiff = OptPathDiff
        cls.angle       = np.deg2rad(angle)
        cls.basis       = basis

        # Diagonal retarder in eigenstate basis (symbolic phase)
        retarder = spy.Matrix([
            [spy.exp(_X * spy.I * OptPathDiff), 0],
            [0, 1],
        ])

        T_eig = State.basis_change(eigenstate[0], eigenstate[1])
        retarder = T_eig * retarder * T_eig.inv()
        R = Rotation(angle) if (basis == (0, 0) or basis == [0, 0]) else Rotation(angle, basis)
        retarder = R.inv() * retarder * R

        if not (basis == (0, 0) or basis == [0, 0]):
            T_basis  = State.basis_change(basis[0], basis[1])
            retarder = T_basis.inv() * retarder * T_eig
            retarder = spy.nsimplify(spy.simplify(retarder), rational=True)

        return super(Waveplate, cls).__new__(cls, retarder)

    def __init__(self, OptPathDiff, angle=0, eigenstate=(0, 0), basis=(0, 0)):
        if isinstance(OptPathDiff, spy.MatrixBase):
            return
        self.OptPathDiff = OptPathDiff
        self.angle       = np.deg2rad(angle)
        self.basis       = basis
        self.eigenstate  = eigenstate

    @property
    def mueller(self):
        """Mueller matrix of this waveplate."""
        M = spy.kronecker_product(self, self.conjugate())
        return spy.simplify(_STOKES_T * M * _STOKES_T_INV)

    def rotate(self, angle):
        """Return a new Waveplate rotated by the given angle (degrees)."""
        return Waveplate(self.OptPathDiff,
                         np.rad2deg(self.angle) + angle,
                         self.eigenstate,
                         self.basis)

    def list_of_phases(self):
        """Return (OPD, sine_symbol, cosine_symbol) for this waveplate."""
        return (self.OptPathDiff,
                spy.sin(_X * self.OptPathDiff),
                spy.cos(_X * self.OptPathDiff))

    def operate(self, states, coherence):
        """
        Apply this waveplate to a list of polarization states.

        Parameters
        ----------
        states : list of State / Partial_State
        coherence : Coherence

        Returns
        -------
        list of Partial_State
        """
        op_M = spy.Matrix(self)
        OPD, sine, cosine = self.list_of_phases()
        coh_val   = coherence.eval(OPD)   # computed once for all states
        imag_val  = np.imag(coh_val)
        real_val  = np.real(coh_val)

        operated_state = []
        for state in states:
            s = spy.expand(op_M * spy.Matrix(state) * op_M.adjoint()).rewrite(spy.cos).expand()
            s = s.subs(sine,   imag_val)
            s = s.subs(cosine, real_val)
            operated_state.append(Partial_State(s[0], s[1], s[2], s[3]))
        return operated_state


class Composite_waveplate(spy.Matrix):
    """Equivalent Jones operator for an ordered sequence of waveplates."""

    @staticmethod
    def _subsetSums(nums):
        """Generate all unique |±n1 ± n2 ± ...| sums from nums, excluding zero."""
        s = [0]
        for num in nums:
            prev = list(s)
            for x in prev:
                s.append( x + num)
                s.append(-x + num)
        del s[0]
        return sorted(set(abs(x) for x in s) - {0})

    def __new__(cls, Waveplates):
        # SymPy internally calls cls(matrix) when unifying types during multiplication.
        if isinstance(Waveplates, spy.MatrixBase):
            return super(Composite_waveplate, cls).__new__(cls, Waveplates)
        cls.waveplates = Waveplates
        cls.operator   = spy.expand(spy.prod(cls.waveplates[::-1]))
        return super(Composite_waveplate, cls).__new__(cls, cls.operator)

    def __init__(self, Waveplates):
        if isinstance(Waveplates, spy.MatrixBase):
            return
        self.waveplates = Waveplates
        super().__init__()

    @property
    def stokes(self):
        """Mueller matrix in Stokes space."""
        if spy.shape(self) == (4, 4):
            return spy.simplify(_STOKES_T * self * _STOKES_T_INV)
        M = spy.kronecker_product(self, self.conjugate())
        return spy.simplify(_STOKES_T * M * _STOKES_T_INV)

    def rotate(self, angle):
        """Return a new Composite_waveplate with all waveplates rotated by angle."""
        return Composite_waveplate([wp.rotate(angle) for wp in self.waveplates])

    def list_of_phases(self):
        """Return (OPD_list, sine_list, cosine_list) for all interference terms."""
        opd_list   = [wp.OptPathDiff for wp in self.waveplates]
        list_of_OPD = Composite_waveplate._subsetSums(opd_list)
        list_of_sines   = [spy.sin(_X * OPD) for OPD in list_of_OPD]
        list_of_cosines = [spy.cos(_X * OPD) for OPD in list_of_OPD]
        return list_of_OPD, list_of_sines, list_of_cosines

    def operate(self, states, coherence):
        """
        Apply this composite waveplate to a list of polarization states.

        Parameters
        ----------
        states : list of State / Partial_State
        coherence : Coherence

        Returns
        -------
        list of Partial_State
        """
        op_M = spy.Matrix(self.operator)
        list_of_OPD, list_of_sines, list_of_cosines = self.list_of_phases()

        # Pre-compute all coherence values before the state loop
        coh_cache     = {opd:  coherence.eval(opd)  for opd in list_of_OPD}
        neg_coh_cache = {opd:  coherence.eval(-opd) for opd in list_of_OPD}

        operated_state = []
        for state in states:
            s = spy.expand(op_M * spy.Matrix(state) * op_M.adjoint()).rewrite(spy.cos).expand()
            for OPD, sine, cosine in zip(list_of_OPD, list_of_sines, list_of_cosines):
                s = s.subs(sine,   np.imag(coh_cache[OPD]))
                s = s.subs(cosine, np.real(neg_coh_cache[OPD]))
            operated_state.append(Partial_State(s[0], s[1], s[2], s[3]))
        return operated_state


class Polarimeter_data:
    """Reader for experimental polarimeter CSV files."""

    @staticmethod
    def read_data(path):
        """
        Return a sorted list of all .csv files in path.
        Sorted in natural (Windows-style numeric) order.
        """
        all_files = glob.glob(path + '*.csv')
        all_files.sort(key=lambda x: [int(c) if c.isdigit() else c
                                       for c in re.split(r'(\d+)', x)])
        return all_files

    def __init__(self, path):
        self.path      = path
        self.all_files = Polarimeter_data.read_data(path)

    def mean(self):
        return np.array([
            pd.read_csv(f, engine='python', index_col=False, header=23,
                        usecols=[1, 2, 3, 4, 5, 8], encoding='latin1').mean(axis=None)
            for f in self.all_files
        ])

    def std(self):
        return np.array([
            pd.read_csv(f, engine='python', index_col=False, header=23,
                        usecols=[1, 2, 3, 4, 5, 8], encoding='latin1').std(axis=None)
            for f in self.all_files
        ])


def Graphic(states: list):
    """
    Plot a list of polarization states on the Poincaré sphere.
    Each state is normalised by its degree of polarisation.
    """
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(xs, ys, zs, rstride=5, cstride=6, color='grey', alpha=0.3, linewidth=1.3)
    ax.plot_surface(xs, ys, zs,   rstride=1, cstride=1, color='grey', alpha=0.2, linewidth=0)

    for xs_ax, ys_ax, zs_ax in [
        ([-1.1, 1.1], [0, 0],      [0, 0]),
        ([0, 0],      [-1.1, 1.1], [0, 0]),
        ([0, 0],      [0, 0],      [-1.1, 1.1]),
    ]:
        ax.plot(xs_ax, ys_ax, zs_ax, color='black', linewidth=1.5, alpha=0.7)

    theta = np.linspace(0, 2 * np.pi, 100)
    c, s  = np.cos(theta), np.sin(theta)
    ax.plot(c, s, np.zeros_like(theta), color='gray', linewidth=1.5)
    ax.plot(s, np.zeros_like(theta), c, color='gray', linewidth=1.5)

    # Compute stokes once per state (not once per component)
    stokes_list = [state.stokes for state in states]
    dops        = [state.dop    for state in states]
    s1 = [sv[1] / d for sv, d in zip(stokes_list, dops)]
    s2 = [sv[2] / d for sv, d in zip(stokes_list, dops)]
    s3 = [sv[3] / d for sv, d in zip(stokes_list, dops)]

    ax.scatter(s1, s2, s3, color='red', alpha=1, s=30)

    fig.set_size_inches(9, 9)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_aspect("equal")
    ax.patch.set_alpha(0)
    plt.tight_layout()
    plt.axis('off')
    ax.text(1.15, 0,    0,    '$S_1$', fontsize=18)
    ax.text(0,    1.15, 0,    '$S_2$', fontsize=18)
    ax.text(0,    0,    1.15, '$S_3$', fontsize=18)
    plt.show()


def stokes_transformation():
    """Return the 4×4 Stokes–Jones transformation matrix (cached module constant)."""
    return _STOKES_T
