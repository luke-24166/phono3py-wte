"""Velocity solver for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets

from phono3py.conductivity.grid_point_data import VelocityResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS, get_kappa_star_operations
try:
    from phonopy.phonon.grid import (
        get_grid_points_by_rotations,
        get_qpoints_from_bz_grid_points,
    )
except ImportError:
    from phono3py.phonon.grid import (
        get_grid_points_by_rotations,
        get_qpoints_from_bz_grid_points,
    )
from phono3py.phonon3.interaction import Interaction

from wte.velocity_operator import VelocityOperator


class VelocityOperatorSolver:
    """Compute velocity operators and their symmetrised outer products.

    This solver implements the ``VelocitySolver`` protocol for the Wigner
    transport equation.  It wraps phonopy's ``VelocityOperator`` and computes
    the k-star-averaged outer product of the velocity operator matrix.

    Notes
    -----
    ``num_band0 == num_band`` is assumed (Wigner requires all phonon branches).
    The velocity operator is Hermitian, so ``V(s,s') = V*(s',s)``.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix()`` must have been called.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for velocity operator. Default None.
    log_level : int, optional
        Verbosity level. Default 0.

    """

    produces_gv_by_gv: bool = False
    produces_vm_by_vm: bool = True

    def __init__(
        self,
        pp: Interaction,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._is_kappa_star = is_kappa_star
        self._log_level = log_level
        self._point_operations, self._rotations_cartesian = get_kappa_star_operations(
            pp.bz_grid, is_kappa_star
        )
        self._velocity_obj = VelocityOperator(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=pp.primitive_symmetry,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    def compute(self, grid_points: int | NDArray[np.int64]) -> VelocityResult:
        """Compute velocities and Voigt-packed operator products in input order.

        A sequence, including a singleton, adds a leading grid-point axis.
        A scalar returns the tensors for one grid point.
        """
        indices = np.asarray(grid_points)
        if indices.ndim == 0:
            gv, vm_by_vm = self._compute_at_grid_point(indices.item())
        elif indices.ndim == 1:
            num_band0 = len(self._pp.band_indices)
            num_band = 3 * len(self._pp.primitive)
            gv = np.empty((len(indices), num_band0, 3), dtype="double")
            vm_by_vm = np.empty((len(indices), num_band0, num_band, 6), dtype="cdouble")
            for i, grid_point in enumerate(indices):
                gv[i], vm_by_vm[i] = self._compute_at_grid_point(grid_point)
        else:
            raise ValueError("Expected one grid point or a one-dimensional grid.")
        return VelocityResult(group_velocities=gv, vm_by_vm=vm_by_vm)

    def _compute_at_grid_point(
        self, grid_point: int
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble]]:
        q_point = get_qpoints_from_bz_grid_points(grid_point, self._pp.bz_grid)
        self._velocity_obj.run([q_point])
        assert self._velocity_obj.velocity_operators is not None
        gv_op_full = self._velocity_obj.velocity_operators[0]
        gv_op = gv_op_full[self._pp.band_indices]
        return (
            self._get_group_velocities(grid_point, gv_op_full),
            self._get_gv_by_gv_operator(grid_point, gv_op),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_group_velocities(
        self,
        grid_point: int,
        gv_op_full: NDArray[np.cdouble],
    ) -> NDArray[np.double]:
        """Return diagonal group velocities with degenerate-subspace correction.

        For degenerate phonon branches the velocity operator restricted to the
        degenerate subspace is diagonalised so that the group velocities are
        well-defined.  This follows the same procedure as
        ``ConductivityWignerComponents._set_gv_operator``.

        """
        nat3 = gv_op_full.shape[0]
        # Diagonal of operator gives standard group velocities
        gv = np.einsum("iij->ij", gv_op_full).real  # (nat3, 3)
        frequencies = self._pp.get_phonons()[0][grid_point]
        deg_sets = degenerate_sets(frequencies)
        pos = 0
        for deg in deg_sets:
            if len(deg) > 1:
                for id_dir in range(3):
                    sl = slice(pos, pos + len(deg))
                    block = gv_op_full[sl, sl, id_dir]
                    gv[sl, id_dir] = np.linalg.eigvalsh(block)
            pos += len(deg)
        assert pos == nat3
        return gv[self._pp.band_indices, :]  # (num_band0, 3)

    def _get_gv_by_gv_operator(
        self,
        grid_point: int,
        gv_op: NDArray[np.cdouble],
    ) -> NDArray[np.cdouble]:
        """Return k-star-averaged velocity-operator outer product.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.
        gv_op : (num_band0, nat3, 3) complex
            Velocity operator filtered to selected bands.

        Returns
        -------
        gv_by_gv_op : (num_band0, nat3, 6) complex
            Symmetry-averaged outer product packed in Voigt order.

        """
        if self._is_kappa_star:
            rotation_map = get_grid_points_by_rotations(grid_point, self._pp.bz_grid)
        else:
            rotation_map = get_grid_points_by_rotations(
                grid_point,
                self._pp.bz_grid,
                reciprocal_rotations=self._point_operations,
            )

        num_band0, nat3, _ = gv_op.shape
        gv_by_gv_op = np.zeros((num_band0, nat3, 6), dtype="complex128")

        for r in self._rotations_cartesian:
            # gv_rot[s, s', i] = sum_j gv_op[s, s', j] * r.T[j, i]
            gv_rot = np.dot(gv_op, r.T)  # (num_band0, nat3, 3)
            for j, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
                gv_by_gv_op[:, :, j] += gv_rot[:, :, a] * np.conj(gv_rot[:, :, b])

        # Divide by site multiplicity (number of rotations per k-star arm)
        order_kstar = len(np.unique(rotation_map))
        site_multi = len(rotation_map) // order_kstar
        gv_by_gv_op /= site_multi

        return gv_by_gv_op
