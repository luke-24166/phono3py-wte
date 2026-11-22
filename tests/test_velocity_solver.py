"""Grid-point shapes and ordering for the WTE velocity solver."""

import numpy as np

from wte.velocity_solvers import VelocityOperatorSolver


def test_velocity_batch_order_and_shape(si_pbesol):
    si_pbesol.mesh_numbers = [3, 3, 3]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_phonon_solver()
    solver = VelocityOperatorSolver(si_pbesol.phph_interaction)

    reference = {gp: solver.compute(gp) for gp in (1, 4)}
    num_bands = 3 * len(si_pbesol.primitive)
    for points in ([], [4], [4, 1, 4]):
        result = solver.compute(np.array(points, dtype="int64"))
        assert result.group_velocities.shape == (len(points), num_bands, 3)
        assert result.vm_by_vm.shape == (len(points), num_bands, num_bands, 6)
        for i, gp in enumerate(points):
            np.testing.assert_allclose(result.group_velocities[i], reference[gp].group_velocities)
            np.testing.assert_allclose(result.vm_by_vm[i], reference[gp].vm_by_vm)
