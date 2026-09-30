import numpy as np

from motor_muscle.motors import DenseMotorArray


def test_fixed_seed_is_reproducible():
    dofs = np.arange(8)
    a = DenseMotorArray(2_000, dofs, 5)
    b = DenseMotorArray(2_000, dofs, 5)
    assert np.array_equal(a.group, b.group)
    assert np.allclose(a.kt, b.kt)


def test_failure_saturation_and_finite_torque():
    motors = DenseMotorArray(2_000, np.arange(8), 4)
    motors.reset(9, 0.1)
    command = np.ones(16, dtype=np.float32)
    torque, diagnostics = motors.step(command, np.zeros(16), np.zeros(8), 0.001)
    assert np.all(np.isfinite(torque))
    assert np.all(motors.current[motors.failed] == 0)
    assert 0.89 <= diagnostics.active_ratio <= 0.91
    assert np.max(motors.current) <= 4.0


def test_hot_motors_are_derated():
    motors = DenseMotorArray(1_000, np.arange(4), 3)
    motors.temperature[:100] = 100
    motors.step(np.ones(8), np.zeros(8), np.zeros(4), 0.001)
    assert np.all(motors.current[:100] == 0)


def test_concentrated_failure_targets_only_left_joints_and_preserves_ratio():
    names = ["spine", "left_hip", "right_hip", "left_knee", "right_knee"]
    motors = DenseMotorArray(1_000, np.arange(len(names)), 3, dof_names=names)
    motors.reset(9, 0.2, concentrated=True)
    assert np.count_nonzero(motors.failed) == 200
    assert set(motors.group_dof[motors.failed]) == {1, 3}
    assert not np.any(motors.failed[np.isin(motors.group_dof, [0, 2, 4])])


def test_concentrated_failure_requires_anatomical_names():
    motors = DenseMotorArray(1_000, np.arange(4), 3)
    try:
        motors.reset(9, 0.1, concentrated=True)
    except ValueError as error:
        assert "left-side DOF" in str(error)
    else:
        raise AssertionError("concentrated failure accepted unnamed joints")
