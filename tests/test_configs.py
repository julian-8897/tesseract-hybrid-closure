import pytest

from tesseract_hybrid_closure.configs import SolverConfig
from tesseract_hybrid_closure.constants import COARSE_NUM_POINTS, ETDRK_ORDER


def test_solver_config_pins_locked_numerics():
    config = SolverConfig()

    assert config.num_points == COARSE_NUM_POINTS == 64
    assert config.order == ETDRK_ORDER == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_points", 32), ("order", 4), ("dt", 0.0), ("diffusivity", -1.0)],
)
def test_solver_config_rejects_invalid_or_unlocked_values(field, value):
    with pytest.raises(ValueError):
        SolverConfig(**{field: value})
