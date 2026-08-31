import pytest

from tesseract_hybrid_closure.closure import parameter_count
from tesseract_hybrid_closure.engine import (
    assert_smoke_passes,
    run_gradient_smoke,
)


@pytest.mark.gradient_smoke
def test_jax_grad_reaches_all_pytorch_closure_weights():
    result = run_gradient_smoke()

    assert_smoke_passes(result)
    assert result.gradient_size == parameter_count()
    assert result.state_shape == (1, 64, 64)
    assert result.state_dtype == "float32"
    assert result.loss > 0.0
    assert result.gradient_norm > 0.0
