import numpy as np

from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.component_loader import load_tesseract_api


def test_local_tesseract_apis_expose_expected_shapes():
    closure_api = load_tesseract_api("scalar_closure")
    solver_api = load_tesseract_api("coarse_solver")
    omega = np.zeros((1, 64, 64), dtype=np.float32)

    closure_inputs = closure_api.InputSchema(
        omega=omega,
        params_flat=initial_parameters(),
    )
    closure_output = closure_api.apply(closure_inputs)
    solver_inputs = solver_api.InputSchema(omega=omega)
    solver_output = solver_api.apply(solver_inputs)

    assert closure_output["tendency"].shape == omega.shape
    assert closure_output["tendency"].dtype == np.float32
    assert solver_output["omega_next"].shape == omega.shape
    assert solver_output["omega_next"].dtype.name == "float32"
