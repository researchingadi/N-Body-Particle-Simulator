"""Tests for the FastAPI layer (api/).

These test the HTTP API's own concerns -- request/response shapes, status
codes, and registry lifecycle -- not the underlying physics, which is
already covered by tests/test_forces.py, tests/test_conservation.py, etc.
A passing test here should mean "the API correctly wraps the engine", not
re-prove the engine itself is correct.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app, manager


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the in-memory simulation registry before and after each test,
    so tests don't leak simulations into one another via the shared
    module-level `manager` instance.
    """
    manager._simulations.clear()
    yield
    manager._simulations.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_ok_and_solver_list(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert set(data["available_solvers"]) == {"direct", "barnes_hut", "taichi_direct"}
    assert set(data["available_integrators"]) == {"euler", "leapfrog", "velocity_verlet", "rk4"}
    assert data["active_simulations"] == 0


def test_presets_returns_all_nine_presets(client):
    response = client.get("/presets")
    assert response.status_code == 200
    presets = response.json()
    names = {p["name"] for p in presets}
    assert names == {
        "binary_orbit",
        "figure_eight",
        "solar_system",
        "plummer_sphere",
        "star_cluster",
        "disk_galaxy",
        "ring_system",
        "galaxy_merger",
        "random_cloud",
    }
    for preset in presets:
        assert "description" in preset
        assert isinstance(preset["default_params"], dict)


def test_create_simulation_returns_id_and_echoes_config(client):
    response = client.post(
        "/simulations",
        json={"preset": "binary_orbit", "preset_params": {"separation": 2.0}, "dt": 0.005, "softening": 0.01},
    )
    assert response.status_code == 201
    data = response.json()
    assert "simulation_id" in data and len(data["simulation_id"]) > 0
    assert data["n_particles"] == 2
    assert data["config"]["preset"] == "binary_orbit"
    assert data["config"]["dt"] == 0.005


def test_create_simulation_increments_active_count(client):
    client.post("/simulations", json={"preset": "binary_orbit"})
    response = client.get("/health")
    assert response.json()["active_simulations"] == 1


def test_step_advances_time_and_returns_state(client):
    create = client.post(
        "/simulations", json={"preset": "binary_orbit", "preset_params": {"separation": 2.0}, "dt": 0.01}
    )
    sim_id = create.json()["simulation_id"]

    response = client.post(f"/simulations/{sim_id}/step", json={"n_steps": 50})
    assert response.status_code == 200
    data = response.json()
    assert data["step_count"] == 50
    assert data["time"] == pytest.approx(0.5, rel=1e-6)
    assert data["n_particles"] == 2
    assert len(data["particles"]["positions"]) == 2
    assert len(data["particles"]["positions"][0]) == 3


def test_step_with_default_body_advances_one_step(client):
    """POST with no body should default to a single step."""
    create = client.post("/simulations", json={"preset": "binary_orbit"})
    sim_id = create.json()["simulation_id"]

    response = client.post(f"/simulations/{sim_id}/step")
    assert response.status_code == 200
    assert response.json()["step_count"] == 1


def test_state_has_expected_shape(client):
    create = client.post("/simulations", json={"preset": "plummer_sphere", "preset_params": {"n": 20, "seed": 1}})
    sim_id = create.json()["simulation_id"]

    response = client.get(f"/simulations/{sim_id}/state")
    assert response.status_code == 200
    data = response.json()
    assert data["n_particles"] == 20
    particles = data["particles"]
    assert len(particles["positions"]) == 20
    assert len(particles["velocities"]) == 20
    assert len(particles["masses"]) == 20
    assert all(len(p) == 3 for p in particles["positions"])
    assert all(len(v) == 3 for v in particles["velocities"])


def test_diagnostics_endpoint_returns_sensible_values(client):
    create = client.post(
        "/simulations", json={"preset": "binary_orbit", "preset_params": {"separation": 2.0}, "softening": 0.01}
    )
    sim_id = create.json()["simulation_id"]

    response = client.get(f"/simulations/{sim_id}/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_energy"] == pytest.approx(data["kinetic_energy"] + data["potential_energy"])
    assert data["potential_energy"] < 0  # bound two-body system
    assert len(data["momentum"]) == 3
    assert len(data["angular_momentum"]) == 3
    assert len(data["center_of_mass"]) == 3
    # symmetric binary orbit at t=0: momentum and COM should be ~zero
    assert data["momentum"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)
    assert data["center_of_mass"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)


def test_diagnostics_reflect_stepped_state_not_initial_state(client):
    """Diagnostics must be computed live -- stepping should change the
    reported time even though total energy stays ~conserved."""
    create = client.post("/simulations", json={"preset": "binary_orbit", "dt": 0.01})
    sim_id = create.json()["simulation_id"]

    before = client.get(f"/simulations/{sim_id}/diagnostics").json()
    client.post(f"/simulations/{sim_id}/step", json={"n_steps": 10})
    after = client.get(f"/simulations/{sim_id}/diagnostics").json()

    assert after["time"] > before["time"]
    assert after["total_energy"] == pytest.approx(before["total_energy"], rel=1e-3)


def test_invalid_solver_name_is_rejected(client):
    response = client.post("/simulations", json={"preset": "binary_orbit", "solver": "not_a_real_solver"})
    assert response.status_code == 422


def test_invalid_integrator_name_is_rejected(client):
    response = client.post("/simulations", json={"preset": "binary_orbit", "integrator": "not_a_real_integrator"})
    assert response.status_code == 422


def test_invalid_preset_name_is_rejected(client):
    response = client.post("/simulations", json={"preset": "not_a_real_preset"})
    assert response.status_code == 422


def test_delete_simulation_removes_it(client):
    create = client.post("/simulations", json={"preset": "binary_orbit"})
    sim_id = create.json()["simulation_id"]

    response = client.delete(f"/simulations/{sim_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    response = client.get(f"/simulations/{sim_id}/state")
    assert response.status_code == 404


def test_operations_on_nonexistent_simulation_return_404(client):
    fake_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/simulations/{fake_id}/state").status_code == 404
    assert client.get(f"/simulations/{fake_id}/diagnostics").status_code == 404
    assert client.post(f"/simulations/{fake_id}/step").status_code == 404
    assert client.delete(f"/simulations/{fake_id}").status_code == 404


def test_taichi_direct_solver_selectable_via_api(client):
    """Solver selection should actually reach the engine, not just validate."""
    response = client.post(
        "/simulations", json={"preset": "binary_orbit", "solver": "taichi_direct", "dt": 0.01}
    )
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]

    step_response = client.post(f"/simulations/{sim_id}/step", json={"n_steps": 20})
    assert step_response.status_code == 200
    assert step_response.json()["step_count"] == 20


def test_barnes_hut_solver_selectable_via_api(client):
    response = client.post(
        "/simulations",
        json={"preset": "plummer_sphere", "preset_params": {"n": 30, "seed": 2}, "solver": "barnes_hut", "theta": 0.6},
    )
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]

    step_response = client.post(f"/simulations/{sim_id}/step", json={"n_steps": 5})
    assert step_response.status_code == 200


def test_preset_params_override_defaults(client):
    response = client.post(
        "/simulations", json={"preset": "plummer_sphere", "preset_params": {"n": 77, "seed": 3}}
    )
    assert response.status_code == 201
    assert response.json()["n_particles"] == 77


def test_invalid_preset_params_returns_400(client):
    response = client.post(
        "/simulations", json={"preset": "binary_orbit", "preset_params": {"not_a_real_kwarg": 1.0}}
    )
    assert response.status_code == 400


def test_cors_allows_local_vite_dev_server(client):
    """The frontend (Stage 4B) runs on http://localhost:5173 by default --
    confirm the API actually sends back CORS headers permitting it, not
    just that CORSMiddleware is imported.
    """
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight_request_succeeds(client):
    """Browsers send an OPTIONS preflight before a cross-origin POST with a
    JSON body -- confirm that preflight is actually handled, since a
    missing/misconfigured CORS setup often passes simple GETs but fails
    exactly this check, which is what actually blocks the frontend.
    """
    response = client.options(
        "/simulations",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
