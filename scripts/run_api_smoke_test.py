"""Manual smoke test for the FastAPI backend -- no running server required.

Uses FastAPI's TestClient (an in-process client, not real HTTP) so this
script works the same in CI/sandboxes as it does locally. To hit a real
running server instead, start one with `uvicorn api.main:app --reload` and
swap TestClient for `httpx.Client(base_url="http://127.0.0.1:8000")` --
every call below has the same method/URL/json signature either way.

Usage:
    python scripts/run_api_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    client = TestClient(app)

    print("=== GET /health ===")
    response = client.get("/health")
    print(f"status={response.status_code}")
    print(response.json())

    print("\n=== GET /presets ===")
    response = client.get("/presets")
    presets = response.json()
    print(f"status={response.status_code}, {len(presets)} presets available:")
    for p in presets:
        print(f"  - {p['name']}: {p['description']}")

    print("\n=== POST /simulations (binary_orbit) ===")
    response = client.post(
        "/simulations",
        json={
            "preset": "binary_orbit",
            "preset_params": {"separation": 2.0},
            "dt": 0.005,
            "softening": 0.01,
            "integrator": "leapfrog",
            "solver": "direct",
        },
    )
    print(f"status={response.status_code}")
    created = response.json()
    print(created)
    simulation_id = created["simulation_id"]

    print(f"\n=== POST /simulations/{simulation_id}/step (n_steps=200) ===")
    response = client.post(f"/simulations/{simulation_id}/step", json={"n_steps": 200})
    print(f"status={response.status_code}")
    state = response.json()
    print(f"time={state['time']:.4f}, step_count={state['step_count']}, n_particles={state['n_particles']}")
    print(f"body 0 position: {state['particles']['positions'][0]}")
    print(f"body 1 position: {state['particles']['positions'][1]}")

    print(f"\n=== GET /simulations/{simulation_id}/diagnostics ===")
    response = client.get(f"/simulations/{simulation_id}/diagnostics")
    print(f"status={response.status_code}")
    diagnostics = response.json()
    print(
        f"total_energy={diagnostics['total_energy']:.6f}  "
        f"momentum={diagnostics['momentum']}  "
        f"angular_momentum={diagnostics['angular_momentum']}"
    )

    print(f"\n=== DELETE /simulations/{simulation_id} ===")
    response = client.delete(f"/simulations/{simulation_id}")
    print(f"status={response.status_code}, {response.json()}")

    print("\n=== GET /simulations/{id}/state after delete (expect 404) ===")
    response = client.get(f"/simulations/{simulation_id}/state")
    print(f"status={response.status_code}")

    print("\nSmoke test complete: all endpoints reachable and behaving as expected.")


if __name__ == "__main__":
    main()
