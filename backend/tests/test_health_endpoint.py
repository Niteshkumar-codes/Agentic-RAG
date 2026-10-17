import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_status_code(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_json_structure_and_values(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert set(data.keys()) == {"status", "message"}
    assert data["status"] == "healthy"
    assert data["message"] == "Agentic-RAG backend is running"
