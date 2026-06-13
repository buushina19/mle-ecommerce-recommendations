import requests

BASE = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    assert r.json()["status"] == "healthy"


def test_recommendations():
    r = requests.get(f"{BASE}/recommendations/123456", timeout=30)
    r.raise_for_status()
    data = r.json()
    assert data["visitorid"] == 123456
    assert len(data["recommendations"]) == 10
    assert data["strategy"] in {"als_rerank", "popularity_cold_start"}
    print("strategy:", data["strategy"])
    print("items:", data["recommendations"][:5])


def test_metrics():
    r = requests.get(f"{BASE}/metrics", timeout=10)
    r.raise_for_status()
    assert "recommendation_requests_total" in r.text


if __name__ == "__main__":
    test_health()
    test_recommendations()
    test_metrics()
    print("All tests passed")
