import pytest
import torch


# Unit tests - test the scoring logic directly without loading model
class TestCategoryScoring:
    """Test category scoring logic without loading the real model"""

    @pytest.fixture
    def label2id(self):
        return {
            "child_abuse": 0,
            "law": 1,
            "rape": 2,
            "suicide": 3,
            "zoo": 4,
            "med": 5,
            "discrimination": 6,
            "financial": 7,
            "cannibalism": 8,
            "necro": 9,
            "politics": 10,
            "religious": 11,
            "incest": 12,
        }

    def compute_scores(self, logits, label2id):
        """Replicate the scoring logic from main.py"""
        s1 = max(logits[label2id.get("cannibalism", 0)].item(), logits[label2id.get("necro", 0)].item())
        s2 = logits[label2id.get("discrimination", 0)].item()
        s3 = max(logits[label2id.get("rape", 0)].item(), logits[label2id.get("zoo", 0)].item())
        s4 = max(logits[label2id.get("child_abuse", 0)].item(), logits[label2id.get("incest", 0)].item())
        s11 = logits[label2id.get("suicide", 0)].item()
        return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S11": s11}

    def test_safe_text_all_negative(self, label2id):
        """All negative logits = safe"""
        logits = torch.tensor([-1.0] * 13)
        scores = self.compute_scores(logits, label2id)
        assert all(s < 0 for s in scores.values())

    def test_unsafe_s4_child_abuse(self, label2id):
        """Positive child_abuse logit = S4 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["child_abuse"]] = 2.0
        scores = self.compute_scores(logits, label2id)
        assert scores["S4"] > 0
        assert scores["S1"] < 0
        assert scores["S2"] < 0
        assert scores["S3"] < 0
        assert scores["S11"] < 0

    def test_unsafe_s4_incest(self, label2id):
        """Positive incest logit = S4 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["incest"]] = 1.5
        scores = self.compute_scores(logits, label2id)
        assert scores["S4"] > 0

    def test_unsafe_s3_rape(self, label2id):
        """Positive rape logit = S3 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["rape"]] = 1.5
        scores = self.compute_scores(logits, label2id)
        assert scores["S3"] > 0

    def test_unsafe_s3_zoo(self, label2id):
        """Positive zoo logit = S3 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["zoo"]] = 1.5
        scores = self.compute_scores(logits, label2id)
        assert scores["S3"] > 0

    def test_unsafe_s11_suicide(self, label2id):
        """Positive suicide logit = S11 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["suicide"]] = 0.5
        scores = self.compute_scores(logits, label2id)
        assert scores["S11"] > 0

    def test_unsafe_s1_cannibalism(self, label2id):
        """Positive cannibalism logit = S1 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["cannibalism"]] = 1.0
        scores = self.compute_scores(logits, label2id)
        assert scores["S1"] > 0

    def test_unsafe_s1_necro(self, label2id):
        """Positive necro logit = S1 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["necro"]] = 1.0
        scores = self.compute_scores(logits, label2id)
        assert scores["S1"] > 0

    def test_unsafe_s2_discrimination(self, label2id):
        """Positive discrimination logit = S2 blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["discrimination"]] = 0.1
        scores = self.compute_scores(logits, label2id)
        assert scores["S2"] > 0

    def test_multiple_categories_blocked(self, label2id):
        """Multiple positive logits = multiple categories blocked"""
        logits = torch.tensor([-1.0] * 13)
        logits[label2id["child_abuse"]] = 2.0
        logits[label2id["rape"]] = 1.5
        logits[label2id["suicide"]] = 0.5
        scores = self.compute_scores(logits, label2id)
        blocked = [cat for cat, score in scores.items() if score > 0]
        assert "S4" in blocked
        assert "S3" in blocked
        assert "S11" in blocked
        assert len(blocked) == 3


# E2E tests - require model to be loaded (skip if model not available)
@pytest.mark.skipif(True, reason="E2E tests require model - run in Docker")
class TestModerationE2E:
    @pytest.fixture
    def client(self):
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data

    def test_moderate_safe_text(self, client):
        response = client.post("/moderate", json={"text": "Hello, how are you today?"})
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "categories" in data
        assert isinstance(data["categories"], list)

    def test_moderate_empty_text(self, client):
        response = client.post("/moderate", json={"text": ""})
        assert response.status_code == 200

    def test_moderate_missing_text(self, client):
        response = client.post("/moderate", json={})
        assert response.status_code == 422  # Validation error
