def test_analyze_no_file():
    r = client.post("/analyze")
    assert r.status_code == 422
