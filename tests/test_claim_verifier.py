from src.backend.claim_verifier import claim_and_citation_verifier


def context(text, rank=1, article_id="a", chunk_id="c"):
    return {
        "text": text,
        "citation_rank": rank,
        "article_id": article_id,
        "chunk_id": chunk_id,
        "title": "Tin",
    }


def test_valid_citation_maps_by_rank_not_list_position_and_is_supported():
    result = claim_and_citation_verifier(
        "Doanh thu năm 2025 đạt 100 tỷ đồng. [Nguồn 7]",
        [context("Doanh thu năm 2025 đạt 100 tỷ đồng.", rank=7)],
    )
    assert result["verification_status"] == "PASS"
    assert result["claims"][0]["status"] == "supported"
    assert result["citations"][0]["citation_rank"] == 7


def test_missing_and_invalid_citations_fail():
    missing = claim_and_citation_verifier("Doanh thu đạt 100 tỷ đồng.", [context("Doanh thu đạt 100 tỷ đồng.")])
    invalid = claim_and_citation_verifier("Doanh thu đạt 100 tỷ đồng. [Nguồn 9]", [context("Doanh thu đạt 100 tỷ đồng.")])
    assert missing["verification_status"] == "FAIL"
    assert any(item["type"] == "missing_citation" for item in missing["verification_errors"])
    assert invalid["verification_status"] == "FAIL"
    assert any(item["type"] == "invalid_citation" for item in invalid["verification_errors"])


def test_orphan_citation_fails():
    result = claim_and_citation_verifier(
        "Doanh thu đạt 100 tỷ đồng.\n[Nguồn 1]",
        [context("Doanh thu đạt 100 tỷ đồng.")],
    )
    assert result["verification_status"] == "FAIL"
    assert any(item["type"] == "orphan_citation" for item in result["verification_errors"])


def test_unknown_claim_is_warning_not_evidence_refusal():
    result = claim_and_citation_verifier(
        "Lợi nhuận tăng gấp mười lần. [Nguồn 1]",
        [context("Doanh thu đạt 100 tỷ đồng.")],
    )
    assert result["verification_status"] == "WARNING"
    assert result["claims"][0]["status"] == "unknown"
    assert result["verification_errors"] == []


def test_contradicted_and_conflicting_claims_fail():
    contradicted = claim_and_citation_verifier(
        "Purin gây bệnh. [Nguồn 1]",
        [context("Purin không gây bệnh.")],
    )
    conflicting = claim_and_citation_verifier(
        "Purin gây bệnh. [Nguồn 1]",
        [context("Purin gây bệnh. Purin không gây bệnh.")],
    )
    assert contradicted["claims"][0]["status"] == "contradicted"
    assert contradicted["verification_status"] == "FAIL"
    assert conflicting["claims"][0]["status"] == "conflicting"
    assert conflicting["verification_status"] == "FAIL"


def test_duplicate_citation_rank_is_invalid_mapping():
    result = claim_and_citation_verifier(
        "Thông tin đúng. [Nguồn 1]",
        [context("Thông tin đúng.", 1, "a"), context("Thông tin đúng.", 1, "b")],
    )
    assert result["verification_status"] == "FAIL"
    assert any(item["type"] == "duplicate_citation_rank" for item in result["verification_errors"])


def test_numeric_paraphrase_is_supported():
    result = claim_and_citation_verifier(
        "Công ty ghi nhận doanh thu 10.000 tỷ đồng. [Nguồn 1]",
        [context("Doanh thu đạt 10 nghìn tỷ đồng.")],
    )
    assert result["verification_status"] == "PASS"
    assert result["claims"][0]["status"] == "supported"


def test_number_mismatch_is_blocked():
    result = claim_and_citation_verifier(
        "Doanh thu đạt 900 tỷ đồng. [Nguồn 1]",
        [context("Doanh thu đạt 100 tỷ đồng.")],
    )
    assert result["verification_status"] == "FAIL"
    assert result["claims"][0]["status"] == "contradicted"
    assert any(item["type"] == "number_mismatch" for item in result["verification_errors"])


def test_entity_mismatch_is_blocked():
    result = claim_and_citation_verifier(
        "Công ty A báo doanh thu tăng. [Nguồn 1]",
        [context("Công ty B báo doanh thu tăng.")],
    )
    assert result["verification_status"] == "FAIL"
    assert result["claims"][0]["status"] == "contradicted"
    assert result["claims"][0]["matches"][0]["entity_mismatch"] is True


def test_direction_antonyms_are_contradicted():
    result = claim_and_citation_verifier(
        "Doanh thu giảm. [Nguồn 1]",
        [context("Doanh thu tăng.")],
    )
    assert result["verification_status"] == "FAIL"
    assert result["claims"][0]["status"] == "contradicted"
