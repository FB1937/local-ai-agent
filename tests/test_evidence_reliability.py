import unittest

from evidence_models import EvidenceSource
from page_extractor import extract_official_pages
from answer_validator import validate_answer, append_validation_notice
from llm import _build_prompts


class EvidenceContractTests(unittest.TestCase):
    def test_evidence_source_accepts_normalized_http_source(self):
        source = EvidenceSource.model_validate({
            "source_id": "S1",
            "url": "https://www.python.org/downloads/latest",
            "title": "Python release",
            "source_type": "official",
        })
        self.assertEqual(source.source_id, "S1")
        self.assertEqual(source.source_type, "official")
        self.assertIsNotNone(source.retrieved_at)

    def test_evidence_source_rejects_invalid_url_and_source_id(self):
        with self.assertRaises(ValueError):
            EvidenceSource.model_validate({
                "source_id": "source-1",
                "url": "file:///etc/passwd",
            })


class OfficialPageExtractionTests(unittest.TestCase):
    def test_extracts_only_first_party_pages_and_preserves_original_url(self):
        calls = []

        def fake_fetcher(url, timeout):
            calls.append(url)
            return "<html>fixture</html>", url

        def fake_extractor(html, url):
            return "Python Release 3.15.2", "Python 3.15.2 is the latest stable release."

        result = extract_official_pages(
            [
                {
                    "source_id": "S1",
                    "url": "https://www.python.org/downloads/latest",
                    "source_type": "official",
                },
                {
                    "source_id": "S2",
                    "url": "https://example.com/article",
                    "source_type": "other",
                },
            ],
            max_pages=1,
            fetcher=fake_fetcher,
            extractor=fake_extractor,
        )

        self.assertEqual(calls, ["https://www.python.org/downloads/latest"])
        self.assertEqual(result["extracted_count"], 1)
        self.assertEqual(result["results"][0]["url"], "https://www.python.org/downloads/latest")
        self.assertEqual(result["results"][0]["extracted_title"], "Python Release 3.15.2")
        self.assertEqual(result["results"][0]["extraction_status"], "extracted")
        self.assertEqual(result["results"][1]["extraction_status"], "skipped_non_official")

    def test_rejects_redirect_from_official_to_non_official_domain(self):
        result = extract_official_pages(
            [{
                "source_id": "S1",
                "url": "https://www.python.org/downloads/latest",
                "source_type": "official",
            }],
            fetcher=lambda url, timeout: ("html", "https://example.com/redirected"),
            extractor=lambda html, url: ("title", "body"),
        )
        self.assertEqual(result["extracted_count"], 0)
        self.assertEqual(result["results"][0]["extraction_status"], "failed")
        self.assertIn("공식 출처가 아닌 주소", result["results"][0]["extraction_error"])


class ClaimEvidenceValidationTests(unittest.TestCase):
    def test_latest_version_mismatch_is_rejected(self):
        sources = [{
            "source_id": "S1",
            "url": "https://www.python.org/downloads/latest",
            "title": "Python Release Python 3.15.2",
            "content": "Python 3.15.2 is the latest stable release.",
            "source_type": "official",
            "official_release_index": True,
            "latest_stable_claims": ["3.15.2"],
        }]
        assessment = {
            "sufficient": True,
            "version_query": True,
            "latest_stable_claims": ["3.15.2"],
            "official_index_versions": ["3.15.2"],
        }
        result = validate_answer(
            "Python 최신 안정 버전은 3.14.10입니다. [S1]",
            sources,
            search_assessment=assessment,
            route="SEARCH",
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            check["claim_type"] == "latest_version"
            and check["status"] == "unsupported"
            for check in result["claim_checks"]
        ))

    def test_unsupported_gartner_percentage_is_rejected(self):
        sources = [{
            "source_id": "S1",
            "url": "https://example.org/rag",
            "title": "RAG overview",
            "content": "RAG combines retrieval with generation.",
            "source_type": "other",
        }]
        result = validate_answer(
            "Gartner는 2026년 기업 AI 도입의 60%가 RAG를 활용할 것으로 전망했습니다. [S1]",
            sources,
            search_assessment={"sufficient": True},
            route="HYBRID",
        )
        self.assertFalse(result["valid"])
        types = {check["claim_type"] for check in result["claim_checks"]}
        self.assertIn("statistic", types)
        self.assertIn("named_source", types)
        rendered = append_validation_notice("답변", result)
        self.assertIn("[주장 검증 경고]", rendered)

    def test_exact_percentage_with_matching_citation_passes_rule_check(self):
        sources = [{
            "source_id": "S1",
            "url": "https://example.org/report",
            "title": "Survey",
            "content": "The survey reports that 60% of respondents use retrieval.",
            "source_type": "other",
        }]
        result = validate_answer(
            "조사 응답자의 60%가 검색 기술을 사용한다고 답했습니다. [S1]",
            sources,
            search_assessment={"sufficient": True},
            route="SEARCH",
        )
        statistic_checks = [
            check for check in result["claim_checks"]
            if check["claim_type"] == "statistic"
        ]
        self.assertEqual(len(statistic_checks), 1)
        self.assertEqual(statistic_checks[0]["status"], "supported")
        self.assertTrue(result["valid"])

    def test_customer_deployment_claim_without_case_evidence_is_rejected(self):
        sources = [{
            "source_id": "S1",
            "url": "https://vendor.example.com/capabilities",
            "title": "RAG capabilities",
            "content": "RAG can support contract review.",
            "source_type": "company_official",
            "company_case_signal": False,
        }]
        result = validate_answer(
            "Acme사는 RAG를 도입해 계약 검토 업무에 활용하고 있습니다. [S1]",
            sources,
            search_assessment={
                "sufficient": True,
                "requires_company_official": True,
                "company_official_count": 1,
                "company_case_signal_count": 0,
            },
            route="HYBRID",
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            check["claim_type"] == "customer_deployment"
            and check["status"] == "unsupported"
            for check in result["claim_checks"]
        ))

    def test_unconfirmed_deployment_statement_is_not_flagged_as_positive_claim(self):
        result = validate_answer(
            "실제 고객 도입 사례는 이번 검색에서 확인되지 않았습니다.",
            [],
            search_assessment={
                "sufficient": False,
                "requires_company_official": True,
                "company_official_count": 0,
                "company_case_signal_count": 0,
            },
            route="HYBRID",
        )
        self.assertFalse(any(
            check["claim_type"] == "customer_deployment"
            for check in result["claim_checks"]
        ))

    def test_hybrid_prompt_separates_three_evidence_zones_in_one_generation(self):
        system_prompt, user_prompt = _build_prompts(
            "RAG 원리와 최신 기업 사례",
            context="[S1] official source",
            route="HYBRID",
            search_assessment={"sufficient": False},
        )
        self.assertIn("한 번의 답변 생성 안에서", system_prompt)
        self.assertIn("개념과 배경", system_prompt)
        self.assertIn("검색으로 확인된 사실", system_prompt)
        self.assertIn("확인 한계", system_prompt)
        self.assertIn("한 번의 답변", user_prompt)

    def test_extracted_release_text_updates_latest_version_assessment(self):
        from search_validator import prepare_results, assess_results

        rows = prepare_results(
            "Python latest stable release",
            [{
                "title": "Python downloads",
                "url": "https://www.python.org/downloads/latest",
                "content": "Official Python downloads.",
                "extracted_title": "Python Release Python 3.15.2",
                "extracted_text": "Python 3.15.2 is the latest stable release.",
            }],
        )
        assessment = assess_results(
            rows, freshness_required=True, requires_official=True, version_query=True
        )
        self.assertEqual(assessment["latest_stable_claims"], ["3.15.2"])
        self.assertTrue(assessment["version_query"])

    def test_response_generator_excludes_sources_that_break_the_contract(self):
        from response_generator import generate_response

        received = {}

        def fake_generation(question, results, route, search_assessment):
            received["results"] = results
            return {"answer": "검색 결과를 확인했습니다.", "ok": True, "incomplete": False}

        result = generate_response(
            "조사해 줘",
            {
                "results": [
                    {"source_id": "S1", "url": "https://example.org/valid", "title": "Valid"},
                    {"source_id": "bad-id", "url": "file:///etc/passwd", "title": "Invalid"},
                ],
                "assessment": {"sufficient": True},
            },
            "SEARCH",
            generation_fn=fake_generation,
        )
        self.assertEqual([row["source_id"] for row in received["results"]], ["S1"])
        self.assertEqual(len(result["evidence_contract_errors"]), 1)
        self.assertFalse(result["validation"]["valid"])

    def test_named_source_must_appear_in_the_cited_source_not_just_any_result(self):
        sources = [
            {
                "source_id": "S1",
                "url": "https://gartner.example/report",
                "title": "Gartner report",
                "content": "Gartner discusses retrieval-augmented generation.",
                "source_type": "other",
            },
            {
                "source_id": "S2",
                "url": "https://medium.example/post",
                "title": "Personal article",
                "content": "A personal discussion of RAG.",
                "source_type": "other",
            },
        ]
        result = validate_answer(
            "Gartner는 RAG 도입을 전망했습니다. [S2]",
            sources,
            search_assessment={"sufficient": True},
            route="SEARCH",
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            check["claim_type"] == "named_source"
            and check["status"] == "unsupported"
            for check in result["claim_checks"]
        ))

    def test_latest_version_must_cite_the_official_release_index(self):
        sources = [
            {
                "source_id": "S1",
                "url": "https://www.python.org/downloads/latest",
                "title": "Python Release Python 3.15.2",
                "content": "Python 3.15.2 is the latest stable release.",
                "source_type": "official",
                "official_release_index": True,
                "latest_stable_claims": ["3.15.2"],
            },
            {
                "source_id": "S2",
                "url": "https://medium.example/python",
                "title": "Python version commentary",
                "content": "Python version discussion.",
                "source_type": "other",
            },
        ]
        result = validate_answer(
            "Python 최신 안정 버전은 3.15.2입니다. [S2]",
            sources,
            search_assessment={
                "sufficient": True,
                "version_query": True,
                "latest_stable_claims": ["3.15.2"],
            },
            route="SEARCH",
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            check["claim_type"] == "latest_version"
            and check["status"] == "unsupported"
            for check in result["claim_checks"]
        ))

    def test_trailing_citation_after_period_stays_attached_to_claim(self):
        sources = [{
            "source_id": "S1",
            "url": "https://example.org/report",
            "title": "Survey",
            "content": "The survey reports that 60% of respondents use retrieval.",
            "source_type": "other",
        }]
        result = validate_answer(
            "조사 응답자의 60%가 검색 기술을 사용한다고 답했습니다. [S1]",
            sources,
            search_assessment={"sufficient": True},
            route="SEARCH",
        )
        self.assertTrue(result["valid"])
        self.assertTrue(any(
            check["claim_type"] == "statistic" and check["status"] == "supported"
            for check in result["claim_checks"]
        ))


if __name__ == "__main__":
    unittest.main()
