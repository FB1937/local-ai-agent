import unittest
from datetime import datetime, timezone

from search_planner import plan_search, build_retry_query
from search_validator import (
    normalize_url, prepare_results, assess_results, search_with_validation,
    classify_source, is_official_release_index,
)
from answer_validator import validate_answer


class SearchQualityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 8, tzinfo=timezone.utc)

    def test_current_query_replaces_stale_router_year(self):
        p = plan_search("Python 최신 안정 버전", "Python latest stable release 2024", self.now)
        self.assertIn("2026", p["query"])
        self.assertNotIn("2024", p["query"])

    def test_user_specified_current_year_overrides_stale_query_year(self):
        p = plan_search("2026년 현재 Python 최신 안정 버전", "Python latest release 2024", self.now)
        self.assertIn("2026", p["query"])
        self.assertNotIn("2024", p["query"])

    def test_historical_query_is_preserved(self):
        p = plan_search("Python 3.10 최초 출시일", "Python 3.10 release 2026", self.now)
        self.assertEqual(p["query"], "Python 3.10 release 2026")
        self.assertFalse(p["freshness_required"])
        self.assertFalse(p["requires_official"])

    def test_time_range_selection(self):
        self.assertEqual(plan_search("오늘 부산 날씨", now=self.now)["time_range"], "day")
        self.assertEqual(plan_search("최근 1개월 보안 공지", now=self.now)["time_range"], "month")
        self.assertIsNone(plan_search("Python 리스트와 튜플 차이", now=self.now)["time_range"])

    def test_latest_version_requires_official_source(self):
        p = plan_search("현재 Python 최신 안정 버전과 공식 배포 정보", now=self.now)
        self.assertTrue(p["version_query"])
        self.assertTrue(p["requires_official"])
        self.assertIn("공식 릴리스 목록", p["retry_query"])

    def test_company_case_requires_first_party_company_source(self):
        p = plan_search("2026년 기업들의 RAG 활용 사례와 최근 동향", now=self.now)
        self.assertTrue(p["requires_company_official"])
        self.assertIn("기업 공식 발표", p["retry_query"])

    def test_url_normalization(self):
        self.assertEqual(
            normalize_url("https://EXAMPLE.com:443/a?utm_source=x&b=2#frag"),
            "https://example.com/a?b=2",
        )

    def test_duplicate_url_removed(self):
        rows = prepare_results("Python release", [
            {"url": "https://python.org/a?utm_source=x", "title": "Python release", "content": "Python release"},
            {"url": "https://python.org/a", "title": "same", "content": "same"},
        ], now=self.now)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_id"], "S1")

    def test_same_title_different_urls_kept(self):
        rows = prepare_results("release", [
            {"url": "https://a.example/x", "title": "Release", "content": "release"},
            {"url": "https://b.example/x", "title": "Release", "content": "release"},
        ], now=self.now)
        self.assertEqual(len(rows), 2)

    def test_core_official_source_classification(self):
        self.assertEqual(classify_source("https://www.python.org/downloads/"), "official")
        self.assertEqual(classify_source("https://www.postgresql.org/docs/"), "official")

    def test_company_official_source_classification(self):
        self.assertEqual(classify_source("https://www.samsungsds.com/"), "company_official")
        self.assertEqual(classify_source("https://www.ibm.com/think/topics/rag"), "company_official")
        self.assertEqual(classify_source("https://cloud.google.com/"), "company_official")

    def test_other_source_classification(self):
        self.assertEqual(classify_source("https://brunch.co.kr/@example/post"), "other")
        self.assertEqual(classify_source("https://github.com/example/repo"), "technical_community")

    def test_official_release_index_detection(self):
        self.assertTrue(is_official_release_index({
            "url": "https://www.python.org/downloads/", "title": "Downloads"
        }))
        self.assertFalse(is_official_release_index({
            "url": "https://www.python.org/downloads/release/python-3150/", "title": "Python 3.15.0"
        }))

    def test_empty_results_insufficient(self):
        self.assertFalse(assess_results([])["sufficient"])

    def test_company_case_not_sufficient_with_blogs_only(self):
        rows = prepare_results("RAG 기업 활용 사례", [
            {"url": "https://blog.example/rag", "title": "RAG 기업 활용 사례", "content": "기업 RAG 활용 사례"},
            {"url": "https://brunch.co.kr/rag", "title": "RAG 도입 사례", "content": "기업 RAG 도입 사례"},
        ], now=self.now)
        result = assess_results(rows, requires_company_official=True)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["company_official_count"], 0)

    def test_company_case_sufficient_with_company_primary_source(self):
        rows = prepare_results("RAG 기업 활용 사례", [
            {"url": "https://blog.example/rag", "title": "RAG 기업 활용 사례", "content": "기업 RAG 활용 사례"},
            {"url": "https://www.samsungsds.com/rag", "title": "RAG 고객 사례", "content": "기업 RAG 업무 적용 사례"},
        ], now=self.now)
        result = assess_results(rows, requires_company_official=True)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["company_official_count"], 1)

    def test_latest_version_without_official_source_is_insufficient(self):
        rows = prepare_results("Python latest stable version", [
            {"url": "https://en.wikipedia.org/wiki/Python", "title": "Python 3.15.2 latest version", "content": "Python 3.15.2 latest stable version"},
            {"url": "https://blog.example/python", "title": "Python 3.15.1", "content": "Python 3.15.1 release"},
        ], now=self.now)
        result = assess_results(rows, requires_official=True, version_query=True)
        self.assertFalse(result["sufficient"])

    def test_version_conflict_without_release_index_is_flagged(self):
        rows = prepare_results("Python latest stable version", [
            {"url": "https://www.python.org/downloads/release/python-3150/", "title": "Python 3.15.0 release", "content": "Python 3.15.0"},
            {"url": "https://en.wikipedia.org/wiki/Python", "title": "Python 3.15.2 latest", "content": "Python 3.15.2 latest version"},
        ], now=self.now)
        result = assess_results(rows, requires_official=True, version_query=True)
        self.assertTrue(result["version_conflict"])
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["version_mentions"], ["3.15.0", "3.15.2"])

    def test_official_release_index_can_resolve_version_source_requirement(self):
        rows = prepare_results("Python latest stable version", [
            {"url": "https://www.python.org/downloads/", "title": "Python downloads 3.15.2", "content": "Latest stable release Python 3.15.2"},
            {"url": "https://www.python.org/downloads/release/python-3152/", "title": "Python 3.15.2", "content": "Python 3.15.2 release"},
        ], now=self.now)
        result = assess_results(rows, requires_official=True, version_query=True)
        self.assertTrue(result["sufficient"])
        self.assertGreaterEqual(result["official_release_index_count"], 1)

    def test_official_source_ranks_above_blog_when_relevance_is_equal(self):
        rows = prepare_results("RAG enterprise", [
            {"url": "https://blog.example/rag", "title": "RAG enterprise", "content": "RAG enterprise"},
            {"url": "https://www.ibm.com/rag", "title": "RAG enterprise", "content": "RAG enterprise"},
        ], now=self.now)
        self.assertEqual(rows[0]["source_type"], "company_official")

    def test_retry_maximum_one(self):
        calls = []
        def fake(query, max_results, time_range=None):
            calls.append(query)
            return []
        out = search_with_validation("latest Python version", "Python latest 2024", fake, self.now)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(out["attempts"]), 2)

    def test_retry_distinct_even_when_suffix_present(self):
        query = "Python 2026 공식 문서 최신 자료"
        self.assertNotEqual(query, build_retry_query(query, 2026))

    def test_company_case_retry_queries_primary_sources(self):
        calls = []
        def fake(query, max_results, time_range=None):
            calls.append(query)
            return []
        search_with_validation("2026년 기업들의 RAG 활용 사례", "2026 RAG enterprise case studies", fake, self.now)
        self.assertEqual(len(calls), 2)
        self.assertIn("기업 공식 발표", calls[1])

    def test_search_exception_is_handled(self):
        def broken(query, max_results, time_range=None):
            raise TimeoutError("test timeout")
        out = search_with_validation("latest Python version", "Python latest", broken, self.now)
        self.assertEqual(len(out["attempts"]), 2)
        self.assertIn("TimeoutError", out["attempts"][0]["error"])

    def test_unknown_citation_detected(self):
        result = validate_answer("최신 버전 [S9]", [{"source_id": "S1", "url": "https://python.org"}])
        self.assertEqual(result["unknown_ids"], ["S9"])

    def test_valid_citation(self):
        result = validate_answer("공식 문서 [S1]", [{"source_id": "S1", "url": "https://python.org"}])
        self.assertTrue(result["valid"])

    def test_semantic_grounding_not_claimed(self):
        result = validate_answer("[S1]", [{"source_id": "S1", "url": "https://python.org"}])
        self.assertFalse(result["semantic_grounding_checked"])


if __name__ == "__main__":
    unittest.main()
