import unittest
from datetime import datetime, timezone
from search_planner import plan_search, build_retry_query
from search_validator import normalize_url, prepare_results, assess_results, search_with_validation
from answer_validator import validate_answer

class SearchQualityTests(unittest.TestCase):
    def test_current_query_replaces_stale_router_year(self):
        now = datetime(2026, 11, 8, tzinfo=timezone.utc)
        p = plan_search("Python 최신 안정 버전", "Python latest stable release 2024", now)
        self.assertIn("2026", p["query"])
        self.assertNotIn("2024", p["query"])

    def test_user_specified_current_year_overrides_stale_query_year(self):
        now = datetime(2026, 11, 8, tzinfo=timezone.utc)
        p = plan_search("2026년 현재 Python 최신 안정 버전", "Python latest release 2024", now)
        self.assertIn("2026", p["query"])
        self.assertNotIn("2024", p["query"])

    def test_historical_query_is_preserved(self):
        now = datetime(2026, 11, 8, tzinfo=timezone.utc)
        p = plan_search("Python 3.10 최초 출시일", "Python 3.10 release 2026", now)
        self.assertEqual(p["query"], "Python 3.10 release 2026")
        self.assertFalse(p["freshness_required"])

    def test_time_range_selection(self):
        self.assertEqual(plan_search("오늘 부산 날씨")["time_range"], "day")
        self.assertEqual(plan_search("최근 1개월 보안 공지")["time_range"], "month")
        self.assertIsNone(plan_search("Python 리스트와 튜플 차이")["time_range"])

    def test_url_normalization(self):
        self.assertEqual(normalize_url("https://EXAMPLE.com:443/a?utm_source=x&b=2#frag"),
                         "https://example.com/a?b=2")

    def test_duplicate_url_removed(self):
        rows = prepare_results("Python release", [
            {"url":"https://python.org/a?utm_source=x","title":"Python release","content":"Python release"},
            {"url":"https://python.org/a","title":"same","content":"same"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_id"], "S1")

    def test_same_title_different_urls_kept(self):
        rows = prepare_results("release", [
            {"url":"https://a.example/x","title":"Release","content":"release"},
            {"url":"https://b.example/x","title":"Release","content":"release"}])
        self.assertEqual(len(rows), 2)

    def test_source_classification(self):
        official = prepare_results("Python release", [
            {"url":"https://www.python.org/downloads/","title":"Python downloads","content":"Python release"}])
        community = prepare_results("GitHub code", [
            {"url":"https://github.com/example/project","title":"GitHub code","content":"GitHub project"}])
        self.assertEqual(official[0]["source_type"], "official")
        self.assertEqual(community[0]["source_type"], "technical_community")

    def test_empty_results_insufficient(self):
        self.assertFalse(assess_results([])["sufficient"])

    def test_retry_maximum_one(self):
        calls = []
        def fake(query, max_results, time_range=None):
            calls.append(query)
            return []
        out = search_with_validation("latest Python version", "Python latest 2024", fake,
                                     datetime(2026, 11, 8, tzinfo=timezone.utc))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(out["attempts"]), 2)

    def test_retry_distinct_even_when_suffix_present(self):
        query = "Python 2026 공식 문서 최신 자료"
        self.assertNotEqual(query, build_retry_query(query, 2026))

    def test_search_exception_is_handled(self):
        def broken(query, max_results, time_range=None):
            raise TimeoutError("test timeout")
        out = search_with_validation("latest Python version", "Python latest", broken)
        self.assertEqual(len(out["attempts"]), 2)
        self.assertIn("TimeoutError", out["attempts"][0]["error"])

    def test_unknown_citation_detected(self):
        r = validate_answer("최신 버전 [S9]", [{"source_id":"S1","url":"https://python.org"}])
        self.assertEqual(r["unknown_ids"], ["S9"])

    def test_valid_citation(self):
        r = validate_answer("공식 문서 [S1]", [{"source_id":"S1","url":"https://python.org"}])
        self.assertTrue(r["valid"])

    def test_semantic_grounding_not_claimed(self):
        r = validate_answer("[S1]", [{"source_id":"S1","url":"https://python.org"}])
        self.assertFalse(r["semantic_grounding_checked"])

if __name__ == "__main__":
    unittest.main()
