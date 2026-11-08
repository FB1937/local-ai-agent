import unittest
from datetime import datetime,timezone
from search_planner import plan_search
from search_validator import normalize_url,prepare_results,assess_results,search_with_validation
from answer_validator import validate_answer
class SearchQualityTests(unittest.TestCase):
 def test_current_query_replaces_stale_router_year(self):
  p=plan_search("Python 최신 안정 버전","Python latest stable release 2024",datetime(2026,11,8,tzinfo=timezone.utc));self.assertIn("2026",p["query"]);self.assertNotIn("2024",p["query"])
 def test_explicit_historical_year_is_preserved(self):
  p=plan_search("Python 3.10 최초 출시일","Python 3.10 release 2026",datetime(2026,11,8,tzinfo=timezone.utc));self.assertFalse(p["freshness_required"])
 def test_time_range_only_for_short_window(self):
  self.assertEqual(plan_search("오늘 부산 날씨")["time_range"],"day");self.assertIsNone(plan_search("Python 리스트와 튜플 차이")["time_range"])
 def test_url_tracking_removed(self):
  self.assertEqual(normalize_url("https://EXAMPLE.com/a?utm_source=x&b=2#frag"),"https://example.com/a?b=2")
 def test_duplicate_normalized_url(self):
  rows=prepare_results("Python release",[{"url":"https://python.org/a?utm_source=x","title":"Python release","content":"Python release"},{"url":"https://python.org/a","title":"same","content":"same"}]);self.assertEqual(len(rows),1)
 def test_same_title_different_urls_kept(self):
  rows=prepare_results("release",[{"url":"https://a.example/x","title":"Release","content":"release"},{"url":"https://b.example/x","title":"Release","content":"release"}]);self.assertEqual(len(rows),2)
 def test_official_domain_classified(self):
  rows=prepare_results("Python release",[{"url":"https://www.python.org/downloads/","title":"Python downloads","content":"Python release"}]);self.assertEqual(rows[0]["source_type"],"official")
 def test_empty_results_insufficient(self):self.assertFalse(assess_results([])["sufficient"])
 def test_at_most_one_retry(self):
  calls=[]
  def fake(q,max_results,time_range=None):calls.append(q);return []
  out=search_with_validation("latest Python version","Python latest 2024",fake,datetime(2026,11,8,tzinfo=timezone.utc));self.assertEqual(len(calls),2);self.assertEqual(len(out["attempts"]),2)
 def test_unknown_citation_detected(self):
  r=validate_answer("latest [S9]",[{"source_id":"S1","url":"https://python.org"}]);self.assertEqual(r["unknown_ids"],["S9"])
 def test_valid_citation(self):
  r=validate_answer("official [S1]",[{"source_id":"S1","url":"https://python.org"}]);self.assertTrue(r["valid"])
 def test_no_semantic_claim(self):
  self.assertFalse(validate_answer("[S1]",[{"source_id":"S1","url":"https://python.org"}])["semantic_grounding_checked"])
if __name__=="__main__":unittest.main()
