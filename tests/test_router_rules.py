import unittest

from router import apply_safety_rules


class RouterSafetyRuleTests(unittest.TestCase):
    def make_decision(self, route="DIRECT", confidence=0.95, search_query=""):
        return {
            "model_route": route,
            "route": route,
            "reason": "단위 테스트용 초기 판단",
            "search_query": search_query,
            "confidence": confidence,
            "overridden": False,
            "fallback": False,
        }

    def test_explicit_search_request_promotes_direct_to_search(self):
        result = apply_safety_rules("Python 공식 문서를 웹에서 검색해 줘.", self.make_decision())
        self.assertEqual(result["route"], "SEARCH")
        self.assertTrue(result["overridden"])

    def test_current_exchange_rate_promotes_direct_to_search(self):
        result = apply_safety_rules("2026년 현재 원달러 환율을 알려 줘.", self.make_decision())
        self.assertEqual(result["route"], "SEARCH")

    def test_today_alone_does_not_force_search_for_text_translation(self):
        question = "이 문장을 영어로 번역해 줘: '오늘은 데이터베이스 과제를 마무리할 예정이다.'"
        result = apply_safety_rules(question, self.make_decision())
        self.assertEqual(result["route"], "DIRECT")

    def test_hybrid_route_is_preserved_for_mixed_question(self):
        decision = self.make_decision(route="HYBRID")
        result = apply_safety_rules("개념을 설명하고 최신 사례도 조사해 줘.", decision)
        self.assertEqual(result["route"], "HYBRID")

    def test_low_confidence_falls_back_to_search(self):
        result = apply_safety_rules("이 개념을 설명해 줘.", self.make_decision(confidence=0.2))
        self.assertEqual(result["route"], "SEARCH")
        self.assertTrue(result["overridden"])

    def test_missing_search_query_defaults_to_original_question(self):
        question = "최근 Python 릴리스 정보를 알려 줘."
        result = apply_safety_rules(question, self.make_decision(route="SEARCH"))
        self.assertEqual(result["search_query"], question)


if __name__ == "__main__":
    unittest.main()
