import unittest
from unittest.mock import patch

from input_processor import process_input
from context_manager import ContextManager
from task_classifier import classify_task
from planner import create_plan
from executor import execute_plan
from response_generator import generate_response
from agent import LocalAIAgent


class InputAndContextTests(unittest.TestCase):
    def test_input_is_normalized_and_timestamped(self):
        result = process_input("  Python으로 CSV를 읽어 줘.  ")
        self.assertEqual(result["text"], "Python으로 CSV를 읽어 줘.")
        self.assertEqual(result["input_type"], "text")
        self.assertEqual(result["tokenization"], "delegated_to_ollama")
        self.assertTrue(result["received_at"])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            process_input("  ")

    def test_context_is_bounded_to_recent_turns(self):
        context = ContextManager(max_turns=2)
        context.add_turn("질문1", "답변1")
        context.add_turn("질문2", "답변2")
        context.add_turn("질문3", "답변3")
        rendered = context.build_context()
        self.assertNotIn("질문1", rendered)
        self.assertIn("질문2", rendered)
        self.assertIn("질문3", rendered)

    def test_empty_turn_is_not_saved(self):
        context = ContextManager()
        context.add_turn("질문", "")
        self.assertEqual(len(context.turns), 0)


class TaskAndPlanningTests(unittest.TestCase):
    def test_classifier_extracts_intent_domain_and_freshness(self):
        result = classify_task("2026년 최신 데이터 엔지니어 채용 동향을 조사해 줘.")
        self.assertEqual(result["intent"], "information_seeking")
        self.assertEqual(result["domain"], "data_engineering")
        self.assertTrue(result["freshness_required"])
        self.assertIn("web_search", result["tool_candidates"])

    def test_classifier_identifies_creation_task(self):
        result = classify_task("Python으로 CSV 파이프라인 코드를 만들어 줘.")
        self.assertEqual(result["intent"], "creation")
        self.assertIn("generation", result["task_types"])

    def test_planner_keeps_direct_route_local(self):
        plan = create_plan(
            "리스트와 튜플의 차이를 설명해 줘.",
            {"intent": "learning"},
            {"route": "DIRECT", "reason": "stable concept"},
        )
        self.assertEqual(plan["route"], "DIRECT")
        self.assertFalse(plan["requires_external_execution"])
        self.assertIsNone(plan["search"])

    def test_planner_adds_search_steps(self):
        plan = create_plan(
            "최신 Python 버전을 조사해 줘.",
            {"intent": "information_seeking"},
            {"route": "SEARCH", "search_query": "Python latest stable release"},
        )
        self.assertTrue(plan["requires_external_execution"])
        self.assertEqual(plan["steps"][0]["tool"], "searxng")
        self.assertEqual(plan["retry_policy"]["max_search_attempts"], 2)


class ExecutionAndResponseTests(unittest.TestCase):
    def test_direct_execution_does_not_call_search(self):
        plan = {"route": "DIRECT"}
        with patch("executor.search_with_validation") as search:
            result = execute_plan("질문", plan)
        search.assert_not_called()
        self.assertEqual(result["status"], "skipped")

    @patch("executor.search_with_validation")
    def test_search_execution_returns_evidence(self, search):
        search.return_value = {
            "results": [{"source_id": "S1", "url": "https://example.com"}],
            "assessment": {"sufficient": True, "reason": "ok"},
            "attempts": [{"count": 1, "assessment": {"reason": "ok"}}],
        }
        result = execute_plan(
            "최신 정보", {"route": "SEARCH", "search": {"query": "query"}}
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], ["searxng"])
        self.assertEqual(result["results"][0]["source_id"], "S1")

    def test_response_generation_receives_prior_context(self):
        seen = {}

        def fake_generation(question, results, route, search_assessment):
            seen["question"] = question
            return {"answer": "이전 문맥을 반영한 답변", "ok": True, "incomplete": False}

        result = generate_response(
            "현재 질문",
            {"results": [], "assessment": {"sufficient": True}},
            "DIRECT",
            context_history="사용자: 이전 질문\n에이전트: 이전 답변",
            generation_fn=fake_generation,
        )
        self.assertIn("이전 질문", seen["question"])
        self.assertIn("현재 질문", seen["question"])
        self.assertTrue(result["validation"]["valid"])


class AgentOrchestrationTests(unittest.TestCase):
    def test_agent_runs_six_stages_and_remembers_completed_turn(self):
        calls = []

        def fake_router(question):
            return {
                "route": "DIRECT",
                "reason": "단위 테스트",
                "search_query": "",
                "confidence": 0.99,
                "overridden": False,
                "fallback": False,
            }

        def fake_generation(question, results, route, search_assessment):
            calls.append(question)
            return {
                "answer": "테스트 답변",
                "ok": True,
                "incomplete": False,
                "attempts": [],
                "retry_count": 0,
            }

        agent = LocalAIAgent(router_fn=fake_router, generation_fn=fake_generation)
        first = agent.run("첫 번째 질문")
        second = agent.run("두 번째 질문")

        expected = {
            "input_processing", "context_understanding", "task_classification",
            "planning_routing", "execution", "response_generation",
        }
        self.assertEqual(set(first["stages"]), expected)
        self.assertEqual(first["execution"]["status"], "skipped")
        self.assertIn("첫 번째 질문", calls[1])
        self.assertEqual(len(agent.context_manager.turns), 2)

    def test_context_dependent_follow_up_is_resolved_before_routing(self):
        routed_questions = []

        def fake_router(question):
            routed_questions.append(question)
            return {
                "route": "DIRECT", "reason": "test", "search_query": "",
                "confidence": 0.99, "overridden": False, "fallback": False,
            }

        def fake_generation(question, results, route, search_assessment):
            return {
                "answer": "테스트 답변", "ok": True, "incomplete": False,
                "attempts": [], "retry_count": 0,
            }

        agent = LocalAIAgent(router_fn=fake_router, generation_fn=fake_generation)
        agent.run("Python 3.15.2가 최신 안정 버전인지 확인하고 싶어.")
        second = agent.run("그럼 그 버전은 언제 출시됐어?")

        self.assertIn("Python 3.15.2", second["contextual_question"])
        self.assertIn("Python 3.15.2", routed_questions[1])
        self.assertTrue(second["stages"]["context_understanding"]["follow_up_resolved"])


if __name__ == "__main__":
    unittest.main()
