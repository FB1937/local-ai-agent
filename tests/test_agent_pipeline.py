import unittest
from unittest.mock import patch

from input_processor import process_input
from context_manager import ContextManager
from task_classifier import classify_task
from planner import create_plan
from executor import execute_plan
from response_generator import generate_response
from agent import LocalAIAgent
from router import classify_question
from search_validator import prepare_results, assess_results
from llm import _build_prompts


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
    def test_router_safety_uses_current_turn_not_old_context(self):
        contextual = (
            "[이전 대화 문맥]\n사용자: 현재 Python 최신 버전을 찾아줘.\n"
            "에이전트: 검색 결과를 확인했습니다.\n\n"
            "[현재 질문]\n튜플 안에 리스트가 있으면 어떻게 돼?"
        )
        model_decision = {
            "route": "DIRECT",
            "reason": "기초 Python 개념",
            "search_query": "",
            "confidence": 0.99,
        }
        with patch("router._call_router_model", return_value=model_decision):
            result = classify_question(
                contextual,
                current_question="튜플 안에 리스트가 있으면 어떻게 돼?",
            )
        self.assertEqual(result["route"], "DIRECT")
        self.assertFalse(result["overridden"])

    def test_simple_conceptual_question_uses_fast_direct_without_router_llm(self):
        with patch("router._call_router_model") as router_model:
            result = classify_question("Python 리스트와 튜플의 차이를 설명해 줘.")
        router_model.assert_not_called()
        self.assertEqual(result["route"], "DIRECT")
        self.assertTrue(result["fast_path"])
        self.assertEqual(result["model_route"], "RULE_DIRECT")

    def test_freshness_request_does_not_use_fast_direct(self):
        model_decision = {
            "route": "SEARCH",
            "reason": "최신 정보 확인",
            "search_query": "Python latest stable release",
            "confidence": 0.95,
        }
        with patch("router._call_router_model", return_value=model_decision) as router_model:
            result = classify_question("현재 Python 최신 안정 버전을 확인해 줘.")
        router_model.assert_called_once()
        self.assertEqual(result["route"], "SEARCH")
        self.assertFalse(result.get("fast_path", False))

    def test_follow_up_conceptual_question_uses_fast_direct(self):
        with patch("router._call_router_model") as router_model:
            result = classify_question(
                "[이전 대화 문맥]\\n사용자: Python 리스트와 튜플의 차이\\n"
                "[현재 질문]\\n그럼 튜플 안에 리스트가 있으면 어떻게 돼?",
                current_question="그럼 튜플 안에 리스트가 있으면 어떻게 돼?",
            )
        router_model.assert_not_called()
        self.assertEqual(result["route"], "DIRECT")
        self.assertTrue(result["fast_path"])

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

    def test_follow_up_concept_question_does_not_request_search(self):
        result = classify_task("그럼 튜플 안에 리스트가 있으면 어떻게 돼?")
        self.assertEqual(result["intent"], "learning")
        self.assertFalse(result["freshness_required"])
        self.assertNotIn("web_search", result["tool_candidates"])

    def test_generic_tell_me_word_does_not_request_search(self):
        result = classify_task("Python 리스트를 알려줘.")
        self.assertNotIn("web_search", result["tool_candidates"])

    def test_python_latest_canonical_endpoint_extracts_patch_version(self):
        rows = prepare_results(
            "Python latest stable release",
            [{
                "title": "Python Release Python 3.15.2",
                "url": "https://www.python.org/downloads/latest",
                "content": "Python 3.15.2 is the latest release.",
                "source_type": "official",
            }],
        )
        assessment = assess_results(
            rows, freshness_required=True, requires_official=True, version_query=True
        )
        self.assertEqual(assessment["latest_stable_claims"], ["3.15.2"])
        self.assertFalse(assessment["unresolved_version_conflict"])

    def test_canonical_release_index_is_authoritative_with_low_snippet_overlap(self):
        rows = prepare_results(
            "unrelated query words",
            [{
                "title": "Python Release Python 3.15.2",
                "url": "https://www.python.org/downloads/latest",
                "content": "Python 3.15.2",
                "source_type": "official",
            }],
        )
        assessment = assess_results(
            rows, freshness_required=True, requires_official=True, version_query=True
        )
        self.assertEqual(assessment["latest_stable_claims"], ["3.15.2"])
        self.assertEqual(assessment["official_release_index_count"], 1)
        self.assertTrue(assessment["sufficient"])

    def test_generic_company_implementation_article_is_not_customer_case_signal(self):
        rows = prepare_results(
            "RAG enterprise customer deployment case study",
            [{
                "title": "Gen AI 해커톤에서 선보인 기업 맞춤형 RAG 적용 사례",
                "url": "https://www.samsungsds.com/kr/insights/rag-customization.html",
                "content": "기업 맞춤형 RAG 기술 소개와 해커톤 결과를 설명합니다.",
                "source_type": "company_official",
            }],
        )
        self.assertEqual(rows[0]["source_type"], "company_official")
        self.assertFalse(rows[0]["company_case_signal"])

    def test_korean_enterprise_first_party_domain_is_company_official(self):
        from search_validator import classify_source
        self.assertEqual(
            classify_source("https://www.naver.com/tech/rag"),
            "company_official",
        )

    def test_named_customer_deployment_is_candidate_signal(self):
        rows = prepare_results(
            "RAG customer deployment",
            [{
                "title": "Named customer deployment",
                "url": "https://www.samsungsds.com/kr/insights/customer-story.html",
                "content": "Customer success story: deployed at Acme Corp for contract review.",
                "source_type": "company_official",
            }],
        )
        self.assertTrue(rows[0]["company_case_signal"])

    def test_direct_prompt_spells_out_tuple_nested_mutability(self):
        system_prompt, _ = _build_prompts(
            "Python 리스트와 튜플 차이", route="DIRECT",
            context="", search_assessment=None
        )
        self.assertIn("튜플은 리스트를 포함할 수 있다", system_prompt)
        self.assertIn("재귀적으로 불변", system_prompt)


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

    def test_insufficient_hybrid_evidence_adds_mandatory_boundary_warning(self):
        from answer_validator import validate_answer, append_validation_notice

        validation = validate_answer(
            "RAG가 업계 표준으로 자리 잡았습니다.",
            [],
            search_assessment={
                "sufficient": False,
                "requires_company_official": True,
                "company_official_count": 0,
                "company_case_signal_count": 0,
            },
            route="HYBRID",
        )
        rendered = append_validation_notice("RAG가 업계 표준으로 자리 잡았습니다.", validation)
        self.assertIn("검색 근거가 충분하지 않습니다", rendered)
        self.assertIn("실제 고객 도입 사례는 확인되지 않았습니다", rendered)

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
        self.assertIn("router_decision", first["timings"])
        self.assertIn("plan_creation", first["timings"])
        self.assertAlmostEqual(
            first["timings"]["planning_routing"],
            first["timings"]["router_decision"] + first["timings"]["plan_creation"],
        )
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
