"""Top-level six-stage local AI Agent orchestration."""
from time import perf_counter

from input_processor import process_input
from context_manager import ContextManager
from task_classifier import classify_task
from router import classify_question
from planner import create_plan
from executor import execute_plan
from response_generator import generate_response


class LocalAIAgent:
    def __init__(self, context_manager=None, router_fn=None, search_fn=None,
                 generation_fn=None):
        self.context_manager = context_manager or ContextManager()
        self.router_fn = router_fn or classify_question
        self.search_fn = search_fn
        self.generation_fn = generation_fn

    def run(self, raw_input: str) -> dict:
        timings = {}
        stages = {}

        started = perf_counter()
        input_data = process_input(raw_input)
        question = input_data["text"]
        timings["input_processing"] = perf_counter() - started
        stages["input_processing"] = {"status": "completed", "data": input_data}

        started = perf_counter()
        context_history = self.context_manager.build_context()
        timings["context_understanding"] = perf_counter() - started
        stages["context_understanding"] = {
            "status": "available" if context_history else "empty",
            "prior_turns": len(self.context_manager.turns),
        }

        started = perf_counter()
        task = classify_task(question)
        timings["task_classification"] = perf_counter() - started
        stages["task_classification"] = {"status": "completed", "data": task}

        started = perf_counter()
        route_decision = self.router_fn(question)
        plan = create_plan(question, task, route_decision)
        timings["planning_routing"] = perf_counter() - started
        stages["planning_routing"] = {
            "status": "completed",
            "route_decision": route_decision,
            "plan": plan,
        }

        started = perf_counter()
        execution = execute_plan(question, plan, search_fn=self.search_fn)
        timings["execution"] = perf_counter() - started
        stages["execution"] = execution

        started = perf_counter()
        response = generate_response(
            question,
            execution,
            plan["route"],
            context_history=context_history,
            generation_fn=self.generation_fn,
        )
        timings["response_generation"] = perf_counter() - started
        stages["response_generation"] = {
            "status": "completed" if response["answer"] else "failed",
            "validation": response["validation"],
            "generation": response["generation"],
        }

        if response["answer"]:
            self.context_manager.add_turn(question, response["answer"])

        return {
            "question": question,
            "task": task,
            "route_decision": route_decision,
            "plan": plan,
            "execution": execution,
            "response": response,
            "stages": stages,
            "timings": timings,
        }
