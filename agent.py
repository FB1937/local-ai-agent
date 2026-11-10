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
        contextual_question = self.context_manager.build_contextual_question(question)
        timings["context_understanding"] = perf_counter() - started
        stages["context_understanding"] = {
            "status": "available" if context_history else "empty",
            "prior_turns": len(self.context_manager.turns),
            "follow_up_resolved": contextual_question != question,
        }

        started = perf_counter()
        # Classify only the current user turn. Prior assistant wording must not
        # leak keywords (e.g. "latest", "search", "error") into intent/freshness.
        task = classify_task(question)
        timings["task_classification"] = perf_counter() - started
        stages["task_classification"] = {
            "status": "completed",
            "data": task,
            "input_scope": "current_user_turn_only",
        }

        # Keep router latency separate from deterministic plan construction.
        # This makes the main Planning bottleneck measurable in live runs.
        started = perf_counter()
        # The router needs contextual wording to resolve pronouns, but its
        # safety rules must inspect the current turn for freshness/search intent.
        if self.router_fn is classify_question:
            route_decision = self.router_fn(
                contextual_question, current_question=question
            )
        else:
            route_decision = self.router_fn(contextual_question)
        timings["router_decision"] = perf_counter() - started

        started = perf_counter()
        plan = create_plan(contextual_question, task, route_decision)
        timings["plan_creation"] = perf_counter() - started
        timings["planning_routing"] = (
            timings["router_decision"] + timings["plan_creation"]
        )
        stages["planning_routing"] = {
            "status": "completed",
            "route_decision": route_decision,
            "plan": plan,
        }

        started = perf_counter()
        execution = execute_plan(
            contextual_question, plan, search_fn=self.search_fn
        )
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
        generation_complete = (
            response["generation"].get("ok", False)
            and not response["generation"].get("incomplete", False)
        )
        stages["response_generation"] = {
            "status": (
                "failed" if not response["answer"]
                else "completed" if generation_complete
                else "incomplete"
            ),
            "validation": response["validation"],
            "generation": response["generation"],
        }

        # Do not carry truncated output into the next turn as trusted context.
        if response["answer"] and generation_complete:
            self.context_manager.add_turn(question, response["answer"])

        return {
            "question": question,
            "contextual_question": contextual_question,
            "task": task,
            "route_decision": route_decision,
            "plan": plan,
            "execution": execution,
            "response": response,
            "stages": stages,
            "timings": timings,
        }
