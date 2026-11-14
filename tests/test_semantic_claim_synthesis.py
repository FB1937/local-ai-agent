import json
import unittest
from unittest.mock import Mock
from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage
from semantic_claim_synthesis import synthesize_claim_evidence_semantically

class FakeResponse:
    def __init__(self, content): self.content = content
    def raise_for_status(self): return None
    def json(self): return {"message": {"content": self.content}}

def sample():
    return ClaimAnalysisResult(
        claims=[Claim(claim_id="C1", text="계약은 2027년 1월 1일부터 효력이 발생한다.")],
        passages=[
            EvidencePassage(evidence_id="E1", source_id="S1", text="계약의 효력 발생일은 2027년 1월이다.", location="p1"),
            EvidencePassage(evidence_id="E2", source_id="S2", text="효력 발생일은 해당 월의 첫날로 정한다.", location="p2"),
        ],
        links=[ClaimEvidenceLink(claim_id="C1", evidence_id="E1", relation="insufficient"),
               ClaimEvidenceLink(claim_id="C1", evidence_id="E2", relation="insufficient")],
    )

class SemanticSynthesisTests(unittest.TestCase):
    def respond(self, payload):
        return Mock(return_value=FakeResponse(json.dumps(payload, ensure_ascii=False)))
    def test_joint_support_upgrade_preserves_pairwise_links(self):
        analysis=sample(); before=[x.model_dump() for x in analysis.links]
        result, report=synthesize_claim_evidence_semantically(analysis, request_fn=self.respond({
            "relation":"supports","evidence_ids":["E1","E2"],"reason":"월과 첫날 규칙을 결합함"
        }))
        self.assertEqual(result.claims[0].status,"supported")
        self.assertEqual(result.claims[0].joint_supporting_evidence_ids,["E1","E2"])
        self.assertEqual(result.claims[0].insufficient_evidence_ids,["E1","E2"])
        self.assertEqual(report.assessed_count,1)
        self.assertEqual([x.model_dump() for x in analysis.links],before)
    def test_same_source_group_with_different_passages_is_not_independent(self):
        analysis = sample()
        analysis.passages[0].source_group = "same-article"
        analysis.passages[1].source_group = "same-article"
        request = self.respond({
            "relation": "supports", "evidence_ids": ["E1", "E2"], "reason": "not independent"
        })
        result, report = synthesize_claim_evidence_semantically(analysis, request_fn=request)
        request.assert_not_called()
        self.assertEqual(result.claims[0].status, "insufficient")
        self.assertEqual(report.assessed_count, 0)
        self.assertEqual(report.skipped_count, 1)

    def test_same_source_id_with_multiple_passages_is_not_independent(self):
        analysis = sample()
        analysis.passages[0].source_id = "S1"
        analysis.passages[1].source_id = "S1"
        analysis.passages[0].source_group = None
        analysis.passages[1].source_group = None
        request = self.respond({
            "relation": "supports", "evidence_ids": ["E1", "E2"], "reason": "not independent"
        })
        result, report = synthesize_claim_evidence_semantically(analysis, request_fn=request)
        request.assert_not_called()
        self.assertEqual(result.claims[0].status, "insufficient")
        self.assertEqual(report.skipped_count, 1)

    def test_selection_prioritizes_distinct_source_groups(self):
        analysis = sample()
        analysis.passages[0].source_group = "origin-A"
        analysis.passages[1].source_group = "origin-A"
        analysis.passages.append(EvidencePassage(
            evidence_id="E3", source_id="S3", source_group="origin-B",
            text="별도 출처의 보완 전제", location="p3",
        ))
        analysis.links.append(ClaimEvidenceLink(
            claim_id="C1", evidence_id="E3", relation="insufficient",
        ))
        request = self.respond({
            "relation": "supports", "evidence_ids": ["E1", "E3"], "reason": "independent premises"
        })
        result, report = synthesize_claim_evidence_semantically(analysis, request_fn=request, max_evidence_per_claim=2)
        self.assertEqual(result.claims[0].status, "supported")
        sent = json.loads(request.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertEqual([item["evidence_id"] for item in sent["evidence"]], ["E1", "E3"])

    def test_unknown_evidence_id_fails_closed(self):
        result,report=synthesize_claim_evidence_semantically(sample(),request_fn=self.respond({
            "relation":"supports","evidence_ids":["E1","E99"],"reason":"invalid"
        }))
        self.assertEqual(result.claims[0].status,"insufficient")
        self.assertEqual(report.unassessed_count,1)
    def test_one_id_cannot_claim_joint_entailment(self):
        result,report=synthesize_claim_evidence_semantically(sample(),request_fn=self.respond({
            "relation":"supports","evidence_ids":["E1"],"reason":"one premise"
        }))
        self.assertEqual(result.claims[0].status,"insufficient")
        self.assertEqual(report.unassessed_count,1)
    def test_model_failure_preserves_baseline(self):
        result,report=synthesize_claim_evidence_semantically(sample(),request_fn=Mock(side_effect=TimeoutError()))
        self.assertEqual(result.claims[0].status,"insufficient")
        self.assertEqual(report.unassessed_count,1)
    def test_non_insufficient_claim_is_not_sent_to_model(self):
        analysis=sample(); analysis.links[0].relation="supports"
        request=self.respond({"relation":"supports","evidence_ids":["E1","E2"],"reason":"unused"})
        result,report=synthesize_claim_evidence_semantically(analysis,request_fn=request)
        request.assert_not_called()
        self.assertEqual(result.claims[0].status,"supported")
        self.assertEqual(report.assessed_count,0)
    def test_prompt_treats_passages_as_untrusted_data(self):
        request=self.respond({"relation":"insufficient","evidence_ids":["E1","E2"],"reason":"gap"})
        synthesize_claim_evidence_semantically(sample(),request_fn=request)
        self.assertIn("untrusted quoted data",request.call_args.kwargs["json"]["messages"][0]["content"])

if __name__=="__main__": unittest.main()
