"""Structural citation checks; semantic entailment is not inferred."""
import re
CITATION_RE=re.compile(r"\\[(S\\d+)\\]")
def validate_answer(answer:str,sources:list|None)->dict:
    sources=sources or []
    valid={str(s.get("source_id")) for s in sources if s.get("source_id")}
    found=CITATION_RE.findall(answer or "")
    unknown=sorted(set(found)-valid)
    missing=sorted(sid for sid in set(found) if sid in valid and not next((s.get("url") for s in sources if s.get("source_id")==sid and s.get("url")),None))
    return {"valid":not unknown and not missing,"found_ids":found,"unknown_ids":unknown,"missing_url_ids":missing,
            "semantic_grounding_checked":False,"note":"인용 ID와 URL의 구조만 검사했습니다. 의미적 근거성은 별도 검토가 필요합니다."}
def append_validation_notice(answer:str,validation:dict)->str:
    if validation["unknown_ids"]:return answer+"\\n\\n[검증 경고] 존재하지 않는 출처 ID: "+", ".join(validation["unknown_ids"])
    if validation["missing_url_ids"]:return answer+"\\n\\n[검증 경고] URL이 없는 출처 ID: "+", ".join(validation["missing_url_ids"])
    return answer
