"""Deterministic search-result normalization and bounded retry."""
from __future__ import annotations
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
import re
from search_planner import plan_search,build_retry_query
TRACKING={"fbclid","gclid","dclid","mc_cid","mc_eid"}
OFFICIAL=(".gov",".go.kr",".gov.uk",".gov.au",".edu",".ac.kr","python.org","postgresql.org","openai.com","microsoft.com","docs.")
NEWS=("reuters.","apnews.","bbc.","yonhapnews.","yna.co.kr","bloomberg.")
def normalize_url(url:str)->str:
    try:
        p=urlsplit((url or "").strip())
        if p.scheme not in {"http","https"} or not p.netloc:return ""
        path=p.path or "/"
        if path!="/":path=path.rstrip("/")
        query=[(k,v) for k,v in parse_qsl(p.query,keep_blank_values=True) if not(k.lower().startswith("utm_") or k.lower() in TRACKING)]
        return urlunsplit((p.scheme.lower(),p.netloc.lower(),path,urlencode(sorted(query)),""))
    except (TypeError,ValueError):return ""
def classify_source(url:str)->str:
    host=(urlsplit(url).hostname or "").lower()
    if any(x in host for x in OFFICIAL):return "official"
    if any(x in host for x in NEWS):return "news"
    if any(x in host for x in ("arxiv.org","acm.org","ieee.org","doi.org","springer.com")):return "academic"
    if any(x in host for x in ("github.com","gitlab.com","stackoverflow.com")):return "technical_community"
    return "other"
def _tokens(text:str)->set:
    return {x.lower() for x in re.findall(r"[A-Za-z0-9가-힣+#.]{2,}",text or "") if x.lower() not in {"현재","최신","최근","알려","찾아","조사","공식","자료"}}
def relevance_score(query:str,result:dict)->float:
    q=_tokens(query); d=_tokens(" ".join([result.get("title",""),result.get("content","")]))
    if not q or not d:return 0.0
    overlap=len(q&d)/len(q); title=_tokens(result.get("title",""))
    return min(1.0,overlap+(0.12 if q&title else 0.0))
def prepare_results(query:str,raw_results:list,max_results:int=8)->list:
    seen=set(); rows=[]
    for item in raw_results or []:
        if not isinstance(item,dict):continue
        url=normalize_url(item.get("url",""))
        if not url or url in seen:continue
        seen.add(url); row=dict(item); row["url"]=url
        row["source_type"]=classify_source(url); row["relevance_score"]=round(relevance_score(query,row),3)
        rows.append(row)
    rows.sort(key=lambda x:(x["relevance_score"],x["source_type"]=="official",bool(x.get("published"))),reverse=True)
    for i,row in enumerate(rows[:max_results],1):row["source_id"]=f"S{i}"
    return rows[:max_results]
def assess_results(results:list,freshness_required:bool=False)->dict:
    if not results:return {"sufficient":False,"reason":"검색 결과가 없습니다.","count":0}
    usable=[r for r in results if r.get("relevance_score",0)>=0.08]
    dated=[r for r in usable if r.get("published")]
    official=[r for r in usable if r.get("source_type")=="official"]
    sufficient=len(usable)>=2
    reason="관련 결과가 부족합니다." if not sufficient else "관련 검색 결과를 확보했습니다."
    if freshness_required and not dated and not official:
        sufficient=False;reason="최신성을 확인할 날짜나 공식 출처가 충분하지 않습니다."
    return {"sufficient":sufficient,"reason":reason,"count":len(results),"usable_count":len(usable),"dated_count":len(dated),"official_count":len(official)}
def search_with_validation(question:str,proposed_query:str,search_fn,now=None,max_results:int=8)->dict:
    plan=plan_search(question,proposed_query,now=now)
    first=search_fn(plan["query"],max_results=max_results,time_range=plan["time_range"])
    results=prepare_results(plan["query"],first,max_results)
    assessment=assess_results(results,plan["freshness_required"])
    attempts=[{"query":plan["query"],"count":len(results),"assessment":assessment}]
    if not assessment["sufficient"]:
        retry=build_retry_query(plan["query"],plan["current_year"])
        if retry!=plan["query"]:
            second=search_fn(retry,max_results=max_results,time_range=plan["time_range"])
            results=prepare_results(retry,results+second,max_results)
            assessment=assess_results(results,plan["freshness_required"])
            attempts.append({"query":retry,"count":len(second),"assessment":assessment})
    return {"results":results,"plan":plan,"assessment":assessment,"attempts":attempts}
