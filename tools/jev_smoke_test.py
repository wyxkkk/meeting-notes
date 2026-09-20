#!/usr/bin/env python3
"""Small Chinese Jev capability test. Only synthetic examples are sent.

Usage: configure TYPESAFE_API_KEY, then run python3 tools/jev_smoke_test.py
Credentials come only from TYPESAFE_API_KEY and are never saved.
This script makes one models request and three paid inference requests.
"""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE = "https://api.typesafe.ai"
PRICE_PER_MILLION_INPUT = 0.042  # Official models documentation, 2026-09-20.


def make_groups():
    tickets = {
        "c1": "同一笔订单被扣了两次款，请退回多扣的钱。",
        "c2": "从早上开始，登录接口一直返回500，所有用户都无法登录。",
        "c3": "我们想购买企业版，想了解50个席位的报价。",
        "c4": "包裹物流显示签收，但我还没有收到，请帮我查一下。",
        "c5": "还是不行，你们看着办吧。",
    }
    categories = {
        "billing": "扣款、发票、退款等账务问题。",
        "technical": "软件错误、接口故障、无法登录等技术问题。",
        "sales": "购买前咨询、报价、升级套餐。",
        "shipping": "物流、配送、收货问题。",
        "other_or_unknown": "其他主题，或缺少足够信息判断所属部门。",
    }
    choice_questions = {
        k: {
            "type": "choice",
            "instructions": f"只看 `tickets.{k}` 这条工单，它应该分配给哪个部门？不要用其他工单补全信息。",
            "criteria": categories,
        }
        for k in tickets
    }
    bugs = {
        "s1": "设置页按钮向右偏了2像素，所有功能都正常。",
        "s2": "PDF导出失败，但CSV导出正常，用户可以先导出CSV再转换。",
        "s3": "所有用户都无法登录，重试和换浏览器也没用，目前没有替代办法。",
        "s4": "不是完全无法导出，只是PDF格式失败。CSV仍能正常导出，可以先用CSV转换成所需格式。",
    }
    levels = [
        "仅外观问题，功能正常，不妨碍用户完成任务。",
        "功能故障或降级，但存在可用的替代办法完成任务。",
        "功能完全阻断用户任务，而且没有可用的替代办法。",
    ]
    score_questions = {
        k: {
            "type": "score",
            "instructions": f"只根据 `bugs.{k}`，评价该故障对用户完成任务的阻碍程度。",
            "criteria": levels,
        }
        for k in bugs
    }
    document = (
        "产品发布说明：新版本已于9月20日向10%的用户灰度发布，"
        "计划9月27日向全部用户开放。目前支持Windows和macOS，Linux尚未支持。"
        "企业版支持单点登录，个人版不支持单点登录。"
        "本说明未披露付费客户数量。"
    )
    claims = {
        "n1": "新版本已经向部分用户开放。",
        "n2": "新版本已经向全部用户开放。",
        "n3": "目前可以在Linux上使用。",
        "n4": "个人版目前不支持单点登录。",
        "n5": "该产品已经有一万家付费客户。",
        "n6": "Windows和macOS目前都受到支持。",
    }
    noul_questions = {
        k: {
            "type": "noul",
            "instructions": f"`document` 是否支持 `claims.{k}` 的说法？只判断所给材料是否支持，不判断现实世界的真伪。",
            "criteria": {
                "true": "材料明确支持该说法，或者该说法是材料的直接同义改写。",
                "false": "材料与该说法矛盾，或没有提供足够证据支持该说法。",
            },
        }
        for k in claims
    }
    return [
        {
            "name": "中文工单分类", "type": "choice",
            "state": {"tickets": tickets}, "questions": choice_questions,
            "expected": dict(zip(tickets, ["billing", "technical", "sales", "shipping", "other_or_unknown"])),
        },
        {
            "name": "故障严重程度", "type": "score",
            "state": {"bugs": bugs}, "questions": score_questions,
            "expected": dict(zip(bugs, [0, 1, 2, 1])),
        },
        {
            "name": "材料支持判断", "type": "noul",
            "state": {"document": document, "claims": claims}, "questions": noul_questions,
            "expected": dict(zip(claims, [True, False, False, True, False, True])),
        },
    ]


def request(path, key, payload=None):
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from None
    except Exception as exc:
        raise RuntimeError(str(exc).replace(key, "[REDACTED]")) from None
    return result, round((time.perf_counter() - start) * 1000, 2)


def validate_and_grade(group, response):
    answers = response["answers"]
    if set(answers) != set(group["questions"]):
        raise ValueError("Returned question IDs do not match the request")
    checks = []
    for qid, expected in group["expected"].items():
        answer = answers[qid]
        if answer["type"] != group["type"]:
            raise ValueError("Returned answer type does not match the question")
        if answer["type"] == "noul":
            value = answer["noul"]
            assert 0 <= value <= 1
            observed = value >= 0.5
            numeric = value
        else:
            probs = answer["probabilities"]
            assert all(0 <= p <= 1 for p in probs.values())
            assert abs(sum(probs.values()) - 1) <= 0.02
            assert 0 <= answer["confidence"] <= 1
            if answer["type"] == "choice":
                assert set(probs) == set(group["questions"][qid]["criteria"])
                observed = answer["choice"]
                assert probs[observed] >= max(probs.values()) - 1e-6
                numeric = probs[observed]
            else:
                levels = group["questions"][qid]["criteria"]
                assert set(probs) == {str(i) for i in range(len(levels))}
                numeric = answer["score"]
                assert 0 <= numeric <= len(levels) - 1
                assert abs(numeric - sum(int(k) * v for k, v in probs.items())) <= 0.03
                observed = int(max(probs, key=probs.get))
        checks.append({
            "id": qid, "expected": expected, "observed": observed,
            "value": numeric, "matches_expected": observed == expected,
        })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write request examples without API calls")
    parser.add_argument("--out", type=Path, default=Path("runs/jev-smoke/results.json"))
    args = parser.parse_args()
    groups = make_groups()
    report = {
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "kind": "synthetic_smoke_test_not_benchmark",
        "model_requested": "jev-latest",
        "input_price_usd_per_million": PRICE_PER_MILLION_INPUT,
        "pricing_source": "https://docs.typesafe.ai/models",
        "grading": "Choice: exact label; Score: most probable level; Noul: threshold 0.5. Expectations set before calling API; not included in requests.",
        "groups": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report["groups"] = groups
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("Dry run: 3 requests / 15 questions prepared; no network calls.")
        return
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set; configure it in your environment first.")
    report["available_models"], report["models_request_ms"] = request("/v1/models", key)
    print("Authentication OK; models:", json.dumps(report["available_models"], ensure_ascii=False))
    for group in groups:
        payload = {"model": "jev-latest", "state": group["state"], "questions": group["questions"]}
        response, elapsed = request("/v1/systemone", key, payload)
        checks = validate_and_grade(group, response)
        record = {"name": group["name"], "elapsed_ms": elapsed, "request": payload, "response": response, "checks": checks}
        report["groups"].append(record)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"group": group["name"], "model": response["model"], "elapsed_ms": elapsed, "usage": response.get("usage"), "checks": checks}, ensure_ascii=False))
    records = report["groups"]
    input_tokens = sum(r["response"]["usage"]["input_tokens"] for r in records)
    output_tokens = sum(r["response"]["usage"]["output_tokens"] for r in records)
    report["summary"] = {
        "inference_requests": len(records),
        "questions": sum(len(r["checks"]) for r in records),
        "matches_expected": sum(c["matches_expected"] for r in records for c in r["checks"]),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "estimated_usd": input_tokens / 1_000_000 * PRICE_PER_MILLION_INPUT,
        "inference_total_ms": round(sum(r["elapsed_ms"] for r in records), 2),
        "inference_median_ms": statistics.median(r["elapsed_ms"] for r in records),
        "timing_note": "Client wall time includes networking, TLS and API latency; each group uses a fresh connection. Three requests are not a latency benchmark.",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Summary:", json.dumps(report["summary"], ensure_ascii=False))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
