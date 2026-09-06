#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm điểm CHẤT LƯỢNG và TỐC ĐỘ của từng model trên cùng một bộ dữ liệu.

Trả lời bằng số đo hai câu hỏi mà mọi dự án NLP đều phải trả lời sớm:

    1. Model tự huấn luyện có đủ tốt so với gọi thẳng một LLM không?
    2. GPU nhanh hơn CPU bao nhiêu lần trên đúng bài toán của mình?

Khác với `/v1/ner/compare` (chỉ đo hai model KHỚP NHAU tới đâu), ở đây có nhãn
người gán làm chân lý, nên con số là ĐỘ CHÍNH XÁC thật.

Cách dùng:
    python3 benchmarks/run_benchmark.py                       # tất cả model sẵn sàng
    python3 benchmarks/run_benchmark.py --models phobert-ner-gpu groq/compound
    python3 benchmarks/run_benchmark.py --repeat 5 --out results.json

Yêu cầu: hệ thống đang chạy (`docker compose up -d`). Script chỉ dùng thư viện
chuẩn của Python, không cần cài gì thêm.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_API = "http://localhost:8080"
LABELS = ("PER", "ORG", "LOC")
Span = Tuple[int, int, str]


def post(api: str, path: str, body: dict, timeout: float = 120):
    req = urllib.request.Request(api + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def spans(entities: List[dict]) -> Set[Span]:
    return {(e["start"], e["end"], e["label"]) for e in entities if e.get("start", -1) >= 0}


def prf(matched: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    """Precision, recall, F1. Quy ước 0/0 = 0 giống seqeval."""
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(api: str, model: str, gold: List[dict], repeat: int,
             is_llm: bool = False, llm_delay: float = 1.0) -> Dict:
    """Chạy toàn bộ tập gold qua một model, trả về điểm chất lượng và độ trễ.

    Độ trễ đo bằng chính con số máy chủ báo về (`latency_ms`), tức thời gian xử
    lý phía server, không tính đường mạng từ máy chạy script. Đó là con số so
    sánh được giữa các model; muốn biết người dùng cuối cảm nhận bao lâu thì
    xem phần đo tải trong `load_test/`.
    """
    # Với dịch vụ tính tiền theo lượt gọi, lặp lại chỉ để đo độ trễ là đốt tiền
    # và đốt hạn mức. Gọi đúng một lần cho mỗi câu.
    if is_llm:
        repeat = 1
    latencies: List[float] = []
    matched = pred_total = gold_total = 0
    per_label = {lab: {"matched": 0, "pred": 0, "gold": 0} for lab in LABELS}
    per_group: Dict[str, Dict[str, int]] = {}
    errors: List[str] = []
    mistakes: List[dict] = []

    for sample in gold:
        gold_spans = spans(sample["entities"])
        pred_spans: Set[Span] = set()

        for attempt in range(repeat):
            try:
                res = post(api, "/v1/ner", {"text": sample["text"], "model": model})
            except urllib.error.HTTPError as e:
                errors.append(f"{sample['id']}: HTTP {e.code} {e.read()[:160]!r}")
                break
            except Exception as e:                       # noqa: BLE001
                errors.append(f"{sample['id']}: {type(e).__name__} {e}")
                break
            latencies.append(res["latency_ms"])
            if attempt == 0:
                pred_spans = spans(res["entities"])
        # Nghỉ giữa các câu khi gọi dịch vụ ngoài. Hạn mức miễn phí tính theo
        # số request mỗi phút, chạy liên tục là chạm trần ngay. Phía máy chủ đã
        # có cơ chế thử lại, nhưng chờ sẵn ở đây vẫn nhanh hơn là để bị chặn rồi
        # mới lùi.
        if is_llm and llm_delay > 0:
            time.sleep(llm_delay)

        hit = gold_spans & pred_spans
        matched += len(hit)
        pred_total += len(pred_spans)
        gold_total += len(gold_spans)

        group = sample["group"]
        bucket = per_group.setdefault(group, {"matched": 0, "pred": 0, "gold": 0})
        bucket["matched"] += len(hit)
        bucket["pred"] += len(pred_spans)
        bucket["gold"] += len(gold_spans)

        for lab in LABELS:
            per_label[lab]["matched"] += len({s for s in hit if s[2] == lab})
            per_label[lab]["pred"] += len({s for s in pred_spans if s[2] == lab})
            per_label[lab]["gold"] += len({s for s in gold_spans if s[2] == lab})

        if gold_spans != pred_spans:
            text = sample["text"]
            mistakes.append({
                "id": sample["id"], "group": group, "text": text,
                "bỏ sót": sorted(f"{text[a:b]} [{lab}]" for a, b, lab in gold_spans - pred_spans),
                "thừa": sorted(f"{text[a:b]} [{lab}]" for a, b, lab in pred_spans - gold_spans),
            })

    p, r, f1 = prf(matched, pred_total, gold_total)
    return {
        "model": model,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "matched": matched, "predicted": pred_total, "gold": gold_total,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 2) if latencies else None,
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "per_label": {lab: dict(zip(("precision", "recall", "f1"),
                                    (round(x, 4) for x in prf(v["matched"], v["pred"], v["gold"]))))
                      for lab, v in per_label.items()},
        "per_group": {g: round(prf(v["matched"], v["pred"], v["gold"])[2], 4)
                      for g, v in per_group.items()},
        "errors": errors[:10],
        "mistakes": mistakes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Mặc định: model tự host, cộng model LLM mặc định nếu có.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Số lần gọi mỗi câu để lấy độ trễ ổn định (mặc định 3). "
                             "Chất lượng luôn lấy từ lần đầu.")
    parser.add_argument("--llm-delay", type=float, default=2.0,
                        help="Giây nghỉ giữa hai lần gọi LLM, tránh chạm hạn mức "
                             "tần suất của gói miễn phí (mặc định 2).")
    parser.add_argument("--gold", default=str(Path(__file__).with_name("gold_vi.json")))
    parser.add_argument("--out", default=None, help="Ghi kết quả đầy đủ ra file JSON.")
    parser.add_argument("--show-mistakes", type=int, default=5,
                        help="In ra bao nhiêu câu sai của mỗi model (0 = không in).")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    try:
        info = json.load(urllib.request.urlopen(args.api + "/v1/models", timeout=10))
    except Exception as e:                                # noqa: BLE001
        print(f"Không gọi được {args.api}/v1/models: {e}")
        print("Hệ thống đã chạy chưa? Thử: docker compose up -d")
        return 1

    ready = {m["name"]: m["runtime"] for m in info["models"] if m["ready"]}
    if args.models:
        models = args.models
    else:
        # Mặc định: mọi model tự host, cộng ĐÚNG MỘT model LLM. Chạy hết danh
        # sách model của nhà cung cấp sẽ đốt sạch hạn mức mà không thêm thông tin.
        local = [n for n, rt in ready.items() if rt != "llm"]
        llm = next((n for n, rt in ready.items() if rt == "llm"), None)
        models = local + ([llm] if llm else [])
    skipped = [m for m in models if m not in ready]
    models = [m for m in models if m in ready]
    if skipped:
        print(f"Bỏ qua (chưa sẵn sàng): {', '.join(skipped)}\n")
    if not models:
        print("Không có model nào sẵn sàng.")
        return 1

    print(f"Bộ dữ liệu: {len(gold)} câu, "
          f"{sum(len(s['entities']) for s in gold)} thực thể.")
    print("LƯU Ý: bộ này sinh ra bằng công cụ AI trong lúc dựng repo, không qua kiểm")
    print("chứng chéo. Dùng để soi dạng lỗi, KHÔNG dùng để công bố năng lực model.")
    print(f"Mỗi câu gọi {args.repeat} lần để đo độ trễ.\n")

    results = []
    for model in models:
        print(f"  đang chạy {model} ...", flush=True)
        results.append(evaluate(args.api, model, gold, args.repeat,
                                ready[model] == "llm", args.llm_delay))

    # ---- Bảng tổng hợp ----
    print("\n" + "=" * 92)
    print("CHẤT LƯỢNG (khớp nghiêm: đúng cả vị trí lẫn loại) và TỐC ĐỘ")
    print("=" * 92)
    print(f"{'model':19s} {'P':>7s} {'R':>7s} {'F1':>7s} | "
          f"{'p50':>9s} {'p90':>9s} | {'PER':>6s} {'ORG':>6s} {'LOC':>6s}")
    print("-" * 92)
    for r in results:
        pl = r["per_label"]
        p50 = f"{r['latency_p50_ms']:.1f}ms" if r["latency_p50_ms"] else "-"
        p90 = f"{r['latency_p90_ms']:.1f}ms" if r["latency_p90_ms"] else "-"
        print(f"{r['model']:19s} {r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} | "
              f"{p50:>9s} {p90:>9s} | "
              f"{pl['PER']['f1']:6.3f} {pl['ORG']['f1']:6.3f} {pl['LOC']['f1']:6.3f}")

    # ---- F1 theo nhóm tình huống: chỗ lộ ra điểm yếu thật ----
    groups = sorted({g for r in results for g in r["per_group"]})
    print("\n" + "=" * 92)
    print("F1 THEO NHÓM TÌNH HUỐNG (nhóm nào tụt là điểm yếu thật của hệ thống)")
    print("=" * 92)
    print(f"{'model':19s} " + " ".join(f"{g:>12s}" for g in groups))
    print("-" * 92)
    for r in results:
        print(f"{r['model']:19s} "
              + " ".join(f"{r['per_group'].get(g, float('nan')):12.3f}" for g in groups))

    if args.show_mistakes:
        print("\n" + "=" * 92)
        print("VÍ DỤ SAI (đọc phần này quan trọng hơn nhìn con số F1)")
        print("=" * 92)
        for r in results:
            print(f"\n### {r['model']}  ({len(r['mistakes'])}/{len(gold)} câu chưa khớp hoàn toàn)")
            for m in r["mistakes"][:args.show_mistakes]:
                print(f"  [{m['group']}] {m['text'][:88]}")
                if m["bỏ sót"]:
                    print(f"      bỏ sót : {', '.join(m['bỏ sót'])}")
                if m["thừa"]:
                    print(f"      thừa   : {', '.join(m['thừa'])}")

    for r in results:
        if r["errors"]:
            print(f"\nLỗi khi chạy {r['model']}:")
            for e in r["errors"]:
                print("   ", e)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nĐã ghi kết quả đầy đủ vào {args.out}")

    print("\nĐọc kèm benchmarks/eval_testset.py (tập test WikiAnn 10.000 câu). Hai bộ cho")
    print("kết quả NGƯỢC NHAU về thứ hạng model, và lý do nằm ở cách gán nhãn của bộ dữ")
    print("liệu chứ không ở model. Giải thích đầy đủ: benchmarks/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
