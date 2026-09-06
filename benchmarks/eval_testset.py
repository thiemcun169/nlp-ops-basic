#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm điểm trên TẬP TEST THẬT của WikiAnn, đi qua API.

Khác với `run_benchmark.py` (bộ nhỏ viết tay để soi tình huống), file này dùng
đúng tập test 10.000 câu mà notebook đã đánh giá lúc huấn luyện. Hai giá trị:

1. **Kiểm chứng đường phục vụ.** F1 ở đây phải xấp xỉ F1 mà notebook báo. Lệch
   nhiều nghĩa là khâu tiền xử lý lúc phục vụ khác lúc huấn luyện, và đó là lỗi
   phải sửa trước khi tin bất cứ con số nào khác.
2. **Đo trên dữ liệu đủ lớn.** Vài chục câu viết tay không đủ để công bố một con số.

Cách dùng:
    python3 benchmarks/eval_testset.py                       # model tự host, toàn bộ 10k câu
    python3 benchmarks/eval_testset.py --limit 500
    python3 benchmarks/eval_testset.py --models phobert-ner-gpu groq/compound --limit 200

Cần `datasets` để tải WikiAnn:  pip install datasets
"""
import argparse
import json
import random
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Set, Tuple

LABELS = ("PER", "ORG", "LOC")
Span = Tuple[int, int, str]


def post(api: str, body: dict, timeout: float = 180):
    req = urllib.request.Request(api + "/v1/ner", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def build_sample(tokens: List[str], tags: List[int], names: List[str]) -> Dict:
    """Ghép token thành câu, đồng thời tính offset ký tự cho từng thực thể.

    WikiAnn lưu dữ liệu ở dạng danh sách token kèm nhãn BIO. API thì nhận văn bản
    thô và trả span theo offset ký tự. Hàm này nối token bằng dấu cách và ghi lại
    vị trí từng token, nên nhãn vàng và nhãn dự đoán so được trực tiếp với nhau.

    Lưu ý khi đọc kết quả: câu ghép kiểu này có dấu câu ĐÃ tách sẵn, tức là dạng
    dễ nhất với model. Văn bản ngoài đời không như vậy.
    """
    text_parts, offsets, pos = [], [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        text_parts.append(tok)
        pos += len(tok) + 1
    text = " ".join(text_parts)

    spans: List[Span] = []
    current = None
    for (start, end), tag in zip(offsets, tags):
        name = names[tag]
        if name == "O":
            current = None
            continue
        etype = name.split("-")[-1]
        if name.startswith("B-") or current is None or current[2] != etype:
            current = [start, end, etype]
            spans.append(current)
        else:
            current[1] = end
    return {"text": text, "tokens": tokens, "spans": {(s, e, t) for s, e, t in spans}}


def count_retokenization_gap(samples: List[Dict]) -> Tuple[int, List[Tuple[str, str]]]:
    """Đếm số câu mà cách tách từ lúc phục vụ KHÁC cách tách của bộ dữ liệu.

    Đây là phép đo quan trọng nhất trong file này. WikiAnn đã tách sẵn token theo
    quy ước của nó; API thì nhận văn bản thô và tự tách lại bằng
    `services/api/app/core/tokenization.py`. Hai quy ước không trùng nhau hoàn
    toàn: `''` bị tách thành hai dấu nháy, `1905-1990` bị tách thành ba token.

    Mỗi chỗ lệch làm ranh giới thực thể xê dịch, và F1 tụt xuống mà KHÔNG PHẢI vì
    model kém. Không đo con số này thì rất dễ đổ oan cho model.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "services/api"))
    from app.core.tokenization import split_words

    mismatched, examples = 0, []
    for s in samples:
        got = [w.text for w in split_words(s["text"])]
        if got != s["tokens"]:
            mismatched += 1
            if len(examples) < 5:
                a = next((x for x, y in zip(s["tokens"], got) if x != y), "")
                examples.append((s["text"][:60], a))
    return mismatched, examples


def prf(matched: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(api: str, model: str, samples: List[Dict], delay: float) -> Dict:
    matched = pred_total = gold_total = 0
    per_label = {lab: {"m": 0, "p": 0, "g": 0} for lab in LABELS}
    latencies, errors = [], []
    worst: List[Dict] = []

    for i, sample in enumerate(samples, 1):
        try:
            res = post(api, {"text": sample["text"], "model": model})
        except urllib.error.HTTPError as e:
            errors.append(f"HTTP {e.code}: {e.read()[:120]!r}")
            if len(errors) > 20:
                print("    quá nhiều lỗi, dừng sớm."); break
            continue
        except Exception as e:                                  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            if len(errors) > 20:
                break
            continue

        latencies.append(res["latency_ms"])
        pred: Set[Span] = {(e["start"], e["end"], e["label"])
                           for e in res["entities"] if e.get("start", -1) >= 0}
        gold = sample["spans"]
        hit = gold & pred
        matched += len(hit)
        pred_total += len(pred)
        gold_total += len(gold)
        for lab in LABELS:
            per_label[lab]["m"] += len({s for s in hit if s[2] == lab})
            per_label[lab]["p"] += len({s for s in pred if s[2] == lab})
            per_label[lab]["g"] += len({s for s in gold if s[2] == lab})

        if gold != pred and len(worst) < 8:
            t = sample["text"]
            worst.append({
                "text": t,
                "bỏ sót": sorted(f"{t[a:b]} [{l}]" for a, b, l in gold - pred),
                "thừa": sorted(f"{t[a:b]} [{l}]" for a, b, l in pred - gold),
            })

        if delay:
            time.sleep(delay)
        if i % 500 == 0:
            print(f"    {i}/{len(samples)} câu ...", flush=True)

    p, r, f1 = prf(matched, pred_total, gold_total)
    return {
        "model": model, "n": len(latencies),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "per_label": {lab: round(prf(v["m"], v["p"], v["g"])[2], 4)
                      for lab, v in per_label.items()},
        "support": {lab: v["g"] for lab, v in per_label.items()},
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "errors": errors[:5], "worst": worst,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--models", nargs="*", default=["phobert-ner-gpu"])
    ap.add_argument("--limit", type=int, default=0,
                    help="Chỉ lấy N câu đầu (0 = toàn bộ 10.000). Bắt buộc dùng "
                         "khi chấm model LLM, nếu không sẽ hết hạn mức và rất tốn.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Hạt giống lấy mẫu, để chạy lại ra cùng tập câu.")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Giây nghỉ giữa hai request. Đặt 1.5 trở lên khi chấm LLM.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-mistakes", type=int, default=4)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Thiếu thư viện datasets. Cài bằng:  pip install datasets")
        return 1

    print("Đang tải tập test WikiAnn (tiếng Việt) ...", flush=True)
    ds = load_dataset("wikiann", "vi", trust_remote_code=True)["test"]
    names = ds.features["ner_tags"].feature.names

    samples = [build_sample(ex["tokens"], ex["ner_tags"], names) for ex in ds]
    if args.limit and args.limit < len(samples):
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.limit]

    total_gold = sum(len(s["spans"]) for s in samples)
    print(f"{len(samples)} câu, {total_gold} thực thể.")

    n_bad, examples = count_retokenization_gap(samples)
    print(f"Tách từ lệch so với bộ dữ liệu: {n_bad}/{len(samples)} câu "
          f"({n_bad / len(samples) * 100:.1f}%)")
    if examples:
        print("  ví dụ token bị tách khác:", ", ".join(repr(a) for _t, a in examples))
    print()

    results = []
    for model in args.models:
        print(f"  đang chấm {model} ...", flush=True)
        results.append(evaluate(args.api, model, samples, args.delay))

    print("\n" + "=" * 84)
    print(f"KẾT QUẢ TRÊN TẬP TEST WIKIANN ({len(samples)} câu)")
    print("=" * 84)
    print(f"{'model':20s} {'P':>7s} {'R':>7s} {'F1':>7s} | "
          f"{'PER':>6s} {'ORG':>6s} {'LOC':>6s} | {'p50':>9s}")
    print("-" * 84)
    for r in results:
        pl = r["per_label"]
        p50 = f"{r['latency_p50_ms']:.1f}ms" if r["latency_p50_ms"] else "-"
        print(f"{r['model']:20s} {r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} | "
              f"{pl['PER']:6.3f} {pl['ORG']:6.3f} {pl['LOC']:6.3f} | {p50:>9s}")

    sup = results[0]["support"]
    print(f"\nSố thực thể vàng: " + ", ".join(f"{k}={v}" for k, v in sup.items()))

    if args.show_mistakes:
        print("\n" + "=" * 84)
        print("VÍ DỤ SAI")
        print("=" * 84)
        for r in results:
            print(f"\n### {r['model']}")
            for m in r["worst"][:args.show_mistakes]:
                print(f"  {m['text'][:80]}")
                if m["bỏ sót"]:
                    print(f"      bỏ sót : {', '.join(m['bỏ sót'])}")
                if m["thừa"]:
                    print(f"      thừa   : {', '.join(m['thừa'])}")

    for r in results:
        if r["errors"]:
            print(f"\nLỗi khi chấm {r['model']}:")
            for e in r["errors"]:
                print("   ", e)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã ghi {args.out}")

    print("\nHai điều cần nhớ khi đọc bảng này:")
    print()
    print(f"1. Con số ở đây THẤP HƠN F1 mà notebook báo, và phần lớn chênh lệch đến từ")
    print(f"   {n_bad / len(samples) * 100:.1f}% số câu bị tách từ lệch so với bộ dữ liệu, chứ không phải model kém đi.")
    print("   Notebook chấm trên chính token của WikiAnn; API nhận văn bản thô và tự tách lại.")
    print("   Đây là cái giá của việc phục vụ văn bản thật, và nó có thật.")
    print()
    print("2. WikiAnn lấy từ Wikipedia: câu ngắn, viết hoa chuẩn, dấu câu tách sẵn. Đây là")
    print("   dạng dữ liệu DỄ NHẤT với model. So với benchmarks/run_benchmark.py (văn bản")
    print("   giống ngoài đời) để thấy model rớt bao nhiêu khi rời khỏi phân phối huấn luyện.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
