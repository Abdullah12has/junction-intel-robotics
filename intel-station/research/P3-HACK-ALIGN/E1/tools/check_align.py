import json
import os
import re
import sys
from datetime import datetime, timezone

req_path = sys.argv[1]
outdir = sys.argv[2]

with open(req_path, encoding="utf-8") as f:
    data = json.load(f)


def norm(text):
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


cache = {}
rows = []
for r in data["requirements"]:
    quote = r["quote"]
    words = len(quote.split())
    check_file = r.get("check_file", "")
    if r["quote_check"] == "image":
        result = "image"
    elif not check_file or not os.path.exists(check_file):
        result = "not-here"
    else:
        if check_file not in cache:
            with open(check_file, encoding="utf-8", errors="replace") as f:
                cache[check_file] = norm(f.read())
        result = "ok" if norm(quote) in cache[check_file] else "MISMATCH"
    rows.append({
        "id": r["id"],
        "kind": r["kind"],
        "status": r["status"],
        "words": words,
        "word_ok": words <= 15,
        "check": result,
    })

countable = [x for x in rows if x["kind"] in ("requirement", "criterion", "recommended", "format")]
metrics = {"requirements_total": len(countable)}
for s in ("met", "partial", "missing", "unknown", "not_applicable"):
    metrics[s] = sum(1 for x in countable if x["status"] == s)
metrics["sum_check"] = sum(metrics[s] for s in ("met", "partial", "missing", "unknown", "not_applicable"))
metrics["entries_all"] = len(rows)
metrics["background_or_open"] = len(rows) - len(countable)
metrics["quotes_over_15_words"] = sum(1 for x in rows if not x["word_ok"])
metrics["quote_text_ok"] = sum(1 for x in rows if x["check"] == "ok")
metrics["quote_text_mismatch"] = sum(1 for x in rows if x["check"] == "MISMATCH")
metrics["quote_image_transcribed"] = sum(1 for x in rows if x["check"] == "image")
metrics["quote_not_checked_here"] = sum(1 for x in rows if x["check"] == "not-here")

utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
verify = {"stage": "align-v1", "utc": utc, "rows": rows}
payload = {"stage": "align-v1", "utc": utc, "metrics": metrics, "previous": None, "delta": None}

with open(os.path.join(outdir, "verify.json"), "w", encoding="utf-8") as f:
    json.dump(verify, f, ensure_ascii=False, indent=1)
with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)

print(json.dumps(payload, ensure_ascii=False))
bad = [x["id"] for x in rows if (not x["word_ok"]) or x["check"] == "MISMATCH"]
print("bad:", bad)
