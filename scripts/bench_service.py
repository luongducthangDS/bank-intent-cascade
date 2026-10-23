"""End-to-end latency of the real service (HTTP layer + embedding + LR), in-process, 1 request at a time."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load  # noqa: E402
from cascade.service import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
d = load()
client = TestClient(create_app())
client.post("/classify", json={"text": "warm up"})
lat, routes = [], []
for text in d.test.texts[:500]:
    t = time.perf_counter()
    r = client.post("/classify", json={"text": text}).json()
    lat.append((time.perf_counter() - t) * 1000)
    routes.append(r["route"])
out = {"requests": len(lat), "p50_ms": round(float(np.percentile(lat, 50)), 1),
       "p95_ms": round(float(np.percentile(lat, 95)), 1), "p99_ms": round(float(np.percentile(lat, 99)), 1),
       "throughput_rps_single_worker": round(1000 / float(np.mean(lat)), 1),
       "human_rate": round(routes.count("human") / len(routes), 4), "metrics_endpoint": client.get("/metrics").json()}
(ROOT / "reports/service_bench.json").write_text(json.dumps(out, indent=2))
print(out)
