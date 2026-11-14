"""UNTESTED DRAFT: sorting-cell state machine over Studio's runtime WebSocket. Never run on the station.

Message shapes come from reading the pinned backend (application/backend/src/api/runtime_ws.py,
runtime_session/session.py, schemas/runtime_session.py). Verify each command against the live
backend before trusting it. Close Studio's own runtime page first: only one session may own the arm.
A person stays at the follower's power switch for every run; Ctrl-C only holds the pose.

Run in the anomalib venv (needs: websockets, opencv-python, numpy):  python cell/run_cell.py [n_cycles]
"""
import asyncio
import csv
import json
import os
import sys
import time
import uuid

import cv2
import numpy as np
import websockets

import inspection

PID = "ab767473-d6fd-47ff-812e-8239c455dabb"
FOLLOWER, LEADER = "e5600356-6296-496e-8344-17995e419baf", "d361855b-0f89-4b7c-ab27-2c12d6341ca5"
CAMERA_IDS = ["FILL: overhead id", "FILL: wrist id"]  # same cameras and names as the dataset
WS = f"ws://localhost:7860/api/projects/{PID}/runtime/ws?hz=20"
ENV = {"name": "sort", "robots": [{"robot_id": FOLLOWER, "tele_operator": {"type": "robot", "robot_id": LEADER}}],
       "cameras": [{"camera_id": c} for c in CAMERA_IDS]}

# Exact training strings. SmolVLA: one model for all three. ACT: one model id per string.
T_INSPECT, T_GOOD, T_BAD = ("move the brick to the inspection spot", "put the brick in the green bowl",
                            "put the brick in the red bowl")
MODEL = {T_INSPECT: "FILL", T_GOOD: "FILL", T_BAD: "FILL"}  # same id three times for SmolVLA
DEVICE = "CPU"            # SmolVLA: CPU until the GPU NaN check passes. ACT: "GPU".
HOME = None               # FILL: joint vector read from the state stream with the arm parked
HOME_TOL, MIN_S, TIMEOUT_S = 3.0, 4.0, 40.0


class Studio:
    def __init__(self, ws):
        self.ws, self.acks, self.state, self.loaded = ws, {}, None, None

    async def pump(self):
        async for m in self.ws:
            if isinstance(m, bytes):  # [u32 header length][JSON header][JPEGs]; we only need joint state
                n = int.from_bytes(m[:4], "big")
                h = json.loads(m[4:4 + n])
                if h.get("state"):
                    self.state = np.array(list(h["state"].values()), dtype=float)
            else:
                j = json.loads(m)
                if j.get("type") == "ack":
                    self.acks.pop(j["request_id"]).set_result(j)
                elif j.get("type") == "error" or j.get("state") == "failed":
                    print("STUDIO", j)

    async def cmd(self, command, **kw):
        rid = uuid.uuid4().hex
        self.acks[rid] = fut = asyncio.get_running_loop().create_future()
        await self.ws.send(json.dumps({"command": command, "request_id": rid, **kw}))
        a = await asyncio.wait_for(fut, 60)
        if not a["ok"]:
            raise RuntimeError(a.get("error"))

    async def skill(self, task):
        """Run one policy skill until the arm is back at HOME. True on completion, False on timeout."""
        if self.loaded != MODEL[task]:  # ACT fallback swaps models; SmolVLA never re-enters here
            await self.cmd("load_model", model_id=MODEL[task], backend="openvino", device=DEVICE)
            self.loaded = MODEL[task]
        await self.cmd("set_control", control={"type": "model", "task": task, "hz": 30,  # hz = dataset fps
                                               "execution": {"mode": "async", "request_threshold": 0.5},
                                               "grace_s": 1.0})  # each set_control starts with fresh policy state
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < TIMEOUT_S:
                await asyncio.sleep(0.05)
                if time.monotonic() - t0 > MIN_S and np.abs(self.state - HOME).max() < HOME_TOL:
                    return True
            return False
        finally:
            await self.cmd("set_control", control=None)  # hold pose


async def cycle(st, th):
    """-> dict for the trial log. One retry per stage, then FAULT."""
    row = {"t_start": time.time(), "retries": 0, "fault": ""}
    t0 = time.monotonic()
    for attempt in range(2):                                  # stage 1: brick must arrive on the mark
        await st.skill(T_INSPECT)
        await asyncio.sleep(0.3)                              # settle
        c = inspection.crop(inspection.grab())
        if inspection.present(c):
            break
        row["retries"] += 1
    else:
        return row | {"fault": "no brick at inspection spot"}
    row["pick_s"] = time.monotonic() - t0
    verdict, s, ms = inspection.classify(c, th)
    cv2.imwrite(f"evidence/crops/{int(row['t_start'])}_{verdict}.png", c)
    row |= {"verdict": verdict, "score": round(s, 4), "inspect_ms": round(ms, 1)}
    task = T_GOOD if verdict == "good" else T_BAD             # ambiguous -> red bowl
    for attempt in range(2):                                  # stage 2: brick must leave the mark
        await st.skill(task)
        await asyncio.sleep(0.3)
        if not inspection.present(inspection.crop(inspection.grab())):
            break
        row["retries"] += 1
    else:
        return row | {"fault": "brick still at inspection spot"}
    return row | {"destination": "green" if task == T_GOOD else "red", "total_s": round(time.monotonic() - t0, 2)}


FIELDS = ["t_start", "verdict", "score", "inspect_ms", "destination", "pick_s", "total_s", "retries", "fault",
          "true_label", "correct"]  # last two filled by a human: the script must not grade itself


async def main(n):
    assert HOME is not None, "set HOME first"
    th = json.load(open("thresholds.json"))
    os.makedirs("evidence/crops", exist_ok=True)
    async with websockets.connect(WS, max_size=None) as ws:
        st = Studio(ws)
        pump = asyncio.create_task(st.pump())
        await st.cmd("load_environment", environment=ENV, unload_incompatible=True)
        with open("evidence/trials.csv", "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if f.tell() == 0:
                w.writeheader()
            for i in range(n):
                input(f"[{i + 1}/{n}] brick in the pick zone, hands clear, then Enter ")
                row = await cycle(st, th)
                w.writerow(row)
                f.flush()
                print(row)
                if row["fault"]:
                    print("FAULT: holding pose. Fix by hand, then rerun.")
                    break
        pump.cancel()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
