"""Sorting-cell state machine over Studio's runtime WebSocket.

Protocol checked against schemas/runtime_session.py and api/runtime_ws.py of the backend at 17714488.
The `check` and `home` modes have not yet run against the live backend; `run` has never moved the arm.

Close Studio's own runtime page first: only one session may own the arms. A person stays at the
follower's power switch for every run; Ctrl-C only holds the pose.

Run from ~/junction-cell with the anomalib venv:
  python cell/run_cell.py check      # load the environment, print joints and camera keys; no motion
  python cell/run_cell.py home       # park the arm at HOME by teleop first; saves the joint values
  python cell/run_cell.py run 10     # autonomous cycles, one trial row each
Settings live in cell/config.json (copy cell/config.example.json).
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

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
PID = CFG["project_id"]
WS = f"ws://127.0.0.1:7860/api/projects/{PID}/runtime/ws?hz=20"
T_INSPECT, T_GOOD, T_BAD = CFG["tasks"]["inspect"], CFG["tasks"]["good"], CFG["tasks"]["bad"]
HOME_TOL, MIN_S, TIMEOUT_S = 3.0, 4.0, 40.0  # HOME_TOL in Studio's normalised joint units


class Studio:
    def __init__(self, ws):
        self.ws, self.acks, self.state, self.loaded, self.cameras = ws, {}, None, None, []

    async def pump(self):
        async for m in self.ws:
            if isinstance(m, bytes):  # [u32 header length][ObservationMessage JSON][JPEGs]
                n = int.from_bytes(m[:4], "big")
                h = json.loads(m[4:4 + n])
                if h.get("state"):
                    self.state = h["state"]  # {joint name: value}
                continue
            j = json.loads(m)
            t = j.get("type")
            if t == "ack" and j["request_id"] in self.acks:
                self.acks.pop(j["request_id"]).set_result(j)
            elif t == "state" and j["data"].get("environment"):
                self.cameras = [c["key"] for c in j["data"]["environment"]["cameras"]]
            elif t == "error" or (t == "status" and j.get("state") == "failed"):
                print("STUDIO", j)
                if t == "error" and not j.get("session_alive", True):
                    raise RuntimeError("Studio ended the session")

    async def cmd(self, command, **kw):
        rid = uuid.uuid4().hex
        self.acks[rid] = fut = asyncio.get_running_loop().create_future()
        await self.ws.send(json.dumps({"command": command, "request_id": rid, **kw}))
        a = await asyncio.wait_for(fut, 120)
        if not a["ok"]:
            raise RuntimeError(f"{command}: {a.get('error')}")
        return a.get("result")

    async def wait_state(self, timeout=10):
        t0 = time.monotonic()
        while self.state is None:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no joint state from Studio")
            await asyncio.sleep(0.05)

    def at_home(self):
        home = CFG["home"]
        return max(abs(self.state[k] - v) for k, v in home.items()) < HOME_TOL

    async def skill(self, task):
        """Run one policy skill until the arm is back at HOME. True on completion, False on timeout."""
        model_id = CFG["models"][task]
        if self.loaded != model_id:  # ACT fallback swaps models; a single SmolVLA loads once
            await self.cmd("load_model", model_id=model_id, backend="openvino", device=CFG["device"])
            self.loaded = model_id
        await self.cmd("set_control", control={"type": "model", "task": task, "hz": CFG["fps"],
                                               "execution": {"mode": CFG.get("execution", "sync"),
                                                             "request_threshold": 0.5},
                                               "grace_s": 1.0})  # each set_control starts fresh policy state
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < TIMEOUT_S:
                await asyncio.sleep(0.05)
                if time.monotonic() - t0 > MIN_S and self.at_home():
                    return True
            return False
        finally:
            await self.cmd("set_control", control=None)  # hold pose


def heatmap(c, amap, path):
    a = cv2.normalize(np.asarray(amap, np.float32).squeeze(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    a = cv2.resize(a, (c.shape[1], c.shape[0]))
    cv2.imwrite(path, np.hstack([c, cv2.addWeighted(c, 0.5, cv2.applyColorMap(a, cv2.COLORMAP_JET), 0.5, 0)]))


async def cycle(st, th):
    """-> dict for the trial log. One retry per stage, then FAULT."""
    row = {"t_start": round(time.time(), 3), "retries": 0, "fault": ""}
    t0 = time.monotonic()
    for _ in range(2):                                        # stage 1: brick must arrive on the mark
        ok = await st.skill(T_INSPECT)
        await asyncio.sleep(0.3)                              # settle
        c = inspection.crop(inspection.grab())
        if ok and inspection.present(c):
            break
        row["retries"] += 1
    else:
        return row | {"fault": "no brick at inspection spot"}
    row["pick_s"] = round(time.monotonic() - t0, 2)
    verdict, s, ms, amap = inspection.classify(c, th)
    heatmap(c, amap, f"evidence/crops/{int(row['t_start'])}_{verdict}.png")
    row |= {"verdict": verdict, "score": round(s, 4), "inspect_ms": round(ms, 1)}
    task = T_GOOD if verdict == "good" else T_BAD             # ambiguous -> red bowl: a false reject beats a false accept
    for _ in range(2):                                        # stage 2: brick must leave the mark
        ok = await st.skill(task)
        await asyncio.sleep(0.3)
        if ok and not inspection.present(inspection.crop(inspection.grab())):
            break
        row["retries"] += 1
    else:
        return row | {"fault": "brick still at inspection spot"}
    return row | {"destination": "green" if task == T_GOOD else "red", "total_s": round(time.monotonic() - t0, 2)}


FIELDS = ["t_start", "verdict", "score", "inspect_ms", "destination", "pick_s", "total_s", "retries", "fault",
          "true_label", "correct"]  # last two filled by a human: the script must not grade itself


async def main(mode, n):
    env = json.loads(os.popen(f"curl -s http://127.0.0.1:7860/api/projects/{PID}/environments/{CFG['environment_id']}").read())
    async with websockets.connect(WS, max_size=None) as ws:
        st = Studio(ws)
        pump = asyncio.create_task(st.pump())
        await st.cmd("load_environment", environment=env, unload_incompatible=True)
        await st.wait_state()
        if mode == "check":
            await asyncio.sleep(1)
            print("joints:", json.dumps(st.state))
            print("camera keys:", st.cameras)
        elif mode == "home":
            CFG["home"] = st.state
            json.dump(CFG, open(os.path.join(os.path.dirname(__file__), "config.json"), "w"), indent=2)
            print("saved HOME:", st.state)
        else:
            assert CFG.get("home"), "run `home` first"
            th = json.load(open("thresholds.json"))
            os.makedirs("evidence/crops", exist_ok=True)
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
        await st.cmd("unload_environment")
        pump.cancel()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "check", int(sys.argv[2]) if len(sys.argv) > 2 else 10))
