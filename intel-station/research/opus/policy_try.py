"""Opus robot-slot tool: run one of OUR Studio policies for a bounded time and record what happens.

Usage: python policy_try.py MODEL_ID "task text" SECONDS [DEVICE]
1. Loads the two-camera + follower environment through Studio's runtime, saves both RealSense frames
   to OUT/pre_<key>.jpg and waits up to 180 s for the file R/run/policy_go (Opus checks the frames
   for hands first). No go -> unload and exit without motion.
2. On go: start_policy (backend openvino, explicit device, default GPU), log joint state at ~10 Hz and
   save frames every 1 s, stop after SECONDS (follower holds its pose), save post frames, unload.
A file R/run/policy_stop stops the policy early. Everything goes to R/logs/policy_try/<HHMMSS>/.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

R = Path("/home/intel/intel-junction-2026")
sys.path.insert(0, str(R / "app" / "loop"))
from studio_runtime_client import StudioRuntimeClient  # noqa: E402

PROJECT = "ab767473-d6fd-47ff-812e-8239c455dabb"
FOLLOWER = "e5600356-6296-496e-8344-17995e419baf"
CAMERAS = ["1d80a0dd-8f62-4f23-ae29-2f1fe406dde9", "b6e5da09-28f2-4208-9d15-526580074c75"]
DRAFT = {
    "name": "OPUS-POLICY-TRY",
    "robots": [{"robot_id": FOLLOWER, "tele_operator": {"type": "none"}}],
    "cameras": [{"camera_id": c} for c in CAMERAS],
}
GO = R / "run" / "policy_go"
STOP = R / "run" / "policy_stop"


def now() -> str:
    return time.strftime("%H:%M:%SZ", time.gmtime())


async def main(model: str, task: str, seconds: float, device: str) -> None:
    out = R / "logs" / "policy_try" / time.strftime("%H%M%S", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "w")

    def ev(text: str) -> None:
        line = f"{now()} {text}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    for f in (GO, STOP):
        f.unlink(missing_ok=True)
    states = []
    async with StudioRuntimeClient(PROJECT, hz=10) as c:
        ev(f"connected; environment={c.state.get('environment') is not None} model={c.state.get('model')}")
        await c.load_environment(DRAFT, wait_status=True)
        ev(f"environment loaded; failures={c.failures()}")
        keys = [cam["key"] for cam in (c.state.get("environment") or {}).get("cameras", [])]
        await c.wait_for_frames(keys, timeout_s=20)
        await asyncio.sleep(1.5)
        for k in keys:
            fr = c.latest_frame(k)
            if fr is not None:
                (out / f"pre_{k.replace(' ', '_')}.jpg").write_bytes(fr.jpeg)
        ev(f"PRE frames saved in {out}; waiting up to 180 s for {GO}")
        t0 = time.monotonic()
        while not GO.exists():
            if time.monotonic() - t0 > 180:
                ev("no go: unloading without motion")
                await c.unload_environment()
                return
            await asyncio.sleep(0.5)
        GO.unlink(missing_ok=True)
        ev(f"GO: start_policy model={model} device={device} task={task!r}")
        t_load = time.monotonic()
        await c.start_policy(model, task, backend="openvino", device=device, hz=30.0)
        ev(f"policy running (start took {time.monotonic() - t_load:.1f} s)")
        t1 = time.monotonic()
        last_frame = 0.0
        n = 0
        async for obs in c.observations():
            el = time.monotonic() - t1
            if obs.state:
                states.append({"t": round(el, 3), **{k: round(float(v), 2) for k, v in obs.state.items()}})
            if el - last_frame >= 1.0:
                last_frame = el
                for k in keys:
                    fr = c.latest_frame(k)
                    if fr is not None:
                        (out / f"f{n:03d}_{k.replace(' ', '_')}.jpg").write_bytes(fr.jpeg)
                n += 1
            if el >= seconds or STOP.exists():
                break
        ev(f"stopping after {time.monotonic() - t1:.1f} s ({len(states)} states)")
        await c.stop()
        ev("policy stopped; follower holds")
        await asyncio.sleep(1.0)
        for k in keys:
            fr = c.latest_frame(k)
            if fr is not None:
                (out / f"post_{k.replace(' ', '_')}.jpg").write_bytes(fr.jpeg)
        (out / "states.json").write_text(json.dumps(states))
        ev(f"errors={list(c.errors)[-5:]}")
        await c.unload_environment()
        ev("environment unloaded")
    STOP.unlink(missing_ok=True)


if __name__ == "__main__":
    dev = sys.argv[4] if len(sys.argv) > 4 else "GPU"
    asyncio.run(main(sys.argv[1], sys.argv[2], float(sys.argv[3]), dev))
