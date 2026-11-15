"""Command line of loop_v2. Run through tools/run_policy.py, which puts the app folder on sys.path under python3 -I.

  scenarios                    list the dry-run scenarios
  dry SCENARIO LOGS_ROOT       one dry-run policy cycle with fakes (no Studio, camera, port or robot)
  live RULES                   live policy run; refused unless the presence flag exists and GO is typed
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loop_v2")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenarios")
    dry = commands.add_parser("dry")
    dry.add_argument("scenario")
    dry.add_argument("logs_root")
    live = commands.add_parser("live")
    live.add_argument("rules")
    args = parser.parse_args(argv)

    from loop.policy_fakes import SCENARIOS, run_dry

    if args.command == "scenarios":
        for name in sorted(SCENARIOS):
            print(f"{name}: {SCENARIOS[name].description} -> expect {SCENARIOS[name].expect}")
        return 0
    if args.command == "dry":
        if args.scenario not in SCENARIOS:
            parser.error(f"unknown scenario {args.scenario!r}; see the scenarios command")
        summary = asyncio.run(run_dry(args.scenario, Path(args.logs_root)))
        print(json.dumps(summary, sort_keys=True))
        return 0

    from loop.policy_live import LiveRefused, run_live

    try:
        answer = input("type GO to start the live policy run: ")
    except EOFError:
        answer = ""
    if answer.strip() != "GO":
        print(json.dumps({"reason": "refused", "message": "GO was not typed"}))
        return 2
    try:
        reason = asyncio.run(run_live(Path(args.rules)))
    except LiveRefused as error:
        print(json.dumps({"reason": "refused", "message": str(error)}))
        return 2
    print(json.dumps({"reason": reason}))
    return 0 if reason in ("max_cycles", "stopped", "quit", "no_brick") else 1
