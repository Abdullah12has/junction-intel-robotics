"""Add the open-source Desktop Commander (local, no call quota) to the Claude desktop app's MCP servers.

Backs up the config first, keeps every existing key, and only adds mcpServers["desktop-commander"] if absent.
The app reads the file at start-up, so the server appears after the next app restart.
"""
import json
import shutil
import subprocess
import time

CFG = "/home/intel/.config/Claude/claude_desktop_config.json"
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
backup = f"{CFG}.bak-{stamp}"
shutil.copy2(CFG, backup)
cfg = json.load(open(CFG))
servers = cfg.setdefault("mcpServers", {})
print("existing mcpServers:", sorted(servers))
if "desktop-commander" not in servers:
    servers["desktop-commander"] = {"command": "npx", "args": ["-y", "@wonderwhy-er/desktop-commander@latest"]}
tmp = CFG + ".tmp"
with open(tmp, "w") as f:
    json.dump(cfg, f, indent=2)
json.load(open(tmp))  # must parse before it replaces the live file
shutil.move(tmp, CFG)
print("backup:", backup)
print("mcpServers now:", json.dumps(cfg["mcpServers"]))
r = subprocess.run(["timeout", "180", "npx", "-y", "@wonderwhy-er/desktop-commander@latest", "--version"],
                   capture_output=True, text=True)
print("npx prefetch rc", r.returncode, (r.stdout + r.stderr).strip()[-400:])
