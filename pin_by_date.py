#!/usr/bin/env python3
import sys, datetime, json, urllib.request

ALIASES = {
    "pyyaml": "PyYAML",
    "pillow": "Pillow",
    "torch-geometric": "torch-geometric",
    "torch-scatter": "torch-scatter",
    # add more aliases if needed
}

def latest_on_or_before(pkg: str, cutoff: datetime.datetime) -> str:
    name = ALIASES.get(pkg.lower(), pkg)
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    best_ver, best_time = None, None
    for ver, files in data.get("releases", {}).items():
        for f in files:
            up = f.get("upload_time_iso_8601") or f.get("upload_time")
            if not up:
                continue
            t = datetime.datetime.fromisoformat(up.replace("Z","+00:00"))
            if t <= cutoff and (best_time is None or t > best_time):
                best_ver, best_time = ver, t
    return best_ver or data["info"]["version"]

def main():
    if len(sys.argv) != 3:
        print("Usage: pin_by_date.py requirements_base.txt YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    base = sys.argv[1]
    cutoff = datetime.datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    with open(base) as f:
        pkgs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    special = {"torch", "torchvision", "torch-geometric", "torch-scatter",
               "torch-sparse", "torch-cluster", "torch-spline-conv", "pyg-lib"}
    for p in pkgs:
        try:
            if p in special:
                # Leave special stack unpinned here; we’ll handle it explicitly later
                print(p)
            else:
                ver = latest_on_or_before(p, cutoff)
                print(f"{p}=={ver}")
        except Exception as e:
            print(f"# Could not pin {p}: {e}")
            print(p)
if __name__ == "__main__":
    main()
