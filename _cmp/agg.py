import hashlib, os, glob
ROOT = r"C:\Users\vickey\Desktop\Rust-SWE-Bench-main"
def agg(base, rel_root):
    d = hashlib.sha256()
    for p in sorted(p for p in glob.glob(os.path.join(base, "**", "*.py"), recursive=True) if os.path.isfile(p)):
        d.update(os.path.relpath(p, rel_root).encode())
        d.update(open(p, "rb").read())
    return d.hexdigest()
# emulate: ROOT/baselines/test -> relpath from ROOT
for name in ["remote_test_patch", "remote_test_stage_20261115", "baselines/test"]:
    base = os.path.join(ROOT, name)
    rel_root = os.path.join(ROOT, "baselines", "test")
    d = hashlib.sha256()
    for p in sorted(p for p in glob.glob(os.path.join(base, "**", "*.py"), recursive=True) if os.path.isfile(p)):
        d.update(os.path.relpath(p, rel_root).encode())
        d.update(open(p, "rb").read())
    print(d.hexdigest(), name)
