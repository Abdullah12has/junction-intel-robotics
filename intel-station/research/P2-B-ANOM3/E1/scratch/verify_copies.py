"""Text check of the copies in R/app/anomaly/anom3/extract_crops.py against the sorter sources.

Nothing is imported. Each function or table is taken from its source file with ast and compared as text.
"""
import ast
import difflib

R = "/home/intel/intel-junction-2026"
OURS = R + "/app/anomaly/anom3/extract_crops.py"
SRC = {
    "pick_any": R + "/app/tools/pick_any.py",
    "sort_loop": R + "/app/tools/sort_loop.py",
    "sort_any": R + "/app/tools/sort_any.py",
}
PAIRS = [
    ("pick_any", "COLOUR_CLASSES", "COLOUR_CLASSES"),
    ("pick_any", "PLATE_HUES", "PLATE_HUES"),
    ("pick_any", "blobs", "blobs"),
    ("sort_loop", "mask_jaws", "mask_jaws"),
    ("sort_any", "close_pre_verdict", "close_pre_verdict"),
    ("sort_loop", "save_crop", "close_crop"),
]


def parse(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return text, ast.parse(text)


def segment(text, tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.get_source_segment(text, node)
    return None


def main():
    ours_text, ours_tree = parse(OURS)
    srcs = {k: parse(v) for k, v in SRC.items()}
    for mod, name, ours_name in PAIRS:
        text, tree = srcs[mod]
        a = segment(text, tree, name)
        b = segment(ours_text, ours_tree, ours_name)
        print("=" * 60)
        print("%s.%s  vs  extract_crops.%s" % (mod, name, ours_name))
        if a is None or b is None:
            print("MISSING: sorter=%s ours=%s" % (a is not None, b is not None))
            continue
        diff = list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "sorter", "ours", n=1))
        print("IDENTICAL" if not diff else "".join(diff))
    with open(R + "/app/tools/grasp_auto.py", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if "JAW_MASK" in line or "TARGET_PX" in line:
                print("grasp_auto.py:%d %s" % (i, line.rstrip()))


main()
