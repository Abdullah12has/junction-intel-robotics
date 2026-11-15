import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = 'firsts.extend(arm_speed(E["S"], ds["fps"])[:15].tolist())'
b = 'firsts.append(float(np.median(arm_speed(E["S"], ds["fps"])[1:15])))'
c = 'return round(max(0.5, 2.0 * float(np.percentile(firsts, 95))), 3), len(firsts)'
d = 'return round(max(0.5, 3.0 * float(np.median(firsts))), 3), len(firsts)'
e = 'f"2 x p95 of arm speed over the first 15 frames of {n_first} episode starts, floor 0.5"'
f = 'f"3 x median over episodes of per-episode median arm speed (frames 1-14), floor 0.5; n={n_first}"'
assert all(x in s for x in (a, c, e)), "pattern missing"
s = s.replace(a, b).replace(c, d).replace(e, f)
open(p, "w", encoding="utf-8").write(s)
print("PATCHED", p)
