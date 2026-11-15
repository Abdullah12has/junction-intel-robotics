import sys, json, time, subprocess, cv2, numpy as np
sys.path.insert(0, '/home/intel/intel-junction-2026/app/tools')
import grasp_auto as ga
def tips(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H >= 165) | (H <= 3)) & (S >= 90) & (V >= 60)).astype(np.uint8); red[:200] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    out = []
    for i in range(1, n):
        if st[i, 4] > 1500 and st[i, 1] + st[i, 3] >= img.shape[0] - 3:
            ys, xs = np.nonzero(lab == i); top = ys.min(); sel = ys <= top + 8
            out.append((round(float(xs[sel].mean()), 1), int(top), int(st[i, 4])))
    return sorted(out)
ref = cv2.imread('/home/intel/intel-junction-2026/logs/depth/jprobe/164223/base/live_wrist.jpg')
print('ref tips', tips(ref))
for g in (25, 30, 35, 40, 45, 25):
    ga.cmd({"op": "grip", "value": float(g), "speed": 20}); time.sleep(0.9)
    st = ga.state(); img = ga.fresh('wrist', time.time() + 0.2)
    cv2.imwrite('/home/intel/intel-junction-2026/logs/depth/servo/grip_%d.jpg' % g, img)
    print('grip', g, 'present', st['present'][5], 'tips', tips(img))
