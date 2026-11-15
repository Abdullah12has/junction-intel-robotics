import json, math
FK = json.load(open("/home/intel/intel-junction-2026/work/P2-D-SIM2/W2/out/fk_check.json"))
base = (0.0388353, 0.0388353)
plates = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}
for dest in ("blue", "green"):
    x, y, z = FK["real_place_tcp_base_xyz_median"][dest]["tcp_base_xyz_median"]
    wx, wy = y + base[0], -x + base[1]  # Rz(-90 deg): (x,y)->(y,-x), then base offset
    d = math.hypot(wx - plates[dest][0], wy - plates[dest][1])
    print("REAL_DROP_WORLD", dest, round(wx, 4), round(wy, 4), "dist_to_plate_m", round(d, 4), "z_base", z)
    # sanity: same transform on the sim median TCP from fk_check
    sx, sy = FK["base_rot_-90"][dest]["sim_place_tcp_world_xy_median"]
    print("SIM_MEDIAN_XY_WORLD", dest, sx, sy, "dist_to_plate_m", round(math.hypot(sx - plates[dest][0], sy - plates[dest][1]), 4))
