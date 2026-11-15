"""Annotate recorded video, or replay measured wrist alignment samples. Never opens an arm.

video: config JSON contains hsv_low, hsv_high, roi, optional min_area/max_area.
replay: NPZ arrays state[N,6], target_uv[N,2], timestamps[N]; outputs proposals only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from visual_servo import Calibration, AlignmentController, detect_target, fit_jacobian


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    v = sub.add_parser('video')
    v.add_argument('video'); v.add_argument('config'); v.add_argument('output', type=Path)
    f = sub.add_parser('fit')
    f.add_argument('samples', help='NPZ joint_deltas, pixel_deltas, validation_joint_deltas, validation_pixel_deltas')
    f.add_argument('metadata', help='JSON camera_id/image_size/reference_state/reference_uv/joints/domain/mapping_sha256')
    f.add_argument('output', type=Path)
    r = sub.add_parser('replay')
    r.add_argument('samples'); r.add_argument('calibration'); r.add_argument('output', type=Path)
    r.add_argument('--camera-id', required=True); r.add_argument('--mapping-sha256', required=True)
    r.add_argument('--image-size', type=int, nargs=2, required=True, metavar=('WIDTH','HEIGHT'))
    a = p.parse_args()
    if a.output.exists(): p.error('Output already exists; choose a new path')
    if a.mode == 'fit':
        c = json.loads(Path(a.metadata).read_text())
        with np.load(a.samples, allow_pickle=False) as n:
            jac, rmse = fit_jacobian(n['joint_deltas'], n['pixel_deltas'], c['joints'])
            x, y = n['validation_joint_deltas'], n['validation_pixel_deltas']
            if x.ndim != 2 or len(x) < 4 or x.shape[1] != len(c['joints']) or y.shape != (len(x),2) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Independent validation samples required')
            validation = float(np.sqrt(np.mean((x @ jac.T-y)**2)))
        # A fit is a draft. A person must verify sample provenance and local domain.
        c.update(schema=1, validated=False, jacobian=jac.tolist(), fit_rmse_px=rmse, validation_rmse_px=validation)
        a.output.write_text(json.dumps(c, indent=2)+'\n')
    elif a.mode == 'replay':
        c = Calibration.load(a.calibration, a.camera_id, a.image_size, a.mapping_sha256)
        controller = AlignmentController(c)
        rows = []
        with np.load(a.samples, allow_pickle=False) as n:
            for state, uv, stamp in zip(n['state'], n['target_uv'], n['timestamps'], strict=True):
                try:
                    command, status = controller.update(state, uv, float(stamp), float(stamp))
                    rows.append(dict(timestamp=float(stamp), command=command.tolist(), status=status))
                except ValueError as e:
                    rows.append(dict(timestamp=float(stamp), status='hold', error=str(e)))
                    break
        a.output.write_text(json.dumps({'mode':'offline proposals; not executed', 'samples':rows}, indent=2)+'\n')
    else:
        import cv2
        config = json.loads(Path(a.config).read_text())
        a.output.mkdir(parents=True)
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened(): raise ValueError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps > 0: raise ValueError('Video has no usable FPS')
        writer = None; rows = []; index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                if writer is None:
                    writer = cv2.VideoWriter(str(a.output/'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (bgr.shape[1],bgr.shape[0]))
                    if not writer.isOpened(): raise ValueError('Cannot create overlay video')
                row = dict(frame=index, seconds=index/fps)
                try:
                    uv, box, area = detect_target(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), **config)
                    x,y,w,h = box
                    cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.drawMarker(bgr,tuple(np.rint(uv).astype(int)),(0,255,0))
                    row.update(target_uv=uv.tolist(), area=area, status='detected')
                except ValueError as e:
                    row.update(status='hold', error=str(e))
                cv2.putText(bgr,row['status'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                writer.write(bgr); rows.append(row); index += 1
        finally:
            cap.release()
            if writer is not None: writer.release()
            (a.output/'detections.json').write_text(json.dumps(rows,indent=2)+'\n')
        if not rows: raise ValueError('Video contained no decoded frames')

if __name__ == '__main__': main()
