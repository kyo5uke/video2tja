"""音声の相関が失敗する動画のために、小節線から Δ(VID2OGG)を求める。
手順: 広い区間を検出 → 検出した小節線を参照譜面の小節頭に合わせる(粗→細)→ Δ = シフト + 参照の OFFSET
使い方: find_delta_bars.py <video> <参照tja> <コース> [検出の開始 終了]
"""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib, detect2
from eval_timing import parse_tja_bars

video, tja, crs = sys.argv[1], sys.argv[2], sys.argv[3]
s0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
s1 = float(sys.argv[5]) if len(sys.argv) > 5 else None
import cv2
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
cap.release()
if s1 is None: s1 = dur - 0.5
mid = (s0 + s1) / 2
cal = calib.calibrate(video, max(s0, mid - 20), min(s1, mid + 20))
fps, ons = detect2.run_parallel(video, s0, s1, dbg=False, cal=cal)
bars = sorted(detect2.run_parallel.bars)
heads = parse_tja_bars(tja, crs)
off = float(re.search(r'^OFFSET:\s*([-\d.]+)', open(tja, encoding='utf-8-sig', errors='replace').read(), re.M).group(1))
print('検出: 音符 %d、小節線 %d 本 / 参照の小節頭 %d' % (len(ons), len(bars), len(heads)))
b = np.asarray(bars); h = np.asarray(sorted(heads))


def score(sh):
    idx = np.searchsorted(h, b - sh)
    lo = np.clip(idx - 1, 0, len(h) - 1); hi = np.clip(idx, 0, len(h) - 1)
    d = np.minimum(np.abs(b - sh - h[lo]), np.abs(b - sh - h[hi]))
    return float(np.median(d)), int((d < 0.03).sum())


best = None
for sh in np.arange(0.0, max(1.0, dur - (h[-1] - h[0]) + 2.0), 0.02):
    m, n = score(sh)
    if best is None or (n, -m) > (best[2], -best[1]): best = (sh, m, n)
sh0 = best[0]
for sh in np.arange(sh0 - 0.05, sh0 + 0.05, 0.0005):
    m, n = score(sh)
    if (n, -m) > (best[2], -best[1]): best = (sh, m, n)
sh, m, n = best
print('小節線の整列: シフト %.4f  ずれの中央値 %.1fms  30ms 以内 %d/%d' % (sh, m * 1000, n, len(b)))
print('VID2OGG = %.4f' % (sh + off))
