"""参照 TJA との相違を「検出の誤り」か「参照側のずれ」かに自動で振り分ける。
根拠: 動画の音符の位置は追跡(多数の点の直線当てはめ)で決まっているので、
  ・長く綺麗なトラック(素の点 20 以上、または縁の行の当てはめ 40 点以上)で、
  ・その前後 ±2 秒の他の音符が参照と ±6ms 以内で揃っていて(局所シフトの問題ではない)、
  ・相手(参照)の同じ種類の音符が 25〜150ms ずれた所にある
なら「参照側のずれ」と判定する。証拠画像(参照の予測位置=緑、こちら=紫)を out_dir に書く。
使い方(run_song から): judge(video, onsets, ref_list, misses, extras, vid_offset, out_dir, cal=None)
  ref_list: [(t_chart, sym)]、misses/extras: evaluate() の返り値(動画秒)
"""
import os, numpy as np, cv2
from eval_ons import bestshift

SYM = {'1': ('don', False), '2': ('ka', False), '3': ('don', True), '4': ('ka', True)}

def _local_shift(mt_small, tt, vo, t):
    win = mt_small[(mt_small >= t - 2.5) & (mt_small < t + 2.5)]
    return bestshift(win, tt, vo, 0.25) if len(win) >= 3 else vo

def judge(video, onsets, ref, misses, extras, vo, out_dir=None, cal=None, evidence=True):
    dk = [o for o in onsets if o['color'] in ('don', 'ka')]
    mt = np.array([o['t'] for o in dk]); small = np.array([not o['big'] for o in dk])
    tt = np.array([r[0] for r in ref]); tc = [str(r[1]) for r in ref]
    verdicts = []
    for m in misses:
        # 参照の音符(動画秒 m は g0 基準なので、局所シフトで捉え直す)
        sh = _local_shift(mt[small], tt, vo, m)
        j = int(np.argmin(np.abs(tt + sh - m))); t_ref = tt[j] + sh; col, big = SYM.get(tc[j], ('?', False))
        # 相手: 同種のこちらの音符で 25〜150ms 以内(余分に数えられたもの)
        cands = [o for o in dk if o['color'] == col and o['big'] == big and 0.02 <= abs(o['t'] - t_ref) <= 0.15 and any(abs(o['t'] - e) < 0.003 for e in extras)]
        if not cands: verdicts.append((m, col, big, 'unmatched', None, '')); continue
        o = min(cands, key=lambda q: abs(q['t'] - t_ref)); dt = (o['t'] - t_ref) * 1000
        strong = (o.get('hidden') and o['npts'] >= 40) or (not o.get('hidden') and not o.get('rescued') and not o.get('emerge') and o['npts'] >= 20)
        # 前後の音符が参照と揃っているか(±2s の小音符の中央値の残差)
        neigh = []
        for q in dk:
            if q is o or abs(q['t'] - o['t']) > 2.0 or q['big']: continue
            k = int(np.argmin(np.abs(tt + sh - q['t'])))
            if abs(tt[k] + sh - q['t']) < 0.05: neigh.append((q['t'] - tt[k] - sh) * 1000)
        aligned = len(neigh) >= 4 and abs(float(np.median(neigh))) <= 6.0
        verdict = 'reference-off' if (strong and aligned) else 'uncertain'
        why = 'track n=%d%s, neighbors median %+.1fms (n=%d)' % (o['npts'], ' rim-fit' if o.get('hidden') else (' rescued' if o.get('rescued') else ''), float(np.median(neigh)) if neigh else 0.0, len(neigh))
        verdicts.append((m, col, big, verdict, (o, t_ref, dt), why))
    if evidence and out_dir and video:
        os.makedirs(out_dir, exist_ok=True)
        _evidence(video, dk, verdicts, out_dir, cal)
    return verdicts

def _evidence(video, dk, verdicts, out_dir, cal):
    """相違の音符が判定枠の 250〜600px 手前にある 3 フレームを切り出し、参照の予測位置(緑)とこちらの推定(紫)を描く"""
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    M = None
    if cal is not None:
        import calib
        if not calib.is_canonical(cal): M = calib.affine(cal)[0]
    for m, col, big, verdict, info, why in verdicts:
        if info is None: continue
        o, t_ref, dt = info; v = abs(o['vel']) * fps
        rows = []
        for xa in (600, 420, 250):
            t = o['t'] - xa / v; fr = int(round(t * fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
            if not ok: continue
            if M is not None: img = cv2.warpAffine(img, M, (1920, 1080))
            tf = fr / fps; x_ref = int(615 + v * (t_ref - tf)); x_me = int(615 + v * (o['t'] - tf))
            crop = img[286:486, 500:1500].copy()
            cv2.line(crop, (x_ref - 500, 0), (x_ref - 500, 199), (0, 255, 0), 2); cv2.line(crop, (x_me - 500, 0), (x_me - 500, 199), (255, 0, 255), 2)
            cv2.putText(crop, 'video %.3fs  green=reference(%.3f)  magenta=mine(%.3f)  %+.0fms' % (tf, t_ref, o['t'], dt), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            rows.append(crop)
        if rows:
            cv2.imwrite(os.path.join(out_dir, 'refcheck_%07.2f_%s%s_%s.png' % (m, col, 'B' if big else '', verdict)), np.vstack(rows))
    cap.release()

def report(verdicts):
    lines = []
    for m, col, big, verdict, info, why in verdicts:
        if info is None: lines.append('  %.2fs %s%s: 相手なし(取りこぼし)' % (m, col, 'B' if big else '')); continue
        o, t_ref, dt = info
        lab = {'reference-off': '参照側のずれ(実機準拠で採用)', 'uncertain': '判定保留(要フレーム確認)'}[verdict]
        lines.append('  %.2fs %s%s: 参照 %.3f / こちら %.3f (%+.0fms) → %s [%s]' % (m, col, 'B' if big else '', t_ref, o['t'], dt, lab, why))
    return '\n'.join(lines)
