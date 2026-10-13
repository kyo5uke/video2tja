"""太鼓プレイ動画 → 音符オンセット列。
本家アーケード(1920x1080, 60fps)のレイアウト用にキャリブレーション:
  判定枠中心 x ≈ JUDGE_X、レーン中心 y ≈ LANE_Y。音符は右→左に流れる。
各フレームで音符ブロブを検出し、色でトラックにまとめ、判定枠xを横切る瞬間(サブフレーム補間)を叩く時刻とする。
"""
import cv2, numpy as np, sys, json

JUDGE_X = 615
LANE_Y = 386
Y_LO, Y_HI = 300, 470          # レーン帯(この外の色は無視)
AREA_MIN, AREA_MAX = 2500, 12000
BIG_AREA = 5800                # これ以上は大音符
CY_LO, CY_HI = 355, 418        # 音符中心yの許容
MAXDX = 32                     # 音符は約21px/frame。隣の16分(89px先)へ飛ばないよう抑える
MINPTS = 2                     # トラックとして認める最小点数

# HSV 色域 (OpenCV: H 0-179)
COLORS = {
    'don':    [((0, 110, 110), (10, 255, 255)), ((168, 110, 110), (180, 255, 255))],  # 赤
    'ka':     [((82, 60, 110), (108, 255, 255))],                                       # 水色
    'renda':  [((23, 120, 140), (34, 255, 255))],                                       # 黄
    'balloon':[((12, 140, 140), (20, 255, 255))],                                       # 橙
}

def detect_blobs(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = []
    for color, ranges in COLORS.items():
        m = None
        for lo, hi in ranges:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = mm if m is None else (m | mm)
        m[:Y_LO, :] = 0; m[Y_HI:, :] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            area = st[i][4]
            cx, cy = cen[i]
            if area < AREA_MIN or area > AREA_MAX: continue
            if cy < CY_LO or cy > CY_HI: continue
            w, h = int(st[i][2]), int(st[i][3])
            fill = area / float(w * h)
            # 音符は円: 縦横比≈1・充填率高め。巨大顔演出(背が高い/三日月=低充填)を弾く
            if not (50 <= h <= 118 and 50 <= w <= 130): continue
            if not (0.60 <= w / h <= 1.7): continue
            if fill < 0.55: continue
            if color == 'balloon' and w > h * 1.6:   # 風船の袋部分は除外(丸い頭だけ拾う)
                continue
            x0 = cx - w / 2.0
            nsub = int(round(w / 62.0)) if w > 100 else 1   # くっついた連打を音符数に分割
            nsub = max(1, nsub)
            for si in range(nsub):
                sx = x0 + w * (si + 0.5) / nsub
                out.append({'color': color, 'cx': float(sx), 'cy': float(cy),
                            'area': int(area / nsub), 'w': int(w / nsub), 'h': h,
                            'big': bool(area > BIG_AREA and nsub == 1)})
    return out

def track(frames_blobs, fps, f0):
    """等速予測トラッカー。音符は右→左へほぼ一定速度。予測位置の近くのブロブだけ対応付ける
    ので、フレーム抜けを跨げて、かつ隣の音符へ飛び移らない。"""
    # まず全体のスクロール速度(px/frame)を推定
    dxs = []
    for i in range(1, len(frames_blobs)):
        prev = [b['cx'] for b in frames_blobs[i - 1]]
        for b in frames_blobs[i]:
            cand = [p - b['cx'] for p in prev if 2 < p - b['cx'] < 60]
            if cand: dxs.append(min(cand))
    V = float(np.median(dxs)) if dxs else 21.0     # 既定21
    GATE = max(11.0, V * 0.85)                        # 予測からの許容ずれ
    tracks = []; active = []
    for fi, blobs in enumerate(frames_blobs):
        n_active = len(active)
        used = [False] * n_active
        order = sorted(range(len(blobs)), key=lambda k: blobs[k]['cx'])  # 左から
        for bk in order:
            b = blobs[bk]
            best, bd = -1, 1e9
            for ai in range(n_active):
                tr = active[ai]
                if used[ai] or tr['color'] != b['color']: continue
                lf, lx = tr['pts'][-1]
                dt = fi - lf
                if dt < 1 or dt > 5: continue
                pred = lx - tr['vel'] * dt          # 予測位置
                e = abs(pred - b['cx'])
                if e < GATE and e < bd:
                    best, bd = ai, e
            if best >= 0:
                tr = active[best]
                tr['pts'].append((fi, b['cx'])); tr['big'].append(b['big'])
                f_first, x_first = tr['pts'][0]
                if fi > f_first:
                    tr['vel'] = (x_first - b['cx']) / (fi - f_first)   # 全体傾きで速度更新
                used[best] = True
            else:
                active.append({'color': b['color'], 'pts': [(fi, b['cx'])],
                               'big': [b['big']], 'vel': V})
        stillA = []
        for tr in active:
            if fi - tr['pts'][-1][0] > 5:
                tracks.append(tr)
            else:
                stillA.append(tr)
        active = stillA
    tracks.extend(active)

    onsets = []
    for tr in tracks:
        pts = tr['pts']
        if len(pts) < MINPTS: continue
        fs = np.array([p[0] for p in pts], float)
        xs = np.array([p[1] for p in pts], float)
        if xs[0] - xs[-1] < 25: continue          # 全体として左へ流れていない → 効果/静止物
        # 直線回帰 x = a*f + b で判定枠到達フレームを求める(叩かれて枠で消えるので外挿する)
        a, b = np.polyfit(fs, xs, 1)
        if a >= -3: continue                       # 左へ十分動いていない
        fcross = (JUDGE_X - b) / a
        # 観測範囲からかけ離れた外挿は捨てる(枠の少し先まで)
        if fcross < fs[0] - 3 or fcross > fs[-1] + 10: continue
        # 最も右の観測が枠より十分右(=枠より手前で現れただけ)なら捨てない: 最小xが枠付近まで来ているか
        if xs.min() > JUDGE_X + 75: continue
        tcross = (f0 + fcross) / fps
        big = bool(sum(tr['big']) >= max(1, len(tr['big']) // 2))
        onsets.append({'t': round(tcross, 4), 'color': tr['color'], 'big': big, 'npts': len(pts)})
    onsets.sort(key=lambda o: o['t'])
    # 近すぎる重複(同フレーム二重検出)を統合
    merged = []
    for o in onsets:
        if merged and abs(o['t'] - merged[-1]['t']) < 0.025 and o["color"] == merged[-1]["color"]:
            continue
        merged.append(o)
    return merged

def run(video, start, end, dump=None):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(start * fps)); f1 = int(round(end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames_blobs = []
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok: break
        frames_blobs.append(detect_blobs(frame))
        fi += 1
    cap.release()
    onsets = track(frames_blobs, fps, f0)
    if dump:
        json.dump({'fps': fps, 'start': start, 'end': end, 'onsets': onsets},
                  open(dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return fps, onsets

if __name__ == '__main__':
    video = sys.argv[1]; start = float(sys.argv[2]); end = float(sys.argv[3])
    dump = sys.argv[4] if len(sys.argv) > 4 else None
    fps, onsets = run(video, start, end, dump)
    print('fps', fps, '区間', start, '-', end, 's  検出音符', len(onsets))
    from collections import Counter
    c = Counter((o['color'], o['big']) for o in onsets)
    print('内訳', dict(c))
    for o in onsets[:40]:
        print('  %7.3fs %-7s %s' % (o['t'], o['color'], '大' if o['big'] else ''))
