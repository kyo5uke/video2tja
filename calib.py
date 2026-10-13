"""レイアウトの自動キャリブレーション。
動画から 判定枠の x・レーンの y・小音符の直径・レーン右端 を推定し、各フレームを「基準配置」
(1080p アーケード配置: 判定枠 x=615, レーン y=386, 小音符内円半径 40)へ写すアフィン変換を作る。
detect2 の画素単位の定数はすべて基準配置で決めてあるので、変換後のフレームをそのまま通せばよい。

推定量はどれも「基準動画(XHcmLQ7vC5o, 1080p)に同じ推定器を掛けた値」との比で使うので、
推定器の癖(ヒット演出で音符が少し早く消える、など)は打ち消される。

使い方: calib.py <video> <start秒> <end秒> [out.json]
"""
import sys, json, cv2, numpy as np

# 基準動画に同じ推定器を掛けたときの値。`calib.py --set-ref <video> <t0> <t1>` で calib_ref.json に保存する
import os
_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib_ref.json')
REF = json.load(open(_REF_PATH)) if os.path.exists(_REF_PATH) else {'judge_x': 616.0, 'lane_y': 385.2, 'd_small': 76.6, 'x_right': 1920.0}
CANON_W, CANON_H = 1920, 1080

HSV = {
    'don': [((0, 150, 160), (10, 255, 255)), ((168, 150, 160), (180, 255, 255))],
    'ka':  [((82, 60, 110), (108, 255, 255))],
}

def _blobs(frame):
    """丸い赤/水色ブロブ (cx, cy, w, h) を返す(画面全体、解像度に依存しない条件)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    out = []
    for name, ranges in HSV.items():
        m = None
        for lo, hi in ranges:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < 0.01 * W or w > 0.12 * W: continue
            if h < 0.6 * w or h > 1.4 * w: continue                # 縦横比 ≈ 1(顔の目で少し欠ける)
            if area < 0.55 * w * h: continue                        # 充填率(円 ≈ 0.78、顔の目で下がる)
            out.append((cen[i][0], cen[i][1], w, h, name))
    return out

def sample_frames(video, t0, t1, step_frames=6):
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(t0 * fps); f1 = int(t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []; fi = f0
    while fi < f1:
        ok = cap.grab()
        if not ok: break
        if (fi - f0) % step_frames == 0:
            ok, fr = cap.retrieve()
            if ok: frames.append(fr)
        fi += 1
    cap.release()
    return frames, fps, W, H

def sample_keyframes(video, t0, t1, min_frames=60):
    """キーフレームだけデコードして集める(速い)。足りなければ従来の連続サンプリングにフォールバック"""
    import subprocess
    info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
                           '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')
    W, H = int(info[0]), int(info[1]); num, den = info[2].split('/'); fps = float(num) / float(den)
    # キーフレームだけデコードし、先頭から最大 100 枚読んだら止める(ffprobe で枚数を数えるとそれ自体がデコードで遅い)
    vf = []
    cmd = ['ffmpeg', '-v', 'error', '-skip_frame', 'nokey', '-ss', '%.3f' % t0, '-i', video, '-t', '%.3f' % (t1 - t0)] + vf + [
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    nbytes = W * H * 3; frames = []
    while len(frames) < 100:                              # 100 枚で十分(1080p のパイプは 1 枚 50ms かかる)
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes: break
        frames.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy())
    p.stdout.close()
    try: p.kill()
    except Exception: pass
    p.wait()
    if len(frames) >= min_frames: return frames, fps, W, H
    mid = (t0 + t1) / 2
    return sample_frames(video, max(t0, mid - 20), min(t1, mid + 20), 6)

def calibrate(video, t0, t1, step_frames=6, verbose=True, fast=True, window=15.0):
    """較正。動画・区間・calib.py が同じなら前回の結果を使う(detect2.cache_dir)"""
    import detect2, pickle
    cdir = detect2.cache_dir(); cpath = None
    if cdir:
        cpath = os.path.join(cdir, 'calib_%s.pkl' % detect2.src_hash(detect2.video_id(video), round(t0, 3), round(t1, 3), step_frames, fast, window,
                                                                    open(__file__, encoding='utf-8').read()))
        if os.path.exists(cpath):
            with open(cpath, 'rb') as fh: c = pickle.load(fh)
            if verbose: print('較正のキャッシュを使う: lane_y %.1f judge_x %.1f d %.1f' % (c['lane_y'], c['judge_x'], c['d_small']))
            return c
    c = _calibrate(video, t0, t1, step_frames, verbose, fast, window)
    if cpath:
        with open(cpath + '.tmp%d' % os.getpid(), 'wb') as fh: pickle.dump(c, fh)
        os.replace(cpath + '.tmp%d' % os.getpid(), cpath)
    return c


def _calibrate(video, t0, t1, step_frames=6, verbose=True, fast=True, window=15.0):
    # 区間の中ほど window 秒を step_frames おきに読む(1080p/60fps で約 6 秒)。キーフレームだけだと疎すぎて(5 秒に 1 枚)統計が取れない
    mid = (t0 + t1) / 2
    frames, fps, W, H = sample_frames(video, max(t0, mid - window / 2), min(t1, mid + window / 2), step_frames)
    per = [_blobs(fr) for fr in frames]
    # 動かないブロブ(左端の太鼓アイコンなど UI)を除く: 同じ位置(±3px)に 60% 以上のフレームで現れるもの
    key = lambda b: (int(b[0] // 6), int(b[1] // 6))
    from collections import Counter
    occ = Counter(key(b) for bs in per for b in set(bs))
    static = {k for k, c in occ.items() if c >= 0.6 * len(frames)}
    per = [[b for b in bs if key(b) not in static] for bs in per]
    bl = [b for bs in per for b in bs]
    if len(bl) < 30:
        raise RuntimeError('音符のブロブが足りない(%d)。音符が流れている区間を指定してください' % len(bl))
    cx = np.array([b[0] for b in bl]); cy = np.array([b[1] for b in bl]); w = np.array([b[2] for b in bl])
    # 直径は縦横がほぼ等しい(±10%)ブロブだけで測る。密な曲では重なった音符の三日月(幅 56 × 高さ 77)が最多になり、
    # 直径を 56.7 と誤って 1.35 倍に拡大していた(vs.VIGVANGS 裏)
    hh_ = np.array([b[3] for b in bl]); rnd = np.abs(w - hh_) <= 0.10 * np.maximum(w, hh_)
    # レーン y: ブロブ中心 y のヒストグラムの山。ただし山が複数あるときは「左へ流れているブロブ」が一番多い山を採る。
    # ふつう・かんたん譜面は音符がまばらで、下の踊り子(水色・赤の丸い体)の方が数が多いことがある
    # (愛♡スクリ～ム! ふつう: 踊り子の列 y=863 をレーンと誤認し、音符の直径 42px、一致率 14% になった)。
    # 音符は 0.1 秒で 30〜300px 左へ動くが、踊り子はその場で揺れるだけ
    hist, edges = np.histogram(cy, bins=np.arange(0, H + 4, 4))
    hs = np.convolve(hist, np.ones(3), mode='same')
    peaks = []
    # 候補の数は絞らない(判定枠のリングで選ぶ)。背景に丸い絵が多い曲では本物のレーンのブロブが上位 4 つに入らなかった(Hurtling Beams: 背景の顔・踊り子)。
    # 丸いブロブが 10 個未満の高さは下の cand_stats で落ちる
    for i in np.argsort(-hs):
        if hs[i] < 10: break
        yy = float(edges[i] + 2)
        if all(abs(yy - p) > 0.05 * H for p in peaks): peaks.append(yy)

    # 各候補で「判定枠のリング」が見つかるかを調べ、スコアが最大の候補を採る(判定枠は画面に 1 つしかない UI)。
    # 「左へ流れるブロブの数」では分けられなかった(背景も左へ流れる)
    def cand_stats(y0):
        m_ = np.abs(cy - y0) < 0.05 * H
        ww = w[m_ & rnd] if (m_ & rnd).sum() >= 10 else w[m_]
        if len(ww) < 10: return (-1.0, 0, 0.0)
        # 最頻の大きさは比で刻んだ箱(8% ずつ)で数える。1px 刻みだと、細かい飛び散り(直径 20px)が 1 つの箱に集まって、72〜76px に散らばる
        # 小音符より多くなった(Hurtling Beams)
        ee = 8.0 * 1.08 ** np.arange(0, int(np.log(0.15 * W / 8.0) / np.log(1.08)) + 2)
        hh, _ = np.histogram(ww, bins=ee)
        k_ = int(np.argmax(hh)); d0 = float(np.median(ww[(ww >= ee[k_]) & (ww < ee[k_ + 1])]))
        # 大きさのそろい具合: 音符は小と大の 2 つの大きさ(大は小の BIG_D 倍)。最頻値が大のこともある
        # (大音符ばかりの区間で 1 つの大きさだけを数えると本物のレーンが 0.24 になり、背景の踊り子の列 0.85 に負けた: Hurtling Beams)
        same = float(np.mean((np.abs(ww - d0) < 0.12 * d0) | (np.abs(ww - d0 * BIG_D) < 0.12 * d0 * BIG_D) | (np.abs(ww - d0 / BIG_D) < 0.12 * d0 / BIG_D)))
        _judge_ring(frames, y0, d0, W, H)
        jx = float(getattr(_judge_ring, 'last_x', 0.0))
        left = float(np.mean(cx[m_] < jx - d0))                 # 判定枠より左にあるブロブの割合(本物のレーンは ≈0)
        return (float(getattr(_judge_ring, 'last_score', -1.0)), int(m_.sum()), same, left)
    # 左へ流れる割合: 次の読んだフレーム(step_frames 後)に、同じ大きさ(±12%)・同じ高さで左へ 30〜300px(1920 幅換算)ずれたブロブがあるもの。
    # 音符は一定の速さで左へ流れるが、踊り子や背景の絵はその場で揺れる
    def moving(y0):
        n_ = m_ = 0
        for k in range(len(per) - 1):
            nxt = [b for b in per[k + 1] if abs(b[1] - y0) < 0.05 * H]
            for b in per[k]:
                if abs(b[1] - y0) >= 0.05 * H: continue
                n_ += 1
                if any(abs(q[2] - b[2]) < 0.12 * b[2] and abs(q[1] - b[1]) < 0.02 * H and 30 * W / 1920 * step_frames / 6 <= b[0] - q[0] <= 300 * W / 1920 * step_frames / 6
                       for q in nxt): m_ += 1
        return m_ / n_ if n_ else 0.0
    st_ = [(cand_stats(p), p) for p in peaks]
    # 判定枠のリング × 大きさのそろい × √(数の比)。リングだけだと OBS 録画(Xa)で 36 対 34 と差が小さかった。
    # 実測: 本物のレーンは そろい 0.77〜0.98、背景の踊り子などは 0.19〜0.78
    # 数の項は入れない(ふつう譜面は本物のレーンの方がブロブが少ない: 愛♡スクリ～ム! で 458 対 1507)。
    # かわりに「判定枠より左にブロブが無いこと」を掛ける(音符は判定枠で消える。踊り子の列は偽の枠の左右に散らばる)
    sc_list = [(max(a[0], 0.0) * a[2] * (1.0 - a[3]) * moving(p), p) for a, p in st_]
    lane_y = max(sc_list)[1] if sc_list else float(edges[np.argmax(hist)] + 2)
    if verbose and len(peaks) > 1:
        print('calib: レーン候補 ' + ', '.join('y=%.0f(判定枠 %.0f, 数 %d, そろい %.2f, 枠の左 %.2f, 流れ %.2f → %.1f)' % (p, a[0], a[1], a[2], a[3], moving(p), c)
                                          for (a, p), (c, _) in zip(st_, sc_list)))
    on_lane = np.abs(cy - lane_y) < 0.05 * H
    cx, cy, w, rnd = cx[on_lane], cy[on_lane], w[on_lane], rnd[on_lane]
    lane_y = float(np.median(cy))
    # 小音符の直径: 幅ヒストグラムの最頻値(大音符は ≈1.45 倍で別の山)。丸いブロブが十分あればそれだけで
    wr = w[rnd] if rnd.sum() >= 30 else w
    # 比で刻んだ箱(8% ずつ)の最頻の箱の中央値(上の cand_stats と同じ。1px 刻みだと細かい飛び散りの 20px が勝った: Hurtling Beams)
    ee_ = 8.0 * 1.08 ** np.arange(0, int(np.log(0.15 * W / 8.0) / np.log(1.08)) + 2)
    hist, _ = np.histogram(wr, bins=ee_); k_ = int(np.argmax(hist))
    d_small = float(np.median(wr[(wr >= ee_[k_]) & (wr < ee_[k_ + 1])]))
    sel = np.abs(wr - d_small) < 0.12 * d_small
    d_small = float(np.mean(wr[sel]))                    # 最頻の山(±12%)の平均(1px 刻みの量子化を均す)
    # 最頻の大きさが小音符か大音符か: レーンの上下の黒い線の間の高さは小音符の直径の約 2.62 倍(基準の配置で 200px ÷ 76.2px)。
    # 測った大きさで割った比が、小と見た場合(2.62)と大と見た場合(2.62 ÷ 1.475)のどちらに近いかで決める
    # (大音符ばかりの曲で大音符の直径を小音符と測り、画面を 0.64 倍に縮めて大音符を全部小音符と読んだ: 大打音)
    lh = _lane_height(frames, lane_y, d_small, W, H)
    if lh is not None:
        r_ = lh / d_small
        if abs(r_ - LANE_H_RATIO / BIG_D) < abs(r_ - LANE_H_RATIO):
            if verbose: print('calib: 最頻の大きさ %.1f は大音符(レーンの高さ %.1f = %.2f 倍)→ 小音符 %.1f' % (d_small, lh, r_, d_small / BIG_D))
            d_small = d_small / BIG_D
    # 判定枠 (x, y): ①ヒット時の黄色いバースト演出の出現頻度マップの重心を初期値に、
    # ②明るい演出を除いた背景(輝度の 20% 分位)で判定枠のリング(半径≈音符直径)の輝度が最大になる中心を探す。
    # 同じ配置なら曲が違っても ±1px で揃う(バースト重心だけだと曲により 9px ぶれる)
    judge_x, judge_y, ring_r = _judge_ring(frames, lane_y, d_small, W, H)
    if judge_y is not None:
        lane_y = judge_y                                  # リング中心 y はブロブ中央値より安定
    # 予備: ブロブ中心 x の分布の左端(バーストが無い動画向け)
    bw = 2
    hist, edges = np.histogram(cx, bins=np.arange(0, W + bw, bw))
    hist = np.convolve(hist, np.ones(7) / 7.0, mode='same')
    plateau = np.median(hist[(edges[:-1] > 0.45 * W) & (edges[:-1] < 0.85 * W)])
    hi = hist >= 0.5 * plateau
    best = (0, 0, 0); i = 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j + 1 < len(hi) and hi[j + 1]: j += 1
            if j - i > best[1] - best[0]: best = (i, j, 0)
            i = j + 1
        else: i += 1
    i0 = best[0]
    if i0 > 0 and hist[i0] != hist[i0 - 1]:
        frac = (0.5 * plateau - hist[i0 - 1]) / (hist[i0] - hist[i0 - 1])
        judge_plateau = float(edges[i0 - 1] + bw * frac)
    else:
        judge_plateau = float(edges[i0])
    method = 'ring'
    if judge_x is None:
        judge_x = judge_plateau - 1.85 * d_small; method = 'plateau'   # 基準動画では プラトー左端 ≈ 判定枠 + 1.85×直径
    # レーン右端: ブロブが最初に現れる x(右端の 99.5% 点 + 半径)
    x_right = float(min(W, np.percentile(cx, 99.5) + d_small / 2))
    cal = {'video': video, 'fps': fps, 'width': W, 'height': H, 'lane_y': lane_y, 'd_small': d_small,
           'judge_x': judge_x, 'x_right': x_right, 'n_blobs': int(len(cx)), 'judge_method': method, 'ring_r': ring_r}
    if verbose:
        print('calib: %dx%d @%.2ffps  blobs %d  lane_y %.1f  d_small %.1f  judge_x %.1f (%s, ring r %.1f)  x_right %.0f' % (W, H, fps, len(cx), lane_y, d_small, judge_x, method, ring_r or -1, x_right))
    return cal

def _ring_scores(img, cxs, cys, r, dr):
    """複数の中心 (cxs, cys) について、半径 r のリング上の平均輝度 − 内外(r±dr)の平均 をまとめて計算(ベクトル化)"""
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    H, W = img.shape
    def ring_mean(rr):
        xs = np.clip((cxs[:, None] + rr * np.cos(ang)[None, :]).astype(int), 0, W - 1)
        ys = np.clip((cys[:, None] + rr * np.sin(ang)[None, :]).astype(int), 0, H - 1)
        return img[ys, xs].astype(np.float32).mean(1)
    return ring_mean(r) - 0.5 * (ring_mean(r - dr) + ring_mean(r + dr))

LANE_H_RATIO = 200.0 / 76.2      # レーンの上下の黒い線(基準の配置で 285・485 行)の間 ÷ 小音符の直径
BIG_D = 1.55                     # 大音符の塊の直径 ÷ 小音符の塊の直径(基準の配置で 119 ÷ 77。白い縁を含むので内円の比 59/40 より大きい)


def _lane_height(frames, lane_y, d, W, H):
    """レーンの上下の黒い線(最も暗い行)の間の高さ。レーンの中心から上下 0.5〜2 × d の範囲で、フレームの中央値の行ごとの明るさが最も暗い行"""
    try:
        x0, x1 = int(0.45 * W), int(0.95 * W)
        ys = np.arange(max(0, int(lane_y - 2.0 * d)), min(H, int(lane_y + 2.0 * d)))
        prof = np.median(np.array([np.median(fr[ys, x0:x1].max(2), axis=1) for fr in frames[::max(1, len(frames) // 30)]]), axis=0)
        up = (ys <= lane_y - 0.5 * d); dn = (ys >= lane_y + 0.5 * d)
        if not up.any() or not dn.any(): return None
        # 暗い行(最も暗い行 + 10 以内)のうち、レーンの中心に最も近いもの(その外側の画面の飾りにも同じくらい暗い帯がある)
        top = ys[up][prof[up] <= prof[up].min() + 10].max(); bot = ys[dn][prof[dn] <= prof[dn].min() + 10].min()
        return float(bot - top)
    except Exception:
        return None


def _judge_ring(frames, lane_y, d, W, H):
    """判定枠の中心 (x, y) とリング半径。明るい演出を除いた背景(輝度 20% 分位)で、判定枠リング(半径≈音符直径)の
    輝度差が最大になる中心を、粗い格子(2px)→細かい格子(1px, 半径も振る)で探す。見つからなければ (None, None, None)"""
    sub = frames[::2] if len(frames) > 80 else frames
    dr = 0.12 * d
    # 分位点はリングが届く範囲(探索の格子 ± 半径 1.2d)だけで計算する(全面だとレーン候補ごとに 1 秒かかっていた)。画素ごとの計算なので結果は同じ
    m = int(np.ceil(1.2 * d)) + 8
    xa = max(0, int(0.27 * W) - m); xb = min(W, int(np.ceil(0.40 * W)) + m)
    ya = max(0, int(lane_y - 0.12 * d) - m); yb = min(H, int(np.ceil(lane_y + 0.12 * d)) + m + 1)
    gray = np.stack([cv2.cvtColor(f[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY) for f in sub])
    p20 = np.zeros((H, W), np.uint8)
    p20[ya:yb, xa:xb] = np.percentile(gray, 20, axis=0).astype(np.uint8)
    # x の下限は 0.24W: その左にはコンボ数の太鼓(同じ高さの丸い枠、x≈0.21W)があり、音符がまばらな譜面では
    # そちらのリングの方が強く出ることがある(愛♡スクリ～ム! ふつう: 判定枠を x=412 と誤認)。本物の判定枠は 0.32W 付近
    # 探す範囲は 0.27W〜0.40W(試験スイート 70 本の判定枠は 0.313〜0.322W: 縮尺違いの IRIS OUT 601.8/1920、さいたま2000 613.4)。
    # 0.55W まで探すと、第六天魔王 裏の動画で x=482(0.25W)の輪を判定枠と誤認し、音符の速さごとに 60〜120ms ずれた
    gx = np.arange(0.27 * W, 0.40 * W, 2.0); gy = np.arange(lane_y - 0.12 * d, lane_y + 0.12 * d + 1, 2.0)
    cx = np.repeat(gx, len(gy)); cy = np.tile(gy, len(gx))
    sc = _ring_scores(p20, cx, cy, d, dr)
    k = int(np.argmax(sc)); bx, by = cx[k], cy[k]
    best = None
    for r in np.arange(0.85 * d, 1.08 * d, 1.0):
        fx = np.arange(bx - 4, bx + 5, 1.0); fy = np.arange(by - 4, by + 5, 1.0)
        cx2 = np.repeat(fx, len(fy)); cy2 = np.tile(fy, len(fx))
        s2 = _ring_scores(p20, cx2, cy2, r, dr); k2 = int(np.argmax(s2))
        if best is None or s2[k2] > best[0]: best = (float(s2[k2]), float(cx2[k2]), float(cy2[k2]), float(r))
    sc_, cxb, cyb, rb = best
    _judge_ring.last_score = sc_; _judge_ring.last_x = cxb
    if sc_ < 20: return None, None, None                   # リングが見えない(スコア: 基準動画で 50-63)
    return cxb, cyb, rb

def affine(cal):
    """cal の配置 → 基準配置 への 2x3 アフィン行列(cv2.warpAffine 用)と、基準配置での右端 X1"""
    s = REF['d_small'] / cal['d_small']
    tx = REF['judge_x'] - s * cal['judge_x']
    ty = REF['lane_y'] - s * cal['lane_y']
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    x1 = int(min(CANON_W - 10, s * cal['x_right'] + tx))
    return M, x1

def is_canonical(cal, tol=0.02):
    """基準配置そのもの(変換不要)か"""
    return (abs(cal['d_small'] / REF['d_small'] - 1) < tol and abs(cal['judge_x'] - REF['judge_x']) < 4
            and abs(cal['lane_y'] - REF['lane_y']) < 4 and cal['width'] == CANON_W and cal['height'] == CANON_H)

if __name__ == '__main__':
    if sys.argv[1] == '--set-ref':                    # 基準動画で推定器を掛け、その値を基準として保存
        cal = calibrate(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
        ref = {k: cal[k] for k in ('judge_x', 'lane_y', 'd_small', 'x_right')}
        json.dump(ref, open(_REF_PATH, 'w'), indent=1); print('saved REF', ref, '->', _REF_PATH); sys.exit(0)
    video, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    cal = calibrate(video, t0, t1)
    M, x1 = affine(cal)
    print('affine scale %.4f  tx %.1f  ty %.1f  X1(canon) %d  canonical=%s' % (M[0, 0], M[0, 2], M[1, 2], x1, is_canonical(cal)))
    if len(sys.argv) > 4:
        json.dump(cal, open(sys.argv[4], 'w'), indent=1); print('saved', sys.argv[4])


def low_priority():
    """このプロセスを「通常以下」の優先度にする(子のプロセス・ffmpeg も引き継ぐ)。全スレッドを使うので、通常の優先度だと
    処理中に画面の操作が固まる。空いている CPU は全部使うので速さはほぼ変わらない。V2T_NICE=0 で無効"""
    if os.environ.get('V2T_NICE') == '0' or os.name != 'nt': return
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
    except Exception: pass
