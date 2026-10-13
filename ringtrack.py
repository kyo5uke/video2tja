"""隠れて流れる音符を、白い縁の円を追いかけて拾う(軽い方式。絵は描かない)。

速さの違う音符が追い越し合う譜面(カオスタイム・チャーリーダッシュ!)では、譜面で後の音符が先の音符の下に隠れて流れ、色の塊の追跡では拾えない。
隠れていても白い縁の一部(弧)は見えることが多い。
 1. 追い越しのある区間の毎フレームで、白い縁の円(ハフ変換)の中心と中身の色を拾う
 2. 見つかっている音符の予想位置で説明できない円だけを残す
 3. 一直線に並ぶ組を探す。音符 1 個の速さは流れている間一定だが、音符ごとに違ってよい(速さが徐々に変わる区間)ので、速さは 1% 刻みで全部試す
 4. 右端から判定枠まで追いかけ、上に重なる音符(譜面で先の音符)に隠れていないはずのフレームの過半数で円が見えていれば本物
 5. 途中から現れた短い軌跡の音符も同じ基準で確かめ、見えるはずの所でほとんど見えないものは偽物として外す
"""
import os, numpy as np

JX0 = JX = 615.0          # 判定枠(基準配置)。detect2.JUDGE_X と同じ(音符の時刻はこの位置に着く時刻。以前は 618 で 3px ずれていた)
X_IN = 1850         # これより右は出現途中
X0_BAND = 750       # 検出の帯の左端(これより左は判定枠の演出)
R_S, R_B = 50, 84   # 小音符・大音符の外周(黒い輪郭まで)の半径(実測: 白い縁は小 42〜49px・大 72〜82px)
NEAR = 6            # 直線に乗るとみなす円の中心のずれ(px)
RIN_S_DET = 40      # detect2.RIN_S(小音符の内円の半径)
MIN_PTS = 5         # 直線に要る円の数(投票の最小)
VSTEP = 1.01        # 速さを試す刻み(1%)


def _x(o, f, fps):
    return JX + (o['t'] * fps - f) * abs(o['vel']) + o.get('_dx', 0.0)


def _windows(notes, fps, t0, t1):
    """追い越しのある区間: 譜面で先の音符 i が後の音符 j より後から画面に入る(速さが違う)とき、j は i に上から追い越されて隠れうる"""
    nt = np.array([o['t'] for o in notes]); nv = np.array([abs(o['vel']) for o in notes])
    enter = nt - (X_IN + 30 - JX) / (nv * fps)
    order = np.argsort(nt); wins = []
    for a in range(len(order)):
        i = order[a]
        for b in range(a + 1, len(order)):
            j = order[b]
            if nt[j] - nt[i] > 4.0: break
            if enter[i] > enter[j] + 1.0 / fps: wins.append([min(enter[i], enter[j]) - 0.5, max(nt[i], nt[j]) + 0.2])
    return _merge([[max(t0, a), min(t1, b)] for a, b in wins if b > t0 and a < t1])


def _subtract(ws, cut):
    """区間 ws から区間 cut を除く"""
    out = []
    for a, b in ws:
        segs = [[a, b]]
        for c, d in cut:
            nxt = []
            for x, y in segs:
                if d <= x or c >= y: nxt.append([x, y]); continue
                if x < c: nxt.append([x, c])
                if d < y: nxt.append([d, y])
            segs = nxt
        out += [s_ for s_ in segs if s_[1] - s_[0] > 0.02]
    return out


def _merge(ws):
    ws = sorted(ws); out = []
    for a, b in ws:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out


LX0 = 560           # 画素の種類を持つ列の左端
CY = 386
HB = 90             # 画素の種類を持つ行の半分(大音符の白い縁 82px まで入る)


def _labels(canvas, y_off=0):
    """レーンの帯(CY±HB、x ≥ LX0)を縦横 1/2 にした画素の種類: 0 その他 / 1 ドンの色 / 2 カッの色 / 3 白"""
    import cv2
    hsv = cv2.cvtColor(canvas[CY - HB - y_off:CY + HB - y_off:2, LX0::2], cv2.COLOR_BGR2HSV)
    # 判定は OpenCV の inRange(uint8 のまま。int にして比べるのと同じ結果で 3 倍速い)。後の種類が前を上書きする順は同じ
    # 種類の番号は後のものほど大きいので、上書きの代わりに最大で合わせる
    m1 = cv2.bitwise_and(cv2.inRange(hsv, (0, 150, 160), (10, 255, 255)) | cv2.inRange(hsv, (177, 150, 160), (255, 255, 255)), 1)
    m2 = cv2.bitwise_and(cv2.inRange(hsv, (82, 60, 150), (92, 255, 255)), 2)
    m3 = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 191), (255, 49, 255)), 3)
    # 4 = 風船の頭のオレンジ(detect2.COLORS['balloon'] と同じ範囲。白・ドン・カッとは重ならない): 隠れた円の直線が風船の頭か見分ける
    m4 = cv2.bitwise_and(cv2.inRange(hsv, (12, 140, 140), (20, 255, 255)), 4)
    return cv2.max(cv2.max(cv2.max(m1, m2), m3), m4)


def _binom_tail(k, n, upper):
    """五分五分のとき、n 回中 k 回以上(upper)/ 以下 の起こる確率"""
    from math import comb
    ks = range(k, n + 1) if upper else range(0, k + 1)
    from fractions import Fraction
    return float(Fraction(sum(comb(n, i) for i in ks), 2 ** n))     # 2.0 ** n は n > 1023 で桁あふれする


P_SIG = 0.05        # 偶然(五分五分)ではこれ未満の確率でしか起きないとき、縁が見えた / 見えないと決める


def _eff(f, clk):
    """動画のフレーム f を、音符の時刻と同じ時計のフレームに直す(録画の複製・落ちたフレームで 1 フレームずれる。速さ 50 の音符なら 50px)。
    検出が音符全部からゲームの描画の時刻を測っていれば(detect2 の _game_delta)、そのずれも足す"""
    if clk is None:
        import detect2
        gd = getattr(detect2.run_parallel, 'gdelta', None)
        if gd is not None:
            k = int(f) - gd[0]
            if 0 <= k < len(gd[1]): return float(f) + float(gd[1][k])
        return float(f)
    ks, slip, lin = clk
    j = int(np.clip(np.searchsorted(ks, f), 0, len(ks) - 1))
    return float(np.polyval(lin, f + slip[j]))


def _read(video, wins, cal, fps=None):
    """区間の全フレームの円 {時刻のフレーム: [(x, 色)]} と、2 フレームおきの画素の種類 {時刻のフレーム: L}"""
    import detect2, synth
    clk = getattr(detect2.run_parallel, 'clk', None)
    skip = getattr(detect2.run_parallel, 'skip', set())     # 古い絵のフレーム(30fps の中身の動画の複製): 音符が 1 フレーム前の位置にある
    out = {}; labs = {}
    import time as _time; _read.t_hough = 0.0; _read.n_win = len(wins)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=min(8, detect2.cpu_budget()))                   # ハフ変換は GIL を離すので、フレームを並べて処理する
    import cv2
    ncv = cv2.getNumThreads(); cv2.setNumThreads(1)          # OpenCV 自身の並列とぶつかると半分の速さ(1.9 → 0.95ms/枚)

    def one(fc):
        f, canvas, step, take = fc
        return f, synth.rings(canvas, Y_CUT), (_labels(canvas, Y_CUT) if take else None)
    # 区間ごとに ffmpeg を起動して頭出しする待ちが読み込みの 9 割(Prelude 裏: 37 区間 40 秒のうち円の検出は 3 秒)なので、
    # 先の区間の ffmpeg を AHEAD 個まで先に起動しておく
    wl = list(wins); AHEAD = min(6, detect2.cpu_budget()); hds = {}

    def start(i):
        a, b, step = wl[i]
        hds[i] = detect2.band_start(video, a, b, (CY - HB - 6, CY + HB + 6), step, cal, parts=int(np.clip((b - a) // 4 + 1, 1, 8)), tag=i)   # 長い区間は分けて並べる
    # 検出のデコードが全フレームの円と画素の種類を取ってあれば(detect2.RT_FEAT)、動画を読まずにそこから引く。
    # 区間・間引き・古い絵の除き方・画素の種類を取るフレームは下の動画を読む場合と同じ
    rtf = getattr(detect2.run_parallel, 'rt_feats', None)
    fps_v = None
    if rtf:
        import zlib
        fps_v = fps or 60.0
        LSH = (len(range(CY - HB, CY + HB, 2)), len(range(LX0, 1920, 2)))

    def from_feats(a, b, step):
        fr_ = [f for f in range(int(np.floor(a * fps_v)) - 1, int(np.ceil(b * fps_v)) + 2)
               if a - 1e-6 <= f / fps_v < b - 1e-6 and (step == 1 or (f - int(round(a * fps_v))) % step == 0)]
        if not fr_ or any(f not in rtf for f in fr_): return False
        nk = 0
        for f in fr_:
            if f in skip: continue
            r_, z_ = rtf[f]; fe = _eff(f, clk); out[fe] = r_
            if step > 1 or nk % 2 == 0: labs[fe] = np.frombuffer(zlib.decompress(z_), np.uint8).reshape(LSH)
            nk += 1
        return True
    if rtf:
        wl = [w for w in wl if not from_feats(*w)]
    for i in range(min(AHEAD, len(wl))): start(i)
    for i, (a, b, step) in enumerate(wl):
        detect2.progress('隠れた音符', i + 1, len(wl), '追い越しのある区間の画面を読んで白い縁の円を探す(%.1f〜%.1f 秒)' % (a, b))
        hd = hds.pop(i)
        if i + AHEAD < len(wl): start(i + AHEAD)
        buf = []; nk = 0

        def flush():
            _tf = _time.time()
            for f, r, L in ex.map(one, buf):
                fe = _eff(f, clk); out[fe] = r
                if L is not None: labs[fe] = L
            buf.clear(); _read.t_hough += _time.time() - _tf
        for f, canvas in detect2.band_iter(hd):
            if f in skip: continue
            # 画素の種類は、使えるフレーム(古い絵を除いた)の 1 枚おき(元のフレーム番号の偶奇で選ぶと、30fps の中身の動画で使えるフレームが
            # 全部奇数側になり、縁の確かめが空になった)
            buf.append((f, canvas[Y_CUT:CY + HB].copy(), step, step > 1 or nk % 2 == 0)); nk += 1     # 帯の行だけ写す(全面 6MB を写していた)
            if len(buf) >= 64: flush()
        flush()
    ex.shutdown(); cv2.setNumThreads(ncv)
    return out, labs


Y_CUT = CY - HB     # 読んだ帯のうち写す最初の行(synth.rings は CY±70、_labels は CY±HB を使う)
RIM = (43, 45, 47)  # 小音符の白い縁の半径(実測 42〜49px)
RIM_B = (73, 75, 77)  # 大音符の白い縁の半径(実測 72〜82px)
ANG = np.linspace(0, 2 * np.pi, 64, endpoint=False)


def _shape(t, v, big, labs, occ, others, fps):
    """形の確かめ: 円の白い縁のうち、上に重なる音符(譜面で先の音符・連打・風船)に隠れていない点が白いか。
    戻り値: (縁が 4 分の 1 周以上見えるフレーム数, そのフレームでの白い点の割合の平均, 見えている中身の色の多数決 'don'/'ka'/None)"""
    rims = list(RIM_B if big else RIM); rin = rims[0] - 8
    cx_, cy_ = np.cos(ANG), np.sin(ANG)
    ws = []; cd = ck = ni = cb = 0; n_shared = 0; fd_ = fk_ = 0
    # 既存の音符の軌跡(当てはめ直したものがあればそれ)を配列にしておき、フレームごとに近いものだけ取り出す(1 つずつ辞書を引くと 1 曲 10 秒)
    FT = np.array([o.get('_fit', (o['t'],))[0] for o in occ], float) * fps
    FV = np.array([o.get('_fit', (0, abs(o['vel'])))[1] for o in occ], float)
    FR = np.array([R_B if o.get('big') else R_S for o in occ], float)
    FRO = np.array([(RIM_B if o.get('big') else RIM)[1] for o in occ], float)
    FE = np.array([o['t'] < t for o in occ], bool)            # 譜面で先の音符(上に描かれる)
    for f, L in labs.items():
        x = JX + (t * fps - f) * v
        if not (JX + 40 < x < X_IN): continue
        XO = JX + (FT - f) * FV
        # 連打・風船の棒: 頭の円から、終わりの位置 + 半径 2 つ分まで(棒の右の丸は終わりより外側まで描かれる: HANDOFF 28 番。
        # Emma 裏 玄人 72.25s のドンは大連打の棒の右端の下を流れ、棒が判定枠で消えてから出てくる)
        bars_ = [(JX + (a * fps - f) * v0 - r0, JX + (b * fps - f) * v0 + 2 * r0) for a, b, v0, r0 in others if a < t]
        k_ = np.flatnonzero(FE & (np.abs(XO - x) < FR + rims[-1]))
        oc = list(zip(XO[k_].tolist(), FR[k_].tolist()))
        px = x + rims[1] * cx_; py = rims[1] * cy_
        vis = np.ones(len(ANG), bool)
        for xo, r in oc: vis &= (px - xo) ** 2 + py ** 2 > r * r
        for x0, x1 in bars_: vis &= ~((px > x0) & (px < x1) & (np.abs(py) < R_B))
        vis_occ = vis.copy()                                  # 隠れていない点(既存の縁で除く前)
        # 既にある音符の白い縁の上にも乗る点は、その音符で説明がつくので証拠に数えない(同じ半径の円が 29px ずれて重なると、
        # 間にハフ変換の幻の円ができ、その縁の大半が本物 2 つの白い縁と交わって白く見える: ドンカマ2000 112.8s)
        for k2 in np.flatnonzero(np.abs(XO - x) <= 2 * rims[-1] + 20):
            xo = XO[k2]; ro = FRO[k2]
            vis &= np.abs(np.sqrt((px - xo) ** 2 + py ** 2) - ro) > 6      # 白い縁の太さの半分(3.5px)+ 当てはめ直した位置の誤差(~2.5px)
        if vis.sum() < len(ANG) // 4:
            if vis_occ.sum() >= len(ANG) // 4: n_shared += 1          # 隠れてはいないのに、縁が既存の音符の縁と重なって確かめられない
            continue
        # 予想位置の前後を探して、白い縁が最も乗る所で判定する(縁の幅は ~7px。時刻の誤差 2.5ms(0.15 フレーム)でも速い音符は数 px ずれる)
        dl = max(NEAR, 0.15 * v); best_w = 0.0
        # ずらし(dd)と縁の半径(rr)を全部まとめて 1 回で引く(以前は dd と rr の二重ループで小さな配列の計算を繰り返し、1 曲 11 秒。
        # 位置の計算の順(x + dd + rr·cos − LX0)と切り捨ては同じなので結果も同じ)
        DD = np.arange(-dl, dl + 0.1, 2.0); RR = np.asarray(rims, float)
        XS = (((x + DD[:, None, None]) + RR[None, :, None] * cx_[None, None, :] - LX0) / 2).astype(int)
        YS = np.broadcast_to(((HB + RR[:, None] * cy_[None, :]) / 2).astype(int)[None], XS.shape)
        ok = (XS >= 0) & (XS < L.shape[1]) & (YS >= 0) & (YS < L.shape[0])
        white = ok & (L[np.clip(YS, 0, L.shape[0] - 1), np.clip(XS, 0, L.shape[1] - 1)] == 3)
        vv_ = vis[None, :] & ok.all(axis=1); nv_ = vv_.sum(1)
        good = nv_ >= len(ANG) // 4
        if good.any(): best_w = max(best_w, float(((white.any(axis=1) & vv_).sum(1)[good] / nv_[good]).max()))
        ws.append(best_w)
        # 見えている中身の色(フレームごとの多数も数える)
        cd0, ck0 = cd, ck
        for rr in (rin * 0.4, rin * 0.7, rin * 0.95):          # 縁のすぐ内側まで(はみ出して見える三日月が細くても中身が見える)
            ix = x + rr * cx_; iy = rr * cy_
            vi = np.ones(len(ANG), bool)
            for xo, r in oc: vi &= (ix - xo) ** 2 + iy ** 2 > r * r
            xs = ((ix[vi] - LX0) / 2).astype(int); ys = ((HB + iy[vi]) / 2).astype(int)
            ok = (xs >= 0) & (xs < L.shape[1]) & (ys >= 0) & (ys < L.shape[0])
            lv = L[ys[ok], xs[ok]]; cd += int((lv == 1).sum()); ck += int((lv == 2).sum()); cb += int((lv == 4).sum()); ni += len(lv)
        if cd - cd0 > ck - ck0: fd_ += 1
        elif ck - ck0 > cd - cd0: fk_ += 1
    # 中身の色: 見えている中身の点の赤と青の多数(顔の目や口は黒なので数えない)。どちらも無ければ None
    col = None if cd + ck == 0 else ('don' if cd > ck else 'ka')
    _shape.seen = sum(1 for w in ws if w >= 0.5)            # 縁が見えた(白い点が過半)フレームの数
    _shape.shared = n_shared                                 # 縁が既存の音符の縁と重なって確かめられなかったフレームの数
    _shape.ni = ni                                           # 見えていた中身の点の数
    _shape.fvote = (fd_, fk_)                                # 中身がドンの色だったフレーム数 / カッの色だったフレーム数
    _shape.cdkb = (cd, ck, cb)                                # 見えていた中身の点: ドンの色・カッの色・風船のオレンジ
    return len(ws), (float(np.mean(ws)) if ws else 0.0), col


def _left_red(t, v, labs, fps):
    """時刻 t・速さ v の音符のすぐ左(判定枠の側、中心から小音符の半径 + 10〜30px)のレーンの中心の行が、ドンの色(赤)のフレームの数。戻り値: (見たフレーム数, 赤のフレーム数)"""
    n = r = 0
    for f, L in labs.items():
        x = JX + (t * fps - f) * v
        if not (JX + 40 + R_S + 30 < x < X_IN): continue
        cols = ((np.arange(x - R_S - 30, x - R_S - 10, 2) - LX0) / 2).astype(int)
        rows = np.arange(HB // 2 - 6, HB // 2 + 7)
        ok = (cols >= 0) & (cols < L.shape[1])
        if ok.sum() < 3: continue
        blk = L[np.ix_(rows, cols[ok])]
        n += 1; r += int((blk == 1).mean() > 0.5)
    return n, r


def _covered(x, f, t, occ, fps):
    """時刻 t の音符の中心 x が、譜面で先の音符(上に描かれる)の円の中にあるか(円の半分以上が隠れ、白い縁の円として取れない)"""
    for o in occ:
        if o['t'] >= t: continue
        r = R_B if o.get('big') else R_S
        tt, vv = o.get('_fit', (o['t'], abs(o['vel'])))
        if abs(JX + (tt * fps - f) * vv - x) < r: return True
    return False


def _support(t, v, col, frames, rings, occ, others, fps, any_color=False):
    """(見えるはずのフレーム数, そのうち円が見えたフレーム数)"""
    n_vis = n_hit = 0
    for f in frames:
        x = JX + (t * fps - f) * v
        if not (JX + 40 < x < X_IN): continue
        if _covered(x, f, t, occ, fps): continue
        if any(a < t and JX + (a * fps - f) * v0 - r0 < x < JX + (b * fps - f) * v0 + 2 * r0 for a, b, v0, r0 in others): continue
        n_vis += 1
        if any(abs(x2 - x) <= NEAR and (any_color or c2 == col) for x2, c2 in rings.get(f, ())): n_hit += 1
    return n_vis, n_hit


def recover(video, onsets, fps, t0, t1, cal=None, dbg=False):
    """戻り値: (足す音符, 外す音符)"""
    global JX
    JX = JX0                      # 判定枠(円の座標)は曲ごとに測り直す(下)
    notes = [o for o in onsets if o['color'] in ('don', 'ka') and o.get('vel') and not o.get('stationary')]
    if len(notes) < 20: return [], []
    # 弱い音符: 途中から現れた・途中で消えた・一部だけ見えた軌跡。最初から最後まで見えていた軌跡は検出の時刻を信じる
    # (密な重なりで円を追いかけ直すと隣の円に引っ張られる: チャーリーダッシュ! 18.63s)
    # 判定枠の手前(帯の外)でだけ見えた音符(emerge: 速さは前後の音符から借りたもの)は、それより前の軌跡を測っていないので確かめない
    # (Emma 裏 玄人 72.25s)。確かめるのは帯の中で軌跡を測った音符だけ
    # 弱い = 軌跡が途中までしか見えなかった(点の数で数えると、30fps の中身の動画では同じ絵が 2 枚ずつ続くので全部の音符が弱くなった:
    # 段位 CARMINE ふつう。見えていた長さの「半分」で線を引くのもあいまいだった: カオスタイム 64.82s の幽霊は x1715〜987 だけ)
    def _partial(o):
        # 途中までしか見えなかった軌跡: 右端(出現位置)より小音符の直径以上手前で現れた、または帯の左端(判定枠の手前)より直径以上右で消えた
        xr = o.get('xr')
        return xr is not None and (xr[0] < X_IN - 2 * R_S or xr[1] > X0_BAND + 2 * R_S)
    # 断片を直線にまとめて拾い直した音符(line)も弱い音符(追い越す音符と一緒に動いて見える下の音符の三日月をつないだ幽霊がある: Hurtling Boys 40.59s)
    # 遅い音符(detect2 の slow: 1 フレームの動きが候補の位置のぶれ程度)は、軌跡から出した時刻が当てにならない(0.9px/フレームで 2 本の軌跡が 0.57 秒ずれる)ので
    # 軌跡の上で縁を確かめない。時刻は synth.slow_hits が「叩かれて消えたフレーム」で決め、同じ音符の軌跡をまとめる(笑える 102.03s)
    # 点が直線に乗らない軌跡(残差の標準偏差が NEAR を超える)も、2 つの音符の点が混ざったものかもしれないので弱い音符として確かめる
    # (彁 27.50s: 速さ 9.8(周りは 10.8)・残差 7.2px の軌跡が、隣の本物のカッと 1 つ置いた次の音符の点をつないだ幽霊だった)
    # 連打の終わりの丸の位置にあるドン(detect2 の capsus)も白い縁で確かめる
    weak = [o for o in notes if not o.get('emerge') and not o.get('slow') and (o.get('emerged') or o.get('rescued') or o.get('line') or o.get('ghost_suspect') or _partial(o)
                                                                             or o.get('res_sd', 0.0) > NEAR or o.get('capsus'))]
    wins_o = _windows(notes, fps, t0, t1)                   # 追い越しのある区間(隠れた音符を探す): 全フレーム
    # 同じ速さで重なった組: 後ろの音符は前の音符からはみ出した細い三日月(色の塊の幅が内円の半径未満)の軌跡として取れる。追い越しが無くても、
    # その通り道を全フレーム調べる(挟まれた真ん中の音符は三日月も取れない: 検出テスト 66.12s の 25ms 間隔のドン 3 つ)
    thin_ = [o for o in notes if not o.get('big') and 0 < o.get('wmed', 99) < RIN_S_DET]
    if thin_:
        if dbg: print('  ringtrack: 重なった組の三日月の軌跡 %d 本' % len(thin_))
        wins_o = _merge(wins_o + [[o['t'] - (X_IN - JX) / (abs(o['vel']) * fps) - 0.1, o['t'] + 0.05] for o in thin_])
    wins_w = _merge([[o['t'] - (X_IN - JX) / (abs(o['vel']) * fps) - 0.1, o['t'] + 0.05] for o in weak])   # 弱い音符の通り道(確かめるだけ): 3 フレームおき
    wins_w = _subtract(wins_w, wins_o)
    wins = _merge(wins_o + wins_w)
    if not wins: return [], []
    if dbg: print('  ringtrack: 区間 ' + ' '.join('%.1f-%.1f' % tuple(w) for w in wins))
    import detect2
    import time as _tm; _t0 = _tm.time()
    rings, labs = _read(video, [(a, b, 1) for a, b in wins_o] + [(a, b, 3) for a, b in wins_w], cal, fps)
    if dbg: print('  ringtrack: 読み込み %.1fs(%d 枚、区間 %d、円と画素の種類 %.1fs)' % (_tm.time() - _t0, len(rings), _read.n_win, _read.t_hough))
    frames = sorted(rings)
    # 連打・風船(描かれる棒・しっぽ)も上に重なる物として扱う: (頭, 終わり, 速さ)
    # 風船には棒が無い(頭と右の風船の絵だけ: 頭〜終わりを 0 にして、下の「終わり + 半径 2 つ」で風船の絵を覆う)。連打の終わりは次の連打・風船の頭より前のものだけ
    # (終わりの取れなかった風船に、ずっと後の連打の終わりまでの棒を付け、その間の画面を全部隠れている扱いにしていた: UNDEAD HEART 88.76s の
    # 風船を叩く間の赤い風船の演出を確かめられず残した)
    ends = sorted(o['t'] for o in onsets if o['color'] == 'roll_end')
    heads = sorted(o['t'] for o in onsets if o['color'] in ('renda', 'balloon'))
    others = []
    for o in onsets:
        if o['color'] in ('renda', 'balloon') and o.get('vel'):
            nh = next((t for t in heads if t > o['t']), 1e9)
            te = o['t'] if o['color'] == 'balloon' else next((t for t in ends if o['t'] < t < nh), o['t'])
            others.append((o['t'], te, abs(o['vel']), R_B if o.get('big') else R_S))
    # 音符ごとの位置のずれ(密な連打では検出の時刻が数 ms ずれる): その音符で説明できた円の中心と予想位置の差の中央値
    nt = np.array([o['t'] for o in notes]); nv = np.array([abs(o['vel']) for o in notes])
    # 見つかっている音符も、その音符の円でずっと追いかけて軌跡を当てはめ直す(速さが徐々に変わる区間や密な連打では、検出の時刻・速さから出した
    # 予想位置が 15px 以上ずれ、その音符自身の円を「説明できない円」として重複を足していた)。
    # 予想位置から小音符の半径の半分(大音符はさらに半径の差)以内の円を集め、外れ(NEAR 超)を除いて直線を当てる
    gate = np.array([R_S / 2 + (R_B - R_S if o.get('big') else 0) for o in notes], float)
    allp = [(f, x, col) for f in frames for x, col in rings[f] if JX + 40 < x < X_IN]
    Fa = np.array([p[0] for p in allp], float); Xa = np.array([p[1] for p in allp], float); Ca_ = np.array([p[2] for p in allp])
    fit = {}; rng = np.random.default_rng(0)
    # 各円を、予想位置が最も近い音符に割り当てる(密な連打では隣の音符の円も近くにあり、固定の範囲で集めると隣に乗り移った: ドンカマ2000 112.8s)
    # 判定枠の位置を円の座標で測り直す: 検出(色の塊の中心)の時刻から出した予想位置と、白い縁の円の中心は 2px ほどずれる(チャーリー +1.9、
    # カオスタイム +1.8)。確かな小音符に最も近い円(12px 以内)の差の中央値だけ判定枠をずらす(3px のずれで、隣と 29px で並ぶ小ドンの縁を
    # 1 枚も確かめられず本物を外した: チャーリー 18.63s)
    weak_ids = {id(w) for w in weak}
    for _ in range(2):
        D = np.abs(Xa[:, None] - (JX + (nt[None, :] * fps - Fa[:, None]) * nv[None, :]))
        owner = np.argmin(D, axis=1) if len(Xa) else np.zeros(0, int)
        dmin = D[np.arange(len(Xa)), owner] if len(Xa) else np.zeros(0)
        sm = np.array([not notes[k].get('big') and id(notes[k]) not in weak_ids for k in owner], bool) & (dmin < 12) if len(Xa) else np.zeros(0, bool)
        if sm.sum() < 50: break
        dx_ = float(np.median((Xa - (JX + (nt[owner] * fps - Fa) * nv[owner]))[sm]))
        if dbg and _ == 0: print('  ringtrack: 円の中心と予想位置の差 %+.2fpx(%d 個)' % (dx_, sm.sum()))
        JX += dx_
    for k, o in enumerate(notes):
        # 自分に割り当てられた円(予想位置から小音符の半径以内。大音符はさらに半径の差)のうち、一直線(速さは検出の ±10%)に最も多く乗る組を選んで
        # 当てはめる(RANSAC)。見えていた場所(判定枠の近く・右側だけ)によらない
        if o.get('big'): continue                 # 円の検出は小音符の大きさ(半径 36〜50)だけなので、大音符自身の円は無い(隣の円に乗り移る)
        if not any(o is w for w in weak): continue   # 追いかけ直すのは弱い音符だけ
        idx = np.flatnonzero((owner == k) & (dmin < gate[k] + R_S / 2))
        if len(idx) < 3: continue
        best = None
        for _ in range(60):
            i, j = rng.choice(idx, 2, replace=False)
            if Fa[i] == Fa[j]: continue
            vv = (Xa[i] - Xa[j]) / (Fa[j] - Fa[i])
            if vv <= 0 or abs(vv / nv[k] - 1) > 0.1: continue
            fc = Fa[i] + (Xa[i] - JX) / vv
            inl = idx[np.abs(JX + (fc - Fa[idx]) * vv - Xa[idx]) <= NEAR]
            if best is None or len(inl) > len(best): best = inl
        if best is None or len(best) < 3: continue
        (c0, vv), *_ = np.linalg.lstsq(np.vstack([np.ones(len(best)), -Fa[best]]).T, Xa[best] - JX, rcond=None)
        if vv <= 0 or abs(vv / nv[k] - 1) > 0.1: continue
        # よく測れた軌跡(残差 NEAR/2 以下)の当て直しが元の軌跡から 2×NEAR 以上ずれたら、隣の別の音符の円に乗り移ったもの。採らない
        # (乱数調整 116.68s: 残差 0.4px のドンの当て直しが 21px 後ろの隠れたドンの円に乗り移り、その円が説明済みになって隠れたドンを足せなかった)
        if o.get('res_sd', 99.0) <= NEAR / 2 and np.median(np.abs(JX + (c0 / vv - Fa[best]) * vv - (JX + (nt[k] * fps - Fa[best]) * nv[k]))) > 2 * NEAR: continue
        fit[k] = (c0 / vv, vv); o['_fit'] = (c0 / vv / fps, vv); o['_nin'] = len(best)
        o['_dx'] = float(np.median(JX + (c0 / vv - Fa[best]) * vv - (JX + (nt[k] * fps - Fa[best]) * nv[k])))
    # 途中から現れた短い軌跡は判定枠の近くの数点で測っているので、追いかけ直した軌跡(円がずっと多い)の時刻と速さに置き換える
    # (カオスタイム 63.025 → 62.974。参照 62.98)
    for o in notes: o['_obs'] = (o['t'], abs(o['vel']))        # 検出が見た軌跡(見えていた区間の比較に使う)
    refitted_ = []                                             # 当て直した軌跡に時刻を置き換えた音符(下の判定枠の手前だけの音符とのまとめに使う)
    for k, o in enumerate(notes):
        if k in fit and (o.get('emerged') or o.get('emerge') or o.get('rescued')) and not o.get('front_ok'):     # 判定枠で時刻を確かめた音符は置き換えない
            o['t'] = round(fit[k][0] / fps, 4); o['vel'] = round(float(fit[k][1]), 2); refitted_.append(o)
    expl = np.zeros(len(allp), bool)
    # 連打・風船の頭、連打の終わりの丸も白い縁の円(resume:story 108.84s・Xa 裏 113.54s で、連打の終わりの丸を音符として足していた)
    for o in onsets:
        if o['color'] in ('renda', 'balloon', 'roll_end') and o.get('vel'):
            expl |= np.abs(JX + (o['t'] * fps - Fa) * abs(o['vel']) - Xa) < R_S
    for k in range(len(notes)):
        if k in fit:
            fc, v_ = fit[k]; expl |= np.abs(JX + (fc - Fa) * v_ - Xa) <= NEAR * 2
        else:
            # 説明済みとする幅は、その音符の位置の確かさ(軌跡の残差)で決める。直線にきれいに乗る音符の円は 3×NEAR(18px)以内にしか出ない
            # (重なった円の中心は 15px ずれて測られることがある: ドッペルゲンガー 裏 57.93s)。
            # 固定の 25px だと、22px ずつずれて重なった組の真ん中の円を前後の音符の円として片付けた(検出テスト 66.12s: 25ms 間隔のドン 3 つ)
            # 遅い音符(detect2 の slow)は時刻そのものが当てにならない(叩かれて消えたフレームで決める)ので狭めない(笑える2000 102.51s: 速さ 0.9 の音符の円が
            # 説明できずに重複の直線になった)
            # 大音符は縁の内側にも小音符の大きさの円が出るので、半径の差(R_B − R_S)の分は残す(雲の彼方 76.51s: 大ドンの 23ms 後ろに小ドンの直線を足した)
            # 狭めるのは右端から判定枠の手前まで追えた軌跡の音符だけ。途中から現れた・拾い直した・判定枠の手前でだけ見えた音符は、残差が小さくても
            # 見えていた部分の中心が本当の位置からずれる(半分隠れた円の中心は 4R/3π まで)ので 25px のまま(いろはすてっぷ! 33.50s: 右上の演出に隠れて
            # 途中から現れたドン(残差 0.4px)の縁の円が軌跡の 13〜29px 後ろに並び、18px の外の円で重複の直線を足した)
            full_k = not (notes[k].get('emerged') or notes[k].get('rescued') or notes[k].get('emerge') or notes[k].get('line') or _partial(notes[k]))
            sd_k = notes[k].get('res_sd', 99.0) if full_k else 99.0
            gk_ = gate[k] if notes[k].get('slow') else min(gate[k], max(3 * NEAR, 2.5 * sd_k) + (R_B - R_S if notes[k].get('big') else 0))
            dk_ = np.abs(JX + (nt[k] * fps - Fa) * nv[k] - Xa)
            # 最も近い円だけにするのは、予想位置が確かな音符(右端から判定枠の手前まで追えた軌跡)だけ。判定枠の手前でだけ見えた・途中から現れた・
            # 拾い直した音符は、速さを数点や前後の音符から借りていて、右の方の予想位置が本物の円から外れる。外れた予想位置の「最も近い円」は隣の音符の円で、
            # 自分の円が説明されずに残り、同じ音符の直線をもう 1 本足した(一括試験 FUJIN Rumble 113.67s: 5 点・速さ 8.93 の emerge、本当は 9.5)
            if notes[k].get('big') or notes[k].get('slow') or not full_k or os.environ.get('V2T_RING1', '1') == '0':
                expl |= dk_ < gk_
            else:
                # 1 つの音符が 1 フレームに写す白い縁は 1 つ。フレームごとに予想位置に最も近い円と、その二重の検出(白い縁の帯の内と外の縁で
                # 中心が縁の幅(R_S − RIN_S_DET = 10px)+ 2px までずれる: いろはすてっぷ! 33.5s は 4〜12px)だけを説明済みにする。幅の中のそれより
                # 離れた円は、重なった別の音符の縁(検出テスト ファイナル 49.8s: 96 分の 3 連は 15px 間隔で、真ん中の円を先頭の音符の円として片付けた)
                ik_ = np.flatnonzero(dk_ < gk_)
                if len(ik_):
                    o_ = np.lexsort((dk_[ik_], Fa[ik_])); ik_ = ik_[o_]
                    first_ = np.r_[True, Fa[ik_][1:] != Fa[ik_][:-1]]
                    head_ = np.maximum.accumulate(np.where(first_, np.arange(len(ik_)), 0))
                    expl[ik_[np.abs(Xa[ik_] - Xa[ik_][head_]) <= R_S - RIN_S_DET + 2]] = True
    pts = [p for p, e in zip(allp, expl) if not e]
    if dbg and os.environ.get('V2T_RT_TRACE'):
        tq = float(os.environ['V2T_RT_TRACE']) * fps
        near = [(p[0], round(p[1]), p[2], bool(e)) for p, e in zip(allp, expl) if abs(p[0] + (p[1] - JX) / float(os.environ.get('V2T_RT_TRACE_V', '13.4')) - tq) < 3]
        print('  ringtrack: 追跡用の円(t 近く) ' + ' '.join('%d:%d%s%s' % (f, x, c[0], '*' if e else '') for f, x, c, e in near[:400]))
    # 途中から現れた短い軌跡: 右端からの全行程で、見えるはずの所の過半数で円が見えなければ偽物
    if dbg: print('  ringtrack: 追いかけ直し %.1fs(弱い音符 %d 個)' % (_tm.time() - _t0, len(weak)))
    gone = []
    # 1 つの軌跡に音符は 1 つ: 追いかけ直した軌跡が同じになった音符どうしは、点の多いほうを残す(色の塊の追跡が 1 つのカッを 2 本に分けた:
    # チャーリーダッシュ! 17.42 / 17.44s)
    wn = sorted([o for o in notes if '_fit' in o], key=lambda o: -o.get('npts', 0))
    for i, o in enumerate(wn):
        if any(o is g for g in gone): continue
        for q in wn[i + 1:]:
            if not any(q is g for g in gone) and _same_path(o, q, fps):
                if dbg: print('  ringtrack: 同じ軌跡 %s t=%.3f(点 %d) と %s t=%.3f(点 %d) → 後者を外す' % (o['color'], o['t'], o.get('npts', 0), q['color'], q['t'], q.get('npts', 0)))
                gone.append(q)
                # 同じと決めたのは当て直した軌跡どうしなので、残す方の時刻も当て直した軌跡にする(彁 27.50s: 2 つの音符の点が混ざった軌跡 27.501/9.8 の
                # 当て直しが 27.460/10.78 で、本物のカッ 27.458 と同じ軌跡。点の多い混ざった方を残し、時刻は混ざったままだった)
                if '_fit' in o: o['t'] = round(o['_fit'][0], 4); o['vel'] = round(float(o['_fit'][1]), 2)
    # 弱い軌跡が見えていた区間で、確かな音符と位置の差が「半円の重心のずれ(4R/3π ≈ 21px)+ 測りの幅」未満なら、その確かな音符を
    # 半分隠れた状態で別の軌跡として拾ったもの(半分隠れた円の中心は見えている側にずれて測られる: チャーリーダッシュ! 17.44s と 17.418s のカッ)
    strong_ = [q for q in notes if not any(q is w for w in weak) and not q.get('emerge')]
    tol_c = 4 * R_S / (3 * np.pi) + NEAR
    for o in weak:
        xr = o.get('xr')
        if xr is None or any(o is g for g in gone): continue
        # 半分隠れた状態で拾ったのは、途中までしか見えなかった軌跡だけ。端から端まで見えていて残差が大きいだけのもの(下の「点が直線に乗らない」)は
        # 対象にしない(Rotter Tarmination 98.13s・105.33s: 残差 7.9px で右端から判定枠まで見えていた本物のドンを、前のドンの半分隠れた姿として外した)
        if not (o.get('emerged') or o.get('rescued') or o.get('line') or _partial(o)): continue
        t0_, v0 = o.get('_obs', (o['t'], abs(o['vel'])))       # 見えていた区間は検出が測った軌跡で比べる(当てはめ直す前)
        fs_ = np.arange(t0_ * fps - (xr[0] - JX) / v0, t0_ * fps - (xr[1] - JX) / v0 + 1, 1.0)
        xo_ = JX + (t0_ * fps - fs_) * v0
        for q in strong_:
            if abs(q['t'] - o['t']) > 1.0: continue          # 色は問わない(重なった塊の色には隣の音符の色が混ざる: チャーリーダッシュ! 17.35s)
            # 全体が見えている円は半分隠れた円より正確に測れる。確かな音符の方が当てはめが荒いなら、それは 2 つの音符が混ざった軌跡で、弱い方が本物の 1 つ
            # (平等院鳳凰 39.57s: 30px 間隔の 2 つのドンで、確かな方の残差 7.1px、弱い方 0.3px)
            if q.get('res_sd', 0.0) > max(o.get('res_sd', 0.0), NEAR / 2): continue
            # 半分隠れた円の中心のずれは、見えている量が変わるにつれて 0〜4R/3π の間で変わる。だから見えていた区間の両端での差の変化(速さの差 × 区間の
            # フレーム数)も tol_c 以内。それを超えて速さが違う 2 つの軌跡は別の音符(区間の途中で交差したのを中央値で同じとみなし、本物のカッを外した:
            # 彁 58.59s は速さ 10.9 と 12.3 で 76 フレーム → 差の変化 106px。チャーリーダッシュ! 17.22s の半分隠れたカッは 15.4 と 15.8 で数 px)
            d_ = JX + (q['t'] * fps - fs_) * abs(q['vel']) - xo_
            # 上に描かれる音符(譜面で先)に隠れて見えるのは後ろ側(右)なので、半分隠れた円の中心は本当の位置より右(判定枠から遠い側)にずれる。
            # 弱い軌跡が確かな音符より測りの幅以上前(判定枠の側)にあるなら、隠れた姿ではない(乱数調整 123.18s: 4.4px/フレームで積み重なったドンの
            # 23px 後ろの同じ色のドンを、前にある方の隠れた姿とみなして消していた)
            if np.median(d_) > NEAR: continue
            if np.median(np.abs(d_)) < tol_c and abs(d_[-1] - d_[0]) < tol_c:
                # 1 つの音符から 1 フレームに写る円は 1 つ。弱い軌跡と確かな音符の両方の位置(NEAR より離れている)に同じフレームで白い縁の円が写るフレームが
                # MIN_PTS 以上あれば、別々の 2 つの音符(とける 102.78s: 1 フレーム 5.4px の遅いカッが 22px 間隔で積み重なり、本物を隣の半分隠れた姿として消した)
                both_ = 0
                for f_ in np.unique(Fa[(Fa >= fs_.min() - 0.5) & (Fa <= fs_.max() + 0.5)]):     # 円を探したフレーム(ゲームの時計の小数のフレーム)
                    xw_ = JX + (t0_ * fps - f_) * v0; xs_ = JX + (q['t'] * fps - f_) * abs(q['vel'])
                    if abs(xw_ - xs_) <= NEAR: continue
                    sel_ = Fa == f_; xr_f = Xa[sel_]; cr_f = Ca_[sel_]
                    # 弱い軌跡の位置の円は、中身が弱い軌跡と同じ色で、ほかの音符(比べている 2 つ以外)の予想位置で説明できないものだけ
                    # (チャーリーダッシュ! 17.45s: 近くの別の音符の円や、カッの円のぶれをドンの弱い軌跡の円として数えた)
                    oth_x = np.array([JX + (n_['t'] * fps - f_) * abs(n_['vel']) for n_ in notes if n_ is not o and n_ is not q and n_.get('vel')])
                    xw_c = [x for x, c_ in zip(xr_f, cr_f) if c_ == o['color'] and abs(x - xw_) <= NEAR and not (len(oth_x) and np.min(np.abs(oth_x - x)) <= NEAR)]
                    if xw_c and np.min(np.abs(xr_f - xs_)) <= NEAR: both_ += 1
                if both_ >= MIN_PTS:
                    if dbg: print('  ringtrack: 弱い軌跡 %s t=%.3f と確かな音符 t=%.3f は同じフレームに別々の円が %d 枚 → 別の音符' % (o['color'], o['t'], q['t'], both_))
                    continue
                # 色が違うとき: 確かな音符に隠れずにはみ出して見える所の中身が、弱い軌跡の色だったフレームが偶然とは言えないほど多ければ、
                # 下にもう 1 つ別の色の音符がある(乱数調整 123.18s: 4.4px/フレームで 24px 後ろに積み重なったドン。円は隠れて取れないが、
                # はみ出した三日月はずっと赤)
                # 位置の差が測りの幅(2×NEAR)未満なら、はみ出した中身の色は別の音符の証拠にならない(クラシックメドレー(冬休み編) 72.83s: カッの 9px 後ろの
                # ドンの軌跡を、赤が 5 フレーム見えたとして別の音符にした)
                # 距離は円で追い直した軌跡があればそれで測る(色の塊の中心は半分隠れると後ろへずれる: 見えていた区間では 28px、円では 9px)
                sep_ = abs(o['_fit'][0] - q['t']) * abs(q['vel']) * fps if '_fit' in o else abs(float(np.median(d_)))
                if o['color'] != q['color'] and sep_ >= 2 * NEAR:
                    _shape(t0_, v0, bool(o.get('big')), labs, [n_ for n_ in notes if n_ is not o], others, fps)
                    fd_, fk_ = _shape.fvote
                    n_own, n_oth = (fk_, fd_) if o['color'] == 'ka' else (fd_, fk_)
                    if n_own >= MIN_PTS and _binom_tail(n_own, n_own + n_oth, True) < P_SIG:
                        if dbg: print('  ringtrack: 弱い軌跡 %s t=%.3f ははみ出した中身が自分の色(%d 対 %d フレーム)→ 確かな音符 t=%.3f とは別' % (o['color'], o['t'], n_own, n_oth, q['t']))
                        continue
                if dbg: print('  ringtrack: 弱い軌跡 %s t=%.3f は確かな音符 t=%.3f を半分隠れた状態で拾ったもの' % (o['color'], o['t'], q['t']))
                gone.append(o); break
    # 点が直線に乗らない(残差 > NEAR)小音符で、その速さ(±10%)で一直線に並ぶ円が 3 つも無いものは、動く円の裏付けが無い。2 つの音符の点を混ぜた幽霊
    # (λ7708 72.90s・73.08s: 残差 7〜9px・速さ 12.6/12.8 で、周りの本物は 11.1)
    for o in weak:
        if not o.get('big') and o.get('res_sd', 0.0) > NEAR and '_fit' not in o and not any(o is g for g in gone):
            if dbg: print('  ringtrack: 点が直線に乗らず円の裏付けも無い %s t=%.3f(残差 %.1f)' % (o['color'], o['t'], o['res_sd']))
            gone.append(o)
    # この動画で縁の白が確かめられるか: 確かな音符(最初から最後まで見えた・直線にきれいに乗る)自身の縁の白の割合の中央値。本物でも縁が白く写らない
    # 動画(音符の絵柄・色の違う録画)では「縁が見えない」は偽物の証拠にならない(Lightning Beams: 本物の縁の白 0.23〜0.24 で、49 個の本物を外した)
    ref_ = [q for q in notes if not any(q is w for w in weak) and not q.get('emerge') and not q.get('big') and q.get('res_sd', 99) <= NEAR / 2]
    # 縁を確かめられるのは読んだ区間(labs)を通った音符だけ(区間の外の音符は確かめるフレームが無い)
    wr_ = []
    for q in ref_:
        r_ = _shape(q['t'], abs(q['vel']), False, labs, [n_ for n_ in notes if n_ is not q], others, fps)
        if r_[0] >= MIN_PTS: wr_.append(r_[1])
        if len(wr_) >= 40: break
    w_ref = float(np.median(wr_)) if len(wr_) >= MIN_PTS else 1.0
    rim_ok = w_ref >= 0.6
    if dbg: print('  ringtrack: 確かな音符の縁の白の割合 %.2f(%d 個)%s' % (w_ref, len(wr_), '' if rim_ok else ' → 縁が白く写らない動画なので、縁が見えないことでは外さない'))
    # 確かめる音符: 弱い音符(追い越す音符の点を乗り継いだ幽霊もここに入る: カオスタイム 64.82s は点 31 / 横切る間 77 フレーム)
    wk_ = [w for w in weak if not any(w is g for g in gone)]
    for k_, o in enumerate(wk_):
        if k_ % max(1, len(wk_) // 20) == 0: detect2.progress('隠れた音符', k_ + 1, len(wk_), '途中から現れた軌跡が本物か、白い縁で確かめる')
        occ_ = [q for q in notes if q is not o]
        # 確かめる軌跡: 円で当て直した軌跡と、追跡が測った元の軌跡の両方で確かめ、自分の大きさの縁がよく見える方を採る。偽物と決めるのは、どちらの
        # 位置にも縁が無いときだけ(同じ大きさの音符が等間隔に並ぶ所で当て直しが 19px ずれ、縁の無い所を確かめて本物を外した: ドキドキ胸きゅん 84.17s)
        paths_ = []
        for tv in (o.get('_fit'), o.get('_obs'), (o['t'], abs(o['vel']))):
            if tv is not None and not any(abs(tv[0] - q[0]) < 1e-4 and abs(tv[1] - q[1]) < 1e-3 for q in paths_): paths_.append(tv)
        best_ = None
        for t_c, v_c in paths_:
            # 大きさは円の大きさで決まる: 小・大の両方の白い縁で確かめ、よく乗るほうを採る(ドンカマ2000 44.86s: 速さ 90 の大カッを小音符として検出していた)
            rs_c = _shape(t_c, v_c, False, labs, occ_, others, fps); ss_c = _shape.seen; fv_s = _shape.fvote
            rb_c = _shape(t_c, v_c, True, labs, occ_, others, fps); sb_c = _shape.seen; fv_b = _shape.fvote
            own_c = (rb_c, sb_c) if o.get('big') else (rs_c, ss_c)
            key_ = own_c[1] / max(own_c[0][0], 1)
            if best_ is None or key_ > best_[0]: best_ = (key_, t_c, v_c, rs_c, ss_c, rb_c, sb_c, fv_b if o.get('big') else fv_s)
        _, t_, v_, rs_, ss_, rb_, sb_, fvote_ = best_
        # 大きさは縁がよく乗るほう。ただし外すのは「その音符自身の大きさの縁が、確かめられて、見えない」ときだけで、もう一方の大きさの縁が
        # 見えているなら外さない(チャーリー 18.63s: 29px 間隔の小ドンの縁は隣の縁と重なってどのフレームでも確かめられず、合うはずのない
        # 大の縁の「73 フレーム中 0」で本物を外していた)
        own_, alt_ = ((rb_, sb_), (rs_, ss_)) if o.get('big') else ((rs_, ss_), (rb_, sb_))
        o['_rim'] = (own_[0][0], own_[1])                       # 自分の大きさの縁: (見えるはずのフレーム, 見えた)
        (n_f, w_, col_), seen = (rb_, sb_) if rb_[1] > rs_[1] else (rs_, ss_)
        if dbg: print('  ringtrack: 現れた音符 %s t=%.3f 速さ %.1f(当て直し %.3f/%.2f 円 %s) 大 %s: 縁が見えるはずのフレーム %d 枚、見えた %d 枚、白い縁 %.2f、中身の色 %s' % (o['color'], o['t'], abs(o['vel']), t_, v_, o.get('_nin'), bool(o.get('big')), n_f, seen, w_, col_))
        # 見えている中身の色がその音符の色と違う(上の音符からはみ出して見える所に、下の別の音符の色が見えている)なら、その音符は無い
        # (チャーリーダッシュ! 17.44s: 17.42s のカッを 2 本に分けた片方。はみ出して見えるはずの所は下のドンの赤)
        # 違う色と決めるのは、中身が違う色だったフレームが、自分の色だったフレームより偶然とは言えないほど多いときだけ(画素を全部足した多数だと、
        # 密な重なりで見えている中身が数画素のとき隣の色が少し混ざるだけで逆転し、縁が 63 フレームとも見えている本物を外した: 乱数調整のリバースシンデレラ 117.30s)
        n_other = fvote_[0] if o['color'] == 'ka' else fvote_[1]; n_own = fvote_[1] if o['color'] == 'ka' else fvote_[0]
        if n_other + n_own >= MIN_PTS and _binom_tail(n_other, n_other + n_own, True) < P_SIG and n_f >= MIN_PTS:
            if dbg: print('  ringtrack: 現れた音符 %s t=%.3f の中身が別の色(フレーム 別 %d / 自分 %d、当て直し %.3f/%.2f)→ 外す' % (o['color'], o['t'], n_other, n_own, t_, v_))
            gone.append(o); o['_gone_col'] = (t_, v_); continue
        if dbg and os.environ.get('V2T_RT_DUMP') and abs(o['t'] - float(os.environ['V2T_RT_DUMP'])) < 0.01: _dump(o, t_, v_, labs, occ_, fps)
        # 小音符の追い越しの覗きの疑い(detect2._peek_small: 速い音符の縁のすぐ後ろを一緒に動き、下に速さの違う同じ色の音符がある)は、位置の証拠が
        # 覗きの幽霊と同じなので、自分の大きさの白い縁が偶然以上に見えたときだけ残す(検出テスト ファイナル 72.79s・72.88s: 縁が見えるはずの 8〜9 枚のうち
        # 見えたのは 2〜3 枚。速い音符は写るフレームが少なく、下の「見えないことがはっきりしたら外す」では決めきれなかった)
        if o.get('peek_suspect') and not (own_[0][0] >= MIN_PTS and _binom_tail(own_[1], own_[0][0], True) < P_SIG):
            if dbg: print('  ringtrack: 覗きの疑いの %s t=%.3f は自分の縁が見えない(%d / %d 枚)→ 外す' % (o['color'], o['t'], own_[1], own_[0][0]))
            gone.append(o); continue
        # 縁が見えないフレームが偶然とは言えないほど多いものだけ外す(弱い音符に「見えた証拠」を求めると、ゴーゴーの演出などで縁を確かめられない
        # 本物を外した: POLARiSNAUT 13 個)
        # 縁が見えたフレームが偶然とは言えないほど少なく、しかも縁の点の白の割合の平均が「見えた」の基準(過半)の 8 割(0.4)に届かないものだけ外す。
        # 縁の半径・太さがわずかにずれた本物は白の割合が 0.43〜0.49 に並び、見えた 0〜5 枚になる(ネクロファンタジア 163.76s・169.01s、いろはすてっぷ! 33.51s)。
        # 偽物は 0.33 以下(Turquoise Tachometer 24.15s 0.33、トロいか2000 100.41s 0.25、チャーリーダッシュ! 19.52s 0.11)
        # 連打の終わりの丸の位置のドン(capsus)は、位置からは丸と見分けられないので、白い縁が偶然以上に見えたこと(本物の証拠)を求める
        # (そして勇者は眠りにつく 164.91s: 32 フレーム中 0 枚。後ろに白い縁の黄色い連打の頭が並ぶので白の割合は 0.43)
        # 縁で分からないとき(後ろに並んだ連打の頭の縁と重なり確かめられない: 本物の resume:story 71.72s も同じ)は、すぐ左(判定枠の側)の棒の位置を見る。
        # 叩いている連打の棒は赤く、その端の丸なら左に接する所もずっと赤い。棒と別の本物のドンの左は棒の端の丸の外で赤くない
        if o.get('capsus') and not (own_[0][0] >= MIN_PTS and _binom_tail(own_[1], own_[0][0], True) < P_SIG):
            # 連打の棒は「終わり + 半径 2 つ」まで覆う扱いなので、棒のすぐ後ろの本物のドンは縁を確かめられない。棒で隠れていないものとして縁を見直し、
            # 棒からはみ出した側に白い縁が偶然以上に見えれば本物(検出テスト 103.62s: 叩いた大連打の赤い棒の端の下に半分隠れた顔のあるドン。
            # 棒の端の丸には白い縁が無い)
            rq_ = _shape(t_, v_, bool(o.get('big')), labs, occ_, [], fps); sq_ = _shape.seen
            rim_nobar = rq_[0] >= MIN_PTS and _binom_tail(sq_, rq_[0], True) < P_SIG
            nb_, nr_ = _left_red(t_, v_, labs, fps)
            if dbg: print('  ringtrack: 連打の終わりの丸の位置のドン t=%.3f: 棒を除いた縁 %d/%d・左の赤 %d/%d' % (o['t'], sq_, rq_[0], nr_, nb_))
            if not rim_nobar and nb_ >= MIN_PTS and _binom_tail(nr_, nb_, True) < P_SIG:
                if dbg: print('  ringtrack: 連打の終わりの丸の位置のドン t=%.3f は左に赤い棒が続く(%d / %d フレーム)→ 棒の端として外す' % (o['t'], nr_, nb_))
                gone.append(o); continue
        if (rim_ok and own_[0][0] >= MIN_PTS and _binom_tail(own_[1], own_[0][0], False) < P_SIG and own_[0][1] < 0.4
                and not (alt_[0][0] >= MIN_PTS and _binom_tail(alt_[1], alt_[0][0], True) < P_SIG)): gone.append(o)
    notes = [o for o in notes if not any(o is g for g in gone)]
    if dbg: print('  ringtrack: 偽物の確かめ %.1fs' % (_tm.time() - _t0))
    if dbg: print('  ringtrack: 説明できない円 %d 点' % len(pts))
    if len(pts) < MIN_PTS: return [], gone
    # 一直線に並ぶ組: 速さを 1% 刻みで試し、(判定枠に着くフレーム, 速さ, 色) に投票
    F = np.array([p[0] for p in pts], float); X = np.array([p[1] for p in pts]); C = np.array([p[2] for p in pts])
    vmin, vmax = nv.min() / 1.2, nv.max() * 1.2
    vs = [vmin]
    while vs[-1] < vmax: vs.append(vs[-1] * VSTEP)
    votes = {}
    for iv, v in enumerate(vs):
        fc = np.round(F + (X - JX) / v).astype(int)
        for col in ('don', 'ka'):
            m = C == col
            u, n = np.unique(fc[m], return_counts=True)
            for a, c in zip(u, n):
                if c >= MIN_PTS: votes[(int(a), iv, col)] = int(c)
    cands = sorted(((n, a, iv, col) for (a, iv, col), n in votes.items()), reverse=True)
    # 候補の直線を先に全部作り、白い縁の証拠が強いものから採る。採った直線の縁は説明済みとして、残りを確かめ直す
    # (1 つの隠れた音符から 2 本の直線ができたとき、先に確かめた方が後の方の証拠を奪わないように: カオスタイム 58.96 / 59.04s)
    lines = []
    used = np.zeros(len(pts), bool)
    tt_n, vv_n, sd_n = _paths(notes); em_n = _paths.em
    strong = [o for o in notes if not any(o is w for w in weak) and not o.get('emerge')]
    # 速さの手本: 確かな音符と、速さの推定の誤差が許容(3%)より十分小さい(1% 未満)弱い音符。誤差 = 残差 × √(12/点の数) ÷ 見えていたフレーム数
    # (弱い音符を全部外すと、密な区間で同じ速さの音符が全部「弱い」になり本物を落とした: λ7708 73.04s。判定枠の手前の数点から推定した速さは誤差が大きく
    # 手本にならない: チャーリーダッシュ! 17.77s)
    def _v_err(o):
        xr_ = o.get('xr'); v_ = abs(o.get('vel') or 0)
        if not xr_ or v_ <= 0 or not o.get('npts'): return 1.0
        span_ = max((xr_[0] - xr_[1]) / v_, 1.0)
        return max(o.get('res_sd', NEAR), 0.5) * np.sqrt(12.0 / max(o['npts'], 2)) / span_ / v_
    # 点が直線に乗らない軌跡は、隣の音符の点を乗り継いだものかもしれないので手本にしない。点が多く区間が長いと誤差の式では速さが確かに見えるが、
    # 乗り継ぎの誤差は偶然のばらつきではない(検出テスト ファイナル 54.1〜55.0s: 22px 間隔の同じ色の 64 分の壁で、残差 7〜10px・速さ 16〜26 の乗り継ぎの
    # 軌跡を手本にして、格子を斜めに貫く偽の直線を 24 本足した)。「乗らない」はその場の普通の測りのぶれと比べる: 前後 2 秒の確かな音符の残差の中央値の
    # 2 倍(3px 以上)を超える軌跡は手本にしない(確かな音符にも掛ける。54.1s の残差 5.97px・速さ 16.62 の乗り継ぎは確かな音符の扱いで、周りは 0.3〜2.6px。
    # 画面が揺れる動画は全部の音符の残差が大きく、上限も一緒に上がる: 第六天魔王 裏の残差 7px)
    ST_ = np.array([o['t'] for o in strong], float); SR_ = np.array([o.get('res_sd') if o.get('res_sd') is not None else np.nan for o in strong], float)
    def _res_ok(o):
        r_ = o.get('res_sd')
        if r_ is None: return True
        m_ = (np.abs(ST_ - o['t']) <= 2.0) & np.isfinite(SR_)
        lim_ = max(NEAR / 2, 2 * float(np.median(SR_[m_]))) if m_.sum() >= 3 else NEAR
        return r_ <= lim_
    exemplar = ([o for o in strong if _res_ok(o)]
                + [o for o in notes if any(o is w for w in weak) and not o.get('emerge') and _v_err(o) < 0.01 and o.get('res_sd', 0.0) <= NEAR and _res_ok(o)])
    # 積み重なりで隠れる相手は、先に候補になった直線でもよい(POLARiSNAUT 62.71s: 44ms 間隔で 3 つ積み重なったカッの 3 つ目。2 つ目も隠れた直線で、
    # 票の多い 3 つ目を先に調べたときはまだ無かった)。隠れようがないとした候補は、直線が増えたら調べ直し、頼った直線を dep に持たせる
    deferred = []
    wk_ = []          # 速さが手本と 3% を超えて違うことだけで採らなかった直線のうち、隣の音符との重なりで説明できるもの(判定枠の顔の裏付けにだけ使う)
    # 候補ごとに音符を 1 つずつ比べると、密な区間(説明できない円 5000 点・候補 3 万本)で 2 分かかった(検出テスト ファイナル 47〜56s)。
    # 同じ比べを配列にまとめる(結果は同じ)
    OT_ = np.array([o['t'] for o in notes], float)                                   # 追い越し: 譜面で先かどうかは元の時刻
    OTT_ = np.array([o.get('_fit', (o['t'], abs(o['vel'])))[0] for o in notes], float)
    OVV_ = np.array([o.get('_fit', (o['t'], abs(o['vel'])))[1] for o in notes], float)
    QV_ = np.array([abs(o['vel']) if o.get('vel') else 0.0 for o in notes], float)   # 積み重なり: 元の速さ(無いものは 0 = 対象外)
    EV_ = np.array([abs(o['vel']) for o in exemplar], float); ET_ = np.array([o['t'] for o in exemplar], float)
    LT_, LV_ = [], []                                                                  # 採った候補の直線の (時刻, 速さ)
    # 小節線の速さもその小節の音符の速さ(BPM × SCROLL。小節線は偽物の軌跡になりにくいので 1 本で手本にする)。小節の中の音符が 1 つも取れて
    # いないとき(検出テスト ファイナル 59.65s: SCROLL 0.1 の 8 分は 1.6px/フレームで積み重なり、手本が無くて縞の直線を足せなかった)
    import detect2 as _d2
    BT_ = np.array(getattr(_d2.run_parallel, 'bars', None) or [], float); BV_ = np.array(getattr(_d2.run_parallel, 'bar_vels', None) or [], float)
    if os.environ.get('V2T_BAR_EX', '1') == '0' or len(BT_) != len(BV_): BT_ = np.zeros(0); BV_ = np.zeros(0)
    def _bar_speed_ok(t_, v_, tol_):
        if not len(BT_): return False
        k_ = int(np.searchsorted(BT_, t_ + 1e-6)) - 1                      # t_ を含む小節の頭の小節線
        return 0 <= k_ < len(BT_) and BV_[k_] > 0 and abs(BV_[k_] / v_ - 1) <= tol_ and t_ - BT_[k_] < 4.0
    def _overtaken_v(h):
        vh = h['vel']; fs = np.arange(h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps, 1.0)
        xh = JX + (h['t'] * fps - fs) * vh
        k_ = OT_ < h['t']
        if not k_.any(): return False
        d = JX + (OTT_[k_, None] * fps - fs[None, :]) * OVV_[k_, None] - xh[None, :]
        return bool(np.any((d.max(axis=1) - d.min(axis=1) >= 2 * R_S) & np.any(np.abs(d) < R_S, axis=1)))
    def _same_lines(h):
        if not LT_: return False
        vh = abs(h['vel']); fs = np.linspace(h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps, 20)
        d_ = np.abs((h['t'] * fps - fs)[None, :] * vh - (np.array(LT_)[:, None] * fps - fs[None, :]) * np.array(LV_)[:, None])
        return bool(np.any(np.median(d_, axis=1) < 2 * NEAR))
    def _try(n, a, iv, col, retry):
        v = vs[iv]
        m = (C == col) & ~used & (np.abs((F + (X - JX) / v) - a) * v <= NEAR * 2)
        if m.sum() < MIN_PTS: return
        A = np.vstack([np.ones(m.sum()), -F[m]]).T
        sol, *_ = np.linalg.lstsq(A, X[m] - JX, rcond=None)
        v2 = float(sol[1])
        if v2 <= 0: return
        fc2 = sol[0] / v2
        inl = (C == col) & ~used & (np.abs(JX + (fc2 - F) * v2 - X) <= NEAR)
        if inl.sum() < MIN_PTS: return
        # 速さは同じ区間の音符と同じ(BPM×SCROLL)。格子のように並んだ円の中では当てはめた速さが 4〜6% ずれ、判定枠まで延ばした時刻が ±15ms ずれて
        # 21ms 間隔の隣の音符と取り違えた(検出テスト ファイナル 54.0〜55.1s: 64 分の同じ色の壁)。前後 1 秒の手本のうち 3% 以内の音符の速さの中央値に
        # 固定し、時刻だけを当てはめ直す(手本 1 つの速さも 16.6〜18.0 とばらつき、それに揃えると隣の音符と 3% 以上違って積み重なりと認めなかった)
        if os.environ.get('V2T_RT_VSNAP', '0') == '1' and len(EV_):      # 既定は使わない(チャーリーダッシュ! 17.6〜18.1s の重なった実際の曲で直線が割れて 1 つ落とした)
            k_ = np.flatnonzero((np.abs(EV_ / v2 - 1) <= 0.03) & (np.abs(ET_ - fc2 / fps) < 1.0))   # 手本と認める幅(下の 3%)と同じ。広げると半端な速さの直線まで拾った(カオスタイム 68.04s)
            # 揃えるのは、手本の速さとのずれが直線自身の速さの誤差(残差 × √(12/点の数) ÷ 見えていたフレーム数)の 3 倍を超えるときだけ。
            # 普通の直線は測ったままの方が確か(揃えて時刻が 3ms 動き、判定枠の手前だけの音符とのまとめから外れた: カオスタイム 68.04s)
            r_ = X[inl] - (JX + (fc2 - F[inl]) * v2); span_ = max(float(F[inl].max() - F[inl].min()), 1.0)
            ve_ = max(float(r_.std()), 0.5) * np.sqrt(12.0 / max(int(inl.sum()), 2)) / span_ / v2
            if len(k_) and abs(v2 / float(np.median(EV_[k_])) - 1) > 3 * ve_:
                v2 = float(np.median(EV_[k_]))
                fc2 = float(np.median(F[inl] + (X[inl] - JX) / v2))
                inl = (C == col) & ~used & (np.abs(JX + (fc2 - F) * v2 - X) <= NEAR)
                if inl.sum() < MIN_PTS: return
                fc2 = float(np.median(F[inl] + (X[inl] - JX) / v2))
        h = {'t': fc2 / fps, 'vel': v2, 'color': col, 'n': int(inl.sum())}
        if dbg and os.environ.get('V2T_RT_TRACE') and abs(h['t'] - float(os.environ['V2T_RT_TRACE'])) < 0.03:
            print('  ringtrack: 候補 %s t=%.3f 速さ %.2f 円 %d 同じ軌跡 %s' % (col, h['t'], v2, h['n'], _same_any(h, tt_n, vv_n, fps, sd_n, em_n) or any(_same_path(o, h, fps) for o in lines)))
        if _same_any(h, tt_n, vv_n, fps, sd_n, em_n) or _same_lines(h): return      # 1 つの軌跡に音符は 1 つ(色を問わない)
        # 速さの違う音符に上を通り過ぎられる(追い越される)こと。この段階の役目は、追い越されて隠れた音符を拾うこと。追い越されない音符の
        # 取りこぼしは追跡の問題(30fps 版の密な並び)で、ここで拾うと既存の音符の測り違いの重複を足した(第六天魔王 裏 120〜127s・KASSEI 91.13s)
        # 積み重なり: 同じ速さ(±3%)で直前(判定枠に先に着く)の音符が、小音符の直径(2R_S)未満の距離で上に重なっていれば隠れうる
        # (乱数調整 116.71s: 4.3px/フレームのドンが 24〜29px 間隔で積み重なり、追い越しが無くても下の音符は欠片しか見えない)
        def _stk(q): return q.get('vel') and abs(abs(q['vel']) / v2 - 1) <= 0.03 and 0 < (h['t'] - q['t']) * v2 * fps < 2 * R_S
        stacked_ = bool(np.any((QV_ > 0) & (np.abs(QV_ / v2 - 1) <= 0.03) & (0 < (h['t'] - OT_) * v2 * fps) & ((h['t'] - OT_) * v2 * fps < 2 * R_S)))
        dep_ = None
        if not stacked_ and retry:
            dep_ = next((q for q in lines if _stk(q)), None)
        if not stacked_ and dep_ is None and os.environ.get('V2T_STK_SNAP', '1') != '0':
            # 積み重なった円の直線は、当てはめた速さが前の音符と 3〜6% ずれることがある(重なった円の中心は隣に引かれる: 検出テスト ファイナル
            # 49.84s の 96 分の 3 連は 16.7〜17.0 と 17.8)。すぐ前(直径未満)に速さの近い(6% 以内)音符があれば、その速さで時刻だけ当てはめ直し、
            # 円が直線に乗れば積み重なりとして扱う(速さを揃えるのはここだけ。採れる直線の時刻を全部揃えると実際の曲が崩れた: V2T_RT_VSNAP)
            for q in sorted(notes, key=lambda q: abs(q['t'] - h['t'])):
                vq_ = abs(q['vel']) if q.get('vel') else 0.0
                if vq_ <= 0 or abs(v2 / vq_ - 1) > 0.06: continue
                m3_ = (C == col) & ~used & (np.abs((F + (X - JX) / vq_) - fc2) * vq_ <= NEAR * 2)
                if m3_.sum() < MIN_PTS: continue
                fc3_ = float(np.median(F[m3_] + (X[m3_] - JX) / vq_))
                if not (0 < (fc3_ / fps - q['t']) * vq_ * fps < 2 * R_S): continue
                inl3_ = (C == col) & ~used & (np.abs(JX + (fc3_ - F) * vq_ - X) <= NEAR)
                if inl3_.sum() < MIN_PTS: continue
                v2 = vq_; fc2 = fc3_; inl = inl3_
                h = {'t': fc2 / fps, 'vel': v2, 'color': col, 'n': int(inl.sum())}
                if _same_any(h, tt_n, vv_n, fps, sd_n, em_n) or _same_lines(h): return
                stacked_ = True
                if dbg: print('  ringtrack: 直線 %s t=%.3f: 前の音符 t=%.3f の速さ %.2f で当てはめ直して積み重なり' % (col, h['t'], q['t'], vq_))
                break
        if not stacked_ and dep_ is None and not _overtaken_v(h):
            deferred.append((n, a, iv, col))
            if dbg: print('  ringtrack: 直線 %s t=%.3f 速さ %.2f: 上を通り過ぎる音符が無い(隠れようがない)ので足さない' % (col, h['t'], v2))
            return
        # 速さは BPM×SCROLL で決まり、前後の音符と同じか少しずつしか変わらない: 前後 2 秒の確かな音符(最初から最後まで追えた軌跡)のどれかと 3% 以内
        # (判定枠の手前の 4 点から推定した速さを手本にすると、半端な速さの直線が通った: チャーリーダッシュ! 17.77s)
        # (速さを 1% 刻みで全部試すので、関係のない円が偶然 5 個並ぶ直線ができる: チャレンジャーズ 125.17s の速さ 29.25)
        # 手本は 1 つでよい。乗り継ぎの幽霊(残差 7〜10px)を手本にして 64 分の壁を斜めに貫く直線を足したこと(検出テスト ファイナル 54.1s)は、
        # 手本を残差 NEAR 以下の軌跡に限ることで防ぐ。手本を 2 つ以上にすると、前後 2 秒にその速さの音符が 1 つしかない本物の隠れた音符を落とした
        # (一括試験 8OROCHI 125.74s・Rough: CHA0Z 46.17s)
        if int(np.sum((np.abs(EV_ / v2 - 1) <= 0.03) & (np.abs(ET_ - h['t']) < 2.0))) < 1 and not _bar_speed_ok(h['t'], v2, 0.03):
            if dbg: print('  ringtrack: 直線 %s t=%.3f 速さ %.2f: 前後にこの速さの音符が無いので足さない' % (col, h['t'], v2))
            # 円が隣の音符と重なる直線は、当てはめた速さが 3〜6% ずれる(重なった相手の円を途中から拾う。上の積み重なりの注と同じ幅)。判定枠に着く位置が
            # 小音符の直径(2R_S)より近い音符があり、手本と 6% 以内なら、下の「判定枠で見えた顔」の裏付けとして覚えておく。音符としては採らず、円も使用済みに
            # しない(使用済みにすると、隣の音符の直線がつくれなくなる: トロいか2000 122.87s)。
            # トロいか2000 119.39s: BPM 156 の 48 分の 32ms 間隔のドンの組は 15.4px/フレームで 30px 間隔。前のドンの直線は円 7 個・51 フレームで速さ 15.96
            # (当てはめの誤差 0.3%)、後ろのドンの追跡は逆に 14.82 で、平均が本当の速さ
            if np.any(np.abs(OT_ - h['t']) * v2 * fps < 2 * R_S) and np.any((np.abs(EV_ / v2 - 1) <= 0.06) & (np.abs(ET_ - h['t']) < 2.0)):
                wk_.append(h)
            return
        if dep_ is not None:
            h['dep'] = dep_
            if dbg: print('  ringtrack: 直線 %s t=%.3f 速さ %.2f: 直前の直線 t=%.3f に積み重なって隠れうる' % (col, h['t'], v2, dep_['t']))
        # 判定枠に着く位置の測りのぶれ(当てはめの標準誤差): 円が判定枠から遠い所でしか見えていない直線は、速さのわずかな誤差で着く位置が大きくずれる
        # (一括試験 FUJIN Rumble 113.67s: 判定枠から 930px 離れた 12 フレームの円 15 個で、±15px)
        fi_ = F[inl]; r_ = X[inl] - (JX + (h['t'] * fps - fi_) * h['vel']); sff_ = float(np.sum((fi_ - fi_.mean()) ** 2))
        h['sx'] = float(max(r_.std(), 1.0) * np.sqrt(1.0 / len(fi_) + (h['t'] * fps - fi_.mean()) ** 2 / max(sff_, 1e-6)))
        used[:] |= inl
        lines.append(h); LT_.append(h['t']); LV_.append(abs(h['vel']))
    for k_, (n, a, iv, col) in enumerate(cands):
        if k_ % max(1, len(cands) // 20) == 0: detect2.progress('隠れた音符', k_ + 1, len(cands), '一直線に並ぶ円から隠れた音符の候補を確かめる')
        _try(n, a, iv, col, False)
    while deferred:
        k0 = len(lines); dd_ = deferred; deferred = []
        for c_ in dd_: _try(*c_, True)
        if len(lines) == k0: break
    if dbg: print('  ringtrack: 直線の候補 %d 本 %.1fs' % (len(lines), _tm.time() - _t0))
    add = []; acc_ = []; left_ = []                  # left_: 採るには届かなかった直線
    # 直線ごとの縁の確かめ(_shape)は、上に重なる音符(notes + add)が変わったときだけ変わる。1 つ足すたびに全部の直線を確かめ直すと、
    # 候補 258 本・足す 115 本で 6 分かかった(検出テスト ファイナル)。足した音符が、直線の見えている間にその直線から
    # 「重なり・縁の共有を確かめる距離」以内に来るときだけ確かめ直す(それ以外は _shape の中で使われないので結果は同じ)
    sh_cache = {}; rp_cache = {}
    RNG_ = max(R_B + RIM_B[-1], 2 * RIM_B[-1] + 20) + 2
    def _near_path(h, q):
        vh = h['vel']; vq = abs(q['vel'])
        f0_, f1_ = h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps - 40 / vh
        d0_ = (q['t'] * fps - f0_) * vq - (h['t'] * fps - f0_) * vh; d1_ = (q['t'] * fps - f1_) * vq - (h['t'] * fps - f1_) * vh
        return d0_ * d1_ <= 0 or min(abs(d0_), abs(d1_)) <= RNG_
    while lines:
        best = None
        for h in lines:
            if 'dep' in h and not any(h['dep'] is q for q in acc_): continue      # 頼った直線がまだ足されていない
            if id(h) not in sh_cache:
                rs_ = _shape(h['t'], h['vel'], False, labs, notes + add, others, fps); ss_ = _shape.seen; cs_ = _shape.cdkb; nis_ = _shape.ni
                rb_ = _shape(h['t'], h['vel'], True, labs, notes + add, others, fps); sb_ = _shape.seen; cbg_ = _shape.cdkb; nib_ = _shape.ni
                sh_cache[id(h)] = (rs_, ss_, cs_, nis_, rb_, sb_, cbg_, nib_)
            rs_, ss_, cs_, nis_, rb_, sb_, cbg_, nib_ = sh_cache[id(h)]
            big2 = rb_[1] > rs_[1]; (n_f, w_, col2), seen = (rb_, sb_) if big2 else (rs_, ss_)
            # 縁が見えたフレームが偶然とは言えないほど多いものだけ(証拠の強さ = その確率の小ささ)
            p_ = _binom_tail(seen, n_f, True) if n_f else 1.0
            # 中身が見えているのに赤も青も 1 点も無いのはドン・カッではない(黄色い連打の棒など: もぺもぺ 95.29s)
            # 見えている中身が数点だけでは決められない(チャーリーダッシュ! 18.07s: 本物の隠れたドンの中身が 4 点しか見えず、色が無いとして外した)
            ni_ = nib_ if big2 else nis_                           # 選んだ大きさの縁で見えていた中身の点の数(後で呼んだ方の値を使っていた)
            if col2 is None and ni_ >= 20: continue
            # 中身が風船のオレンジの方が多いのは風船の頭(SAVAGE DELIGHT 110.84s: 前のドンの下に半分隠れて流れる遅い風船を、ドンの赤が混ざって
            # ドンとして足した)。大・小のどちらの縁で見ても
            cdkb_ = cbg_ if rb_[1] > rs_[1] else cs_
            if cdkb_[2] > cdkb_[0] + cdkb_[1] and cdkb_[2] >= 20:      # 数点だけでは決めない(チャーリーダッシュ! 18.07s は中身 1 点)
                if dbg: print('  ringtrack: 直線 %s t=%.3f 速さ %.2f: 中身が風船のオレンジ(%d 対 ドン %d・カッ %d)→ 風船の頭' % (h['color'], h['t'], h['vel'], cdkb_[2], cdkb_[0], cdkb_[1]))
                continue
            # 縁を確かめようがない(見えるはずのフレームが MIN_PTS 未満: 密な重なりで縁の大半が隠れるか隣の縁と重なる)ときは、直線をつくった
            # 円の検出(MIN_PTS 以上)を証拠とする(チャーリーダッシュ! 18.07s: 円は 54 フレームで見つかるが縁を確かめられたのは 2 フレーム)
            if n_f < MIN_PTS:
                if id(h) not in rp_cache: rp_cache[id(h)] = _ring_p(h, F, X, C, fps)      # 円の数だけで決まる(上の音符に依らない)
                p_ = rp_cache[id(h)]
            if (n_f >= MIN_PTS or h['n'] >= MIN_PTS) and p_ < P_SIG and (best is None or p_ < best[1]): best = (h, p_, n_f, big2, col2)
        if best is None:
            for h in lines:
                r_ = _shape(h['t'], h['vel'], False, labs, notes + add, others, fps); h['_seen'] = (r_[0], _shape.seen)
            left_ = lines
            if dbg:
                for h in lines:
                    r_ = _shape(h['t'], h['vel'], False, labs, notes + add, others, fps)
                    print('  ringtrack: 直線 %s t=%.3f 速さ %.2f 円 %d → 足さない(縁が見えるはず %d 枚、見えた %d 枚、中身の色 %s、中身の点 %d)' % (
                        h['color'], h['t'], h['vel'], h['n'], r_[0], _shape.seen, r_[2], _shape.ni))
            break
        h, w_, n_f, big2, col2 = best
        lines = [q for q in lines if q is not h]; acc_.append(h)
        if dbg and os.environ.get('V2T_RT_DUMP') and abs(h['t'] - float(os.environ['V2T_RT_DUMP'])) < 0.01: _dump(h, h['t'], h['vel'], labs, notes + add, fps)
        col = col2 or h['color']
        if dbg: print('  ringtrack: 直線 %s t=%.3f 速さ %.2f 円 %d: 縁が見えるはずのフレーム %d 枚、偶然の確率 %.4f、中身の色 %s、大 %s → 足す' % (h['color'], h['t'], h['vel'], h['n'], n_f, w_, col2, big2))
        add.append({'t': round(h['t'], 4), 'color': col, 'big': bool(big2), 'npts': h['n'], 'vel': round(h['vel'], 2), 'hidden': False, 'ringtrack': True, 'sx': round(h.get('sx', 0.0), 1), 'col_unsure': col2 is None})
        for q_ in lines:
            if id(q_) in sh_cache and _near_path(q_, add[-1]): del sh_cache[id(q_)]
    # 判定枠で丸ごと見えた顔の時刻(detect2 の一番前の顔の列)が、その顔で説明した音符から照合の幅以上ずれていて、その音符を円で追い直した時刻も
    # 追跡の時刻と同じ(照合の幅以内)なら、顔は別の音符(トロいか2000 119.39s: 32ms 間隔のドンの組。色の塊の追跡も円の追い直しも後ろのドンを指し、
    # 前のドンは判定枠の手前の 3 フレームだけ顔が丸ごと見えた)
    # 縁を確かめる(_shape)のは、顔を持つ音符の顔の時刻か自分の時刻の近く(0.1 秒)の直線だけ(ほかは裏付けに使われない)
    fts_ = [(o['color'], t_) for o in notes if '_front_t' in o for t_ in (o['_front_t'], o['t'])]
    wk_ = [h for h in wk_ if any(c_ == h['color'] and abs(h['t'] - t_) < 0.1 for c_, t_ in fts_)]
    for h in wk_:
        r_ = _shape(h['t'], h['vel'], False, labs, notes + add, others, fps); h['_seen'] = (r_[0], _shape.seen)
    sup_ = left_ + wk_
    for o in notes:
        if '_front_t' not in o or '_fit' not in o or any(o is g for g in gone): continue
        # 顔で説明した音符自身の時刻が確かなときだけ(途中から現れた・救済した短い軌跡は時刻が 30ms 以上ずれ、顔とのずれで同じドンを二重にした:
        # Caribbean Knight 73.02s は救済の 72.980(円 6 個)と顔の 73.017)
        if o.get('emerged') or o.get('rescued'): continue
        # 別の音符とみなすのは、照合の幅に半フレーム(顔の位置を見たフレームの刻み)を足した幅を超えてずれたときだけ(チャーリーダッシュ! 17.87s・
        # メカデス。 裏 25.47s は 25〜27ms のずれで同じドン)
        tf_, tol_ = o['_front_t'], o['_front_tol'] + 0.5 / fps
        fit_t_ = o['_fit'][0]
        if abs(o['t'] - tf_) <= tol_: continue
        if any(q['color'] == o['color'] and abs(q['t'] - tf_) <= tol_ for q in notes + add if q is not o): continue
        # 顔の時刻だけでは足さない(録画の時計が乱れた動画では顔の時刻も 40ms ずれる: Mood Swing 89.43s)。顔の時刻と、この音符の軌跡の時刻の両方に、
        # 別々の証拠があるときだけ 2 つの音符とする。証拠は「円で追い直した軌跡の時刻が合う」か「採るには届かなかった円の直線(同じ色・縁が見えるはずの
        # フレームの過半で見えた)が合う」で、1 つの証拠を両方には使わない。
        # トロいか2000 119.39s: 追い直しは軌跡に合い(119.436)、顔の側に届かなかった直線(8/11)。122.85s: 2 つの点をつないだ軌跡(122.889)の追い直しは
        # 前のドン(122.855)に移って顔の側に合い、軌跡の側(後ろのドン)に届かなかった直線 122.871(10/16)
        sup_f_ = [h for h in sup_ if h['color'] == o['color'] and abs(h['t'] - tf_) <= tol_ and h['_seen'][1] * 2 > h['_seen'][0]]
        sup_t_ = [h for h in sup_ if h['color'] == o['color'] and abs(h['t'] - o['t']) <= tol_ and abs(h['t'] - tf_) > tol_ and h['_seen'][1] * 2 > h['_seen'][0]]
        fit_f_, fit_t2_ = abs(fit_t_ - tf_) <= tol_, abs(fit_t_ - o['t']) <= tol_
        if not ((fit_t2_ and not fit_f_ and sup_f_) or (fit_f_ and not fit_t2_ and sup_t_)): continue
        if dbg: print('  ringtrack: 判定枠で見えた顔 %s t=%.3f は %.3f(円で追い直して %.3f)と別の音符 → 足す' % (o['color'], tf_, o['t'], o['_fit'][0]))
        add.append({'t': round(tf_, 4), 'color': o['color'], 'big': False, 'npts': 1, 'vel': round(abs(o['vel']), 2), 'hidden': False, 'ringtrack': True, 'front': True})
    # 積み重なりの縞: とても遅い音符が小音符の半径より狭い間隔で積み重なると、白い縁は重なって 1 本の白い帯になり円は取れないが、
    # 音符ごとの白い縁の外周(中心 + 47px)で「白 → 暗い輪郭」に変わる所は 1 本ずつ残る(検出テスト ファイナル 58〜63s: SCROLL 0.1〜0.25 の 13〜18px 間隔)。
    # 既にある音符の縁で説明できない変わり目を、前後 2 秒の手本と同じ速さ(±2%)の直線に投票し、手本の音符のすぐ後ろ(直径未満)に積み重なる直線だけ採る。
    # 色は中心 + 35〜39px(上の音符からはみ出して見える三日月の中身)の画素の多数で、偶然とは言えないほど偏っているときだけ(見えなければ足さない)
    if os.environ.get('V2T_STACK', '1') != '0' and labs and exemplar:
        occ_s = notes + add
        OT_s = np.array([o.get('_fit', (o['t'],))[0] for o in occ_s], float) * fps
        OV_s = np.array([o.get('_fit', (0, abs(o['vel'])))[1] for o in occ_s], float)
        OR_s = np.array([RIM_B[-1] if o.get('big') else RIM[-1] for o in occ_s], float)
        OB_s = np.array([bool(o.get('big')) for o in occ_s], bool)
        hc_ = HB // 2
        # 白 → 暗の変わり目(白い縁の外周)を全部集める
        EF_, EX_ = [], []
        for fe, L in labs.items():
            rows_ = L[hc_ - 2:hc_ + 3]
            wh_ = (rows_ == 3).mean(0) > 0.5; dk_ = (rows_ == 0).mean(0) > 0.5
            for e_ in np.flatnonzero(wh_[:-1] & dk_[1:]):
                EF_.append(fe); EX_.append(LX0 + 2.0 * (e_ + 1))
        EF_ = np.array(EF_, float); EX_ = np.array(EX_, float)
        # 変わり目が中心から何 px に出るかを、既にある小音符で速さごとに測る(白の判定の境目・動きのぼけで 47px から数 px ずれる:
        # 同 59.8s の 1.6px/フレームでは −4px、9〜18px/フレームでは +4〜8px ずれ、遅い音符の時刻が 40ms ずれた)
        cal_v, cal_r = [], []
        OS_ = np.flatnonzero(~OB_s) if len(OT_s) else np.zeros(0, int)
        for k2 in OS_:
            xq_ = JX + (OT_s[k2] - EF_) * OV_s[k2]
            m2_ = np.abs(EX_ - xq_ - RIM[-1]) < 12
            cal_v.extend([OV_s[k2]] * int(m2_.sum())); cal_r.extend((EX_[m2_] - xq_[m2_]).tolist())
        cal_v = np.array(cal_v, float); cal_r = np.array(cal_r, float)
        def _edge_off(v_):
            if len(cal_r):
                m3_ = np.abs(cal_v / v_ - 1) <= 0.1
                if m3_.sum() >= 20: return float(np.median(cal_r[m3_]))
                if len(cal_r) >= 20: return float(np.median(cal_r))
            return float(RIM[-1])
        # 既にある音符の縁で説明できる変わり目と、前にいる大音符の円の中の変わり目を除く
        keep_ = np.ones(len(EF_), bool)
        # 判定枠の手前でだけ見えた音符(emerge)は除く相手にしない: 測れているのは判定枠に着く時刻だけで、速さは前後の音符から借りたもの。
        # 画面の右の方までの道筋は当てにならず、時刻が 17〜25ms ずれた emerge が、隣の本物の音符の変わり目の 8 割を「自分の縁」として消していた
        # (ファイナル 61.2〜62.2s: 3.94px/フレームで 15px 間隔の積み重なり。変わり目は 24 音とも譜面の ±4ms に鋭い山を作るのに、直線が 1 本も採れなかった)
        OEm_ = np.array([bool(o.get('emerge')) for o in occ_s], bool)
        for k2 in range(len(OT_s)):
            if OEm_[k2]: continue
            xq_ = JX + (OT_s[k2] - EF_) * OV_s[k2]
            if OB_s[k2]:
                keep_ &= ~(np.abs(EX_ - xq_ - RIM_B[-1]) < NEAR)
                # 大音符の円の中の変わり目は大音符の顔や縁のもの(小音符の輪郭は上の大音符に隠れる: 同 62.8〜63.7s)。隠せるのは前にいる大音符だけ
                keep_ &= ~((xq_ < EX_ - RIM[-1]) & (np.abs(xq_ - EX_) < R_B))
            else:
                keep_ &= ~(np.abs(EX_ - xq_ - _edge_off(OV_s[k2])) < NEAR)
        SF_ = EF_[keep_]; SE_ = EX_[keep_]
        SX_ = SE_ - RIM[-1]                                   # 仮の中心(速さごとに下で測った値で直す)
        inr_ = (SX_ > JX + 60) & (SX_ < X_IN - 10); SF_ = SF_[inr_]; SE_ = SE_[inr_]; SX_ = SX_[inr_]
        EV_s = np.array([abs(o['vel']) for o in exemplar], float); ET_s = np.array([o['t'] for o in exemplar], float)
        new_ = []; em_claim_ = set(); full_cache_ = {}
        # 候補の直線を全部の速さで先に集め、票の多い順に採る。採った直線の変わり目は使用済みにする(ringtrack の円の直線と同じ)。速さの小さい順に
        # 採ると、隣の小節の手本の速さ(3.14)の斜めの直線が、3.94 で並ぶ 15px 間隔の縞を横切って先に採られ、同じ変わり目を本物の直線と二重に使った
        # (ファイナル 61.5〜62.2s)。積み重なりの前提(すぐ前の音符)が後で採る直線に頼るものは、ほかを採った後に調べ直す
        cand_s = []
        if len(SF_) >= 10:
            for v_ in sorted(set(np.round(EV_s, 2)) | set(np.round(BV_[BV_ > 0], 2))):
                fc_ = SF_ + (SE_ - _edge_off(v_) - JX) / v_
                h_, e_ = np.histogram(fc_, bins=np.arange(fc_.min() - 1, fc_.max() + 2, 1.0))
                for k_ in np.flatnonzero(h_ >= 10):
                    if h_[k_] < h_[max(0, k_ - 1):k_ + 2].max(): continue      # 山の頂だけ
                    cand_s.append((int(h_[k_]), float(v_), float(e_[k_] + 0.5)))
        cand_s.sort(key=lambda c: -c[0])
        used_s = np.zeros(len(SF_), bool)

        def _stk_try(v_, fcb_):
            """候補の直線を確かめる。'ok'(足した)・'defer'(前の音符がまだ無い)・'no'"""
            SX_ = SE_ - _edge_off(v_)                 # この速さで測った「変わり目 − 中心」で中心に直す
            sel_ = ~used_s & (np.abs(JX + (fcb_ - SF_) * v_ - SX_) <= NEAR)
            # 速さの違う大音符の上を通り過ぎる所の変わり目は、上の音符の縁と下の大音符の境目(上の音符と一緒に動くが音符ではない。
            # ドンカマ2000 124.50s: 最後の大ドンが曲の間ずっとゆっくり流れ、その上を 5.9px の小ドンが通るたびに 45px 後ろに縞の直線ができた)
            if sel_.any() and np.any(OB_s):
                ib_ = np.flatnonzero(OB_s & (np.abs(OV_s / v_ - 1) > 0.03))
                if len(ib_):
                    xb2_ = JX + (OT_s[ib_][None, :] - SF_[sel_][:, None]) * OV_s[ib_][None, :]
                    inb_ = np.any(np.abs(xb2_ - (SX_[sel_][:, None] + RIM[-1])) < R_B, axis=1)
                    idx_ = np.flatnonzero(sel_); sel_ = sel_.copy(); sel_[idx_[inb_]] = False
            if sel_.sum() < 20 or np.ptp(SF_[sel_]) < 30: return 'no'       # 変わり目 20 点以上(普通の曲の偽物は 10〜15 点、ファイナルの積み重なりは 28〜134 点)
            # 速さの違う音符の上を通り過ぎる間だけできる縞は、重なっている間(大音符の直径ぶんほど)しか続かない。本物の積み重なりの音符の縞は
            # 画面の広い範囲に続く(止まって記録された大音符は位置を予想できないので、動いた距離で見る: ドンカマ2000 124.50s)
            if np.ptp(SX_[sel_]) <= 2 * R_B + 2 * NEAR: return 'no'
            # 直線の速さと時刻は、選んだ変わり目そのものに当てはめ直す。投票の速さは手本の速さ(小数 2 桁に丸めた値・手本自身の測りの誤差)で、
            # 遅い音符は判定枠から 1000px 離れた所で見えるので、速さの 0.6% のずれが判定枠に着く時刻の 60ms のずれになった(ファイナル 59.8〜61.0s:
            # 1.59 で投票し、本当は 1.58。変わり目は速さによらず中心 + 48.5px で、時刻を合わせれば 8 つとも譜面の ±8ms に山がある)。外れた点を除いて 2 回
            vl_ = v_; fcm_ = float(np.median(SF_[sel_] + (SX_[sel_] - JX) / v_))
            for _it in range(2):
                A_ = np.vstack([np.ones(int(sel_.sum())), -SF_[sel_]]).T
                sol_ = np.linalg.lstsq(A_, SX_[sel_] - JX, rcond=None)[0]
                if sol_[1] <= 0: break
                vf_ = float(sol_[1]); ff_ = float(sol_[0]) / vf_
                s2_ = ~used_s & (np.abs(JX + (ff_ - SF_) * vf_ - SX_) <= NEAR)
                if s2_.sum() < 20: break
                sel_ = s2_; vl_ = vf_; fcm_ = ff_
            if abs(vl_ / v_ - 1) > 0.02 or sel_.sum() < 20 or np.ptp(SX_[sel_]) <= 2 * R_B + 2 * NEAR: return 'no'
            tm_ = fcm_ / fps
            if int(np.sum((np.abs(EV_s / vl_ - 1) <= 0.02) & (np.abs(ET_s - tm_) < 2.0))) < 2 and not _bar_speed_ok(tm_, vl_, 0.02): return 'no'
            h2_ = {'t': tm_, 'vel': vl_, 'n': int(sel_.sum())}
            # 手本の音符(か足した音符)のすぐ後ろに積み重なる(同じ速さ ±3%、中心の差が直径未満)
            # 縞が要るのは白い縁どうしが重なって円が取れないほど詰まった積み重なりだけ: 前の音符との中心の距離が小音符の半径未満
            # (直径未満まで許すと、普通の曲の 16 分(60〜100px)の顔や縁の模様の変わり目から偽の音符を足した: 全曲試験 28 曲)
            # 追い越されて隠れる直線も(ringtrack の直線と同じ: 同 59.65s の SCROLL 0.1 の列は、先に着く 2.37px の列に上を通り過ぎられる)
            # 前の音符が大音符なら外周 R_B まで(検出テスト ファイナル 115.71s: 大カッの 53px 後ろの小ドン。以前は大カッの影の偽の小カッ(27px 前)を前の音符にしていた)
            if (not any(q.get('vel') and abs(abs(q['vel']) / vl_ - 1) <= 0.03 and 0 < (tm_ - q['t']) * vl_ * fps < (R_B if q.get('big') else R_S) for q in occ_s + new_)
                    and not _overtaken_v({'t': tm_, 'vel': vl_})): return 'defer'
            # 判定枠の手前でだけ見えた音符(emerge)の道筋とは比べない(上の「除く相手」と同じ理由: 借りた速さで延ばした道筋。同じ音符なら下でまとめる)
            if _same_any(h2_, tt_n[~em_n], vv_n[~em_n], fps, sd_n[~em_n], em_n[~em_n]) or any(_same_path(q, h2_, fps) for q in add + new_): return 'no'
            # 1 本のレーンに同じ時刻に 2 つは着かない: 判定枠での位置の差(時刻の差 × 遅い方の速さ)が 2×NEAR 未満の音符があれば同じ音符
            # (ドンカマ2000 38.64s: 速さ 11.85 で取れていたカッに、速さ 5.9 の縞の直線を 19ms 差で重ねた)
            # 判定枠の手前でだけ見えた音符(emerge)は除く: 数点から出した速さで時刻が 80ms ずれることがあり(ファイナル 60.77s・60.96s は速さ 3.9 と測り
            # 本当は 1.58)、同じ音符なら下の「判定枠の手前だけの音符は足した音符と同じ」で全行程の証拠がある縞の方を残す
            if any(q.get('vel') and not q.get('emerge') and abs(tm_ - q['t']) * fps * min(vl_, abs(q['vel'])) < 2 * NEAR for q in occ_s + new_): return 'no'
            # 色: 中心 + 35〜39px の中央の行の画素
            cd_ = ck_ = nfc_ = ntry_ = 0
            for fe, L in labs.items():
                x_ = JX + (fcm_ - fe) * vl_
                if not (JX < x_ < X_IN - 45): continue
                cols_ = ((np.arange(x_ + 35, x_ + 40) - LX0) / 2).astype(int)
                if cols_.min() < 0 or cols_.max() >= L.shape[1]: continue
                # 速さの違う音符(追い越す音符)が近くを通るフレームの色は、その音符の色(ドンカマ2000 37.49s: 5.9px の遅いドンの列の上を
                # 11.85px のカッが追い越し、その青を縞の直線の色と数えた)
                # 除くのは上に描かれる(先に着く)音符だけ。後から着く音符は下に描かれ、三日月の色を隠さない(ファイナル 61.2〜62.6s: 後ろから来る遅い大音符の
                # せいで試せるフレームが 0 枚になった)
                if len(OT_s) and np.any((np.abs(OV_s / vl_ - 1) > 0.03) & (OT_s < fcm_) & (np.abs(JX + (OT_s - fe) * OV_s - (x_ + 37)) < R_S + 10)): continue
                blk_ = L[hc_ - 3:hc_ + 4][:, cols_]; ntry_ += 1
                d1_, k1_ = int((blk_ == 1).sum()), int((blk_ == 2).sum())
                if d1_ + k1_ >= 4: nfc_ += 1; cd_ += int(d1_ > k1_); ck_ += int(k1_ > d1_)
            # 色は 10 枚以上で見え、多い方が 8 割以上(偽物は 5〜6 枚、遅い音符の上の模様はドン 162・カッ 201 と半々: 笑える2000 102.49s)
            col_ = None
            if cd_ + ck_ < 10 or max(cd_, ck_) < 0.8 * (cd_ + ck_) or nfc_ < 0.6 * ntry_:
                # 三日月の中身が前の音符に覆われて見えない積み重なり(間隔 15px 未満: 前の音符の外周 R_S が中心 + 35px を越える)の色は、
                # 同じ時刻に判定枠に着く、判定枠の手前で顔が見えた音符(emerge)の色。前の音符が叩かれて消えると顔が丸ごと見える。
                # 時刻は変わり目の直線の方が確か(emerge は 17〜46ms ずれる: ファイナル 58.2〜59.6s・61.2〜62.6s)。1 つの emerge は 1 本の直線にだけ
                ce_ = [q for q in occ_s if q.get('emerge') and id(q) not in em_claim_ and abs(q['t'] - tm_) * fps * vl_ <= NEAR]
                if ce_:
                    qe_ = min(ce_, key=lambda q: abs(q['t'] - tm_)); em_claim_.add(id(qe_)); col_ = qe_['color']
                    if dbg: print('  ringtrack: 積み重なりの縞 t=%.3f の色は判定枠の手前で見えた %s %.3f(三日月の色 ドン %d / カッ %d・%d / %d 枚)' % (tm_, col_, qe_['t'], cd_, ck_, nfc_, ntry_))
                else:
                    # 判定枠で顔が見えなければ、三日月を元の解像度で読む(すぐ前の同じ速さの音符との距離で三日月の形を決める)
                    fr_ = [((tm_ - q['t']) * vl_ * fps, R_B if q.get('big') else R_S) for q in occ_s + new_
                           if q.get('vel') and abs(abs(q['vel']) / vl_ - 1) <= 0.03 and 0 < (tm_ - q['t']) * vl_ * fps < R_B]
                    if not fr_: return 'no'
                    sf_, rf_ = min(fr_, key=lambda z: z[0] - z[1])                   # 三日月の左端を決める前の音符(外周が最も右まで来るもの)
                    if sf_ - rf_ + RIN_S_DET <= 2: return 'no'                        # 前の音符の外周が色の円を覆う(三日月に色が無い: 大音符の 44px 以内の後ろ)
                    # 元の解像度で読むのは、見える色の幅がラベル画像(半分の解像度)では読めないほど細いとき(12px 以下)だけ。それより太い三日月の色が
                    # ラベル画像で読めなければ音符ではない(前の音符から 42〜43px 後ろの偽の縞の直線で、後ろにいる別の音符の色を読んだ: 段位 弾丸ノーツ 裏 119.07s・
                    # Prelude 裏 31.68s・ドッペルゲンガー 裏 36.55s)
                    if sf_ - rf_ + RIN_S_DET > 12: return 'no'
                    cd_, ck_, nfc_, ntry_ = _crescent_full(video, cal, fcm_, vl_, sf_, rf_, OT_s, OV_s, fps, full_cache_)
                    if dbg: print('  ringtrack: 積み重なりの縞 t=%.3f の三日月を元の解像度で: ドン %d / カッ %d(%d / %d 枚、前の音符まで %.0fpx%s)' % (tm_, cd_, ck_, nfc_, ntry_, sf_, '・大' if rf_ == R_B else ''))
                    if cd_ + ck_ < 8 or max(cd_, ck_) < 0.8 * (cd_ + ck_) or nfc_ < 0.6 * ntry_: return 'no'
                    col_ = 'don' if cd_ > ck_ else 'ka'
            # 色は確かめたフレームの過半(6 割以上)で見えること。速さの違う音符が近くを通るフレームを除けば、本物の三日月の中身(中心 + 35〜39px)は
            # 前の音符に覆われない限り毎フレーム見える(ファイナルで時刻の合った縞は 71〜100%)。前の音符に覆われるほど狭い積み重なり(15px 未満)は
            # 色がそもそも 10 枚に届かない。偽物は別々の音符の縁の偶然の並びで、色はたまたま重なった分だけ(女神な世界Ⅰ 100.39s・100.75s: 2 つの速さの
            # 音符が重なり合う普通の場面で 28%・45%、下の大音符の色を拾った縞は 150 枚中 10 枚: ドンカマ2000 124.50s)。以前の 15% は、位置が 5px ずれて
            # 時刻が 60ms 早かったファイナル 59.8〜60.3s の直線(18〜35%)を本物と見ていた
            if col_ is None: col_ = 'don' if cd_ > ck_ else 'ka'
            h2_.update({'color': col_})
            new_.append(h2_); used_s[:] |= sel_
            if dbg: print('  ringtrack: 積み重なりの縞 %s t=%.3f 速さ %.2f 変わり目 %d(%.0fpx)・色 ドン %d / カッ %d(%d / %d 枚)→ 足す' % (col_, tm_, vl_, h2_['n'], float(np.ptp(SX_[sel_])), cd_, ck_, nfc_, ntry_))
            return 'ok'
        defer_s = []
        for k_, c_ in enumerate(cand_s):
            if k_ % max(1, len(cand_s) // 20) == 0: detect2.progress('隠れた音符', k_ + 1, len(cand_s), '重なって積み重なった音符を白い縁の縞から読む')
            if _stk_try(c_[1], c_[2]) == 'defer': defer_s.append(c_)
        while defer_s:
            k0_ = len(new_); dd_ = defer_s; defer_s = []
            for c_ in dd_:
                if _stk_try(c_[1], c_[2]) == 'defer': defer_s.append(c_)
            if len(new_) == k0_: break
        for h2_ in new_:
            add.append({'t': round(h2_['t'], 4), 'color': h2_['color'], 'big': False, 'npts': h2_['n'], 'vel': round(h2_['vel'], 2), 'hidden': False, 'ringtrack': True, 'stack': True})
    # 判定枠の手前でだけ見えた音符(emerge)が、足した音符と同じ時刻に判定枠に着くなら同じ音符(足した方は全行程の証拠がある)
    # (チャーリーダッシュ! 17.66s、Emma 裏 玄人 72.25s)
    # 当て直しで時刻を置き換えた音符(救済・途中から現れた軌跡)とも同じようにまとめる(彁 27.46s: 救済の 27.409 が円 76 個の当て直しで 27.460 になり、
    # 判定枠の手前だけのカッ 27.464 と二重になった)
    for o in notes:
        if not o.get('emerge') or any(o is g for g in gone): continue
        best_ = None
        for h in add + [q for q in refitted_ if q is not o and not q.get('emerge') and not any(q is g for g in gone)]:
            # 中身の色を確かめられなかった直線(円の色だけ)は、判定枠の手前で顔の見えた音符と色が違っても同じ音符でありうる。まとめたら顔の色にする
            # (‽Chaos Time‽ 54.36s: 円 8 個・中身の色が分からない直線をドンとし、同時に判定枠の手前で見えたカッと二重になった)
            if h['color'] != o['color'] and not h.get('col_unsure'): continue
            # 判定枠の手前でだけ見えた音符で測れているのは判定枠に着く時刻だけ(速さは前後の音符から借りたもの): 着く時刻の差が判定枠で 2×NEAR 未満なら同じ
            # 判定枠の一番前の顔の列から足した音符(front)は、detect2 で既存の音符が説明するのと同じ幅(_front_tol)の中なら同じ音符
            # (‽Chaos Time‽ 59.01s: 顔の 29ms 前に 5 つの円の直線を足して二重になった)
            # 足した直線の着く位置のぶれ(sx)の 2 倍までは同じ音符とみなす。ぶれが NEAR を超える直線より、判定枠で顔を確かめた音符(front_ok)の時刻の方が確か
            sx_ = h.get('sx', 0.0)
            d_ = abs(o['t'] - h['t']) * h['vel'] * fps / (2 * NEAR + 2 * sx_)
            if o.get('front') and o.get('_front_tol'): d_ = min(d_, abs(o['t'] - h['t']) / o['_front_tol'])
            # 当てはまる音符が複数あれば、幅に対して最も近いもの(ぶれの大きい直線は幅が広く何にでも当てはまる: ファイナル 57.52s・57.61s の
            # 判定枠の手前だけのカッ 2 つを、18ms 差の縞の音符でなく 0.6 秒離れた着く位置のぶれ ±69px の直線とまとめ、直線の時刻を 57.61 に動かして二重にした)
            if d_ < 1.0 and (best_ is None or d_ < best_[0]): best_ = (d_, h)
        if best_ is not None:
            h = best_[1]
            if h.get('sx', 0.0) > NEAR and o.get('front_ok'): h['t'] = o['t']
            if h.get('col_unsure'): h['color'] = o['color']
            if dbg: print('  ringtrack: 判定枠の手前だけの音符 %s t=%.3f は足した t=%.3f と同じ' % (o['color'], o['t'], h['t']))
            gone.append(o)
    # 足した直線のすぐ後ろ(半分隠れた円の重心のずれ 4R/3π + 測りの幅以内)に同じ色・同じ速さ(±10%)の弱い音符があり、その音符自身の縁は
    # 偶然以上には見えていなければ、それは足した音符の一部が隠れた姿(色の塊の中心は見えている後ろ側へずれる)。足した方(縁の証拠がある)を残す
    # (Rough: CHA0Z 22.90s: 追い越す速いカッの下から判定枠の手前 7 点だけ見えたドンと、同じドンの縁の直線 22.893 が二重になった)
    for o in weak:
        if any(o is g for g in gone) or '_rim' not in o: continue
        n_r, s_r = o['_rim']
        if n_r >= MIN_PTS and _binom_tail(s_r, n_r, True) < P_SIG: continue      # 自分の縁が見えている(別の音符)
        for h in add:
            if h['color'] != o['color'] or abs(abs(o['vel']) / h['vel'] - 1) > 0.1: continue
            d_ = (o['t'] - h['t']) * h['vel'] * fps
            if -2 * NEAR < d_ < 2 * NEAR + 4 * R_S / (3 * np.pi):
                if dbg: print('  ringtrack: 弱い音符 %s t=%.3f(縁 %d/%d)は足した t=%.3f の一部が隠れた姿' % (o['color'], o['t'], s_r, n_r, h['t']))
                gone.append(o); break
    # 中身が別の色で外した音符は、その色の音符が後から(円の直線で)上に重なる位置に見つかっていれば戻す。中身を確かめた時点では上の音符を知らず、
    # 上の音符の色を中身と数えていた(検出テスト ファイナル 47.65s: 64 分の交互の列のカッ。26px 前のドン 47.620 は色の塊の追跡で取れず、円の直線で後から見つかった)
    for o in [g for g in gone if g.get('_gone_col')]:
        t_o, v_o = o['_gone_col']; oc_ = 'don' if o['color'] == 'ka' else 'ka'
        if any(q['color'] == oc_ and q['t'] < t_o and abs(abs(q['vel']) / v_o - 1) <= 0.1 and (t_o - q['t']) * v_o * fps < R_S for q in add):
            if dbg: print('  ringtrack: 中身が別の色で外した %s t=%.3f は、後から見つかった上の音符の色だった → 戻す' % (o['color'], o['t']))
            gone = [g for g in gone if g is not o]
    return add, gone


def _crescent_full(video, cal, fcm, v, sf, rf, OT, OV, fps, cache):
    """積み重なりの音符(判定枠に着くフレーム fcm・速さ v、すぐ前の同じ速さの音符との距離 sf px)の三日月の色を、元の解像度の画素で数える。
    戻り値 (ドンのフレーム数, カッのフレーム数, 色の見えたフレーム数, 試したフレーム数)。
    三日月は前の音符の外周(半径 rf: 小 R_S・大 R_B)と自分の色の円(半径 RIN_S_DET)の間: 中央の行で幅 sf − 10px(15px 間隔なら 5px)、上下 ±20 行に続く。
    半分の解像度のラベル画像では 2〜3 画素しかなく色を読めなかった(ファイナル 58.2〜59.6s・61.2〜62.6s の 13〜15px 間隔の積み重なり)。
    cache: {動画のフレーム: レーン中央 ±22 行の帯}(直線ごとに同じ区間を読み直さない)"""
    import detect2, cv2
    clk = getattr(detect2.run_parallel, 'clk', None)
    f_lo, f_hi = fcm - (X_IN - 60 - JX) / v, fcm - 80 / v
    if f_hi - f_lo < 4: return 0, 0, 0, 0
    step = max(1, int((f_hi - f_lo) / 60))                    # 1 本の直線で 60 枚ほど
    want = [f for f in range(int(np.floor(f_lo)), int(np.ceil(f_hi)) + 1, step)]
    miss = [f for f in want if f not in cache]
    if miss:
        hd = detect2.band_start(video, miss[0] / fps, (miss[-1] + 1) / fps, (CY - 30, CY + 30), step, cal, parts=2, tag='cr')
        for f, canvas in detect2.band_iter(hd):
            if f in want: cache[f] = canvas[CY - 22:CY + 23, LX0:].copy()
    cd = ck = nfc = ntry = 0
    ys = np.arange(-20, 21)
    # 読む範囲は前の音符の外周の 2px 手前から、自分の色の円の 4px 外まで(この録画の小音符の色の円は 42〜43px で、決め打ちの 40px より外。
    # 前の音符の黒い輪郭と自分の白い縁は彩度・明るさで外れるので、広げても混ざらない: ファイナル 49.99s の 96 分の 3 連の 2 つ目は中心 + 42〜43px に幅 1〜2px)
    lo_ = np.sqrt(np.maximum(rf ** 2 - ys ** 2, 0)) - 2; hi_ = np.sqrt(np.maximum((RIN_S_DET + 4) ** 2 - ys ** 2, 0))
    for f in want:
        B = cache.get(f)
        if B is None: continue
        fe = _eff(f, clk); x = JX + (fcm - fe) * v
        if not (JX + 60 < x < X_IN - 45): continue
        # 速さの違う音符が上を通るフレームは、その音符の色が混ざる(ラベル画像の色と同じ除き方。後から着く音符は下に描かれるので混ざらない)
        if len(OT) and np.any((np.abs(OV / v - 1) > 0.03) & (OT < fcm) & (np.abs(JX + (OT - fe) * OV - (x + 37)) < R_S + 10)): continue
        ntry += 1; dn = kn = 0
        for y, a_, b_ in zip(ys, x - sf + lo_, x + hi_):
            c0, c1 = int(np.ceil(a_)) - LX0, int(np.floor(b_)) - LX0
            if c1 < c0 or c0 < 0 or c1 >= B.shape[1]: continue
            hsv = cv2.cvtColor(B[y + 22:y + 23, c0:c1 + 1], cv2.COLOR_BGR2HSV)
            # 幅 1〜2px の色は白い縁と混ざって彩度が下がる(カッの色相 83〜87・彩度 69〜128)ので彩度は緩める。明るさはラベル画像と同じ(下げると
            # 赤茶色のレーンの背景(明るさ 110〜140)をドンと数えた: 段位 AtoZ 57.16s・みんなのかき氷 72.39s。どちらも見える色の幅が 19〜25px の太い三日月で、
            # 今は上の「12px 以下」で読まない)
            dn += int((cv2.inRange(hsv, (0, 90, 160), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 160), (255, 255, 255))).astype(bool).sum())
            kn += int(cv2.inRange(hsv, (78, 50, 150), (95, 255, 255)).astype(bool).sum())
        if dn + kn >= 6: nfc += 1; cd += int(dn > kn); ck += int(kn > dn)
    return cd, ck, nfc, ntry


def _overtaken(h, notes, fps):
    """h の上を、速さの違う音符(譜面で先 = 上に描かれる)が通り過ぎるか: 画面にいる間に、h との位置の差が直径ぶん以上変わり、
    その間に一度は重なる。追い越されない音符は画面でずっと見えているので、通常の追跡で取れているはず
    (同じ速さで並ぶ 2 つの音符の間のハフ変換の幻の円: チャレンジャーズ 125.18s・ドンカマ2000 112.8s)"""
    vh = h['vel']; fs = np.arange(h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps, 1.0)
    xh = JX + (h['t'] * fps - fs) * vh
    for o in notes:
        if o['t'] >= h['t']: continue
        tt, vv = o.get('_fit', (o['t'], abs(o['vel'])))
        d = JX + (tt * fps - fs) * vv - xh
        if d.max() - d.min() >= 2 * R_S and np.any(np.abs(d) < R_S): return True
    return False


def _same_path(o, h, fps):
    """同じ軌跡か: h が画面にいる間、o の位置との差の中央値が円の中心の測りの幅(2×NEAR)未満。
    別の音符どうしは 23px まで近づくことがある(チャーリーダッシュ! 18.02/18.07s)ので、それより細かい所だけを同じとみなす"""
    if '_fit' in o: o = {'t': o['_fit'][0], 'vel': o['_fit'][1]}
    if '_fit' in h: h = {'t': h['_fit'][0], 'vel': h['_fit'][1]}
    vh = abs(h['vel']); vo = abs(o['vel'])
    fs = np.linspace(h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps, 20)
    d_ = np.abs((h['t'] * fps - fs) * vh - (o['t'] * fps - fs) * vo)
    return float(np.median(d_)) < 2 * NEAR


def _dump(o, t, v, labs, occ, fps):
    """確かめた点を画像に描く(調べる用): 画素の種類の画像に、白い縁の点(白=白い、赤=白くない)と上に重なる音符の円"""
    import cv2
    ims = []
    for f in sorted(labs)[::max(1, len(labs) // 200)]:
        x = JX + (t * fps - f) * v
        if not (900 < x < 1600): continue
        L = labs[f]; img = np.zeros(L.shape + (3,), np.uint8)
        img[L == 1] = (0, 0, 255); img[L == 2] = (255, 255, 0); img[L == 3] = (255, 255, 255)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for q in occ:
            tt, vv = q.get('_fit', (q['t'], abs(q['vel'])))
            xo = JX + (tt * fps - f) * vv
            if abs(xo - x) < 300: cv2.circle(img, (int(xo - LX0), HB), R_B if q.get('big') else R_S, (0, 255, 0) if q['t'] < t else (128, 128, 128), 1)
        for rr in RIM_B + RIM:
            for a in ANG:
                px, py = x + rr * np.cos(a), rr * np.sin(a)
                cv2.circle(img, (int(px - LX0), int(HB + py)), 1, (255, 0, 255), -1)
        ims.append(img[:, max(0, int(x - LX0) - 250):int(x - LX0) + 250])
        if len(ims) >= 3: break
    if ims:
        w = min(i.shape[1] for i in ims)
        cv2.imwrite(os.path.join(os.environ.get('TEMP', '.'), 'rt_dump.png'), np.vstack([i[:, :w] for i in ims]))


def _paths(notes):
    """音符の軌跡 (時刻, 速さ, 位置のばらつき) の配列(追いかけ直したものはその軌跡)"""
    # 当てはめ直した軌跡と、検出が測った元の軌跡の両方を並べる(どちらかと同じなら同じ音符。揺れる画面では当てはめ直しが狂う: 第六天魔王 裏 125.69s)
    # 判定枠の手前でだけ見えた音符(emerge)には印を付ける(時刻の一致で同じとみなさない。下の recover の最後で足した直線とまとめる)
    tt, vv, sd = [], [], []; em = []
    for o in notes:
        t0_, v0_ = o.get('_obs', (o['t'], abs(o['vel'])))
        tt.append(t0_); vv.append(v0_); sd.append(float(o.get('res_sd') or 0.0)); em.append(bool(o.get('emerge')))
        if '_fit' in o: tt.append(o['_fit'][0]); vv.append(o['_fit'][1]); sd.append(0.0); em.append(False)
    _paths.em = np.array(em, bool)
    return np.array(tt, float), np.array(vv, float), np.array(sd, float)


def _same_any(h, tt, vv, fps, sd=None, em=None):
    """_same_path を音符の配列にまとめて掛ける。許容には既存の音符の軌跡のばらつき(res_sd)の 2 倍を足す(画面が揺れる演出で軌跡が直線から
    7px ばらつくと、その音符の円が 16px ずれて見え、重複として足していた: 第六天魔王 裏 120〜127s)"""
    if len(tt) == 0: return False
    m = np.abs(tt - h['t']) < 4.0
    if not m.any(): return False
    vh = h['vel']; fs = np.linspace(h['t'] * fps - (X_IN - JX) / vh, h['t'] * fps, 20)
    d = np.abs((h['t'] * fps - fs)[None, :] * vh - (tt[m, None] * fps - fs[None, :]) * vv[m, None])
    tol = 2 * NEAR + (2 * sd[m] if sd is not None else np.zeros(int(m.sum())))
    if np.any(np.median(d, axis=1) < tol): return True
    # 判定枠に着く時刻が同じ(差が判定枠で 2×NEAR 未満)で速さも同じ(±10%)なら同じ音符(1 本のレーンで同時に 2 つは着かない)。
    # 判定枠の近くの数点しか見えなかった音符の速さは当てにならず、全行程に延ばして比べると別の軌跡に見える
    # (Rough: CHA0Z 22.90s: x964〜727 の 7 点で速さ 26.5 と測ったドンと、同じドンの縁の直線 25.9 が 6ms 差で二重になった)
    same_t = (np.abs(tt[m] - h['t']) * vh * fps < 2 * NEAR) & (np.abs(vh / vv[m] - 1) <= 0.1)
    if em is not None: same_t &= ~em[m]
    return bool(np.any(same_t))


def _ring_p(h, F, X, C, fps):
    """縁を確かめようがない直線の、円の数の偶然の確率。偶然の率は、同じ速さで判定枠に着く時刻をずらした(音符の直径ぶん以上離した)直線に
    同じ色の説明できない円が乗る割合で測る(揺れる画面では説明できない円がそこら中にある: 第六天魔王 裏 120.4s は 0.5 の 7 乗 = 0.008 で足していた)"""
    v = h['vel']; fc = h['t'] * fps; col = h['color']
    m = C == col
    f_lo, f_hi = fc - (X_IN - JX) / v, fc - 40 / v
    frames = np.unique(F[(F >= f_lo) & (F <= f_hi)])
    n_fr = max(1, int(round(f_hi - f_lo)))
    def hits(fc_):
        on = m & (F >= fc_ - (X_IN - JX) / v) & (F <= fc_ - 40 / v) & (np.abs(JX + (fc_ - F) * v - X) <= NEAR)
        return len(np.unique(F[on]))
    k = hits(fc)
    step = 2 * R_S / v
    ctrl = [hits(fc + s_ * step * j) for s_ in (-1, 1) for j in (1, 2, 3)]
    p0 = min(0.99, max(1.0 / n_fr, (sum(ctrl) + 1) / (len(ctrl) * n_fr + 2)))
    from math import comb
    return sum(comb(n_fr, i) * p0 ** i * (1 - p0) ** (n_fr - i) for i in range(k, n_fr + 1))


def _shape_shared(h, labs, occ, others, fps, big):
    _shape(h['t'], h['vel'], big, labs, occ, others, fps)
    return _shape.shared
