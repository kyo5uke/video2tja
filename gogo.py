"""ゴーゴータイムの区間を動画から取る。

見分け方: ゴーゴー中はレーンの背景が明るい色のグラデーションになる。
・明るさ(V)で分ける。実測(3 曲): ゴーゴー中 V 中央値 123〜125、それ以外 61〜72。
・彩度(S)は使えない。レーン背景は拍に合わせて色が変わり、ゴーゴーでなくても S 145 になる瞬間がある(タベルナ 95s)。
・音符(大は 327-445 行)と判定枠(x<750)を避けた帯の上下(295-325 行と 450-476 行)の中央値を見る。

境目: 画面が明るくなるのは譜面の #GOGOSTART より **+190ms**、暗くなるのは #GOGOEND より **+90ms** 遅れる
(ゲームの演出の立ち上がり。3 曲 17 箇所で開始 +153〜+217ms、終了 +51〜+132ms と安定している)。
その分を引いてから、検出した小節線・音符のうち最も近いものに ±150ms で寄せる
(本家のゴーゴーの境目は小節線に乗っていることがほとんど。まれに小節の途中の音符に乗る: 雲の彼方 93.9s)。
"""
import cv2
import os
import numpy as np
import calib

ROWS = ((295, 325), (450, 476))       # 音符と判定枠を避けた帯の上下(基準配置)
X0, X1 = 800, 1850
STEP = 3                              # 3 フレームおき(20Hz)
# 特徴量は「赤 − 緑」の中央値(2026-09-22 に明るさ V から変更)。明るさだとさいたま2000 の動画のように
# ゴーゴーが暗い(V 92、通常 65)と取りこぼす。赤 − 緑 は ゴーゴー +52〜+153、通常 −14〜+25、拍の瞬き(茶)+21
TH_HI = 42.0                          # これ以上が 3 点続いたらゴーゴー開始
TH_LO = float(__import__('os').environ.get('V2T_GOGO_TH_LO', '40'))                          # これ未満が 3 点続いたら終了。15 曲の実測で通常のレーンは上位 5% でも 38 以下、ゴーゴー中は下位 5% でも 41 以上(regress/gogo_levels.py)。34 だと終わった後が 35 の曲(闇の魔法少女)で遅れた
BRI_HI = 85.0                         # 開始には明るさも要る。ゴーゴー直前のちらつきは 赤 − 緑 が 47 まで上がるが暗い(V 77〜80、雲の彼方 130.3s)。
                                      # ゴーゴーは暗い動画でも V 91 以上(さいたま2000)
RUN = 3                               # 立ち上がり/立ち下がりに必要な連続点数(150ms)
MEDIAN = 3                            # 中央値で瞬きを消す
MIN_LEN = 0.25                        # これより短い区間は捨てる(彁の 0.41 秒のゴーゴーがあるので短めに)。全フレームで確かめた短い区間(_SHORT_OK)は例外
_SHORT_OK = []                        # 全フレームで確かめた短い区間(生の時刻)。MIN_LEN で捨てない
SHORT_MIN_FR = 5                      # 中央値で消えた短い区間を本物とみなすのに要る、ゴーゴーの色が続くフレーム数
LAG_START, LAG_END = 0.190, 0.090     # 演出の立ち上がりの遅れ
LAG_END_FINE = float(__import__('os').environ.get('V2T_GOGO_LAG_END_FINE', '0.065'))   # 全フレームで測った終わりの遅れ(20Hz の最初の点は平均 25ms 遅いので LAG_END − 0.025)
LAG_START_FINE = float(__import__('os').environ.get('V2T_GOGO_LAG_FINE', '0.165'))   # 全フレームで測った開始の遅れ(20Hz の最初の点は平均 25ms 遅い)
SNAP = 0.150                          # 小節線に寄せる窓
SNAP_NOTE = 0.060                     # 小節線が近くに無いときだけ、音符に寄せる窓(狭くする)


# ゴーゴーのレーンの横のグラデーション: 左(判定枠側 x800〜1000)と右(x1650〜1850)の青の差。ゴーゴー中 +31〜+55、それ以外 ≈0(6 曲の実測、regress/grad_probe.py)
# レーンの背景の色を測る窓(_gogo_feat が窓ごとの RGB の中央値と背景の割合を返す。ここを変えると全曲の再デコードになるので広めに持つ)
WINDOWS = ((800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1750, 1880))
W_MID, W_RIGHT = 2, 5            # 指標に使う窓: 中ほど(1200-1400)と右端(1750-1880)。判定枠側はドンを叩くと赤く光るので使わない
BG_MIN = 0.5                     # 背景の画素がこれ未満の窓は測れなかった扱い(連打の棒や風船がレーンを覆う)
GRAD_ON = 12.0           # 中→右の「赤が減って青が増える」量の区間の中央値がこれ以上ならゴーゴー(実測 ゴーゴー +21〜+55 / 分岐の紫 −2〜+4)
# ゴーゴー中も 1〜2 割のフレームは一時的に 15 を下回る(拍の瞬き・大音符)。分岐のレーンは短くても 1.7 秒続く(Emma 裏)ので、1 秒(21 点)の中央値で見る
GRAD_WIN = 21
_GRAD = None             # from_feats が残すグラデーションの時系列
_RGB_R = None            # from_feats が残す右端の背景の色(RGB)の時系列
_FULL = None             # 検出が全フレームで取ったゴーゴーの特徴 (feats, fps)。全フレームの測り直しはここから引く(動画を読み直さない)
_ALL = None              # from_feats が残す全フレームの (時刻, 右端の色, 赤−緑, fps)。色の判定に使う


def _full_frames(t0, t1):
    """[t0, t1) の全フレームの特徴 [(f, feat)]。1 枚でも無ければ None(動画を読む)"""
    if _FULL is None: return None
    feats, fps = _FULL
    fa, fb = int(np.ceil(t0 * fps - 1e-6)), int(np.ceil(t1 * fps - 1e-6))
    if fb <= fa: return None
    out = []
    for f in range(fa, fb):
        g = feats.get(f)
        if g is None: return None
        out.append((f, g))
    return out
BRANCH_PURPLE = (108.0, 26.0, 94.0)   # 譜面分岐(達人譜面)のレーンの色。Emma 裏 111,30,96 / 夢と希望 108,26,93 / エンジェル ドリーム裏 106,23,95。紫の上のゴーゴーは 152〜164,44〜49,105〜118
PURPLE_DIST = 25.0      # 右端の色がこの距離以内なら分岐のレーン(紫の上のゴーゴーは 50 以上離れる)


# レーン右端(x1750〜1880)の背景の色(RGB)。譜面分岐のレーンはゲームの UI として決まった色で、ゴーゴーはどの分岐でもほぼ (+36, +7, +20) が上乗せされる。
# 12 曲の実測(regress/lane_colors.py、ばらつき ±5 前後): 普通 44,42,40 / 80,49,60、玄人 29,73,93 / 74,81,115、達人 108,26,95 / 143,36,116
LANE_REFS = (((44, 42, 40), False), ((80, 49, 60), True), ((29, 73, 93), False), ((74, 81, 115), True), ((108, 26, 95), False), ((143, 36, 116), True))
REF_MAX = 30.0           # 参照色からこれより遠い色は測れない扱い(参照色のばらつきは ±5〜10)
LAG_C_START = float(__import__('os').environ.get('V2T_GOGO_LAG_C_START', '0.175'))   # 色で決めた開始の画面の遅れ(全曲の中央値)
LAG_C_END = float(__import__('os').environ.get('V2T_GOGO_LAG_C_END', '0.009'))       # 色で決めた終わりの画面の遅れ


_REF_OFF = np.zeros(3)   # 録画ごとの色のずれ(_color_intervals が曲ごとに決める)


GOGO_DIR = np.array((36.0, 7.0, 20.0))   # ゴーゴーで分岐のレーンの色に上乗せされる色(12 曲の平均)。録画によって大きさは 1〜1.7 倍(Xa 裏の OBS 録画 62,8,38)だが向きは同じ


def lane_is_gogo(rgb):
    """右端の色 → ゴーゴーか(True/False)。測れない(NaN)ときは None。
    分岐のレーンの色(普通・玄人・達人、録画ごとのずれを足したもの)に近ければゴーゴーではない。そこからの変化がゴーゴーの向き(R:G:B ≒ 36:7:20)に
    そろい、大きさが標準の半分以上ならゴーゴー。どちらでもない見えている色(曲ごとの演出のレーン・音符の色が混ざったもの)はゴーゴーではない"""
    if rgb is None or np.any(np.isnan(rgb)): return None
    c = np.asarray(rgb, float)
    bases = [np.asarray(b, float) + _REF_OFF for b, g in LANE_REFS if not g]
    dists = [float(np.linalg.norm(c - b)) for b in bases]
    u = GOGO_DIR / np.linalg.norm(GOGO_DIR); smag = float(np.linalg.norm(GOGO_DIR))
    best_g = None
    for b in bases:
        dv = c - b; pr = float(dv @ u); perp = float(np.linalg.norm(dv - pr * u))
        if pr >= 0.5 * smag and perp <= max(15.0, 0.35 * pr):
            best_g = pr if best_g is None else max(best_g, pr)
    if min(dists) <= 0.5 * smag: return False        # 分岐のレーンの色そのもの
    if best_g is not None: return True
    return None if os.environ.get('V2T_FAR_HOLD', '1') != '0' else False    # どちらでもない色(連打の棒・風船・演出が右端にかかった)は測れない扱い


def right_rgb(video, t0, t1, cal=None):
    """[t0, t1) の全フレームのレーン右端の背景の色 → (時刻[], RGB[])"""
    ff = _full_frames(t0, t1)
    if ff is not None:
        fps = _FULL[1]; ts = []; cs = []
        for f, g in ff:
            (rgb, frac) = g[4][W_RIGHT]
            ts.append(f / fps); cs.append(rgb if frac >= BG_MIN else (np.nan, np.nan, np.nan))
        return np.array(ts), np.array(cs, float)
    import detect2
    hd = detect2.band_start(video, t0, t1, (250, 516), 1, cal, parts=1)
    ts = []; cs = []
    fps = hd['fps']
    for f, img in detect2.band_iter(hd):
        (rgb, frac) = detect2._gogo_feat(img)[4][W_RIGHT]
        ts.append(f / fps); cs.append(rgb if frac >= BG_MIN else (np.nan, np.nan, np.nan))
    return np.array(ts), np.array(cs, float)


def _short_ok(a, b):
    """全フレームで確かめた短い区間に重なるか(ラグを引いた後でも ±0.3 秒で照合)"""
    return any(a - 0.3 <= y and x <= b + 0.3 for x, y in _SHORT_OK)


def _color_intervals(ts, rgb_, video, cal, verbose, rg_=None, rate=None, bri_=None):
    """右端の色でゴーゴーの区間を決める(分岐の種類によらない)。境目は前後だけ全フレームで測り直す。戻り値は LAG_START/LAG_END を引く前の値"""
    # 録画ごとの色のずれ: 曲の中で最も多い右端の色(ふつうは通常のレーン)と、それに最も近い分岐のレーンの参照色の差。
    # 最も多い色がどの参照色からも REF_MAX 以上離れているとき(曲ごとの演出のレーン)はずらさない(さいたま2000 の明るい録画: 54,53,53)
    global _REF_OFF
    _REF_OFF = np.zeros(3)
    ok_ = ~np.isnan(rgb_).any(1)
    if ok_.sum() >= 50:
        q = np.round(rgb_[ok_] / 4) * 4
        vals, cnts = np.unique(q, axis=0, return_counts=True)
        mode = vals[int(np.argmax(cnts))]
        refs = [np.asarray(c, float) for c, _ in LANE_REFS]          # 最も多い色がゴーゴーの色のこともある(雲の彼方・Replay)ので 6 色のうち最も近いもの
        k = int(np.argmin([np.sum((mode - b) ** 2) for b in refs]))
        if np.sqrt(np.sum((mode - refs[k]) ** 2)) <= REF_MAX: _REF_OFF = mode - refs[k]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-color 録画の色のずれ %s' % np.round(_REF_OFF).astype(int).tolist())
    lab = []; direct = []
    prev = False; base_k = 0; bases_k = []
    for i_, c in enumerate(rgb_):
        g = lane_is_gogo(c); direct.append(g is True)
        if g is not None and not np.isnan(c).any():
            base_k = int(np.argmin([np.linalg.norm(np.asarray(c, float) - (np.asarray(b, float) + _REF_OFF) - (GOGO_DIR if g else 0))
                                    for b, gg in LANE_REFS if not gg]))          # 今の分岐のレーン(0 普通・1 玄人・2 達人)
        if g is None:
            # 右端の色が測れない(隠れている)ときは、普通のレーンならレーン全体の 赤−緑(ゴーゴーは 41 以上、通常は 38 以下)で決める。
            # 玄人・達人のレーンは 赤−緑 がゴーゴーでなくても高い(達人の紫 82〜94)ので直前の状態。
            # 色は測れているのにどのレーンの色でもない(分岐のレーンが切り替わる途中の色など)ときも直前の状態(普通 → 達人の紫へ移る途中で 赤−緑 が
            # 上がり、ゴーゴーと誤った: Emma 裏 17.6s)
            measurable = not np.isnan(c).any()
            g = bool(rg_[i_] >= TH_LO) if (not measurable and base_k == 0 and rg_ is not None and not np.isnan(rg_[i_])) else prev
        bases_k.append(base_k)
        lab.append(g); prev = g
    lab_raw = np.array(lab, bool)
    # 見本の間隔(rate: 1 秒あたりの点の数。既定は STEP おき = 20 点)
    rate = rate or (60.0 / STEP)
    k_med = max(1, int(round(0.25 * rate))) | 1                 # 0.25 秒(20 点/秒なら 5 点)
    n_short = int(round(0.2 * rate))                           # 0.2 秒(20 点/秒なら 4 点)
    lab = _median_filter(np.array(lab, float), k_med) >= 0.5   # 0.25 秒の中央値で瞬きを消す
    iv = []; st = None
    for i in range(len(ts)):
        if lab[i] and st is None: st = i
        if not lab[i] and st is not None: iv.append((st, i)); st = None
    if st is not None: iv.append((st, len(ts) - 1))
    if rate > 30:
        # 全フレームで見るときは、右端の色そのものがゴーゴーの色だったフレームが SHORT_MIN_FR 以上ある区間だけ(瞬きや分岐のレーンが切り替わる途中の色は
        # 1〜2 フレームだけゴーゴーの色に見える: Emma 裏 17.6s は普通 → 達人の紫へ移る途中の 2 フレーム)
        # さらに明るいこと(ゴーゴーのレーンは明るい。ゴーゴー直前のちらつきは色が近づいても暗い: 明るさで決めていた頃の開始の条件 BRI_HI と同じ。
        # エンジェル ドリーム 裏 135.6s・Hopes and Dreams 68.0s・まださいたま 87.7s の 0.15〜0.25 秒の区間)
        dr = np.array(direct, bool)
        if bri_ is not None: dr &= np.nan_to_num(np.asarray(bri_, float), nan=0.0) >= BRI_HI
        iv = [(i0, i1) for i0, i1 in iv if dr[i0:i1 + 1].sum() >= SHORT_MIN_FR]
    _color_intervals.bases = [(bases_k[i0], bases_k[i1]) for i0, i1 in iv]
    out = []
    from concurrent.futures import ThreadPoolExecutor
    # 全フレームで測り直す所(境目の前後・中央値で消えた短い区間)を先に集め、1 秒以内に並ぶものはまとめて 1 回で読む
    # (所ごとに ffmpeg で頭出しすると、キーフレームからのデコードが毎回かかって 1 曲 11 秒: Emma 裏は 48 か所)
    segs = []
    if video is not None and os.environ.get('V2T_GOGO_SHORT', '1') != '0':
        j = 0; n_ts = len(ts)
        while j < n_ts:
            if lab_raw[j] and not lab[j]:
                e = j
                while e + 1 < n_ts and lab_raw[e + 1] and not lab[e + 1]: e += 1
                if e - j + 1 <= n_short and not any(ts[i0] - 0.3 <= ts[j] <= ts[i1] + 0.3 for i0, i1 in iv): segs.append((j, e))
                j = e + 1
            else: j += 1
    spans = [(ts[i] - 0.15, ts[i] + 0.05) for i0, i1 in iv for i in (i0, i1)] + [(ts[j] - 0.1, ts[e] + 0.1) for j, e in segs]
    blocks = []
    for a, b in sorted(spans):
        if blocks and a - blocks[-1][1] < 1.0: blocks[-1][1] = max(blocks[-1][1], b)
        else: blocks.append([a, b])

    def _rb(ab):
        try: return right_rgb(video, max(0.0, ab[0]), ab[1], cal)
        except Exception as ex_b:
            if os.environ.get('V2T_GOGO_DBG'): print('  gogo-color 読み直し失敗 %.2f-%.2f %s' % (ab[0], ab[1], ex_b))
            return np.array([]), np.zeros((0, 3))
    if video is not None and blocks and (os.environ.get('V2T_GOGO_FINE', '1') != '0' or segs):
        _tb = __import__('time').time()
        with ThreadPoolExecutor(max_workers=min(8, __import__('detect2').cpu_budget())) as ex_: got_b = list(ex_.map(_rb, blocks))
        if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-color 読み直し %.1fs' % (__import__('time').time() - _tb))
    else: got_b = []
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-color 読み直し %d か所を %d 回で' % (len(spans), len(blocks)))

    def right_rgb_c(a, b):
        for (ba, bb), (tt, cc) in zip(blocks, got_b):
            if ba <= a + 1e-9 and b <= bb + 1e-9:
                tt = np.asarray(tt); m = (tt >= a - 1e-6) & (tt < b - 1e-6)
                return tt[m], np.asarray(cc)[m]
        raise RuntimeError('読み直していない区間 %.2f-%.2f' % (a, b))

    def _edge(args):
        t_, want = args
        try:
            tt, cc = right_rgb_c(t_ - 0.15, t_ + 0.05)      # 0.25 秒の中央値で境目が最大 2 点(0.1 秒)ずれるので前を広く
        except Exception as ex_e:
            if os.environ.get('V2T_GOGO_DBG'): print('  gogo-color edge 失敗 %.2f %s' % (t_, ex_e))
            return None
        labs = [lane_is_gogo(c) for c in cc]
        k = next((j for j in range(len(tt) - 2) if all(x is want for x in labs[j:j + 3])), None)
        if os.environ.get('V2T_GOGO_DBG') and k is None: print('  gogo-color edge 見つからず %.2f %s %s' % (t_, want, labs))
        return float(tt[k]) if k is not None else None
    jobs = []
    for i0, i1 in iv:
        jobs += [(ts[i0], True), (ts[i1], False)]
    got = [_edge(j_) for j_ in jobs] if (video is not None and os.environ.get('V2T_GOGO_FINE', '1') != '0') else [None] * len(jobs)
    for n_, (i0, i1) in enumerate(iv):
        a = got[2 * n_] if got[2 * n_] is not None else ts[i0]
        b = got[2 * n_ + 1] if got[2 * n_ + 1] is not None else ts[i1]
        out.append((a - LAG_C_START + LAG_START, b - LAG_C_END + LAG_END))
    # 中央値で消えた短い区間(3 フレームおきで 1〜4 点): 全フレームで測り直し、ゴーゴーの色が SHORT_MIN_FR フレーム以上続けば本物
    # (続・〆ドレー2000 51.5s の 0.22 秒のゴーゴー: 画面では 7 フレームだけゴーゴーの色。瞬きの偽物は 1〜2 フレーム)
    if video is not None and os.environ.get('V2T_GOGO_SHORT', '1') != '0':
        reads = []
        for j, e in segs:
            try: reads.append(right_rgb_c(ts[j] - 0.1, ts[e] + 0.1))
            except Exception: reads.append(([], []))
        for (j, e), (tt, cc) in zip(segs, reads):
            labs = [lane_is_gogo(c) is True for c in cc]
            best = (0, None); r0 = None
            for q, g_ in enumerate(labs + [False]):
                if g_ and r0 is None: r0 = q
                if not g_ and r0 is not None:
                    if q - r0 > best[0]: best = (q - r0, r0)
                    r0 = None
            if best[0] >= SHORT_MIN_FR:
                a = float(tt[best[1]]); b = float(tt[best[1] + best[0] - 1]) + (tt[1] - tt[0] if len(tt) > 1 else 1 / 60.0)
                out.append((a - LAG_C_START + LAG_START, b - LAG_C_END + LAG_END)); _color_intervals.bases.append((bases_k[j], bases_k[e]))
                _SHORT_OK.append((a, b))
                if verbose: print('  gogo-color 短い区間 %.2f-%.2f(全フレームでゴーゴーの色が %d フレーム続く)' % (a, b, best[0]))
        if verbose and segs: print('  gogo-color 短い区間の候補 %d 個を読み直した' % len(segs))
        o_ = sorted(range(len(out)), key=lambda k: out[k][0]); out = [out[k] for k in o_]; _color_intervals.bases = [_color_intervals.bases[k] for k in o_]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-color ' + ' '.join('%.2f-%.2f' % (a - LAG_START, b - LAG_END) for a, b in out))
    if rate > 30:
        # 全フレームの色で 0.25 秒の中央値を通った区間は、ゴーゴーの色が続いた根拠があるので長さで捨てない
        # (続・〆ドレー2000 57.6s の 0.3 秒のゴーゴーは、境目の遅れを引くと 0.24 秒になり MIN_LEN 0.25 で落ちた)
        _SHORT_OK.extend(out)
    keep = [k for k, x in enumerate(out) if x[1] - x[0] > MIN_LEN or _short_ok(x[0], x[1])]
    _color_intervals.bases = [_color_intervals.bases[k] for k in keep]          # 区間と同じ並びに
    return [out[k] for k in keep]


def brightness(video, t0, t1, cal=None, step=None, border_check=True, parts=None):
    """レーン背景の「赤 − 緑」の時系列(時刻[秒], 値)。名前は明るさを使っていた頃のまま"""
    if step == 1 and not border_check:
        ff = _full_frames(t0, t1)
        if ff is not None:
            fps = _FULL[1]
            return np.array([f / fps for f, g in ff]), np.array([g[0] for f, g in ff], float), np.array([g[1] for f, g in ff], float)
    M = None
    if cal is not None and not calib.is_canonical(cal):
        M, _ = calib.affine(cal)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    # レーンの上下の縁(黒い線)の行を、この動画で探す(縮尺の違うキャプチャは補正後も数 px〜十数 px ずれる:
    # でぃんぴーでぃんぱー の動画は全フレームで「隠れている」と誤判定し、ゴーゴーを 1/3 しか取れなかった)
    import detect2
    # 帯(基準配置の 250〜515 行)だけを STEP 枚おきに ffmpeg(GPU)で読む(OpenCV で全フレームを読むと 1 曲 9 秒)。
    # ffmpeg を先に起動し、その間に縁の行を探す
    hd = detect2.band_start(video, f0 / fps, f1 / fps, (250, 516), step or STEP, cal, parts=parts)
    border = _find_border(cap, f0, f1, M) if border_check else None     # 縁を探すシークは遅い(AV1 で 1 回数秒)。短い測り直しでは省く
    cap.release()
    ts = []; val = []; bri = []
    for f, img in detect2.band_iter(hd):
        if True:
            rows = np.vstack([img[a:b, X0:X1] for a, b in ROWS])
            ts.append(f / fps)
            rr = rows.astype(np.int16)
            v_ = float(np.median(rr[:, :, 2] - rr[:, :, 1]))      # 赤 − 緑(ゴーゴーのレーンは赤〜桃の帯)
            br_ = float(np.median(rr.max(2)))                      # 明るさ(HSV の V)
            # レーンの上下の縁(y≈285, 485 の黒い線、通常もゴーゴー中も V<10)が明るいフレームはレーンが隠れている
            # (曲ごとのカットイン演出: 倍倍FIGHT! のヒヨコが画面全体を覆い、明るいのでゴーゴーと誤認した)。NaN にして前後で埋める
            # 縁の線は細い(数 px)ので、±6 行の中の最も暗い行で見る(補正で数 px ずれても拾える: さいたま2000 の 1906px 幅の動画)
            if border is not None:
                e1 = cv2.cvtColor(img[border[0] - 3:border[0] + 4, X0:X1], cv2.COLOR_BGR2HSV)[:, :, 2]
                e2 = cv2.cvtColor(img[border[1] - 3:border[1] + 4, X0:X1], cv2.COLOR_BGR2HSV)[:, :, 2]
                dark = min(float(np.median(e1, axis=1).min()), float(np.median(e2, axis=1).min()))   # 上下どちらかの線が見えればレーンは見えている
                if dark > 40: v_ = br_ = float('nan')
            val.append(v_); bri.append(br_)
    return np.array(ts), np.array(val, float), np.array(bri, float)


def _find_border(cap, f0, f1, M):
    """レーンの上下の縁の行(黒い線)。20 枚の行ごとの明るさの中央値で、上側(260-300)・下側(465-505)の最も暗い行。
    見つからなければ None(隠れの判定をしない)"""
    rows = []
    for f in np.linspace(f0, max(f0, f1 - 1), 20).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if not ok: continue
        if M is not None: img = cv2.warpAffine(img, M, (1920, 1080))
        rows.append(np.median(img[250:515, X0:X1].max(2), axis=1))
    if not rows: return None
    r = np.median(np.array(rows), axis=0)
    top = 260 + int(np.argmin(r[10:50])); bot = 465 + int(np.argmin(r[215:255]))
    if r[top - 250] > 40 or r[bot - 250] > 40: return None
    return top, bot


def _median_filter(x, k):
    h = k // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def from_feats(feats, fps):
    """detect2.run_parallel(gogo_range=…) がデコード中に取った特徴から brightness と同じ (ts, val, bri) を作る"""
    global _FULL
    fs = sorted(feats)
    if not fs: return np.array([]), np.array([]), np.array([])
    full = len(fs) > 1 and fs[1] - fs[0] == 1
    _FULL = (feats, fps) if full else None
    if full: fs = [f for f in fs if feats[f][2] is not None]      # 指標は STEP おきの全部の特徴があるフレーム(ほかのフレームの軽い特徴は測り直しに使う)
    # 縁の行: 区間から均等に 20 枚(_find_border と同じ。縁は動かない UI なので最寄りのサンプルで同じ結果)
    pick = sorted(set(fs[int(round(q))] for q in np.linspace(0, len(fs) - 1, 20)))
    tops = np.median(np.array([feats[f][2] for f in pick]), axis=0); bots = np.median(np.array([feats[f][3] for f in pick]), axis=0)
    # _find_border: 上 260-299 行・下 465-504 行の最も暗い行(窓は 257-303 / 462-508 行を持っている)
    top = 260 + int(np.argmin(tops[3:43])); bot = 465 + int(np.argmin(bots[3:43]))
    border = None if (tops[top - 257] > 40 or bots[bot - 462] > 40) else (top, bot)
    ts = []; val = []; bri = []
    gr = []; rgbs = []
    for f in fs:
        v_, br_, tp, bt = feats[f][:4]
        w_ = feats[f][4] if len(feats[f]) > 4 else None
        if isinstance(w_, (list, tuple)) and w_ and isinstance(w_[0], (list, tuple)):
            (rm, gm, bm), fm = w_[W_MID]; (rr, gr2, br2), fr = w_[W_RIGHT]
            gr.append((br2 - bm) - (rr - rm) if (fm >= BG_MIN and fr >= BG_MIN) else np.nan)   # 中→右で「赤が減って青が増える」量
            rgbs.append((rr, gr2, br2) if fr >= BG_MIN else (np.nan, np.nan, np.nan))          # 右端の背景の色
        else:
            gr.append(np.nan); rgbs.append((np.nan, np.nan, np.nan))
        if border is not None:
            dark = min(float(tp[border[0] - 3 - 257:border[0] + 4 - 257].min()), float(bt[border[1] - 3 - 462:border[1] + 4 - 462].min()))
            if dark > 40: v_ = br_ = float('nan')
        ts.append(f / fps); val.append(v_); bri.append(br_)
    global _GRAD, _RGB_R, _ALL
    _GRAD = np.array(gr, float); _RGB_R = np.array(rgbs, float)
    _ALL = None
    if _FULL is not None:
        # 全フレームの右端の色と 赤−緑(色の判定は全フレームで行う。STEP おきの見本だと、見本の位置が Δ で 1〜2 フレーム変わるだけで、
        # 点滅するレーンで偽の短いゴーゴーが出たり消えたりした: 倍倍FIGHT! 61.5s・Emma 裏)。レーンが隠れているか(縁の黒線)は近くの STEP のフレームで見る
        fs_a = np.array(fs); hid_a = np.isnan(np.array(val, float))
        fa = sorted(feats); ta, ra, va, ba = [], [], [], []
        near_i = np.clip(np.searchsorted(fs_a, fa), 0, len(fs_a) - 1) if len(fs_a) else None
        for n_f, f in enumerate(fa):
            g = feats[f]; w_ = g[4] if len(g) > 4 else None
            (rr, gg_, bb_), fr_ = w_[W_RIGHT] if (isinstance(w_, (list, tuple)) and w_ and w_[W_RIGHT] is not None) else ((np.nan,) * 3, 0.0)
            ta.append(f / fps); ra.append((rr, gg_, bb_) if fr_ >= BG_MIN else (np.nan, np.nan, np.nan))
            va.append(np.nan if (near_i is not None and hid_a[near_i[n_f]]) else g[0]); ba.append(g[1])
        _ALL = (np.array(ta), np.array(ra, float), np.array(va, float), fps, np.array(ba, float))
    return np.array(ts), np.array(val, float), np.array(bri, float)


def intervals(video, t0, t1, cal=None, bars=None, notes=None, verbose=True, rolls=None, feats=None, objs=None):
    """ゴーゴー区間 [(開始, 終了), ...](動画秒)。bars/notes を渡すとその格子に寄せる"""
    global _FULL
    _SHORT_OK.clear(); _FULL = None
    ts, val, bri = from_feats(*feats) if feats else brightness(video, t0, t1, cal)
    if len(ts) < 10: return []
    color_mode = _RGB_R is not None and feats and len(_RGB_R) == len(ts) and os.environ.get('V2T_GOGO_COLOR', '1') != '0'
    # レーンが隠れていたサンプルは直前の値を引き継ぐ(隠れている間に状態が変わったことにはしない)
    val = val.copy(); bri = bri.copy()
    for arr in (val, bri):
        for i in range(len(arr)):
            if np.isnan(arr[i]): arr[i] = arr[i - 1] if i > 0 and not np.isnan(arr[i - 1]) else 0.0
    v = _median_filter(val, MEDIAN); vb = _median_filter(bri, MEDIAN)
    # ヒステリシス: 明るいが 3 点続いたら開始、暗いが 3 点続いたら終了。
    # 単純な閾値だと、ゴーゴー直前のちらつき(雲の彼方 129.5-130.2s の 88〜98)で 237ms 早く始まってしまう
    iv = []; st = None
    for i in range(len(ts)):
        if st is None:
            if i + RUN <= len(ts) and np.all(v[i:i + RUN] >= TH_HI) and np.all(vb[i:i + RUN] >= BRI_HI): st = ts[i]
        else:
            if i + RUN <= len(ts) and np.all(v[i:i + RUN] < TH_LO): iv.append((st, ts[i])); st = None
    if st is not None: iv.append((st, ts[-1]))
    iv = [x for x in iv if x[1] - x[0] > MIN_LEN or _short_ok(x[0], x[1])]
    # 終わりは区間ごとの中間の値で決め直す: 固定の閾値(赤−緑 < 34)だと、終わった後のレーンが閾値すれすれの曲で遅れる
    # (闇の魔法少女 72.2s: 終わった後の 赤−緑 が 35 で +0.3 秒)。その区間のゴーゴー中の値(中央値)と、終わった後 1 秒の値(中央値)の中間を
    # 3 点続けて下回った最初の点を終わりにする(区間の後ろ半分から探す)
    if os.environ.get('V2T_GOGO_MIDEND', '0') == '1':
        iv2 = []
        for a, b in iv:
            ins = np.where((ts >= a) & (ts < b))[0]; aft = np.where((ts >= b) & (ts < b + 1.0))[0]
            if len(ins) < 6 or len(aft) < 6: iv2.append((a, b)); continue
            mid = (float(np.median(v[ins])) + float(np.median(v[aft]))) / 2
            half = ins[len(ins) // 2:]
            k = next((i for i in half if i + RUN <= len(ts) and np.all(v[i:i + RUN] < mid)), None)
            iv2.append((a, ts[k]) if k is not None and ts[k] < b else (a, b))
        iv = iv2
    # 開始は 20Hz の点のどこに境目が来るかで最大 50ms 前後する(魔宵月 裏 89.2s: 小節線 +76ms の手前の音符に寄って −70ms)。
    # 開始の前後だけ全フレームで測り直し、3 フレーム続けて明るい最初のフレームにする(遅れ LAG_START_FINE は全フレーム基準の値)
    fine = os.environ.get('V2T_GOGO_FINE', '1') != '0' and video is not None
    if fine:
        from concurrent.futures import ThreadPoolExecutor
        def _fine(a):
            try: return brightness(video, a - STEP / 60.0 - 0.02, a + 0.10, cal, step=1, border_check=False, parts=1)
            except Exception: return [], [], []
        with ThreadPoolExecutor(max_workers=min(8, __import__('detect2').cpu_budget())) as ex_: got = list(ex_.map(_fine, [a for a, _ in iv]))
        iv2 = []
        for (a, b), (t2, v2, b2) in zip(iv, got):
            k = next((i for i in range(len(t2) - 2) if np.all(v2[i:i + 3] >= TH_HI) and np.all(b2[i:i + 3] >= BRI_HI)), None)
            if verbose and os.environ.get("V2T_GOGO_DBG"): print("  gogo-fine %.3f → %s (%d 点)" % (a, "%.3f" % t2[k] if k is not None else "-", len(t2)))
            iv2.append((float(t2[k]) + (LAG_START - LAG_START_FINE), b) if k is not None else (a, b))
        iv = iv2
    # 区間の頭が「その区間の明るさ(75% 点)の 4 分の 3」に満たないまま 0.5 秒以上続くなら、そこはゴーゴーではない。
    # 譜面分岐のレーン(エンジェル ドリーム裏の達人譜面は明るい紫で V=114)がゴーゴー(V=180)の直前に来ると
    # ひと続きに見えるため、これが無いと 1.3 秒早く始まってしまう。
    # 終端側は削らない(ゴーゴーの終わりは元々暗くなっていくので、削ると 0.2〜0.7 秒短くなる)
    # 区間の頭が、明るさと 赤−緑 の両方とも区間の 75% 点の 3/4 未満のまま 0.5 秒以上続くなら、そこは削る。
    # ・分岐の紫(エンジェル ドリーム裏): 明るさ 113 / ゴーゴー 175〜196、赤 − 緑 86 / 153 → 両方低い → 削る
    # ・明るさが 105→140 と上がっていくゴーゴーの頭(愛♡スクリ～ム!): 赤 − 緑 は高い → 削らない
    # (2026-09-22: 分岐とゴーゴーを曲によらず分ける印を探した。判定枠の炎の色・判定枠のまわりの光・レーンの横のグラデーションは
    #  どれも全曲には通じなかった(HANDOFF 43)。見つかるまでは明るさの規則のまま)
    trimmed = []
    for a, b in iv:
        m = (ts >= a) & (ts <= b)
        idx = np.where(m)[0]
        lv = 0.75 * float(np.percentile(vb[m], 75)); lr = 0.75 * float(np.percentile(v[m], 75))
        i0 = idx[0]
        while i0 < idx[-1] and vb[i0] < lv and v[i0] < lr: i0 += 1
        trimmed.append((ts[i0] if ts[i0] - a > 0.5 else a, b))
    # 区間の中で明るさが「区間の 90% 点」の 0.7 未満のまま 3 秒以上続き、その中央値が 0.66 以下なら、そこは譜面分岐の紫(ゴーゴーではない)として区間を割る
    # (夢と希望 おに: 達人譜面の紫 R130 とその上のゴーゴー R174〜196 が 10 秒ずつ交互に来て 4 区間が 2 区間につながっていた。紫は 0.55〜0.61。
    #  ゴーゴー自身の暗い時期は 愛♡スクリ～ム! 0.72(8 秒続く)、エンジェル ドリーム裏 0.72〜0.81 なので割らない)
    rgb_ = _RGB_R if (feats and _RGB_R is not None and len(_RGB_R) == len(ts)) else None
    if rgb_ is not None and os.environ.get('V2T_GOGO_PURPLE', '1') != '0':
        # 明るいレーンのうち、右端が譜面分岐の紫のところはゴーゴーではない(1 秒の中央値で見る)。分岐のレーンはゲームの UI として決まった色で、
        # 叩いた光(判定枠側が赤くなる)も右端までは届かない。以前の明るさの規則(区間の頭の紫を削る・3 秒 0.66 で割る)の代わり(2026-09-23)
        dist = np.sqrt(((rgb_ - np.array(BRANCH_PURPLE)) ** 2).sum(1))
        # レーンの右端が隠れて色が測れないフレーム(カットインの絵など)は直前の状態を引き継ぐ
        for i_ in range(len(dist)):
            if np.isnan(dist[i_]): dist[i_] = dist[i_ - 1] if i_ else 999.0
        purple = _median_filter(dist, 21) <= PURPLE_DIST
        out_ = []
        for a, b in iv:
            idx = np.where((ts >= a) & (ts <= b))[0]
            k = 0
            while k < len(idx):
                if not purple[idx[k]]:
                    j = k
                    while j + 1 < len(idx) and not purple[idx[j + 1]]: j += 1
                    out_.append((a if k == 0 else ts[idx[k]], b if j == len(idx) - 1 else ts[idx[j + 1]]))
                    k = j + 1
                else: k += 1
        # 紫 → ゴーゴーに切り替わる始まり(区間の頭ではないもの)は 20Hz の点で 50ms 前後するので、前後だけ全フレームで測り直す。
        # 切り替わりでは明るさが紫の値からゴーゴーの値へ上がるので、前後の中央値の中間を越えた最初のフレームにする(Emma 裏 101.28s: −47ms)
        heads_ = {a for a, _ in iv}
        if video is not None and os.environ.get('V2T_GOGO_FINE', '1') != '0':
            fixed = []
            for a, b in out_:
                if a in heads_:
                    fixed.append((a, b)); continue
                try:
                    t2, v2, b2 = brightness(video, a - 0.15, a + 0.15, cal, step=1, border_check=False, parts=1)
                    lo_, hi_ = float(np.median(b2[:5])), float(np.median(b2[-5:]))
                    k = next((i for i in range(len(t2)) if b2[i] >= (lo_ + hi_) / 2), None) if hi_ - lo_ > 10 else None
                    fixed.append((float(t2[k]) + (LAG_START - LAG_START_FINE), b) if k is not None else (a, b))
                except Exception:
                    fixed.append((a, b))
            out_ = fixed
        if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-purple %s → %s' % (' '.join('%.2f-%.2f' % x for x in iv), ' '.join('%.2f-%.2f' % x for x in out_)))
        trimmed = out_
    gr_ = _GRAD if (feats and _GRAD is not None and len(_GRAD) == len(ts)) else None
    if gr_ is not None and os.environ.get('V2T_GOGO_GRAD', '0') == '1':
        # 試験中: 区間の中を持続するレーンの色の変化で区切り、区切りごとにゴーゴーのグラデーション(右の青 − 左の青)の中央値で決める
        gr0 = gr_.copy()                                  # 背景が見えないサンプル(NaN)は中央値に入れない
        out_ = []
        for a, b in trimmed:
            idx = np.where((ts >= a) & (ts <= b))[0]
            W_ = max(2, int(round(1.0 / max(1e-6, float(np.median(np.diff(ts)))))))
            if len(idx) < 2 * W_ + 2:
                out_.append((a, b)); continue
            med_b = max(1.0, float(np.median(vb[idx]))); med_r = max(1.0, float(np.median(np.abs(v[idx]))))
            step = np.zeros(len(idx))
            for k in range(W_, len(idx) - W_):
                i = idx[k]
                step[k] = max(abs(np.median(vb[i:i + W_]) - np.median(vb[i - W_:i])) / med_b,
                              abs(np.median(v[i:i + W_]) - np.median(v[i - W_:i])) / med_r,
                              abs(np.nan_to_num(np.nanmedian(gr0[i:i + W_])) - np.nan_to_num(np.nanmedian(gr0[i - W_:i]))) / 30.0)
            cuts = []
            for k in np.argsort(-step):                      # 大きい段差から、前後 1 秒以内に他の区切りが無いものを採る
                if step[k] < 0.3: break
                if all(abs(k - c) > W_ for c in cuts): cuts.append(int(k))
            bounds = [0] + sorted(cuts) + [len(idx)]
            segs = [(ts[idx[k0]] if k0 else a, ts[idx[k1]] if k1 < len(idx) else b, float(np.nan_to_num(np.nanmedian(gr0[idx[k0:k1]]), nan=GRAD_ON))) for k0, k1 in zip(bounds, bounds[1:])]
            if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-grad %.2f-%.2f: ' % (a, b) + ' '.join('%.2f-%.2f(%.0f)' % q for q in segs))
            cur = None
            for sa, sb, gm in segs:
                if gm >= GRAD_ON:
                    if cur is None: cur = [sa, sb]
                    else: cur[1] = sb
                elif cur is not None: out_.append(tuple(cur)); cur = None
            if cur is not None: out_.append(tuple(cur))
        trimmed = out_
    elif rgb_ is None and os.environ.get('V2T_GOGO_SPLIT', '1') != '0':      # 紫の判定が使えないとき(古いキャッシュ・動画を直接読むとき)だけ
        split = []
        for a, b in trimmed:
            m = (ts >= a) & (ts <= b); idx = np.where(m)[0]
            if len(idx) < 60: split.append((a, b)); continue
            sm = np.array([np.median(vb[max(idx[0], i - 10):min(idx[-1], i + 10) + 1]) for i in idx])
            hi = float(np.percentile(sm, 90)); low = sm < 0.70 * hi
            cur = a; i = 0
            while i < len(idx):
                if low[i]:
                    j = i
                    while j + 1 < len(idx) and low[j + 1]: j += 1
                    if ts[idx[j]] - ts[idx[i]] >= 3.0 and np.median(sm[i:j + 1]) <= 0.66 * hi:
                        if ts[idx[i]] - cur > MIN_LEN: split.append((cur, ts[idx[i]]))
                        cur = ts[idx[j]] if j + 1 < len(idx) else None
                    i = j + 1
                else: i += 1
            if cur is not None and b - cur > MIN_LEN: split.append((cur, b))
        trimmed = split
    # 終わりも開始と同じく全フレームで測り直す(20Hz の点のどこに境目が来るかで最大 50ms 前後する)。終わりの前後(−70〜+30ms)で、
    # 赤−緑 が TH_LO を 3 フレーム続けて下回った最初のフレーム − LAG_END_FINE にする。区間の途中で分けた終わり(紫の手前)は対象外
    trimmed_ends = {b for _, b in iv}
    if video is not None and os.environ.get('V2T_GOGO_FINE_END', '1') != '0':
        from concurrent.futures import ThreadPoolExecutor
        def _fine_end(b):
            try: return brightness(video, b - STEP / 60.0 - 0.02, b + 0.05, cal, step=1, border_check=False, parts=1)
            except Exception: return [], [], []
        ends_ = sorted({b for _, b in trimmed if b in trimmed_ends})
        with ThreadPoolExecutor(max_workers=min(8, __import__('detect2').cpu_budget())) as ex_: got_e = dict(zip(ends_, ex_.map(_fine_end, ends_)))
        tr2 = []
        for a, b in trimmed:
            t2, v2, b2 = got_e.get(b, ([], [], []))
            k = next((i for i in range(len(t2) - 2) if np.all(np.asarray(v2[i:i + 3]) < TH_LO)), None) if len(t2) else None
            nb = float(t2[k]) + (LAG_END - LAG_END_FINE) if k is not None else b
            if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-fine-end %.3f → %.3f' % (b, nb))
            tr2.append((a, nb))
        trimmed = tr2
    if color_mode:
        # ゴーゴーかどうかは右端の色で決める(分岐の種類によらない)。境目の時刻は、明るさで決めた境目(普通のレーンでは正確、ゴーゴー直前のちらつきも除く)が 0.5 秒以内にあればそちら、
        # 無ければ(玄人・達人のレーンなど明るさでは決められない所)色の境目を使う(雲の彼方: 色の境目は移り変わりのフレームの判定で +24ms 遅れた)
        old_st = [a for a, _ in trimmed]; old_en = [b for _, b in trimmed]
        if _ALL is not None:
            ta_, ra_, va_, fps_a, ba_ = _ALL
            va_f = va_.copy()
            for i in range(len(va_f)):
                if np.isnan(va_f[i]): va_f[i] = va_f[i - 1] if i > 0 and not np.isnan(va_f[i - 1]) else 0.0
            if objs:
                # 右端の窓に音符・連打の頭や棒が掛かるフレームは、背景の色を測れない(大連打の頭の顔の赤い口や縁を背景と数え、ゴーゴーと誤った: Emma 裏 73.8s)
                ra_ = ra_.copy(); x0w, x1w = WINDOWS[W_RIGHT]; JXg = 615.0
                fr_a = ta_ * fps_a
                for th, te, vv, rr in objs:
                    xh = JXg + (th * fps_a - fr_a) * vv; xe = JXg + (te * fps_a - fr_a) * vv
                    cov = (xh - rr < x1w) & (xe + (2 * rr if te > th else rr) > x0w)
                    ra_[cov] = np.nan
            col = _color_intervals(ta_, ra_, video, cal, verbose, rg_=_median_filter(va_f, MEDIAN * STEP), rate=fps_a, bri_=ba_)

        else:
            col = _color_intervals(ts, _RGB_R, video, cal, verbose, rg_=v)
        def _near(t_, cands):
            c_ = [x for x in cands if abs(x - t_) <= 0.5]
            return min(c_, key=lambda x: abs(x - t_)) if c_ else t_
        bs_ = getattr(_color_intervals, 'bases', [(0, 0)] * len(col))
        # 玄人のレーン(base 1)では明るさの境目は当てにならない(青いレーンの上でゴーゴーの 赤−緑 が弱い)ので色の境目のまま
        # 寄せる相手は、その色の区間と重なる明るさの区間の境目だけ(近いだけの別の区間の終わりに寄せて長さが 0 になり、続・〆ドレー2000 57.6s の
        # 0.3 秒のゴーゴーが消えた)
        def _hyb(a, b, ka_, kb_):
            ov = [(a2, b2) for a2, b2 in zip(old_st, old_en) if a2 < b and b2 > a]
            a_n = _near(a, [x for x, _ in ov]) if ka_ != 1 else a
            b_n = _near(b, [y for _, y in ov]) if kb_ != 1 else b
            return (a_n, b_n) if b_n > a_n else (a, b)
        trimmed = [_hyb(a, b, ka_, kb_) for (a, b), (ka_, kb_) in zip(col, bs_)] if os.environ.get('V2T_GOGO_HYBRID', '1') != '0' else col
    iv = [(a - LAG_START, b - LAG_END) for a, b in trimmed if b - a > MIN_LEN or _short_ok(a, b)]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage568 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    if notes and iv:
        # 音符の無い区間が前の区間の終わりから 0.3 秒以内に始まるなら、前の区間の続き(第六天魔王 裏 153.6s: 曲の終わりの演出で画面が明るくなる
        # 瞬間に 赤−緑 が一瞬下がって割れ、後ろの 6 秒が「音符の無い区間」として捨てられた)。後ろに音符があれば別の区間のまま(銀の黎明の 0.2 秒の切れ目)
        na0 = np.asarray(sorted(notes)); merged_ = [list(iv[0])]
        for a, b in iv[1:]:
            has_note = ((na0 >= a - 0.3) & (na0 <= b + 0.3)).any()
            if not has_note and a - merged_[-1][1] <= 0.3: merged_[-1][1] = max(merged_[-1][1], b)
            else: merged_.append([a, b])
        iv = [tuple(x) for x in merged_]
        if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage577 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    if notes:
        # 曲が終わったあとのクリア演出も明るい。終わりは「最終音符 +1 秒」で切り
        # (本家のゴーゴーは最終音符の 0.7 秒あとまで続くことがある: SORA-VIII。+0.5 秒で切ると 199ms 早まる)、
        # さらに「音符を 1 つも含まない区間」を捨てる(クリア演出はここで消える)
        na_ = np.asarray(sorted(notes))
        n0, n1 = float(na_[0]), float(na_[-1])
        # 終わりは「最終音符 +2 秒」まで(+1 秒を越える分は後で小節線に揃うときだけ残す。幽玄ノ乱: 本家のゴーゴーは最終音符の 1.7 秒後の
        # 小節線まで続き、+1 秒で切ると 793ms 早かった)
        # 最終音符の後も 赤−緑 が開始の基準(TH_HI)以上で 9 割以上続くなら、本物のゴーゴー(第六天魔王 裏: 本家は最終音符の約 9 秒後、空の小節 2 つの
        # 後まで。赤−緑 は 60 前後)。クリア演出は 赤−緑 が 35 前後で開始の基準に届かない(カキゴオリ)。そのときは最終音符 +12 秒まで認める
        long_tail = set()
        for k, (a, b) in enumerate(iv):
            if b > n1 + 1.0:
                m_ = (ts > n1 + 0.2) & (ts < b)
                if m_.sum() >= 10 and np.mean(v[m_] >= TH_HI) >= 0.9: long_tail.add(k)
        iv = [(max(a, n0 - 1.0), min(b, n1 + (12.0 if k in long_tail else 2.0))) for k, (a, b) in enumerate(iv)]
        if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage593 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
        # ・音符は前後 0.3 秒まで見る(彁の 0.41 秒のゴーゴーは、音符が開始の 5ms 手前にあるだけで中には無い)
        # ・最終音符の 0.3 秒後より後ろで始まる区間は捨てる(曲が終わったあとの演出)。
        #   以前は「0.2 秒前より後」だったが、最後の音符を含むゴーゴー(魔宵月裏: 最終音符の 50ms 前に開始)を消していた。
        #   カキゴオリのクリア演出(最終音符の 4ms 後)は 赤−緑 が 35 で、そもそも開始の条件(42)に届かない
        iv = [(a, b) for a, b in iv if (b - a > MIN_LEN or _short_ok(a, b)) and a <= n1 + 0.3
              and ((na_ >= a - 0.3) & (na_ <= b + 0.3)).any()]

    ba = np.array(sorted(bars or [])); na = np.array(sorted(notes or []))

    def snap(t):
        # ①±60ms 以内に小節線か音符があれば、近い方に寄せる(ゴーゴーは小節の途中の音符から始まることもある:
        #   Taiko Drum Monster は 90ms 先の小節線に寄せて +90ms 遅れた)
        # ②無ければ ±150ms 以内の小節線に寄せる。音符は 60ms より遠いものには寄せない
        #   (曲末など小節線を取りこぼした所で 174ms 離れた音符に引っぱられ、SORA-VIII の終わりが 198ms 早くなった)
        # ⓪45ms 以内に小節線があれば小節線(本家のゴーゴーはたいてい小節頭。幽玄ノ乱 116.45s: 3ms 先の音符に寄せて小節頭より 40ms 早かった)
        if len(ba):
            j = int(np.argmin(np.abs(ba - t)))
            if abs(ba[j] - t) <= 0.045: return float(ba[j])
        best = None
        for arr in (ba, na):
            if len(arr):
                j = int(np.argmin(np.abs(arr - t)))
                if abs(arr[j] - t) <= SNAP_NOTE and (best is None or abs(arr[j] - t) < abs(best - t)): best = float(arr[j])
        if best is not None: return best
        if len(ba):
            j = int(np.argmin(np.abs(ba - t)))
            if abs(ba[j] - t) <= SNAP: return float(ba[j])
        return t

    iv = [(snap(a), snap(b)) for a, b in iv]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage623 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    if notes:
        # 最終音符 +1 秒より後ろの終わりは、小節線に揃ったときだけ残す(揃わなければクリア演出などとみなして +1 秒で切る)
        iv = [(a, b if (b <= n1 + 1.0 or (len(ba) and np.min(np.abs(ba - b)) < 1e-6) or b > n1 + 2.0 + 1e-6) else min(b, n1 + 1.0)) for a, b in iv]   # +2 秒を越えるのは上の長い尻尾だけ
    # 短い切れ目(0.2 秒ほど)で分かれたゴーゴーは、前の終わりと次の始まりが同じ小節線に寄ってしまう。本家は切れ目の後の音符から
    # 再開する(銀の黎明 裏 93.628 で終わり、次の音符 93.825 から)ので、重なったら次の始まりを「前の終わりより後の最初の音符」(0.5 秒以内)にする
    for k in range(1, len(iv)):
        pa, pb = iv[k - 1]; a, b = iv[k]
        if a <= pb + 0.02 and len(na):
            nxt = na[(na > pb + 0.02) & (na <= pb + 0.5)]
            if len(nxt) and nxt[0] < b: iv[k] = (float(nxt[0]), b)
    iv = [(a, b) for a, b in iv if b - a > MIN_LEN or _short_ok(a, b)]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage634 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    # 連打・風船の範囲にほぼ収まる短い区間(3 秒未満、6 割以上が連打の中)は捨てる。
    # 長い連打の帯がレーン全体を覆い、赤い帯をゴーゴーと誤認することがある(IRIS OUT むずかしい 89s)。
    # 「連打中は状態を引き継ぐ」やり方は、連打と同時に始まるゴーゴーの開始を連打の後ろまで遅らせた(雲の彼方 +1.2〜1.5 秒)
    def in_roll(a, b):
        # 帯がゴーゴーの判定行(295-325 / 450-476)に掛かる連打(大連打・風船)だけ。小さい連打の帯(高さ 92〜96px)は掛からないので、連打と同時の
        # ゴーゴーを捨てない(憎悪と醜悪の花束 おに 43.0〜44.2s: 小さい連打と同時の 1.2 秒のゴーゴーを捨てていた)
        cov = sum(max(0.0, min(b, e + 0.05) - max(a, h - 0.3)) for h, e, *bg in (rolls or []) if (bg[0] if bg else True))
        return cov / max(b - a, 1e-6)
    def past_roll(a, b):      # 連打の終わりより 0.15 秒以上長く続く(本家のゴーゴーが連打と同時に始まって小節の終わりまで続く: 第六天魔王 裏の大連打)
        return any(h - 0.3 <= a <= e and b > e + 0.15 for h, e, *_ in (rolls or []))
    iv = [(a, b) for a, b in iv if not (b - a < 3.0 and in_roll(a, b) >= 0.6 and not past_roll(a, b))]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage645 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    # 始まりの 0.15 秒後までに連打の頭があれば、開始を連打の頭にする(速い連打の帯は頭より先にレーンを覆うので明るさが早く上がる: 第六天魔王 裏 -94ms)
    heads_ = sorted(r[0] for r in (rolls or []))
    iv = [(next((h for h in heads_ if a < h <= a + 0.15), a), b) for a, b in iv]
    if verbose and os.environ.get('V2T_GOGO_DBG'): print('  gogo-stage648 %s' % ' '.join('%.2f-%.2f' % x for x in iv))
    if verbose:
        print('[gogo] ゴーゴー %d 区間: %s' % (len(iv), ' '.join('%.2f-%.2f' % x for x in iv)))
    return iv
