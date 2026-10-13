import os
"""太鼓プレイ動画 → 音符オンセット列 (方式2: 列プロファイル + 右端リング境界)。
考え方:
  レーン帯(y=386±50)の各列で「音符色の画素数」を数える。音符は左(手前)の音符が上に描かれ、
  重なった音符でも手前音符の白リング+黒縁(約13px)で列プロファイルが 0 近くまで落ちる。
  → 連続した「色つき列」の区間(run)を切り、各 run の右端 = その音符の内円右端 なので
     中心 cx = 右端 - RIN。密連打で半分隠れた音符(三日月)でも右端は必ず見える。
  → 等速予測トラッカーで追い、判定枠 x=615 到達フレームを回帰で外挿。
判定枠付近(x<750)はヒット演出で汚れるので使わない。
"""
import cv2, numpy as np, sys, json

JUDGE_X = 615
X0, X1 = 750, 1910            # 使う列範囲(左はヒット演出、右は出現途中を除外)
Y0, Y1 = 336, 437             # レーン帯 (中心386 ± 50)
RIN_S, RIN_B = 40, 59         # 小/大 音符の内円半径(色つき部分)
# 連打の終わり(8)の点: 色つきの棒の右端から左へ ROLL_END_S(小連打)/ ROLL_END_B(大連打)。追跡は右端 − 2×RIN_S の点で行い(後段の尻尾の丸の位置は
# この点からの距離で決めてある)、書き出す直前に roll_end_point で置き換える。試験スイートと一括試験 317 曲の連打 1536 本で、右端 − 2×RIN_S を
# 終わりにしたときの本家とのずれが速さ(8〜30px/フレーム)・長さによらず一定だった(小 −18.5px 四分位 −20.5〜−16.9、大 +17.5px 四分位 +14〜+20)
ROLL_END_S, ROLL_END_B = 2 * RIN_S - 18.5, 2 * RIN_S + 17.5
R_B_OUT = 84                  # 大音符の外周(黒い輪郭まで)の半径(ringtrack.R_B と同じ)
TH_PRESENT = 18               # 列の色つき画素数がこれ以上で「音符あり」
GAP_MERGE = 3                 # これ以下の穴は同じ音符とみなす(目/口/圧縮ノイズ)
HMAX_MIN = 40                 # run 内の最大高さ(短すぎる=演出)
BIG_H = 86                    # 高さがこれ以上なら大音符 (小≤82, 大=119 が帯でクリップされ~100。画面フラッシュのフレームでは大が88程度まで落ちる)
BIG_W = 100                   # run 幅がこれ以上なら大音符 (小≈75, 大≈119。高さより安定)
W_MIN = 8
GAP_LONG = 45                 # 速度既知トラックが観測無しで生き続ける最大フレーム数(連打棒・速い音符に追い越されて隠れる対策)
LONG_W = 135                  # これより長い run は音符1個ではない(連打棒/風船袋)
SHORT_ROLL_W = 200            # 小連打の run がこれより狭ければ頭と尻尾の丸が重なる短い連打(終わりの追跡の点を右端 − 1 半径にする)

# HSV (OpenCV H:0-179)。ドンはゴーゴー背景と分けるため S/V に下限を置く。
# 背景は曲によって濃さが違う(エンジェルドリームのゴーゴーは H168 S159 V160 とかなり濃く、音符の間の背景が「大ドン」になった)。
# S/V の下限を上げると音符の縁が削れて時刻が荒れる(往復 sd 1.8→3.8ms)ので、色相の回り込み側を H168→174 に狭めて分ける。
# 本物のドンの画素の色相は 4 曲の実測で H0〜2(ゴーゴー中でも +10 まで)、H174 未満に落ちることは無い
# さらに 174→177 に狭めた(2026-09-22): Emma 裏の達人譜面の上のゴーゴーは背景が H173〜176 S200 V196 で、音符の隙間がドンになり、
# 風船の頭と袋もつながった。本物のドンの画素は 6 曲の実測で H174〜176 が 0.2% 以下、H177〜179 が最大 3%(regress/_don_hue.py)
COLORS = {
    # ドン本体の色相は通常 0-4 だがゴーゴー中は 5-10 まで上がるので H≤10。風船の袋(H 5-8)もドン扱いになるが、
    # 袋は「幅>90px なのに右端に白リングが無い」ので下の run 判定で弾く
    'don':     [((0, 150, 160), (10, 255, 255)), ((177, 150, 160), (180, 255, 255))],
    'ka':      [((82, 60, 110), (108, 255, 255))],
    'renda':   [((23, 120, 140), (34, 255, 255))],
    'balloon': [((12, 140, 140), (20, 255, 255))],    # 風船の頭(H 11-14)
}
CNAMES = list(COLORS)
# カッの芯の色相: 本物のカッの画素は 11 曲で 85〜98% が H82〜92(regress/_ka_hue.py)。玄人譜面のレーンの背景は H91〜101 で、明るく光るとカッの色の範囲に入る。
# 塊・縁の行・形の候補などの補助の判定ではこの芯の色相だけを使う(列の数える所はそのまま)
KA_CORE = (np.array((82, 60, 110)), np.array((92, 255, 255)))
KACORE = os.environ.get('V2T_KACORE', '1') != '0'
COLORS_KA_FULL = (np.array(COLORS['ka'][0][0]), np.array(COLORS['ka'][0][1]))
KA_VMED = 150.0          # カッの塊の明るさの中央値の下限(本物 179〜205 / 玄人のレーンの背景 110 前後)

# フレームごとの HSV の使い回し: candidates() の最初に帯の周り(行 Y0−RIM〜Y1+RIM、列 EM_X0〜X1)を 1 回だけ HSV にし、
# 各関数はその切り出しを使う(HSV は画素ごとの変換なので結果は同じ。1 フレームに 6 回変換していて CPU を食っていた)
_HSVC = [None, None, 0, 0]      # [フレーム, HSV, 行の起点, 列の起点]


def _hsv_cache_set(frame):
    ya = max(0, Y0 - RIM); yb = min(frame.shape[0], Y1 + RIM); xa = EM_X0; xb = X1
    _HSVC[0] = frame; _HSVC[1] = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV); _HSVC[2] = ya; _HSVC[3] = xa


def _hsv(sub):
    """sub(フレームの切り出し)の HSV。キャッシュの範囲内なら切り出して返す"""
    fr = _HSVC[0]
    if fr is not None and sub.ndim == 3 and sub.strides == fr.strides:
        off = sub.__array_interface__['data'][0] - fr.__array_interface__['data'][0]
        if off >= 0:
            r, rem = divmod(off, fr.strides[0]); c = rem // fr.strides[1]
            y0 = r - _HSVC[2]; x0 = c - _HSVC[3]; H_, W_ = _HSVC[1].shape[:2]
            if rem % fr.strides[1] == 0 and 0 <= y0 and y0 + sub.shape[0] <= H_ and 0 <= x0 and x0 + sub.shape[1] <= W_:
                return _HSVC[1][y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
    return cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)


def _colsum(mask):
    """inRange の 0/255 マスクの列ごとの画素数(= (mask > 0).sum(0))。cv2.reduce は SIMD で numpy の 5 倍ほど速い"""
    return (cv2.reduce(mask, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0] // 255).astype(np.int64)


def column_counts(bgr_band):
    hsv = _hsv(bgr_band)
    cnt = np.zeros((len(CNAMES), bgr_band.shape[1]), np.int32)
    for i, name in enumerate(CNAMES):
        m = None
        for lo, hi in COLORS[name]:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = mm if m is None else (m | mm)
        cnt[i] = _colsum(m)
    return cnt

def white_counts(bgr_band):
    hsv = _hsv(bgr_band)
    return _colsum(cv2.inRange(hsv, (0, 0, 170), (180, 50, 255)))

def _col_extent(band, a, b, color):
    """列 a〜b の中で、その色の一番上と一番下の行の差 + 1 の最大(画素の数ではなく端で測る: 大音符の顔は目・眉・口の黒が大きい)"""
    hs = _hsv(band[:, a:b + 1]); m = None
    for lo, hi in COLORS[color]:
        mm = cv2.inRange(hs, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
    m = m > 0; best = 0
    for j in range(m.shape[1]):
        rows = np.flatnonzero(m[:, j])
        if len(rows): best = max(best, int(rows[-1] - rows[0] + 1))
    return best


def _right_radius(band, a, b, color):
    """色の塊 [a, b] の右端の丸みから出した円の半径[px]。右端から深さ d(4〜20px)の列で、その色の一番上と一番下の行の差を弦の長さ h とし、
    r = (h²/4 + d²) / 2d の中央値。高さは画素の数ではなく上下の端で測る(大音符の顔は目・眉・口の黒が大きく、画素の数では半径が小さく出た)。
    帯の上下で切れる列と、塊の幅を超える深さは使わない"""
    w = b - a + 1
    if w < 9: return 0.0
    hs = _hsv(band[:, a:b + 1])
    m = None
    for lo, hi in COLORS[color]:
        mm = cv2.inRange(hs, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
    m = m > 0; H = m.shape[0]; rs = []
    for d in range(4, min(21, w - 2)):
        rows = np.flatnonzero(m[:, w - 1 - d])
        if len(rows) < 2 or rows[0] <= 0 or rows[-1] >= H - 1: continue
        h = float(rows[-1] - rows[0] + 1)
        rs.append((h * h / 4.0 + d * d) / (2.0 * d))
    return float(np.median(rs)) if len(rs) >= 5 else 0.0


def ring_big(wc, b):
    """右隣の白リングの量で音符の大小を判定(大音符はリングが大きい)。実測: 小 321-465, 大 599-607(8列の合計)"""
    return int(wc[b + 2:b + 10].sum()) >= 540

def ring_after(wc, b):
    """列 b の右隣に音符の白リングがあるか。音符の右端の直後は白が 30-90px 続く。
    連打棒の尻尾や風船の袋の先端は黒縁だけ(白 0)、小節線は帯いっぱい(≥95)なので区別できる。"""
    seg = wc[b + 2:b + 10]
    return len(seg) >= 6 and int(((seg >= 8) & (seg <= 95)).sum()) >= 5

def round_left(total, a, b):
    """run の左端が丸い頭か。実測: 頭は 22,26,35,36,48,48,52,53,60,60,68,68,68(大)/もっと低い(小) と緩やかに上がる。
    連打棒の左端(大連打では頭のリングの外側に帯の上下からはみ出して見え始める)は 18,19,26,...,71,76,101 と
    12列で帯の上限(101)に達する。頭は12列では上限に達しない。"""
    return b - a >= 20 and int(total[a]) <= 60 and int(total[a + 10:a + 17].max()) < 96

def head_big(total, a, b, w):
    """頭の大小。頭だけの run なら幅(小≈75 / 大≈119)。頭+棒が繋がった長い run なら棒の高さ
    (小連打の棒=92-96px / 大連打の棒=146px→帯でクリップされ101)。大音符の顔(眉)で高さは当てにならない。"""
    if w <= LONG_W:
        if w >= BIG_W and os.environ.get('V2T_SHORTROLL', '1') != '0':
            # 幅 100〜135 は「大きい頭」か「小さい頭 + 短い尻尾」(もぺもぺ 79.7s の 0.05 秒の連打: 幅 132 で大連打と誤判定)。頭の部分の高さで分ける(小≈80 / 大≈100)
            # 高さ 99 以上は大(帯で切れた大きい頭)、90 以下は小(小さい頭 80)。その間(小連打の棒 92〜96 / 動いてぼやけた大きい頭)は幅で:
            # 125 以下は大きい頭だけ(≈119)、それより広いのは小さい頭 + 棒(resume:story 裏 33.3s: 手前のカッに頭を隠されて幅 132・高さ 96)
            hh = float(total[a:a + min(w, 80)].max())
            if hh >= 99: return True
            if hh <= 90: return False
            return w <= 125
        return w >= BIG_W
    bar = total[a + 80:a + 220]
    return len(bar) >= 40 and float(np.median(bar)) >= 99

def runs_of(present, gap_merge):
    """True の連続区間 [a,b] のリスト。gap_merge 未満の穴は埋める。"""
    idx = np.flatnonzero(present)
    if len(idx) == 0: return []
    out = []
    a = b = idx[0]
    for i in idx[1:]:
        if i - b - 1 < gap_merge:
            b = i
        else:
            out.append((a, b)); a = b = i
    out.append((a, b))
    return out

def _ladder_heads(cnt, wc, a, b, hmax):
    """連打のはしご(resume:story: 127ms の連打を 22ms 間隔で 40 個並べる演出)の 2 本目以降の頭。
    帯が重なって 1 本の黄色い run になり、2 本目以降の頭は左半分が前の帯の丸い端に隠れて、白い縁が三日月状にのぞくだけ。
    帯の胴・帯の端の縁は黒く、白い縦線は出ない。白い縦線(白 ≥ 40 行)の両側が黄色なら後続の頭とみなす
    (ドン/カッが帯に重なるときは白線の片側が赤/水色になるので出ない)"""
    ycnt = cnt[CNAMES.index('renda')]; out = []
    # 左側(前の帯の丸い端)は、前の連打が叩かれている間は橙〜赤になる。黄+橙+赤で数える
    lcnt = ycnt + cnt[CNAMES.index('balloon')] + cnt[CNAMES.index('don')]
    for i in range(a + 2 * RIN_S + 10, b - 30):
        if wc[i] >= 40 and wc[i] >= wc[i - 1] and wc[i] > wc[i + 1] and wc[i] >= wc[max(0, i - 6):i + 7].max():
            lft = lcnt[max(a, i - 26):i - 6]; rgt = ycnt[i + 6:i + 26]
            if len(lft) and len(rgt) and lft.mean() >= 0.35 * H_BAND and rgt.mean() >= 0.5 * H_BAND:   # 左は前の帯の丸い端(黄 46〜49 行)、右は頭の顔(黄 80 行以上)
                out.append({'color': 'renda', 'cx': float(X0 + i + LADDER_K), 'big': False, 'w': 60, 'hmax': hmax, 'ladder': True})
    return out


LADDER_K = -60       # はしごの頭: のぞいている白い縁(三日月)から、本家の 5 の位置までの距離[px]。resume:story の 2 つのはしご(速さ 44 と 22px/frame)で最適値が −76 と −43 と違うので間を取った
H_BAND = 101         # 候補を探す帯の高さ(Y1 - Y0)

def _head_cx(cnt, wc, a, b, big):
    """連打・風船の頭の中心。ふつうは左端 + 半径。左にドン/カッが接して頭の左側が隠れているとき(vs.VIGVANGS 裏 67.57s: 55ms 前のカッが頭に重なり、
    左端 + 半径だと +22ms 遅れた)は、見えている右側の白い縁(頭と棒の境)の手前を頭の色の右端として、右端 − 半径にする(ドン・カッと同じ出し方)"""
    r = RIN_B if big else RIN_S
    cx = a + r
    if a >= 12:
        nb = cnt[:2, max(0, a - 12):a].sum(0)                  # すぐ左のドン/カッの色
        if nb.max() >= TH_PRESENT and os.environ.get('V2T_HEADRING', '1') != '0':
            k = next((j for j in range(a + 20, min(b, a + 2 * r + 20)) if wc[j] >= 40), None)
            if k is not None: cx = min(cx, k - 1 - r)
    return cx


def ring_candidates(frame, have):
    """白い縁の円から音符の中心を出す(円のハフ変換)。重なった音符は片側しか見えないことがあるが、縁は円の一部なので中心が決まる。
    カオスタイム おに 59.04s: 速さ 120 の遅いカッが、譜面で先にある速さ 160 のカッに右から追い越されて右側を覆われ、左側だけ見えていた
    (色の塊の「右端 − 半径」では拾えず、追跡が画面に入った直後で切れていた)。既存の候補(have)に同じ色で 20px 以内のものがある音符は足さない"""
    y0, y1 = Y0 - 20, Y1 + 20
    band = frame[y0:y1, X0:X1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    w = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    w = cv2.GaussianBlur(w, (5, 5), 1.5)
    cs = cv2.HoughCircles(w, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=100, param2=14, minRadius=36, maxRadius=68)
    out = []
    if cs is None: return out
    lane_c = (Y0 + Y1) / 2.0
    for x, y, r in cs[0]:
        if abs(y + y0 - lane_c) > 12: continue
        xi, yi = int(round(x)), int(round(y))
        inner = hsv[max(0, yi - 18):yi + 19, max(0, xi - 18):xi + 19].reshape(-1, 3)
        if not len(inner): continue
        don = ((((inner[:, 0] <= 10) | (inner[:, 0] >= 177)) & (inner[:, 1] >= 150) & (inner[:, 2] >= 160))).mean()
        ka = ((inner[:, 0] >= 82) & (inner[:, 0] <= 92) & (inner[:, 1] >= 60) & (inner[:, 2] >= 150)).mean()
        if max(don, ka) < 0.5: continue
        color = 'don' if don > ka else 'ka'
        cx = float(X0 + x); big = r > 54
        # 同じ色の既存の候補が 20px 以内にあれば、その位置を円の中心に置き換える(重なった音符の「右端 − 半径」は 10〜15px ずれる。円は縁の形から中心が決まる)
        near = [c for c in have if c['color'] == color and abs(c['cx'] - cx) < 20 and not c.get('ring_fixed')]
        if near:
            c = min(near, key=lambda c: abs(c['cx'] - cx)); c['cx'] = cx; c['ring_fixed'] = True; continue
        if any(c['color'] == color and abs(c['cx'] - cx) < 20 for c in out): continue
        out.append({'color': color, 'cx': cx, 'big': bool(big), 'w': int(2 * r), 'hmax': int(2 * r), 'ring': True})
    return out


def candidates(frame):
    """1フレーム → 音符候補 [{'color','cx','big','w','hmax'}] (cx はフレーム座標)"""
    _hsv_cache_set(frame)
    band = frame[Y0:Y1, X0:X1]
    cnt = column_counts(band)
    wc = white_counts(band)
    total = cnt.sum(0)
    present = total >= TH_PRESENT
    n = len(present)
    out = []
    runs = []
    cntc_ = [None]; wcc_ = [None]                  # 帯の中央の行だけの色・白の画素数(色の混ざった塊を分けるときだけ作る)
    for a, b in runs_of(present, GAP_MERGE):
        # 隣り合う 16 分音符は色領域が触れて 1 本の run(幅 ≈ 163 = 2 音符)になる。幅 135 超は連打棒扱いになり
        # 右端の音符しか出せず、左の音符は追跡を失う(ゴーゴー中の 103.42s のドンを落とした)。
        # 2 音符の境界には両方の白リングが重なる列(白 ≥ 35 行。音符の中は ≈ 17、連打棒の中は数行)があるので、そこで分割する
        w = b - a + 1
        if 130 <= w <= 260 and a > 1 and b < n - 2:
            inner = wc[a + 40:b - 40]
            cuts = []
            j = 0
            while j < len(inner):
                if inner[j] >= 35:
                    k = j
                    while k + 1 < len(inner) and inner[k + 1] >= 35: k += 1
                    cuts.append(a + 40 + (j + k) // 2); j = k + 1
                else: j += 1
            merged = []                                   # 30px 以内の山は同じ境界(リングの重なりが 2 山に割れることがある)
            for c in cuts:
                if merged and c - merged[-1][-1] <= 30: merged[-1].append(c)
                else: merged.append([c])
            cuts = [int(round(sum(g) / len(g))) for g in merged]
            if cuts:
                bounds = [a] + [c for c in cuts] + [b]
                subs = [(bounds[i] + (1 if i else 0), bounds[i + 1] - (1 if i + 1 < len(bounds) - 1 else 0)) for i in range(len(bounds) - 1)]
                # 両側ともドン/カッの run だけ分割する(連打の頭+棒、風船の頭+袋には掛けない)。
                # 左側の右端は「その色の画素が TH_PRESENT 以上ある最後の列」に詰める(リングの重なりの中心だと 5px ほど右に偏る)
                subs = [(a2, b2) for a2, b2 in subs if b2 - a2 + 1 >= 45]     # リングに挟まれた細い断片(裏の大音符の覗き)は捨てる
                ok = len(subs) >= 2
                if ok:
                    fixed = []
                    for si, (a2, b2) in enumerate(subs):
                        seg2 = cnt[:, a2:b2 + 1]; ci2 = int(np.argmax(seg2.sum(1)))
                        if CNAMES[ci2] not in ('don', 'ka'): ok = False; break
                        if si < len(subs) - 1:
                            cols = np.where(seg2[ci2] >= TH_PRESENT)[0]
                            if len(cols) and cols.max() + 1 >= 45: b2 = a2 + int(cols.max())
                        fixed.append((a2, b2))
                if ok:
                    runs.extend(fixed); continue
        # ドンとカッの色が混ざった塊: 小音符が大音符の上に重なると、小音符の白い縁の列にも大音符の顔が上下にはみ出して写り、塊がつながる
        # (検出テスト ファイナル 116.0s: 大カッの上の小ドンの 9px の赤い三日月が大カッの青とつながり、多い方のカッだけになった)。
        # 帯の中央の行(小音符の縁の高さ ±18 行)だけで見ると小音符の白い縁で途切れるので、そこで色の違う塊に分ける
        if os.environ.get('V2T_MIXSPLIT', '0') == '1':      # 既定は使わない(裏 Ka 21.57s: 大ドンの一部を小ドンとして切り出した)
            dsum_, ksum_ = int(cnt[0, a:b + 1].sum()), int(cnt[1, a:b + 1].sum())
            if min(dsum_, ksum_) >= 0.15 * (dsum_ + ksum_):
                if cntc_[0] is None:
                    hb_ = band.shape[0] // 2; cntc_[0] = column_counts(band[hb_ - 18:hb_ + 19])
                pc_ = cntc_[0][:2, a:b + 1].sum(0) >= 6
                cr_ = [(a + a3, a + b3) for a3, b3 in runs_of(pc_, GAP_MERGE)]
                if len(cr_) >= 2:
                    cuts_ = [(cr_[k][1] + cr_[k + 1][0]) // 2 for k in range(len(cr_) - 1)]
                    bnd_ = [a] + cuts_ + [b]
                    parts_ = [(bnd_[k] + (1 if k else 0), bnd_[k + 1]) for k in range(len(bnd_) - 1)]
                    cis_ = [int(np.argmax(cnt[:2, p0:p1 + 1].sum(1))) for p0, p1 in parts_ if p1 >= p0]
                    # 小音符が大音符の上に重なった形だけ: 片方の色の高さは小音符(BIG_H 未満)、もう片方は大音符(BIG_H 以上)。小音符どうしの
                    # 重なり(普通の曲の 16 分のドン・カッ)を分けると偽の候補ができた(ドンカマ2000 120.19s・裏 Ka 21.57s・笑える2000 102.06s)
                    hts_ = [_col_extent(band, p0, p1, CNAMES[ci_]) for ci_, (p0, p1) in zip(cis_, parts_)]   # 高さは上下の端で(顔の黒で画素の数は減る)
                    # 小音符の部分の右には、その小音符自身の白い縁が中央の行に見える(上に描かれているので隠れない)
                    if wcc_[0] is None: wcc_[0] = white_counts(band[band.shape[0] // 2 - 18:band.shape[0] // 2 + 19])
                    rim_ok_ = all(int((wcc_[0][ce_ + 1:ce_ + 8] >= 12).sum()) >= 3 for (a3_, ce_), h in zip(cr_, hts_) if h < BIG_H and ce_ + 8 <= len(wcc_[0]))   # 中央の行の色の右端のすぐ右
                    if (len(cis_) == len(parts_) and len(set(cis_)) >= 2
                            and any(h < BIG_H for h in hts_) and any(h >= BIG_H for h in hts_) and rim_ok_):
                        runs.extend(parts_); continue
        runs.append((a, b))
    # カッの芯の色相: 本物のカッの画素は 11 曲で 85〜98% が H82〜92(regress/_ka_hue.py)。玄人譜面のレーンの背景は H91〜101 で、叩いた光で明るくなると
    # カッの色の範囲(H82〜108・V110 以上)に入り、音符の隙間が細いカッになった(Emma 裏 玄人 余分 10)。塊の中のカッの画素のうち芯の色相が半分未満なら音符ではない
    ka_core = None
    if os.environ.get('V2T_KACORE', '1') != '0':
        hb_ = _hsv(band); kam_ = cv2.inRange(hb_, *COLORS_KA_FULL) > 0
        ka_core = (cv2.inRange(hb_, *KA_CORE) > 0).sum(0)
        vch_ = hb_[:, :, 2]
        def _ka_vmed(a_, b_):
            m_ = kam_[:, a_:b_ + 1]
            return float(np.median(vch_[:, a_:b_ + 1][m_])) if m_.any() else 255.0
    for a, b in runs:
        rb_ = False
        left_clipped = a <= 1                      # 左端に掛かっても右端が見えれば中心は出せる
        if ka_core is not None:
            ks_ = int(cnt[1, a:b + 1].sum())
            if ks_ > 0 and ks_ >= 0.6 * int(cnt[:, a:b + 1].sum()):
                if int(ka_core[a:b + 1].sum()) < 0.5 * ks_: continue
                # 明るさ: 本物のカッの画素の明るさの中央値は 11 曲で 179〜205(暗い動画でも 179)。玄人譜面のレーンの背景は明るくなっても 110 前後で、
                # カッの色の下限(V110)をわずかに越える(Emma 裏 玄人 114.44s)。塊のカッの画素の明るさの中央値が KA_VMED 未満なら音符ではない
                if _ka_vmed(a, b) < KA_VMED: continue
        w = b - a + 1
        if w < W_MIN: continue
        if b >= n - 2 and w > LONG_W:
            out.extend(_ladder_heads(cnt, wc, a, b, int(total[a:b + 1].max())))      # はしごの帯は画面の右端まで続く
        if b >= n - 2 or (w < 72 and b >= n - 4):
            # 右端に掛かる run は出現途中(右端が見えない)。部分的な run は右端の 4px 手前でも切れていることがある(レーン右端の縁)。ただし画面より長い連打棒は尻尾が入る前に
            # 頭が判定枠まで行ってしまうので、左端が黄/橙の丸い頭ならここで頭だけ出す
            if left_clipped or w < 95: continue
            hs = cnt[:, a:a + 60].sum(1); hci = int(np.argmax(hs)); hcolor = CNAMES[hci]
            if hcolor in ('renda', 'balloon') and hs[hci] >= 0.6 * hs.sum() and round_left(total, a, b):
                hbig = head_big(total, a, b, w)
                out.append({'color': hcolor, 'cx': float(X0 + a + (RIN_B if hbig else RIN_S)), 'big': bool(hbig), 'w': int(w), 'hmax': int(total[a:b + 1].max())})
            continue
        seg = cnt[:, a:b + 1]
        hmax = int(total[a:b + 1].max())
        if hmax < HMAX_MIN: continue
        sums = seg.sum(1)
        ci = int(np.argmax(sums)); color = CNAMES[ci]
        if w > 90:
            # 1個の音符より広い run(連打棒/風船袋と重なった音符、くっついた2音符など)。
            # 右端40列がドン/カ色で、その右に白リングが続けば「右端は音符の内円右端」なので音符として拾う
            # (風船の袋の裏のドン、棒の尻尾と重なった大音符がこれで取れる)
            rs = cnt[:, max(a, b - 39):b + 1].sum(1); rci = int(np.argmax(rs)); rcolor = CNAMES[rci]
            if rcolor in ('don', 'ka') and rs[rci] >= 0.85 * rs.sum() and ring_after(wc, b):   # 右端40列がほぼ単色のときだけ(混色=右端の帰属が曖昧)
                rbig = ring_big(wc, b)          # 棒と繋がった run では高さが棒のものになるので、リングの大きさで大小を決める
                out.append({'color': rcolor, 'cx': float(X0 + b - (RIN_B if rbig else RIN_S)), 'big': bool(rbig), 'w': int(w), 'hmax': hmax})
        if sums[ci] < 0.6 * sums.sum():             # 色が混ざりすぎ → 演出
            # くす玉(黄〜橙の大きな丸い顔、幅 ≈ 148、尻尾なし)は色相 18〜25 で連打(23〜)と風船(〜20)の色が半々に混ざる
            # (Bai Bai FIGHT! 46.82s: 連打 55% 風船 45% で捨てていた)。中心は run の真ん中
            rb = sums[CNAMES.index('renda')] + sums[CNAMES.index('balloon')]
            if (rb >= 0.9 * sums.sum() and 120 <= w <= 175 and hmax >= 90 and not left_clipped and round_left(total, a, b)
                    and os.environ.get('V2T_KUSU', '1') != '0'):
                out.append({'color': 'balloon', 'cx': float(X0 + (a + b) / 2.0), 'big': True, 'w': int(w), 'hmax': hmax, 'kusu': True})
            continue
        big = (w >= BIG_W and w <= LONG_W) or hmax >= BIG_H
        if color in ('renda', 'balloon') or w > LONG_W:
            # 連打の頭の大小は幅(頭だけ)か棒の高さ(頭+棒)で決める
            big = head_big(total, a, b, w)
        if w > LONG_W: out.extend(_ladder_heads(cnt, wc, a, b, hmax))
        if w > LONG_W:
            # 大音符(内径119)より長い run = 連打の棒 or 風船の袋。
            # 右端(棒の終端の丸/袋の先端)は色に関係なく 'roll_end' として追跡 → 連打終了(8)の時刻になる。
            # 追う点は右端から 2*RIN_S(譜面の終わりの点は書き出す直前に roll_end_point で置き換える。上の ROLL_END_S の注)
            # (叩かれると棒が赤くなるので色で分けない。頭が判定枠を過ぎて左が切れていても終端は追う)
            # ただし短い小連打(幅 SHORT_ROLL_W 未満: 頭と尻尾の丸が重なる)は尻尾の丸の中心(右端 − 1 半径)。2 半径だと、もぺもぺ 79.4s の 0.05 秒の
            # 連打の終わりが頭の 10ms 後に出て、譜面側で捨てられていた(追う点をそろえると同 79.8〜80.6s の終わりが崩れた)。
            # 大連打は頭と棒が別の塊になり、棒だけの短い塊(帯 120〜250px の大連打・判定枠の近くで左が切れた棒)にも掛かって終わりが +58px 遅れたので、小連打だけ
            out.append({'color': 'roll_end', 'cx': float(X0 + b - (RIN_S if (w < SHORT_ROLL_W and not big and os.environ.get('V2T_SHORTROLL', '1') != '0') else 2 * RIN_S)), 'big': False, 'w': int(w), 'hmax': hmax})
            # 頭は左端の丸い部分。左端60列の色で種類を決める。赤/水色なら叩かれ中の棒 → 頭は既に拾い済み。
            # 大連打は頭と棒が別 run になるので、棒だけの run(左端が垂直)は頭を出さない
            if left_clipped: continue
            hs = cnt[:, a:a + 60].sum(1); ci = int(np.argmax(hs)); color = CNAMES[ci]
            if color not in ('renda', 'balloon') or not round_left(total, a, b): continue
            cx = _head_cx(cnt, wc, a, b, big)
        elif color in ('renda', 'balloon'):
            if left_clipped: continue
            if color == 'balloon':
                # 風船は「白リング付きの丸い頭」だけ採る。袋だけの run(幅100-130, リング無し)は音符ではない
                if w > 95 or not ring_after(wc, b): continue
                cx = b - RIN_S
            else:
                cx = _head_cx(cnt, wc, a, b, big) if w > 95 else b - RIN_S
                if w > 95 and not big and os.environ.get('V2T_SHORTROLL', '1') != '0':
                    # 小さい頭 + 短い尻尾(幅 95〜135): 尻尾の丸の中心を終わりの候補にする(もぺもぺ 79.7s の 0.05 秒の連打は終わりの候補が無く、
                    # 次の頭の手前に補った終わりが 80ms 遅かった)
                    out.append({'color': 'roll_end', 'cx': float(X0 + b - RIN_S), 'big': False, 'w': int(w), 'hmax': hmax})
        else:
            if left_clipped:
                # 棒の残り(終端側)の可能性があるので終端候補も出す(音符ならトラックにならず消える)
                out.append({'color': 'roll_end', 'cx': float(X0 + b - 2 * RIN_S), 'big': False, 'w': int(w), 'hmax': hmax})
            # 1個の音符より幅が広い赤/水色 = 風船の袋(橙赤, 幅100-130)や連打棒の断片 → 音符ではない。
            # 大音符(幅≈119)と区別するため、幅>90 の run は右端に白リングがあるときだけ音符と認める
            if left_clipped and w > 90:
                big = ring_big(wc, b)           # 左が切れていると幅で大小を決められない(棒の残り+小音符でも幅>100になる)
            if w > (125 if big else 90): continue
            if w > 90 and not ring_after(wc, b): continue
            if (not left_clipped and 12 <= w < 72 and os.environ.get('V2T_RINGBIG', '1') != '0' and b + 10 <= n
                    and (big or (os.environ.get('V2T_RB_NB', '1') != '0' and ring_big(wc, b)))      # 小を大に変えるのは右の白い縁も大のときだけ(丸みだけだと
                    # 普通の曲の小音符を 6 曲で大にした: もぺもぺ・チャーリー・ドンカマ など)
                    and ring_after(wc, b) and (RIN_S + RIN_B) / 2.0 < _right_radius(band, a, b, color) < 1.25 * RIN_B):
                # 左を上の音符に隠された大音符の三日月: 見えている高さが低く小音符と判定し、中心を「右端 − 小の半径」として 19px 遅れた
                # (検出テスト ファイナル 114.2s: SCROLL 0.4 の大音符の 16 分は 35px 間隔で、右端の 35px しか見えない)。右側は隠れないので、
                # 右端の丸み(右端から深さ d の所の色の高さ h から半径 (h²/4 + d²)/2d)で大小を決める。右の白い縁の量では、すぐ後ろに重なった
                # 大音符の縁を数えて小音符を大にした(同 52.5s: 大・小が交互の 48 分)。高さが帯いっぱいで大と判定された三日月も、この印(rb)があれば
                # 下の「幅 60 未満の大音符は大ドンの陰の三日月」の除外に掛けない(同 74.8s: 62px 間隔の大音符で、正しい中心の候補を捨て 12px ずれた候補が残った)。
                # 半径の上限(1.25×RIN_B)は、右側を上の音符に隠された陰の三日月(右端が隠した円の弧で、丸みが大きく出る)を除くため
                big = True; rb_ = True
            face_r = int(total[b + 8:b + 21].max()) >= TH_PRESENT if b + 21 <= n else False
            face_l = int(total[max(0, a - 20):max(0, a - 7)].max()) >= TH_PRESENT if a - 7 > 0 else False
            if not big and 12 <= w < 72 and not left_clipped and face_r and not face_l:
                # 右のリング 1 本ぶん先に他の音符の顔がある(左には無い) = 右隣の上に描かれた音符に右側を隠された三日月/部分顔。
                # 中心は左端 + 半径(右端 − 半径では隠された分だけ左に偏る: 彁 裏で速いドンに追い越されるカッ)。
                # 32 分の流れの三日月(左隣に隠される)は左に顔があるので従来どおり
                cx = a + RIN_S
            else:
                cx = b - (RIN_B if big else RIN_S)
        out.append({'color': color, 'cx': float(X0 + cx), 'big': bool(big), 'w': int(w), 'hmax': hmax})
        if rb_: out[-1]['rb'] = True
    # 同色で 25px 以内の候補は同じ音符(幅広ルールと通常ルールの重複など) → 幅の広い方(=情報の多い方)を残す
    out.sort(key=lambda c: (c['color'], c['cx']))
    dedup = []
    for c in out:
        if dedup and dedup[-1]['color'] == c['color'] and abs(dedup[-1]['cx'] - c['cx']) < 25:
            if c['w'] > dedup[-1]['w']: dedup[-1] = c
            continue
        dedup.append(c)
    hsv_b = _hsv(band)
    outer = {}; masks = {}
    for name in ('don', 'ka'):
        m = None
        for lo, hi in COLORS[name]:
            mm = cv2.inRange(hsv_b, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
        if name == 'ka' and KACORE: m = m & cv2.inRange(hsv_b, *KA_CORE)
        m = m > 0; masks[name] = m
        outer[name] = (m[0:13].sum(0), m[88:101].sum(0))
    kept = []
    for c in dedup:
        if c['big'] and c['color'] in ('don', 'ka') and not c.get('rb'):      # 縁で大と決めた三日月(rb)は縁が証拠
            if c['w'] < 60 and c['cx'] - X0 > 2: continue                       # 幅 60 未満の「大音符」= 大ドンの陰の三日月(中心が違う)
            if not (110 <= c['w'] <= 128):                                    # 素の大音符(幅 110-128)以外の大音符候補は縁の行で中心を検査(三日月・合体 run)
                xi = int(c['cx'] - X0); top, bot = outer[c['color']]
                lo, hi = max(0, xi - 6), min(len(top), xi + 7)
                if lo >= hi or not (top[lo:hi].max() >= 4 and bot[lo:hi].max() >= 4): continue
        kept.append(c)
    dedup = kept
    # 色別の連結成分からの候補(上に描かれた小音符の顔は自分の白リングで切り離されるので、重なっていても別々に取れる)
    heads = [c['cx'] for c in dedup if c['color'] in ('renda', 'balloon', 'roll_end')]
    for b in blob_candidates(masks):          # masks のカッは芯の色相だけ(上)
        if b['color'] == 'ka' and ka_core is not None:
            xa_ = int(b['cx'] - X0 - b['w'] / 2); xb_ = int(b['cx'] - X0 + b['w'] / 2)
            if 0 <= xa_ < xb_ and _ka_vmed(xa_, min(xb_, band.shape[1] - 1)) < KA_VMED: continue
        near = [c for c in dedup if c['color'] == b['color'] and abs(c['cx'] - b['cx']) < 30]
        if b['big'] and (b['w'] > 125 or any(abs(h - b['cx']) < 160 for h in heads)): continue   # 風船の袋・連打棒(赤みが don マスクに掛かる)は大音符ではない
        if not near: dedup.append(b); continue
        if not b['big'] and any(b.get('edge', (False, False))):
            # 片側を隠された小音符: 色ごとの成分の縁から決めた中心の方が確か(run は裏の大音符の面と合体して端がずれる)
            for c in near:
                if not c['big'] and 8 <= abs(c['cx'] - b['cx']) < 60 and not c.get('blob'):
                    dedup.remove(c); dedup.append(b); break
    dedup.extend(hidden_big_candidates(frame, dedup))
    dedup.extend(ring_pair_candidates(band, cnt, wc, present, dedup))
    dedup.extend(emerge_candidates(frame))
    if os.environ.get('V2T_RING', '0') == '1':      # 試験: 既定では使わない(カオスタイムで余分が出る)
        dedup.extend(ring_candidates(frame, [c for c in dedup if c['color'] in ('don', 'ka')]))
    return dedup

EM_X0, EM_X1 = 535, 790     # 判定枠(615)の大音符(半径 59)も入るように。帯の左端 X0 の手前まで。上の音符が判定枠で叩かれて消えた後に、裏に隠れていた音符が数フレームだけ見える

def emerge_candidates(frame):
    """帯(x ≥ 750)に入る前に別の音符の裏に完全に隠れていた音符は、上の音符が判定枠で叩かれて消えた後、
    判定枠の右 (x 630-760) に丸ごと数フレーム見える(彁 裏: 速いドンの流れの裏の遅いカッ)。
    顔の行(y349-424)で色ごとに連結成分を取り、丸い顔 1 個ぶん(幅 66-80・高さ 70-78)の成分だけ候補にする。
    叩き演出(黄/橙の光、上へ飛ぶ音符=右上へ動く)は形か動きで外れる。追跡側では通常の音符に付け足さず、
    同色の音符が無い時刻にだけ音符を足す(emerge フラグ)。"""
    reg = frame[Y0 + 13:Y0 + 88, EM_X0:EM_X1]
    hsv = _hsv(reg)
    out = []
    for name in ('don', 'ka'):
        m = None
        for lo, hi in COLORS[name]:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
        if name == 'ka' and os.environ.get('V2T_KACORE', '1') != '0':
            m = m & cv2.inRange(hsv, np.array((82, 60, 110)), np.array((92, 255, 255)))    # カッの芯の色相(玄人譜面の青いレーンの判定枠の輪を拾わない)
        n, lab, st, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        W = reg.shape[1]
        for j in range(1, n):
            x, y, w, h, a = [int(v) for v in st[j]]
            big = 100 <= w <= 125 and h >= 70 and a >= 4500      # 叩く直前の「怒り顔」(口が大きい)でも拾えるように面積は緩め
            if not big and not (66 <= w <= 80 and 70 <= h <= 78 and a >= 3600): continue
            if x <= 1 or x + w >= W - 2: continue
            cx = EM_X0 + float(cen[j][0])
            if cx >= X0 + 10: continue                     # 帯の中の音符は通常の候補で出る
            # eyes: 色の塊の中の黒い穴(穴の画素の明るさの平均 < 100)の数。顔には黒い目がある。色づいた判定枠の円(両玄ノ舞 25.38s・Emma 裏 23.54s)の穴は白っぽい輪
            comp_ = (lab[y:y + h, x:x + w] == j).astype(np.uint8)
            cnts_, hier_ = cv2.findContours(comp_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            eyes_ = 0
            if hier_ is not None:
                vch_ = hsv[y:y + h, x:x + w, 2]
                for c_, hh_ in zip(cnts_, hier_[0]):
                    if hh_[3] < 0 or cv2.contourArea(c_) < 15: continue
                    mk_ = np.zeros_like(comp_); cv2.drawContours(mk_, [c_], -1, 1, -1)
                    hv_ = vch_[(mk_ > 0) & (comp_ == 0)]
                    if len(hv_) and float(hv_.mean()) < 100: eyes_ += 1       # 目 58〜69・口 85〜90、判定枠の円の輪 123〜140
            out.append({'color': name, 'cx': cx, 'big': big, 'w': int(w), 'hmax': int(h), 'emerge': True, 'eyes': eyes_})
    return out

def blob_candidates(masks):
    """帯の内側(小音符の顔の行 13-87)で色ごとに連結成分を取り、丸い顔 1 個ぶんの大きさなら候補にする。
    幅 60-92・高さ 70 以上 = 小音符、幅 100 以上 = 大音符。帯の端に掛かるものは出さない。
    小音符の中心は「顔の縁の形」で決める: 円の縁は端の列で高さが小さく(先細り)、他の音符に切られた縁は端の列でも高い。
    左が先細りで右が切られている(右隣の上の音符に隠された)なら 左端+半径、逆なら 右端−半径、両方先細り(素の顔)なら重心。
    両方切られていれば出さない。幅 30-59 の三日月も同じ規則で出す(彁 裏: 裏の大ドンの上のカッが速いドンに追い越される)"""
    out = []
    for name in ('don', 'ka'):
        inner = masks[name][13:88].astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(inner, 8)
        W = inner.shape[1]
        for j in range(1, n):
            x, y, w, h, a = [int(v) for v in st[j]]
            if h < 70 or a < 2500 or x <= 1 or x + w >= W - 2: continue
            if w >= 100:
                out.append({'color': name, 'cx': float(X0 + cen[j][0]), 'big': True, 'w': int(w), 'hmax': int(h), 'blob': True}); continue
            if not (30 <= w <= 92): continue
            comp = (lab[:, x:x + w] == j).sum(0)
            hl = float(comp[:3].mean()); hr = float(comp[-3:].mean())
            cut_l = hl >= 45; cut_r = hr >= 45          # 円の縁の端 3 列は高さ ≈ 17-30、切られた縁は幅 60 の三日月でも ≈ 67
            if cut_l and cut_r: continue
            if not cut_l and not cut_r:
                if w < 60: continue
                cx = float(cen[j][0]) if a >= 3600 else float(x + w - 38)
            elif cut_r: cx = float(x + 38)                    # 右を隠された(左の縁が本物)
            else: cx = float(x + w - 38)                      # 左を隠された(右の縁が本物)
            out.append({'color': name, 'cx': float(X0 + cx), 'big': False, 'w': int(w), 'hmax': int(h), 'blob': True, 'edge': (cut_l, cut_r)})
    return out

def ring_pair_candidates(band, cnt, wc, present, normal):
    """大音符(幅 ≈118)の上に小音符が重なると run が 1 本(幅 130-200)になり、右端の音符しか出ない(彁 裏の低速大ドン+16 分)。
    小音符は自分の白リング(直径 ≈ 88)を持つので、run の内側に 70-90px 離れた白の山が 2 つあればその中点が小音符の中心。
    色は中点付近の中央行(小音符が上に描かれる)で決める。近くに同色の通常候補があれば出さない"""
    out = []
    n = len(present)
    for a, b in runs_of(present, GAP_MERGE):
        w = b - a + 1
        if w < 110 or a <= 1 or b >= n - 2: continue
        inner = wc[a + 6:b - 6]
        peaks = []; j = 0
        while j < len(inner):
            if inner[j] >= 30:
                k = j
                while k + 1 < len(inner) and inner[k + 1] >= 30: k += 1
                peaks.append(a + 6 + (j + k) // 2); j = k + 1
            else: j += 1
        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            d = p2 - p1
            if not (70 <= d <= 92): continue
            cx = (p1 + p2) // 2
            seg = cnt[:, max(a, cx - 25):cx + 26]
            sums = seg.sum(1); ci = int(np.argmax(sums)); color = CNAMES[ci]
            if color not in ('don', 'ka') or sums[ci] < 0.6 * sums.sum(): continue
            if any(c['color'] == color and not c['big'] and abs(c['cx'] - (X0 + cx)) < 30 for c in normal): continue
            out.append({'color': color, 'cx': float(X0 + cx), 'big': False, 'w': int(d), 'hmax': int(cnt.sum(0)[cx]), 'ringpair': True})
    return out

RIM = 10   # 大音符の面(y326-446)のうち帯(y336-437)の外にある行数。小音符(リング込みで y339-433)はこの行に掛からない

def hidden_big_candidates(frame, normal):
    """遅い大音符が速い小音符の裏に隠れる仕掛け(銀の黎明か…裏 103.55s の大カッ、彁 裏の低速大ドン)。
    大音符の面(y326-446)は帯(y336-437)の上下に 10 行はみ出し、その行には小音符(リング込みで y339-433)が掛からない。
    上下の縁 10 行それぞれで 2 行以上に色がある列の連結(面の円の弦、幅 ≈ 55px)を大音符候補にする。中心は弦の中央(占有に依らない)。
    連打・風船の頭の近く(棒・袋が同じ色相に入る)は出さない。"""
    top = frame[Y0 - RIM:Y0, X0:X1]; bot = frame[Y1:Y1 + RIM, X0:X1]
    hsv_t = _hsv(top); hsv_b = _hsv(bot)
    heads = [c['cx'] for c in normal if c['color'] in ('renda', 'balloon')]
    out = []; chords_ = {}
    for name in ('don', 'ka'):
        mt = None; mb = None
        for lo, hi in COLORS[name]:
            mt_ = cv2.inRange(hsv_t, np.array(lo), np.array(hi)); mt = mt_ if mt is None else (mt | mt_)
            mb_ = cv2.inRange(hsv_b, np.array(lo), np.array(hi)); mb = mb_ if mb is None else (mb | mb_)
        if name == 'ka' and KACORE:
            mt = mt & cv2.inRange(hsv_t, *KA_CORE); mb = mb & cv2.inRange(hsv_b, *KA_CORE)    # 縁の行はレーンの背景が見えるので芯の色相だけ
        ct = _colsum(mt); cb = _colsum(mb)
        present = (ct >= 2) & (cb >= 2)
        chords_[name] = list(runs_of(present, 3))
    for name in ('don', 'ka'):
        for a, b in chords_[name]:
            w = b - a + 1
            if w < 30 or w > 75 or a <= 1 or b >= X1 - X0 - 2: continue
            cx = X0 + (a + b) / 2.0
            # 左を上の大音符に隠された弦(本来の幅 ≈ 55px より狭い)は、右端が本当の縁なので中心は右端 − 半幅(27.5)
            # (検出テスト ファイナル 74.8s: 62px 間隔の大音符で、弦の中央が 12px 右にずれた)。覆っているかは形で決める: 縁の行(中心から約 55px 上下)では
            # 大音符の外周(R_B = 84)は中心から ±63.5px、顔の色の弦は ±27.5px なので、左の弦の右端から 63.5 − 27.5 + 3 = 39px 以内に始まる弦はその大音符に覆われている
            if os.environ.get('V2T_HBCUT', '1') != '0' and w < 45 and any(0 < a - b2 <= 39 for n2 in ('don', 'ka') for a2, b2 in chords_[n2] if b2 < a):
                cx = X0 + b - 27.5
            if any(abs(h - cx) < 200 for h in heads): continue
            out.append({'color': name, 'cx': float(cx), 'big': True, 'w': int(w), 'hmax': 101, 'hidden': True})
    return out

def hidden_big_candidates_inner(band, normal):
    """遅い大音符が速い小音符の裏に隠れる仕掛け(銀の黎明か…裏 103.55s の大カッ: SCROLL 0.5 で 16 分の流れの裏)。
    大音符の面(y326-446)は小音符(y349-423)より上下に 22px はみ出すので、帯の上下の縁(各 13 行)だけを見ると
    小音符に隠れていても大音符の色が上下両方に残る。上下とも 4 行以上ある幅 20-70px の連結を大音符候補にする
    (面は円なので縁の行に掛かる幅は ≈ 38px、中心は連結の中央)。通常の候補に同色の大音符が近くにあれば重複なので出さない。
    叩かれた連打棒(赤、棒の高さが縁に掛かる)は長さ > 70 で除外、判定枠の叩き演出は帯(x≥750)の外。"""
    hsv = _hsv(band)
    out = []
    for name in ('don', 'ka'):
        m = None
        for lo, hi in COLORS[name]:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi)); m = mm if m is None else (m | mm)
        if name == 'ka' and KACORE: m = m & cv2.inRange(hsv, *KA_CORE)
        m = m > 0
        top = m[0:13].sum(0); bot = m[88:101].sum(0)
        present = (top >= 4) & (bot >= 4)
        for a, b in runs_of(present, 3):
            w = b - a + 1
            if w < 20 or w > 70 or a <= 1 or b >= len(present) - 2: continue      # 実測: 縁の行に掛かる幅は ≈ 38px(面の直径 ≈ 95)
            cx = X0 + (a + b) / 2.0
            if any(c['color'] == name and c['big'] and abs(c['cx'] - cx) < 40 for c in normal): continue
            out.append({'color': name, 'cx': float(cx), 'big': True, 'w': int(w), 'hmax': 101, 'hidden': True})
    return out

TRACK_DEBUG = None   # DETECT_DEBUG_TRACKS=1 で track() が追跡の内部を残す(可視化用)

BAR_STRIPS = ((296, 322), (452, 478))   # 音符(大音符でも y326-446)が掛からないレーン上下の帯
BAR_X0 = 800                             # ヒット演出(x<790)を避ける
BAR_MARGIN = 50                          # 局所背景よりこれだけ明るい細い縦線を小節線とみなす

def bar_candidates(frame):
    """1フレーム → 小節線の x 位置リスト。上下の帯の各行で「局所背景より明るい」画素を数え、
    帯の行の 35% 以上で明るく、幅が細い(<=6px)列を小節線とみなす。
    (行平均だと大連打の棒(y313-459)が帯に掛かって線が薄まり見失う。行ごとの判定なら棒に掛からない行で拾える)"""
    ya = min(y0 for y0, _ in BAR_STRIPS); yb = max(y1 for _, y1 in BAR_STRIPS)
    g = cv2.cvtColor(frame[ya:yb, BAR_X0:X1], cv2.COLOR_BGR2GRAY)     # 使う行だけ変換(全面 1080p を毎フレーム変換すると並列でメモリ帯域を食い合う)
    rows = np.vstack([g[y0 - ya:y1 - ya] for y0, y1 in BAR_STRIPS]).astype(np.uint8)
    bg = cv2.blur(cv2.medianBlur(rows, 5), (61, 1)).astype(np.int16)
    hi = (rows.astype(np.int16) - bg) > BAR_MARGIN
    cnt = hi.sum(0)
    present = cnt >= 0.35 * rows.shape[0]
    out = []
    for a, b in runs_of(present, 1):
        w = b - a + 1
        if w <= 6 and a > 0 and b < len(present) - 1:
            out.append({'color': 'bar', 'cx': float(BAR_X0 + (a + b) / 2.0), 'big': False, 'w': int(w), 'hmax': 0})
    return out

# ゲームの描画間隔の揺れ(フレーム)。速い音符の位置は 速さ × これだけ揺れる(HANDOFF 罠 50: 15.5/17.5/21ms)。ringtrack の確かめ(0.15 × 速さ)と同じ。
# 追跡の段階の定数なので、候補のキャッシュの鍵(大文字で始まる定数)に入れないよう下線で始める
_JITTER_FR = 0.15


def _refine_fit(fs, xs, ws, big, a, b):
    """ドン/カッの直線の当てはめを直す。戻り値 (a, b, 理由) か None(直さない)。
    ・大音符で幅が中央値の 3/4 未満の点(判定枠で消えていく所。小音符の半径で中心を出すので 15px ずれる)は使わない(VERTeX 91.67s: +6ms)
    ・点が「流れ 1 フレームぶん(0.7〜1.3v)離れた 2 本の平行線」に割れていたら先を行く線だけ使う。
      録画が古い絵を出したフレームの点は 1 フレーム遅れた位置にしか来ない(先に出ることは無い)ので、先の線が本物
      (VERTeX 83.95s のドン: 残差 +7/−9px の 2 本で、両方に当てると +10ms)"""
    keep = np.ones(len(fs), bool); why = []
    if big and len(ws) == len(fs) and np.median(ws) > 0:
        k2 = ws >= 0.75 * np.median(ws)
        if (~k2).any() and k2.sum() >= 8: keep = k2; why.append('narrow')
    res = xs - (a * fs + b)
    if keep.sum() >= 12:
        a_k, b_k = np.polyfit(fs[keep], xs[keep], 1); r = xs - (a_k * fs + b_k); rk = r[keep]
        # 既定では使わない(V2T_LEAD=1 で有効): 試験スイートで 290 音符に掛かり 74 音符で逆に悪化(段位 祭 sd 2.1→3.1、Xa 120fps 3.8→4.3)。
        # VERTeX の古い絵はゲームの時計(prep60.game_clock)で直るので不要になった
        if rk.std() > 3.0 and os.environ.get('V2T_LEAD', '0') == '1':
            v = abs(a_k); lo, hi = np.percentile(rk, 15), np.percentile(rk, 85)
            if 0.7 * v <= hi - lo <= 1.3 * v:
                mid = (lo + hi) / 2; lead = keep & (r < mid); lag = keep & (r >= mid)
                if lead.sum() >= 8 and lag.sum() >= 4 and xs[lead].max() - xs[lead].min() >= 300 and r[lead].std() < 3.0:
                    keep = lead; why.append('lead')
    if not why or keep.sum() < 8: return None
    a2, b2 = np.polyfit(fs[keep], xs[keep], 1)
    if a2 > -4: return None
    return a2, b2, '+'.join(why), keep




_SIGHT_PX = 8        # 判定枠での目撃とみなす、判定枠の中心からの位置の差(px)。速い音符(45px/フレーム)でも 1 枚は入る幅
_MIN_PTS_END = 5      # 頭の速さで測り直した連打の終わりに要る点の数(速い連打の終わりの 3 点より多く: 普通の速さなら丸は何フレームも写る)

def progress(stage, done=None, total=None, what=''):
    """画面(web)に進み具合を出す行「[進み] 段階: いくつ/全部 何を」。画面から動かしたとき(V2T_PROGRESS=1)だけ"""
    if os.environ.get('V2T_PROGRESS') != '1': return
    print('[進み] %s: %s%s' % (stage, ('%d/%d ' % (done, total)) if total else '', what), flush=True)


def cpu_budget():
    """使う CPU の本数。V2T_CPU(数 / all)。既定は 8(全スレッドを使うと速いがファンがうるさい。全 32 スレッドの 1/4 でも 1 曲 1 分以内)。
    試験(regress/suite.py)は all"""
    v = os.environ.get('V2T_CPU', '8')
    n = os.cpu_count() or 4
    return max(1, n - 2) if v == 'all' else max(1, min(n, int(v)))

def roll_end_point(onsets, fps):
    """連打の終わりを、追跡に使った点(棒の右端 − 2×RIN_S)から譜面の終わり(8)の点(右端 − ROLL_END_S / ROLL_END_B)の時刻に置き換える。
    書き出す直前に 1 回だけ呼ぶ(後段はどれも追跡の点を前提にしている)。大小は直前の連打の頭で決める。風船の終わりは置き換えない(袋は割れるまで
    判定枠で止まり、終わりは打数から見積もる)。戻り値: 置き換えた数"""
    # 短い小連打(幅 SHORT_ROLL_W 未満)は右端 − 1 半径で追っている。頭の中心は run の左端 + RIN_S なので、その終わりは頭から SHORT_ROLL_W − 2×RIN_S
    # (120px)未満にしか出ない。120px 以上なら右端 − 2 半径で追った連打なので置き換える。それより短い小連打は、どちらの点で追ったか決められないので
    # そのまま(一括試験で帯 70〜100px の短い小連打は +19px。数が少ない)
    n = 0; head = None
    for o in sorted(onsets, key=lambda o: o['t']):
        if o['color'] in ('renda', 'balloon'): head = o
        elif o['color'] == 'roll_end':
            v = abs(o.get('vel') or (head or {}).get('vel') or 0)
            if head is not None and head['color'] == 'renda' and v > 0:
                big = bool(head.get('big'))
                if big or (o['t'] - head['t']) * v * fps >= SHORT_ROLL_W - 2 * RIN_S:
                    d_ = 2 * RIN_S - (ROLL_END_B if big else ROLL_END_S)     # 追跡の点より右(遅く着く)なら正
                    o['t'] = round(o['t'] + d_ / (v * fps), 4); n += 1
            head = None
    return n


def _endcap(merged, fps, dbg=False, mark_only=False):
    """連打の終わりの丸(ドンの色に見える)を捨て、終わりの丸の位置に右端から見えていたドンに capsus の印を付ける(下の注)。戻り値: merged"""
    if os.environ.get('V2T_ENDCAP', '1') != '0':
        ends_o = [q for q in merged if q['color'] == 'roll_end' and q.get('vel')]
        # 丸の中心は連打の終わりの基準点(棒の右端 − 半径 2 つ)から半径 1 つ(RIN_S)後ろ。丸は叩かれて赤くなってからドンに見えるので、ドンの色で見え始めるのは
        # 画面の途中から(右端から見えていた本物のドンとはここで分かれる)。位置の許容は丸の半径の半分 + 両方の残差の 2 倍
        # (VERTeX 17.09s: 残差 13.5px で +28px。彼は誰時の誘惑 75.75s・129.09s: +41px で、以前は丸の中心を終わりの位置として比べて見逃した)
        def _late_start(o):
            xr_ = o.get('xr')
            return bool(o.get('emerged') or o.get('rescued') or (xr_ and xr_[0] < X1 - 2 * RIN_S))
        # 終わりの位置そのもの(差 0)に重なるドン/カッも、右端から見えていたかによらず連打・風船の一部(風船の袋は H 5-8 でドンの色に入り、
        # 終わりと同じ所を同じ速さで流れる: Leviathan 24.00s)。譜面で終わりと同じ時刻に音符は置けない
        def _at_end(o, q, off):
            return (abs((o['t'] - q['t']) * abs(q['vel']) * fps - off) < RIN_S / 2 + 2 * (o.get('res_sd', 0) + q.get('res_sd', 0))
                    and abs(abs(o['vel']) / abs(q['vel']) - 1) <= 0.10)
        cap_ = [o for o in merged if o['color'] in ('don', 'ka') and not o.get('stationary') and o.get('vel')
                and any((o['color'] == 'don' and _late_start(o) and _at_end(o, q, RIN_S)) or _at_end(o, q, 0.0) for q in ends_o)]
        # 右端から見えていた丸の位置のドン: 長い連打を叩いている間は棒が画面の外まで赤くなり、終わりの丸も右端から赤い(そして勇者は眠りにつく 164.91s)。
        # 連打のすぐ後ろの本物のドンとは位置では分けられないので、白い縁で確かめる(赤い棒の端に白い縁は無い)。ringtrack の弱い音符に入れる
        for o in merged:
            if o['color'] == 'don' and o.get('vel') and not o.get('stationary') and not any(o is c_ for c_ in cap_) and any(_at_end(o, q, RIN_S) for q in ends_o):
                o['capsus'] = True
                if dbg: print('  roll-endcap? don %.3f(白い縁で確かめる)' % o['t'])
        if cap_ and mark_only:
            # 拾い直した音符には印だけ付けて白い縁で確かめる(途中から現れる本物のドンが丸の位置にあることがある: resume:story 71.72s は一度外され、
            # 拾い直しで戻っていた)
            for o in cap_: o['capsus'] = True
            if dbg: print('  roll-endcap? ' + ' '.join('%s %.3f n=%d' % (o['color'], o['t'], o['npts']) for o in cap_))
            cap_ = []
        if cap_:
            if dbg: print('  roll-endcap ' + ' '.join('%s %.3f n=%d' % (o['color'], o['t'], o['npts']) for o in cap_))
            merged = [o for o in merged if not any(o is c_ for c_ in cap_)]
    return merged



def _peek_behind(fs, xs, ws, merged, fps, f0):
    """覗きの幽霊か(下の注): 欠けて細く(幅の中央値が小音符の内円の半径未満)、点の 7 割以上が採った大音符の予測位置のすぐ後ろ
    (内円の半径 + 小音符の内円の半径以内)にある。戻り値: (覗きか, 後ろにある点の割合)"""
    bigs = [q for q in merged if q.get('big') and q['color'] in ('don', 'ka') and q.get('vel')]
    if not bigs or len(fs) < 3 or np.median(ws) >= RIN_S: return False, 0.0
    fs = np.asarray(fs, float); xs = np.asarray(xs, float)
    A = np.array([-abs(q['vel']) for q in bigs]); B = np.array([JUDGE_X + abs(q['vel']) * (q['t'] * fps - f0) for q in bigs])
    d = xs[None, :] - (A[:, None] * fs[None, :] + B[:, None])
    fr = float(((d > 0) & (d < RIN_B + RIN_S)).any(0).mean())
    return fr >= 0.7, fr


def _peek_small(fs, xs, ws, o, merged, fps, f0):
    """小音符の追い越しの覗きの幽霊か: 速い音符が遅い同じ色の音符の上を通り過ぎるとき、速い音符の縁のすぐ後ろから下の遅い音符の端が覗き、
    速い音符と一緒に動く細い色の塊になる(検出テスト ファイナル 72.79s・72.88s: 速さ 47 のカッの 44px 後ろに、速さ 6.3 のドンの赤)。
    欠けて細く(幅の中央値が小音符の内円の半径未満)、点の 7 割以上で「同じ速さ(±3%)の採った音符のすぐ後ろ(内円の半径の和以内)」かつ
    「速さの違う(10% 以上)同じ色の採った音符と重なる(中心の差が内円の直径未満)」。本物の積み重なりの音符は下に遅い音符の無い所でも見える。
    戻り値: (覗きか, 当てはまる点の割合)"""
    if len(fs) < 3 or np.median(ws) >= RIN_S or not o.get('vel'): return False, 0.0
    v_ = abs(o['vel'])
    fr_ = [q for q in merged if q is not o and q['color'] in ('don', 'ka') and q.get('vel') and abs(abs(q['vel']) / v_ - 1) <= 0.03]
    sl_ = [q for q in merged if q is not o and q['color'] == o['color'] and q.get('vel') and abs(abs(q['vel']) / v_ - 1) > 0.10]
    if not fr_ or not sl_: return False, 0.0
    fs = np.asarray(fs, float); xs = np.asarray(xs, float)
    Af = np.array([-abs(q['vel']) for q in fr_]); Bf = np.array([JUDGE_X + abs(q['vel']) * (q['t'] * fps - f0) for q in fr_])
    Rf = np.array([RIN_B if q.get('big') else RIN_S for q in fr_], float)
    d = xs[None, :] - (Af[:, None] * fs[None, :] + Bf[:, None])
    behind = ((d > 0) & (d < Rf[:, None] + RIN_S)).any(0)
    As = np.array([-abs(q['vel']) for q in sl_]); Bs = np.array([JUDGE_X + abs(q['vel']) * (q['t'] * fps - f0) for q in sl_])
    under = (np.abs(As[:, None] * fs[None, :] + Bs[:, None] - xs[None, :]) < 2 * RIN_S).any(0)
    fr = float((behind & under).mean())
    return fr >= 0.7, fr


def _line_recover(frames_c, fps, f0, merged, dbg=False):
    """追跡で使われなかったドン/カッの候補点から、「一定の速さで判定枠へ向かう直線」にきれいに乗る点の列を探して音符にする。
    速さの違う音符が追い越し合う所(もぺもぺ おに 121.9s: SCROLL 2.28〜0.58 のカッが重なって届く)では、追跡が点をつなぎ間違えて
    音符を落とす。等間隔に並んで見えている区間の点は各自の直線に乗るので、点の組から (判定枠を通るフレーム, 速さ) に投票して拾う。
    条件: 10 点以上(4px 以内)・ばらつき 2.5px 以下・幅 300px 以上・判定枠の近く(x ≤ 1000)まで来ている(15 点以上・2px 以下・幅 400px 以上なら x ≤ 1300)・同じ色。"""
    if os.environ.get('V2T_LINES', '1') == '0': return []
    fr = fps / 60.0
    ex = [(o['t'] * fps - f0, abs(o['vel']), o['color'], max(8.0, min(2.5 * o.get('res_sd', 0.0), RIN_S / 2))) for o in merged if o.get('vel') and abs(o['vel']) > 1]
    ex_fc = np.array([e[0] for e in ex]); ex_v = np.array([e[1] for e in ex])
    # 採った音符の点のばらつき: 当てはめの荒い軌跡(録画の時計が乱れた動画で残差 6px)の点は予想位置から 8px 以上ずれる。残差の 2.5 倍以内の同じ色の点は
    # その音符自身の点(Mood Swing 58.99s: 残差 6.1px の大カッの点を拾い直して、39ms 前に同じ大カッの直線を作っていた)
    ex_c = np.array([e[2] for e in ex]); ex_r = np.array([e[3] for e in ex])
    # 採った音符の予想位置はフレームごとに 1 回だけ計算し、帯の中(X0〜X1 ± 許容の最大)に来る音符だけと比べる
    # (点ごとに全部の音符と比べていた。帯の外の音符は許容の最大より離れているので結果は同じ)
    ex_tolmax = float(max(8.0, ex_r.max())) if len(ex_r) else 8.0
    ex_tol = {col: np.where(ex_c == col, ex_r, 8.0) for col in ('don', 'ka')}
    pts = []
    for fi, cs in enumerate(frames_c):
        if len(ex_fc) and cs:
            pred_all = JUDGE_X + ex_v * (ex_fc - fi)
            near_ = np.flatnonzero((pred_all >= X0 - ex_tolmax) & (pred_all <= X1 + ex_tolmax))
            pred_n = pred_all[near_]; tol_n = {col: t_[near_] for col, t_ in ex_tol.items()}
        for c in cs:
            # 幅 20px 以上(大きく重なった音符の三日月も入れる: Emma 裏 玄人 77.7s は 33px ずつ重なった 3 つのドンの後ろ 2 つが幅 26〜30 の三日月で、
            # 追跡が隣へ乗り継いで落とした。直線の条件(10 点以上・sd 2.5px 以下・幅 300px 以上・右端から判定枠の近くまで)が細い欠片の幽霊を防ぐ)
            if c['color'] not in ('don', 'ka') or c.get('w', 0) < int(os.environ.get('V2T_LINE_MINW', '20')) or not (X0 <= c['cx'] <= X1): continue
            x = c['cx']
            if len(ex_fc):
                if np.any(np.abs(pred_n - x) <= tol_n[c['color']]): continue          # 採った音符の軌跡の上の点(その音符自身)
            pts.append((fi, x, c['color'], c.get('w', 0), bool(c.get('big'))))
    if len(pts) < 10: return []
    pts.sort()
    F = np.array([p[0] for p in pts], float); Xs = np.array([p[1] for p in pts], float); Cc = np.array([p[2] for p in pts])
    Wd = np.array([p[3] for p in pts], float); Bg = np.array([p[4] for p in pts], bool)
    n_ = len(pts)
    # 5〜20 フレーム離れた同じ色の点の組で速さを求め、(判定枠を通るフレーム, 速さ/2, 色) に投票する(隣同士だと ±5px のぶれで票が散る)。
    # 速さは左向きならどれだけ遅くてもよい(止まっている物は下の「300px 以上」で落ちる)。上限は帯に 10 点入る速さ。
    # 組は配列でまとめて作る(以前は点ごとの二重ループで 1 曲 18 秒)。票の数も、同じ票数の並び(組を i→j の順に見て最初に出た順)も同じ
    lo_ = np.searchsorted(F, F + 5 * fr); hi_ = np.searchsorted(F, F + 20 * fr, side='right')
    cnt_ = np.maximum(hi_ - lo_, 0)
    I_ = np.repeat(np.arange(n_), cnt_)
    J_ = np.arange(int(cnt_.sum())) - np.repeat(np.cumsum(cnt_) - cnt_, cnt_) + np.repeat(lo_, cnt_)
    m_ = Cc[I_] == Cc[J_]; I_, J_ = I_[m_], J_[m_]
    v_ = (Xs[I_] - Xs[J_]) / (F[J_] - F[I_])
    m_ = (0 < v_) & (v_ <= (X1 - X0) / 10.0 / fr); I_, v_ = I_[m_], v_[m_]
    ranked = []
    if len(I_):
        fc_ = F[I_] + (Xs[I_] - JUDGE_X) / v_
        key_ = np.stack([np.round(fc_) + 0.0, np.round(v_ / 2.0) + 0.0, (Cc[I_] == 'ka').astype(float)], 1)   # + 0.0: −0.0 を 0.0 と同じ票にする(元は int(round()) で 0)
        uk_, first_, nv_ = np.unique(key_, axis=0, return_index=True, return_counts=True)
        for q_ in sorted(range(len(uk_)), key=lambda q_: (-nv_[q_], first_[q_])):
            ranked.append(((int(uk_[q_, 0]), int(uk_[q_, 1]), 'ka' if uk_[q_, 2] else 'don'), int(nv_[q_])))
    used = np.zeros(len(pts), bool); out = []
    iscol_ = {c_: (Cc == c_) for c_ in ('don', 'ka')}
    # 点は F(フレーム)の昇順なので、直線が帯(X0〜X1)を通るフレームの範囲の点だけで計算する。点の x は帯の中なので、範囲の外
    # (1 フレームの余裕を取る)の点は条件を満たせず、結果は全部の点で計算したのと同じ(全部の点(1 曲 25 万)で計算して 1 曲 12 秒かかった)
    def _span(f_lo, f_hi):
        return slice(int(np.searchsorted(F, f_lo - 1.0)), int(np.searchsorted(F, f_hi + 1.0, side='right')))
    for (fc, v2, col), nv in ranked:
        if nv < 15: break
        v = v2 * 2.0
        # 票の区切り(1 フレーム × 2px)の中心は真の直線から最大で半フレームぶんずれるので、広めに集めてから当てはめ直して 4px 以内に絞る
        w_ = 1500 / max(v, 1); sl = _span(fc - w_, fc + w_)
        m = np.zeros(n_, bool)
        m[sl] = (~used[sl]) & (np.abs(Xs[sl] - (JUDGE_X + v * (fc - F[sl]))) <= 0.6 * v + 6) & iscol_[col][sl] & (np.abs(F[sl] - fc) <= w_)
        if m.sum() < 10: continue
        ok_ = False
        for tol in (12, 8, 4, 4):
            a, b = np.polyfit(F[m], Xs[m], 1)
            if not (-(X1 - X0) / 10.0 / fr <= a < 0): break
            sl = _span((X1 + tol - b) / a, (X0 - tol - b) / a)      # a < 0: 直線が x = X1 + tol から X0 − tol まで動くフレーム
            m = np.zeros(n_, bool)
            m[sl] = (~used[sl]) & (np.abs(Xs[sl] - (a * F[sl] + b)) <= tol) & iscol_[col][sl]
            if m.sum() < 10: break
        else:
            ok_ = True
        if not ok_ or len(set(F[m])) < 10: continue
        a, b = np.polyfit(F[m], Xs[m], 1); r = Xs[m] - (a * F[m] + b)
        long_clean = m.sum() >= 15 and r.std() <= 2.0 and Xs[m].max() - Xs[m].min() >= 400    # 長くきれいな直線は、判定枠の手前で他の音符に重なって
        #   見えなくなっていてもよい(もぺもぺ 122.10s の vel 22.5 のカッ: x1616→1231 の 17 点のあと、vel 29.7 のカッと重なって届く)
        if r.std() > 2.5 or Xs[m].max() - Xs[m].min() < 300 or (Xs[m].min() > 1000 and not (long_clean and Xs[m].min() <= 1300)): continue
        # 見えるフレームの割合: 本物の音符は通る道の多くのフレームで見える(大きく重なった三日月 92%、追い越しのカッ 43〜70%)。点の寄せ集めの幽霊は 11%(彁 裏 20.33s)
        if m.sum() < 0.3 * ((Xs[m].max() - Xs[m].min()) / abs(a) + 1): continue
        # 細い候補(幅 60 未満)ばかりの列は判定枠の近く(x ≤ 1000)まで来ていること(判定枠の手前で途切れる欠片の列: vs.VIGVANGS 裏 76.39s は x1182 まで)
        if np.median(Wd[m]) < 60 and Xs[m].min() > 1000: continue
        if Xs[m].max() < 1700: continue        # 右端から入ってきた点の列だけ(途中から始まる列は別の音符の点の寄せ集め: バブリー革命 45.94s の幽霊は x1447 から)
        fcross = (JUDGE_X - b) / a; t = round((f0 + fcross) / fps, 4)
        if any(o['color'] == col and abs(o['t'] - t) < 0.03 for o in merged + out): continue
        # 交互の双子(track の注): 近く(2 フレームぶんの動き以内)に同じ色・同じ速さの音符があり、その音符が直線の点と同じフレームにほとんど写っていなければ、
        # 録画の時計の乱れで交互に振り分けられた同じ音符(Mood Swing 58.99s・79.24s)
        v_l = -a; twin_ = False
        for q in merged:
            if q['color'] != col or not q.get('vel') or abs(abs(q['vel']) / v_l - 1) > 0.10: continue
            if abs(q['t'] - (f0 + (JUDGE_X - b) / a) / fps) * v_l * fps > 2 * v_l + 6: continue
            fr_l = sorted(set(int(f) for f in F[m]))
            both = sum(1 for f in fr_l if any(c['color'] == col and abs(c['cx'] - (JUDGE_X + abs(q['vel']) * (q['t'] * fps - f0 - f))) <= 8 for c in frames_c[f]))
            if both <= 0.1 * len(fr_l): twin_ = True; break
        if twin_:
            if dbg: print('  line-twin %s t=%.3f(交互に写る同じ音符 %.3f)' % (col, (f0 + (JUDGE_X - b) / a) / fps, q['t']))
            continue
        # 大音符の後ろの覗き(下の点の帰属の注: Calculator 48.10s。断片の直線でも拾い直していた)
        pk_, pf_ = _peek_behind(F[m], Xs[m], Wd[m], merged, fps, f0) if Bg[m].mean() <= 0.5 else (False, 0.0)
        if pk_:
            if dbg: print('  line-peek %s t=%.3f(大音符の後ろ %.0f%%)' % (col, t, 100 * pf_))
            continue
        used |= m
        # 大小は直線に乗った候補の多数決(常に小にしていた: SUPERNOVA 126.52s・resume:story 72.41s・崖の上のポニョ 47.75s の大ドン)
        o = {'t': t, 'color': col, 'big': bool(Bg[m].mean() > 0.5), 'npts': int(m.sum()), 'vel': round(-a, 2), 'hidden': False, 'line': True, 'res_sd': round(float(r.std()), 2),
             'xr': (round(float(Xs[m].max()), 1), round(float(Xs[m].min()), 1))}
        out.append(o)
        if dbg: print('  line %s t=%.3f vel=%.1f n=%d x%.0f-%.0f sd=%.1f 票 %d' % (col, t, -a, m.sum(), Xs[m].max(), Xs[m].min(), r.std(), nv))
    return out


def _fast_roll_recover(frames_c, fps, f0, merged, dbg=False):
    """1 フレーム 100px を超える速い連打の頭と尻尾(第六天魔王 裏の SCROLL 6 の大連打: 178px/フレーム)は、追跡の移動量の上限(130px)を超えて
    つながらない。頭(連打色の幅 130px 以下の候補)と尻尾(連打の終わりの候補)を、別々に「一定の速さで判定枠へ向かう直線」で拾う。
    条件: 5 点以上・頭と尻尾が組になる(速さ 5% 以内)・ばらつきが速さの 15% 以下(178px なら 27px = 2.5ms)・右側(x ≥ 1500)から判定枠の近く(x ≤ 1100)まで・速さ 100〜260px/フレーム"""
    if os.environ.get('V2T_FASTROLL', '1') == '0': return []
    fr = fps / 60.0; out = []
    end_pts = [(fi, c['cx']) for fi, cs in enumerate(frames_c) for c in cs if c['color'] == 'roll_end']
    for kind in ('head', 'end'):
        pts = []
        for fi, cs in enumerate(frames_c):
            for c in cs:
                if kind == 'head' and c['color'] in ('renda', 'balloon'): pts.append((fi, c['cx'], c['color'], bool(c.get('big'))))   # 小さい連打は頭と棒が一続きの幅広い候補(中心が正しい速さで動く)
                if kind == 'end' and c['color'] == 'roll_end': pts.append((fi, c['cx'], 'roll_end', False))
        if len(pts) < 4: continue
        F = np.array([p[0] for p in pts], float); X = np.array([p[1] for p in pts], float)
        used = np.zeros(len(pts), bool)
        for i in range(len(pts)):
            if used[i] or X[i] < 1500: continue
            # i から始まる直線: 次のフレームで 100〜260px 左の点をつないでいく
            # つないでいく: 最初は次のフレームで 100〜260px 左の点、以後は予想位置(速さ × フレーム差)の ±25% に最も近い点(1 フレーム飛ばしてよい)。
            # 尻尾の候補は位置のぶれが大きい(±30px)ので、最初の一歩の候補ごとに試して一番長い列を採る
            best_chain = None
            # 次の 1〜2 フレーム(30fps の中身の動画は 1 枚おきに絵が無い)で、フレームあたり 100〜260px 左の点
            firsts = [j for j in range(len(pts)) if not used[j] and 1 <= F[j] - F[i] <= 2 * fr and 100 / fr <= (X[i] - X[j]) / (F[j] - F[i]) <= 260 / fr]
            for j1 in firsts:
                chain = [i, j1]; last = j1; v = (X[i] - X[j1]) / (F[j1] - F[i])
                while True:
                    cand = []
                    for j in range(len(pts)):
                        df = F[j] - F[last]
                        if used[j] or not (1 <= df <= 2 * fr): continue
                        pred = X[last] - v * df
                        if abs(X[j] - pred) <= 0.25 * v * df: cand.append((F[j], abs(X[j] - pred), j))
                    if not cand: break
                    j = min(cand)[2]; chain.append(j); last = j; v = (X[chain[0]] - X[j]) / (F[j] - F[chain[0]])
                if best_chain is None or len(chain) > len(best_chain): best_chain = chain
            chain = best_chain or [i]
            if len(chain) < 5 or X[chain[-1]] > 1100: continue
            a, b = np.polyfit(F[chain], X[chain], 1); r = X[chain] - (a * F[chain] + b)
            if r.std() > 0.15 * abs(a) or not (-260 / fr <= a <= -100 / fr): continue
            t = round((f0 + (JUDGE_X - b) / a) / fps, 4)
            col = 'roll_end' if kind == 'end' else ('balloon' if sum(pts[k][2] == 'balloon' for k in chain) * 2 > len(chain) else 'renda')
            big = kind == 'head' and sum(pts[k][3] for k in chain) * 2 >= len(chain)
            same = ('renda', 'balloon') if kind == 'head' else ('roll_end',)
            if any(o['color'] in same and abs(o['t'] - t) < 0.05 for o in merged + out): continue
            used[chain] = True
            out.append({'t': t, 'color': col, 'big': big, 'npts': len(chain), 'vel': round(-a, 2), 'hidden': False, 'fast_roll': True})
            if dbg: print('  fast-roll? %s%s t=%.3f vel=%.1f n=%d x%.0f-%.0f sd=%.1f' % (col, ' B' if big else '', t, -a, len(chain), X[chain[0]], X[chain[-1]], r.std()))
    # 超高速の風船(憎悪と醜悪の花束 裏の SCROLL 13: 380px/フレーム)は 2〜3 フレームしか写らず、尻尾も無い。風船の色の点が 2 点以上、
    # 速さ 250〜450px/フレームで一定、右側(x ≥ 1300)から判定枠の近く(x ≤ 1100)まで来るものを風船にする
    fb = []
    bp = [(fi, c['cx'], bool(c.get('big'))) for fi, cs in enumerate(frames_c) for c in cs if c['color'] == 'balloon']
    for i, (fi, xi, bi) in enumerate(bp):
        if xi < 1300: continue
        chain = [(fi, xi)]
        while True:
            # 次の 1〜2 フレーム(30fps の中身は 1 枚おき)。移動はフレームあたりで比べる
            nx_ = [(f2, x2) for f2, x2, _ in bp if 1 <= f2 - chain[-1][0] <= 2 and 250 / fr <= (chain[-1][1] - x2) / (f2 - chain[-1][0]) <= 450 / fr]
            if len(chain) >= 2:
                v_ = (chain[0][1] - chain[-1][1]) / (chain[-1][0] - chain[0][0])
                nx_ = [q for q in nx_ if abs((chain[-1][1] - q[1]) / (q[0] - chain[-1][0]) - v_) <= 0.15 * v_]
            if not nx_: break
            chain.append(nx_[0])
        if len(chain) < 2 or chain[-1][1] > 1100: continue
        F_ = np.array([q[0] for q in chain], float); X_ = np.array([q[1] for q in chain], float)
        a, b = np.polyfit(F_, X_, 1)
        t = round((f0 + (JUDGE_X - b) / a) / fps, 4)
        if any(o['color'] in ('balloon', 'renda') and abs(o['t'] - t) < 0.1 for o in merged + fb): continue
        fb.append({'t': t, 'color': 'balloon', 'big': False, 'npts': len(chain), 'vel': round(-a, 2), 'hidden': False, 'fast_roll': True})
        if dbg: print('  fast-balloon t=%.3f vel=%.1f n=%d x%.0f-%.0f' % (t, -a, len(chain), X_[0], X_[-1]))
    # 群れで飛ぶもの(0.5 秒以内に他の超高速の風船がある)は演出(東京シャンディ 100.0〜100.7s: 280px/フレームの風船が 0.11 秒おきに 6 つ画面を横切る)。
    # 本物の超高速の風船は 1 つずつ(憎悪と醜悪の花束 裏は 6.6 秒離れた 2 つ)
    fb = [o for o in fb if not any(q is not o and abs(q['t'] - o['t']) < 0.5 for q in fb)]
    # 最後のドン/カッより 1 秒以上後のものは曲の終わりの演出(CARMINE 100.06s: クリアの虹色の文字を 3 点の風船と誤認)
    dk_ = [o['t'] for o in merged if o['color'] in ('don', 'ka')]
    if dk_: fb = [o for o in fb if o['t'] <= max(dk_) + 1.0]
    # 頭と尻尾が組になるものだけ(速さ 5% 以内、尻尾は頭の 0.05〜1 秒後)。速い連打は頭も尻尾も追跡に乗らないので必ず組で出る。
    # 片方だけのものは連打のはしご(resume:story 144s)などの演出で、連打を 2 個増やしていた
    heads = [o for o in out if o['color'] != 'roll_end']; ends = [o for o in out if o['color'] == 'roll_end']
    keep = []
    for h in heads:
        # 尻尾は頭の後で、次の頭より前(1 秒で打ち切ると、1 秒を超える速い連打がまるごと消えた)
        nh_ = min([q['t'] for q in heads if q['t'] > h['t']] or [h['t'] + 60.0])
        e = next((e for e in ends if 0.05 < e['t'] - h['t'] and e['t'] < nh_ and abs(e['vel'] / h['vel'] - 1) <= 0.05 and e not in keep), None)
        if e is None and end_pts and abs(h['vel']) > 130.0 / fr and os.environ.get('V2T_FASTROLL_END', '1') != '0':
            # 追跡が追えない速さ(1 フレーム 130px 超)の頭だけ。尻尾の点が直線に 5 点並ばなくても(1 フレーム 178px 動くので、録画のフレーム間隔の揺れ ±5ms で位置が ±50px ぶれる:
            # 第六天魔王 裏 50.09s)、尻尾は頭と同じ棒なので頭と同じ速さで動く。尻尾の候補の点ごとに頭の速さで判定枠に着く時刻を出し、
            # 0.02 秒(約 1 フレーム)の幅に 3 点以上集まる所の中央値を尻尾の時刻にする
            v_h = abs(h['vel'])
            tt = sorted((f0 + fi + (x - JUDGE_X) / v_h) / fps for fi, x in end_pts)
            # 尻尾はその頭と次の連打の頭の間にしか無い(次の頭の後ろの尻尾は別の連打のもの: resume:story のはしご 145.1s)
            nxt_h = min([o['t'] for o in merged if o['color'] in ('renda', 'balloon') and o['t'] > h['t'] + 0.02] + [h['t'] + 1.0])
            tt = [t for t in tt if 0.05 < t - h['t'] and t < nxt_h]
            best = max(([t for t in tt if t0_ <= t < t0_ + 0.02] for t0_ in tt), key=len, default=[])
            if len(best) >= 3:
                e = {'t': round(float(np.median(best)), 4), 'color': 'roll_end', 'big': False, 'npts': len(best), 'vel': h['vel'], 'hidden': False, 'fast_roll': True}
                if dbg: print('  fast-roll end by head speed %.3f (%d 点)' % (e['t'], len(best)))
        if e is not None:
            keep += [h, e]
            if dbg: print('  fast-roll %s%s %.3f-%.3f vel %.1f/%.1f' % (h['color'], ' B' if h['big'] else '', h['t'], e['t'], h['vel'], e['vel']))
    return keep + fb


V_FAST2 = (6 / 2) / (0.25 / 4)       # 48px/フレーム: 2 本の一致でゲームの時刻を測ってよい速さ(ringtrack.NEAR の半分の位置のぶれが 0.0625 フレーム以下)


def _game_delta(tracks, n, dbg=False, it_n=3, win=61):
    """フレームごとのゲームの時刻のずれ[フレーム](整数のフレーム番号からの差)を、画面の音符全部から測る。同じフレームの音符は全部同じゲームの
    時刻で描かれるので、各軌跡の直線からの位置のずれ ÷ 速さ が、そのフレームの時刻のずれ。フレームごとに軌跡の中央値を取る(3 本以上)。
    当てはめ直して繰り返し、前後 win 枚の平均を引く(録画は平均すれば 60fps)。戻り値: 長さ n の配列"""
    d = np.zeros(n)
    use = []
    # 軌跡は 10 点以上。3 本に届かないフレームの補い(下)だけ 8 点以上も使う(速い音符は画面に 9〜14 フレームしか写らない: 検出テスト 極 89.5s の
    # SCROLL 4・BPM 200 の 16 分)
    for tr in tracks:
        if tr['color'] not in ('don', 'ka') or len(tr['pts']) < 8: continue
        fs = np.array([p[0] for p in tr['pts']], int); xs = np.array([p[1] for p in tr['pts']], float)
        if xs[0] - xs[-1] < 200: continue
        use.append((fs, xs, len(tr['pts']) >= 10, abs(tr['vel'] or 0.0)))
    for _ in range(it_n):
        acc = [[] for _ in range(n)]; acc2 = [[] for _ in range(n)]
        for fs, xs, full_, v_tr in use:
            ft = fs + d[fs]
            a, b = np.polyfit(ft, xs, 1)
            if a > -4: continue
            r = (xs - (a * ft + b)) / a
            ok = np.abs(r) <= 1.5                      # 2 つの音符をつないだ軌跡の外れた点は数えない
            if ok.sum() < 8: continue
            a, b = np.polyfit(ft[ok], xs[ok], 1); r = (xs - (a * ft + b)) / a
            for f_, v_ in zip(fs[np.abs(r) <= 1.5], r[np.abs(r) <= 1.5]):
                if full_: acc[f_].append(v_)
                if v_tr >= V_FAST2: acc2[f_].append(v_)
        # 測れたとするのは 10 点以上の軌跡 3 本以上の中央値。3 本に届かないフレームは、速い(V_FAST2 以上の)8 点以上の軌跡 2 本がそろって同じずれ
        # (0.25 フレーム以内)を示すときだけ補う(音符のまばらな速い区間は画面に 1〜2 個しかいない: 検出テスト 極 89.5s: ゲームの時計が 1 フレーム止まった所を
        # 80px/フレームのドン 2 本がまたいだが測れず、2 本とも「直線に乗らない偽物」になった)。1 本だけでは、別々の音符をつないだ軌跡の段差と区別できない。
        # 2 本の一致が証拠になるのは、1 本ずつの時刻の測りのぶれ(位置のぶれ NEAR/2 ÷ 速さ)が一致の幅の 4 分の 1 以下のときだけ。遅い軌跡の 2 本は
        # 偶然そろう(20px/フレーム以下の密な区間で補うと、検出テスト ファイナルの取りこぼし 71 → 80)。3 本以上で測れるフレームは元のまま
        for f_ in range(n):
            if len(acc[f_]) < 3:
                b_ = acc2[f_]
                acc[f_] = list(b_) if (len(b_) == 2 and abs(b_[0] - b_[1]) <= 0.25) else []
        def _ok(a_): return len(a_) >= 3 or len(a_) == 2
        has = np.array([_ok(a_) for a_ in acc])
        upd = np.array([(float(np.median(a_)) if len(a_) >= 3 else float(np.mean(a_))) if _ok(a_) else 0.0 for a_ in acc])
        d = d + upd
        # 前後 win 枚の(測れたフレームの)平均を引く
        k_ = np.ones(win)
        num = np.convolve(np.where(has, d, 0.0), k_, mode='same'); den = np.convolve(has.astype(float), k_, mode='same')
        d = np.where(has, d - num / np.maximum(den, 1), 0.0)
        # 測れなかったフレーム(軌跡が 3 本未満)は前後の測れたフレームからつなぐ(0 にすると、+0.3 と −0.7 の間に 0 が挟まって時刻が行き来し、
        # つなぎ直しでドンの軌跡が切れた: 検出テスト 82.68s)
        if has.any() and not has.all():
            d = np.interp(np.arange(n), np.flatnonzero(has), d[has])
    # 測りのぶれを超えるずれだけ使う: フレームごとの軌跡のずれの散らばり(1.4826×MAD)から中央値の標準誤差(×1.2533/√本数)を出し、その 3 倍を超える
    # フレームだけ。普通の動画はずれ sd 0.02〜0.05 フレームで測りのぶれと同じ大きさなので使わない(全部に使うと、処理落ちの無い動画の連打の数や
    # 時刻が変わった: 段位 REPLAY・太鼓ドラムモンスター・もぺもぺ)
    se = np.array([1.2533 * 1.4826 * float(np.median(np.abs(np.array(a_) - np.median(a_)))) / np.sqrt(len(a_)) if len(a_) >= 3
                   else (max(abs(a_[0] - a_[1]) / 2, 0.05) if _ok(a_) else np.inf) for a_ in acc])
    sig = has & (np.abs(d) > 3 * np.maximum(se, 0.01))
    # 動画全体の描画の乱れ(ずれの散らばり 1.4826×MAD)が時刻の精度(0.1 フレーム = 1.7ms)に届かない動画では使わない。同じフレームの音符のずれは
    # 互いに関係していて(重なった列の欠けた円の中心は同じ向きにずれる)、標準誤差は本当の測りのぶれより小さく出る(Future 裏: 散らばり 0.026 で
    # 828 枚が「ぶれを超える」となり、使うと書き出しとの往復 sd が 1.2→2.4ms に悪化)。処理落ちの録画は 0.2〜0.3(検出テスト・段位 REPLAY)
    jit = 1.4826 * float(np.median(np.abs(d[has] - np.median(d[has])))) if has.any() else 0.0
    _game_delta.jit = jit
    if jit < 0.1: sig = np.zeros(n, bool)
    _game_delta.has = has; _game_delta.sig = sig
    if not sig.any(): return np.zeros(n)
    # 描画が乱れている動画では、測れたずれを全部使う(フレームごとに「ぶれを超える」ものだけ残して他を 0 にすると、時刻が飛び飛びになり
    # つなぎ直しで 48 分の列がまた乗り移った: 検出テスト ファイナル 47s)。測れなかったフレームは上で前後からつないである
    return d


def track(frames_c, fps, f0, dbg=False, tclock=None, gd_fit=None):
    """tclock: フレームごとのゲームの時刻のずれ(_game_delta)。与えると、つなぎの予測をゲームの時刻で行う(2 回目の追跡)"""
    # フレームレートが 60 と違うとき: 1フレームあたりの移動量は 60/fps 倍、フレーム数の閾値は fps/60 倍
    fr = fps / 60.0
    TT_ = np.arange(len(frames_c), dtype=float) + (np.asarray(tclock, float) if tclock is not None else 0.0)     # 各フレームのゲームの時刻
    # 1 点のトラックに次の点をつなぐ移動の上限は 130px。それを超えて帯の幅の 3 分の 1(帯の中に 3 点以上映る)までは、ドン・カッで色も大小も
    # 同じ候補どうしだけつなぐ(〆ドレー2000 106.7s の 132px/フレームの小ドン。全部の候補で広げると、曲の終わりの演出に 186px/フレームの偽の
    # 連打の終わりができ、チャーリーダッシュ! の大音符 4 個が小音符とつながった)
    d_max = 130.0 / fr; d_far = (X1 - X0) / 3.0 / fr; gap_long = max(4, int(round(GAP_LONG * fr))); dt_short = max(2, int(round(4 * fr)))
    # 全体速度(px/frame)の推定: 同色候補の最も近い左向き変位の中央値
    # 直前の「候補のあるフレーム」(3 枚以内)との差をフレーム数で割る。30fps 相当の動画は複製を外して 1 枚おきに候補が空なので、
    # 隣のフレームとだけ比べると V が測れず(較正のわずかな差で 74px/フレームなどに振れ)、新しいトラックが隣の音符とつながって
    # 音符が細切れになった(IRIS OUT 74.32s)
    dxs = []
    last_i = None
    for i in range(len(frames_c)):
        if not frames_c[i]: continue
        if last_i is not None and i - last_i <= 3:
            dtf = i - last_i
            prev = [c['cx'] for c in frames_c[last_i]]
            for c in frames_c[i]:
                cand = [p - c['cx'] for p in prev if 5 * dtf < p - c['cx'] < d_max * dtf]
                if cand: dxs.append(min(cand) / dtf)
        last_i = i
    V = float(np.median(dxs)) if dxs else 21.0 / fr
    tracks = []; active = []; hidden_pts = {}; emerge_pts = {}; emerge_fill = {}
    global TRACK_DEBUG
    dbgT = {'tracks': [], 'byid': {}, 'f0': f0, 'fps': fps} if os.environ.get('DETECT_DEBUG_TRACKS') else None
    TRACK_DEBUG = dbgT
    def _dbg_add(tr, status, why='', onset=None, pts=None):
        if dbgT is None: return
        e = {'color': tr['color'] if tr else (onset or {}).get('color'), 'pts': list(pts if pts is not None else tr['pts']),
             'ws': list(tr.get('ws', [])) if tr else [], 'status': status, 'why': why, 'onset': onset,
             'dropped': list(tr.get('dropped', [])) if tr else [], 'joined': bool(tr.get('joined')) if tr else False}
        dbgT['tracks'].append(e)
        if onset is not None: dbgT['byid'][id(onset)] = e
    def _dbg_mark(o, status):
        if dbgT is not None and id(o) in dbgT['byid']: dbgT['byid'][id(o)]['status'] = status
    for fi, cands_all in enumerate(frames_c):
        cands = []
        for c in cands_all:
            if c.get('hidden'): hidden_pts.setdefault(c['color'], []).append((fi, c['cx']))
            elif c.get('emerge'): emerge_pts.setdefault(c['color'], []).append((fi, c['cx'], bool(c.get('big')))); emerge_fill[(fi, c['color'], round(c['cx']))] = c.get('eyes', 0)
            else: cands.append(c)
        # フレーム内の (トラック, 候補) 全ペアをコスト順に貪欲割当て。候補ごとの逐次処理だと、
        # 予測誤差の大きいトラックが左の候補に先に取られて本物が押し出される「横取り」が起きる
        pairs = []
        if cands and active:
            # 全トラックを配列にして、候補ごとにまとめて計算する(1 組ずつ Python で回すと 1 曲 10 秒)。式と判定は 1 組ずつの版と同じ
            A_col = np.array([tr['color'] for tr in active]); A_lf = np.array([tr['pts'][-1][0] for tr in active])
            A_lx = np.array([tr['pts'][-1][1] for tr in active], float); A_n = np.array([len(tr['pts']) for tr in active])
            A_v = np.array([tr['vel'] for tr in active], float); A_big = np.array([bool(tr['big'][-1]) for tr in active])
            A_dt = fi - A_lf
            # 連打の棒(先行音符なので上に描かれる)に隠れて十数フレーム見えないことがある。
            # 速度が確からしい(8点以上)トラックは最大 GAP_LONG フレームまで予測位置で橋渡しする
            A_ok = (A_dt >= 1) & (A_dt <= np.where(A_n >= 8, gap_long, dt_short))
            A_two = A_n >= 2
            A_dtf = np.maximum(TT_[fi] - TT_[A_lf], 0.25) if tclock is not None else np.maximum(A_dt, 1).astype(float)
            # 動画のフレーム時刻ジッタ(移動量が 10→21→31px と揺れる)を跨げる幅。隣の音符は ≥45px 先。長い空白は速度既知なので予測誤差ぶんだけ
            A_gate = np.where(A_dt <= dt_short, np.maximum(14.0, 0.6 * np.abs(A_v) * A_dtf), 14.0 + 0.5 * A_dtf / fr)
            A_pred = A_lx - A_v * A_dtf
            for ci, c in enumerate(cands):
                m = A_ok & (A_col == c['color'])
                if not m.any(): continue
                cx = c['cx']
                e = np.abs(A_pred - cx)
                for ai in np.flatnonzero(m & A_two & (e < A_gate)).tolist(): pairs.append((float(e[ai]), ai, ci))
                d = (A_lx - cx) / A_dtf
                far = (A_big == bool(c['big'])) & (d <= d_far) if c['color'] in ('don', 'ka') else np.zeros(len(active), bool)
                # 1 点目と 2 点目は 5px 以上離れていること: 候補の中心は ±5px ぶれるので、それより小さい動きは止まっている演出と区別できない
                # (下げると判定枠の火花などがつながり、遅いトラックが本物の候補を横取りした: チャーリー 69%)。
                # それより遅い音符は、多くのフレームにわたる直線として _line_recover・隠れ大音符の直線探しで拾う(どちらも速さの下限なし)
                for ai in np.flatnonzero(m & ~A_two & (((d >= 5) & (d <= d_max)) | far)).tolist():    # SCROLL 3 の速い音符(≈78px/frame)も繋げる
                    pairs.append((abs(float(d[ai]) - V) + 1000, ai, ci))   # 1点トラックは速度既知トラックより後回し
        pairs.sort()
        used_t = [False] * len(active); used_c = [False] * len(cands)
        for cost, ai, ci in pairs:
            if used_t[ai] or used_c[ci]: continue
            tr = active[ai]; c = cands[ci]
            tr['pts'].append((fi, c['cx'])); tr['big'].append(c['big']); tr.setdefault('ws', []).append(c['w']); tr.setdefault('hs', []).append(c['hmax'])
            tr.setdefault('hid', []).append(bool(c.get('hidden'))); tr.setdefault('ku', []).append(bool(c.get('kusu')))
            ff, xf = tr['pts'][0]
            tr['vel'] = (xf - c['cx']) / ((TT_[fi] - TT_[ff]) if tclock is not None else (fi - ff))
            used_t[ai] = True; used_c[ci] = True
        newborn = [{'color': c['color'], 'pts': [(fi, c['cx'])], 'big': [c['big']], 'vel': V, 'ws': [c['w']], 'hs': [c['hmax']], 'hid': [bool(c.get('hidden'))], 'ku': [bool(c.get('kusu'))]}
                   for ci, c in enumerate(cands) if not used_c[ci]]
        active.extend(newborn)
        keep = []
        for tr in active:
            lim = gap_long if len(tr['pts']) >= 8 else dt_short
            (tracks if fi - tr['pts'][-1][0] > lim else keep).append(tr)
        active = keep
    tracks.extend(active)
    if os.environ.get('V2T_TRACKPRINT'):
        _a, _b = [int(x) for x in os.environ['V2T_TRACKPRINT'].split(':')]
        for tr in tracks:
            fs_ = [f0 + p[0] for p in tr['pts']]
            if (tr['color'] in ('don', 'ka') or os.environ.get('V2T_TRACKALL')) and fs_[-1] >= _a and fs_[0] <= _b and len(fs_) >= 2:
                print('TRK %s n=%d %s' % (tr['color'], len(fs_), ' '.join('%d:%.0f' % (f0 + p[0], p[1]) for p in tr['pts'])))
    # 同色で同じ直線上を並走する 2 本は 1 本にする。動画のフレーム重複(同じ画が 2 枚続く)で音符が 1 フレーム分ずれた直線に乗り換わると
    # ゲートを僅かに外れて新しいトラックが生まれ、以後 2 本が候補を交互に取り合って両方生き残る(各 40 点、時刻差 21ms → 音符が 2 重になる)
    def _line(tr):
        fs = np.array([p[0] for p in tr['pts']], float); xs = np.array([p[1] for p in tr['pts']], float)
        return np.polyfit(fs, xs, 1)
    tracks.sort(key=lambda t: t['pts'][0][0])
    merged_tracks = []
    for tr in tracks:
        if len(tr['pts']) < 3: merged_tracks.append(tr); continue
        a1, b1 = _line(tr); f0_, f1_ = tr['pts'][0][0], tr['pts'][-1][0]; target = None
        for mt in merged_tracks:
            if mt['color'] != tr['color'] or len(mt['pts']) < 3: continue
            ov0, ov1 = max(f0_, mt['pts'][0][0]), min(f1_, mt['pts'][-1][0])
            if ov1 - ov0 < 5: continue
            a2, b2 = _line(mt); fm = (ov0 + ov1) / 2.0; vref = max(abs(a1), abs(a2), 4.0)
            if abs((a1 * fm + b1) - (a2 * fm + b2)) < 1.5 * vref and abs(a1 - a2) < 0.25 * vref: target = mt; break
        if target is None: merged_tracks.append(tr); continue
        for k in ('pts', 'big', 'ws', 'hs', 'hid'):
            if k in tr: target[k] = target.get(k, []) + tr[k]
        order = np.argsort([p[0] for p in target['pts']])
        for k in ('pts', 'big', 'ws', 'hs', 'hid'):
            if k in target and len(target[k]) == len(order): target[k] = [target[k][j] for j in order]
        ff, xf = target['pts'][0]; fl, xl = target['pts'][-1]
        if fl > ff: target['vel'] = (xf - xl) / (fl - ff)
    tracks = merged_tracks
    # 交互の双子: 録画の時計が乱れた動画(処理落ちの組が多い)では、同じ音符の点が 1〜2 フレームずれた 2 本の軌跡に交互に振り分けられる
    # (Mood Swing 74.76s: 4439 の x1289 と 4440 の x1289 が別々の軌跡に入り、24ms ずれた 2 つのドンになった)。別の音符なら同じフレームに両方写るので、
    # 時間が 5 フレーム以上重なるのに共通のフレームが 1 割以下で、速さが 10% 以内・位置の差が 2 フレームぶんの動き以内なら、点の多い方だけ残す
    tracks.sort(key=lambda t: -len(t['pts']))
    twins_ = []
    for i_, tr in enumerate(tracks):
        if tr['color'] not in ('don', 'ka') or len(tr['pts']) < 5 or any(tr is g for g in twins_): continue
        fs1 = {p[0] for p in tr['pts']}; a1, b1 = _line(tr)
        for mt in tracks[i_ + 1:]:
            if mt['color'] != tr['color'] or len(mt['pts']) < 5 or any(mt is g for g in twins_): continue
            fs2 = {p[0] for p in mt['pts']}
            lo_, hi_ = max(min(fs1), min(fs2)), min(max(fs1), max(fs2))
            if hi_ - lo_ < 5: continue
            n1 = sum(1 for f in fs1 if lo_ <= f <= hi_); n2 = sum(1 for f in fs2 if lo_ <= f <= hi_)
            if len(fs1 & fs2) > 0.1 * min(n1, n2): continue
            a2, b2 = _line(mt); vref = max(abs(a1), abs(a2), 4.0)
            if abs(a1 - a2) > 0.10 * vref: continue
            fm = (lo_ + hi_) / 2.0
            if abs((a1 * fm + b1) - (a2 * fm + b2)) > 2.0 * vref + 6: continue
            if dbg: print('  twin %s f%d-%d(n=%d) と f%d-%d(n=%d): 交互に写る同じ音符 → 後者を外す' % (tr['color'], f0 + min(fs1), f0 + max(fs1), len(fs1), f0 + min(fs2), f0 + max(fs2), len(fs2)))
            twins_.append(mt)
    tracks = [t for t in tracks if not any(t is g for g in twins_)]
    # 同色で同じ直線上を時間的に前後する 2 本(途中で他の音符に隠れて追跡が切れ、再出現で別トラックになった)を 1 本にする。
    # 同じ直線に乗る = 同じ位置・同じ速度なので同じ音符(速度が 10% 以上違えば別物)。彁 裏の速いドンの裏の v11 のカッ
    tracks.sort(key=lambda t: t['pts'][0][0])
    joined = []
    for tr in tracks:
        if len(tr['pts']) < 5 or tr['color'] not in ('don', 'ka'): joined.append(tr); continue
        a1, b1 = _line(tr); target = None
        for mt in joined:
            if mt['color'] != tr['color'] or len(mt['pts']) < 5: continue
            gap = tr['pts'][0][0] - mt['pts'][-1][0]
            if gap > gap_long: continue
            if gap < 1:
                # 2 フレームまでの重なりは、重なったフレームで両方の点が同じ所(12px 以内)にあるときだけ(同じフレームの同じ所に別の音符は写らない。
                # Aria 81.69s: x938 で終わる前のフレームに x955 から判定枠の近くの軌跡が始まり、3 つ目のドンが 2 つになった)
                if gap < -2: continue
                a_m, b_m = _line(mt)        # 前の軌跡の点が飛び飛びでも比べられるよう、前の軌跡の直線の位置と比べる
                ov_ = [q for q in tr['pts'] if q[0] <= mt['pts'][-1][0]]
                if not ov_ or any(abs(a_m * q[0] + b_m - q[1]) > 12 for q in ov_): continue
            a2, b2 = _line(mt)
            if abs(a1 - a2) > 0.10 * max(abs(a1), abs(a2), 4.0): continue
            if abs((a2 * tr['pts'][0][0] + b2) - tr['pts'][0][1]) > 25 + 0.5 * gap: continue
            fs_u = np.array([q[0] for q in mt['pts'] + tr['pts']], float); xs_u = np.array([q[1] for q in mt['pts'] + tr['pts']], float)
            au, bu = np.polyfit(fs_u, xs_u, 1)
            if np.abs(au * fs_u + bu - xs_u).max() > max(12.0, _JITTER_FR * abs(au)):     # 描画間隔の揺れ × 速さ
                # 受け渡し: 片方が消えた直後(2 フレーム以内)に、半径未満だけ離れた所からもう片方が現れた。別の音符ならその近さで両方同時に写るので、
                # 中心の測りが跳んだ同じ音符(いろはすてっぷ! 33.50s: x952 で終わった次のフレームに 31px 手前の x906 から始まり、2 つのドンになった)。
                # つなぐと直線に乗らないので、当てはめのきれいな方だけを残す
                if gap <= 2 and abs((a2 * tr['pts'][0][0] + b2) - tr['pts'][0][1]) < RIN_S:
                    def _sd(t_):
                        f_ = np.array([q[0] for q in t_['pts']], float); x_ = np.array([q[1] for q in t_['pts']], float); aa_, bb_ = np.polyfit(f_, x_, 1)
                        return float((x_ - (aa_ * f_ + bb_)).std())
                    target = ('handover', mt, _sd(mt), _sd(tr)); break
                continue
            target = mt; break
        if isinstance(target, tuple):
            _, mt, sd_m, sd_t = target
            # 残すのは点の多い方。ただし点の多い方が直線にきれいに乗らず(残差 > ringtrack.NEAR/2)、少ない方がきれいに乗るなら少ない方
            # (点の少ない軌跡は残差が小さく出やすいので、残差だけで選ぶと 5 点の欠片を残して 138 点の軌跡を捨てた: トロいか2000 74.3s)
            nr_ = __import__('ringtrack').NEAR / 2
            long_t = len(tr['pts']) > len(mt['pts'])
            sh_ = mt if long_t else tr           # 点の少ない方: 単独で音符として採られる大きさ(途中から現れた音符と同じ 6 点・90px 以上)でなければ選ばない
            sh_ok = len(sh_['pts']) >= 6 and (max(q[1] for q in sh_['pts']) - min(q[1] for q in sh_['pts'])) >= 90   # (Destr0yer 156.12s: 53px の欠片を残して音符を落とした)
            keep_t = long_t if not ((sd_t if long_t else sd_m) > nr_ and (sd_m if long_t else sd_t) <= nr_ and sh_ok) else not long_t
            if dbg: print('  handover %s f%d-%d(n=%d sd %.1f) → f%d-%d(n=%d sd %.1f): %s を残す' % (tr['color'], f0 + mt['pts'][0][0], f0 + mt['pts'][-1][0], len(mt['pts']), sd_m, f0 + tr['pts'][0][0], f0 + tr['pts'][-1][0], len(tr['pts']), sd_t, '後' if keep_t else '前'))
            if keep_t: joined.remove(mt); joined.append(tr)
            continue
        if target is None: joined.append(tr); continue
        if dbg: print('  join %s f%d-%d(n=%d) + f%d-%d(n=%d)' % (tr['color'], f0 + target['pts'][0][0], f0 + target['pts'][-1][0], len(target['pts']), f0 + tr['pts'][0][0], f0 + tr['pts'][-1][0], len(tr['pts'])))
        target['joined'] = True
        for k in ('pts', 'big', 'ws', 'hs', 'hid'):
            if k in tr: target[k] = target.get(k, []) + tr[k]
        ff, xf = target['pts'][0]; fl, xl = target['pts'][-1]
        if fl > ff: target['vel'] = (xf - xl) / (fl - ff)
    tracks = joined
    # ゲームの描画の時刻(重い譜面で処理落ちし、描画間隔が 46/60/75px と揺れる: 検出テスト ファイナル)。音符全部から測ったフレームごとのずれを、
    # 軌跡の当てはめに使う(V2T_GCLOCK=0 で使わない)
    GD_ = gd_fit          # 小節線の追跡: 音符から測ったゲームの時刻をそのまま使う(小節線だけ整数のフレームだと、描画の乱れる録画で音符と 13ms 食い違った: Xa 裏 obs120)
    if gd_fit is None and os.environ.get('V2T_GCLOCK', '1') != '0' and getattr(track, 'frac', None) is None:
        GD_ = _game_delta(tracks, len(frames_c), dbg)
        if not _game_delta.sig.any(): GD_ = None
        # 1 回目の追跡(整数のフレーム)でつなぎの予測がずれた量: ずれ × 画面の速さ(中央値)。候補の中心のぶれ(±5px)を超えるフレームがあれば、
        # ゲームの時刻でつなぎ直す(検出テスト ファイナル 47s: 同じ絵が 2 枚続いた次の絵は 1.3 フレームぶんしか進まず、2 フレームぶん先を予測して
        # 29px 隣の三日月に乗り移り、速さ 21 の偽の軌跡が並んだ)。普通の動画はずれ sd 0.02〜0.05 フレームで、つなぎ直さない
        _vs = [abs(t_['vel']) for t_ in tracks if t_['color'] in ('don', 'ka') and len(t_['pts']) >= 10]
        track.relink = tclock is None and GD_ is not None and bool(_vs) and float(np.max(np.abs(GD_))) * float(np.median(_vs)) > 5.0
        if dbg:
            hv_ = _game_delta.has
            print('ゲームの時刻(音符全部から): 測れたフレーム %d、描画の乱れ %.3f フレーム、測りのぶれを超えるずれ %d 枚%s' % (int(hv_.sum()), _game_delta.jit, int(_game_delta.sig.sum()),
                  '(使わない)' if GD_ is None else '、ずれ sd %.3f フレーム' % float(GD_[hv_].std())))
    track.gdelta = GD_
    split_out = []
    for tr in tracks:
        pts = tr['pts']
        if tr['color'] not in ('don', 'ka') or len(pts) < 12: split_out.append(tr); continue
        fs = np.array([p[0] for p in pts], float); xs = np.array([p[1] for p in pts], float)
        if GD_ is not None: fs = fs + GD_[fs.astype(int)]
        a0, b0 = np.polyfit(fs, xs, 1)
        if (xs - (a0 * fs + b0)).std() <= 8.0: split_out.append(tr); continue
        n = len(pts); best = None
        samp = sorted(set(int(round(q)) for q in np.linspace(0, n - 1, min(n, 14))))
        for ii in range(len(samp)):
            for jj in range(ii + 1, len(samp)):
                i, j = samp[ii], samp[jj]
                if fs[j] - fs[i] < 4: continue
                a1 = (xs[j] - xs[i]) / (fs[j] - fs[i]); b1 = xs[i] - a1 * fs[i]
                inl = np.abs(a1 * fs + b1 - xs) <= 4.0
                if best is None or inl.sum() > best[0]: best = (int(inl.sum()), inl)
        if best is None or best[0] < 12: split_out.append(tr); continue
        inl = best[1]
        for _ in range(2):
            a1, b1 = np.polyfit(fs[inl], xs[inl], 1); inl = np.abs(a1 * fs + b1 - xs) <= 4.0
        if inl.sum() < 12 or xs[inl].max() - xs[inl].min() < 300: split_out.append(tr); continue
        if dbg: print('  ransac %s f%d-%d n=%d -> %d inliers vel %.1f (dropped %s)' % (tr['color'], f0 + pts[0][0], f0 + pts[-1][0], n, int(inl.sum()), -a1, ' '.join('f%d' % (f0 + p[0]) for p, k in zip(pts, inl) if not k)[:120]))
        t2 = dict(tr); t2['pts'] = [p for p, k in zip(pts, inl) if k]; t2['dropped'] = [p for p, k in zip(pts, inl) if not k]
        for key in ('big', 'ws', 'hs', 'hid'):
            if key in tr and len(tr[key]) == n: t2[key] = [v for v, k in zip(tr[key], inl) if k]
        ff, xf = t2['pts'][0]; fl, xl = t2['pts'][-1]
        if fl > ff: t2['vel'] = (xf - xl) / (fl - ff)
        split_out.append(t2)
    tracks = split_out
    onsets = []; lost_ends = []; resc = []
    # 録画の止まり: 候補がまったく同じフレームが 3 枚以上続く所(POLARiSNAUT 62.2〜62.6s: 20 フレーム同じ絵)。止まっている間もゲームは進むので、
    # 止まる直前まで見えていた音符が、止まりの後には帯の左端より先(判定枠の手前)まで進んでいて、帯の中に二度と現れないことがある
    frz_ = []; run_s = None; prev_sig = None
    for fi_, cs_ in enumerate(frames_c):
        sig_ = tuple(sorted((c['color'], int(round(c['cx']))) for c in cs_))
        if sig_ and sig_ == prev_sig:
            if run_s is None: run_s = fi_ - 1
        else:
            if run_s is not None and fi_ - run_s >= 3: frz_.append((run_s, fi_ - 1))
            run_s = None
        prev_sig = sig_

    def _frozen_bridge(fl_, a_, b_):
        # 最後の点が止まりの直前(2 フレーム以内)にあり、止まりが明けた時にはもう帯の左端(X0)より先まで進んでいるはずなら、消えたのではなく録画が止まっていた
        for s_, e_ in frz_:
            if 0 <= s_ - fl_ <= 2 and a_ * (e_ + 1) + b_ <= X0 + 2.0 * abs(a_): return True
        return False
    def reject(tr, why):
        _dbg_add(tr, 'reject', why)
        if dbg and len(tr['pts']) >= 5:
            pts = tr['pts']
            print('  reject(%s) %s f%d-%d x%.0f-%.0f n=%d vel=%.1f w_med=%.0f h_med=%.0f' % (why, tr['color'], f0 + pts[0][0], f0 + pts[-1][0], pts[0][1], pts[-1][1], len(pts), tr['vel'], np.median(tr.get('ws', [0])), np.median(tr.get('hs', [0]))))
    FR_ = getattr(track, 'frac', None)
    if FR_ is None and GD_ is not None: FR_ = GD_
    for tr in tracks:
        pts = tr['pts']
        if len(pts) < 3: continue
        fs = np.array([p[0] for p in pts], float); xs = np.array([p[1] for p in pts], float)
        if FR_ is not None: fs = fs + FR_[np.clip(fs.astype(int), 0, len(FR_) - 1)]     # 小数のゲーム時刻(V2T_FRAC=1)
        if xs[0] - xs[-1] < 60: reject(tr, 'span'); continue
        # 点の下限は 5。ただし帯(X0〜X1)を横切る間に写るフレーム数がそれより少ない速さでは、写れる数まで(チルノのパーフェクトさんすうタイム 133.90s:
        # 1 フレーム 258px の大カッは帯に 4 フレームしか写らない)。ただし 4 点未満は直線かどうか確かめられない(3 点では直線の当てはめに 1 点しか余らず、
        # 無関係な候補 3 つが 300px/フレームの幽霊になった: 彁 裏 69.85s・活声 100.80s)
        if len(pts) < max(4, min(5, int((X1 - X0) / max(abs(tr['vel']), 1.0)))) and tr['color'] in ('don', 'ka'): reject(tr, 'few'); continue
        if tr['color'] in ('don', 'ka') and len(pts) < 12 and abs(tr['vel']) >= 30 and np.median(tr.get('ws', [99])) < 35: reject(tr, 'slivers'); continue   # 速い流れの裏の遅い音符の覗きを繋いだ幽霊(遅い本物の欠けた音符は残す: Vrykolakas 374.4s)
        a, b = np.polyfit(fs, xs, 1)
        if a > -4:
            # 遅い音符(わら得る2000 102.03s の大ドン: 0.9px/フレームで 20 秒かけて流れる)は、帯の 1/3 以上をまっすぐ流れる。演出の偽物は移動 100px 以下
            r_s = xs - (a * fs + b)
            # 連打・風船・小節線も同じ(SCROLL の小さい区間: 1 フレーム 4px 未満のものを全部捨てていた)。時刻を「消えたフレーム」で出し直すのは
            # ドン・カッだけなので、印(slow)はドン・カッにだけ付ける
            if not (a < -0.3 and xs[0] - xs[-1] >= 0.33 * (X1 - X0) and r_s.std() <= 5.0):
                reject(tr, 'slow'); continue
            if tr['color'] in ('don', 'ka'): tr['slow'] = True
        # 左端付近まで観測されていること(途中で消える=演出/誤対応)
        # (連打/風船の頭は左端が切れた時点で出さなくなるので、その分(半径)だけ手前で消えてよい。
        #  小節線は x>=BAR_X0 でしか見ないので左端は BAR_X0。遅いスクロールでは 3.5V が BAR_X0-X0 より小さくなる)
        slack = RIN_B if tr['color'] in ('renda', 'balloon') else 0
        x_left = BAR_X0 if tr['color'] == 'bar' else X0
        long_track = len(pts) >= 20 and tr['color'] in ('don', 'ka', 'roll_end')      # 連打の尻尾も、叩かれて形が変わると x≈860 で見えなくなる(もぺもぺ 79.5s)
        ladder_head = tr['color'] in ('renda', 'balloon') and len(pts) >= 15      # 連打のはしごの頭は x≈900 で三日月が見えなくなる(resume:story)。風船も手前の音符に重なって x≈900 で見えなくなる(第六天魔王 裏 78.6s: 33 点あるのに extrap で捨てた)
        bridge_ = tr['color'] in ('don', 'ka') and len(pts) >= 8 and a < -4 and _frozen_bridge(pts[-1][0], a, b)
        if bridge_ and dbg: print('  frozen-bridge %s f%d-%d x%.0f-%.0f vel=%.1f(録画の止まりの間に判定枠の手前まで進んだ)' % (tr['color'], f0 + pts[0][0], f0 + pts[-1][0], xs[0], xs[-1], -a))
        # 帯の 6 割以上をまっすぐ(残差 4px 以下)流れてから判定枠の手前(x ≤ 1000)で見えなくなった軌跡も、演出ではなく判定枠の手前の叩いた演出(爆発)に
        # 入って形が崩れた音符(段位 達人 銀の黎明 144.76s・146.33s・146.53s: 速さ 43〜48 の大音符が x 1807 → 948 を 14〜17 点で流れ、直前の音符の
        # 爆発に入って捨てられた)。点 20 個の決まり(long_track)は、帯を 26 フレームで横切る速い音符には届かない
        # 見えていた間はほぼ毎フレーム写っていること(写った点が軌跡のフレーム数の 6 割以上。大音符は直線から外れて落とした点(dropped: 口の開け閉めや
        # 爆発で重心が揺れた点)も写っていた点に数える: 銀の黎明 146.53s の大ドンは 23 フレームに 14 点 + 外れた 7 点)。ほかの音符に隠れて飛び飛びに
        # しか写らない遅い音符の軌跡は形の崩れた重心で、採ると積み重なりの読み取り(ringtrack)の方が確かだった所を悪くした(検出テスト ファイナル 小節 11・12)
        # 外れた点を数えるのは大音符だけ(大音符の顔の揺れ)。小音符の外れた点は隣の音符を乗り継いだ点(検出テスト ファイナル 小節 8 の 22px 間隔の 64 分の
        # 壁で、乗り継ぎの軌跡がこの例外を通った)
        _fa, _fb = pts[0][0], pts[-1][0]
        _big_ = sum(tr['big']) * 2 >= len(tr['big'])
        n_seen_ = len(pts) + (sum(1 for q in tr.get('dropped', []) if _fa <= q[0] <= _fb) if _big_ else 0)
        span_ok_ = (tr['color'] in ('don', 'ka') and xs.max() - xs.min() >= 0.6 * (X1 - X0) and xs.min() <= 1000
                    and n_seen_ >= 0.6 * (_fb - _fa + 1)
                    and float(np.std(xs - (a * fs + b))) <= 4.0 and os.environ.get('V2T_VANISH_SPAN', '1') != '0')
        if span_ok_ and dbg and xs.min() > x_left + 3.5 * abs(a) + slack and not long_track:
            print('  vanish-span %s f%d-%d x%.0f-%.0f n=%d vel=%.1f(帯の 6 割以上をまっすぐ流れ、判定枠の手前で見えなくなった)' % (tr['color'], f0 + pts[0][0], f0 + pts[-1][0], xs.max(), xs.min(), len(pts), -a))
        if xs.min() > x_left + 3.5 * abs(a) + slack and not ((long_track or ladder_head or span_ok_) and xs.min() <= 1000) and not bridge_:
            if tr['color'] == 'roll_end' and len(pts) >= 4 and a < -3:      # 途中で見えなくなった連打の尻尾(終わりの無い連打の補完に使う)
                lost_ends.append((round((f0 + (JUDGE_X - b) / a) / fps, 4), -a, len(pts)))
            resc.append((tr, 'vanish')); reject(tr, 'vanish'); continue
        # 本物の音符は右端から判定枠まで流れる(移動距離 ≥ 1100px)。短い移動=判定枠付近の演出。
        # ただし速い連打棒/速い音符の裏から判定枠の直前で現れる音符(6点以上、判定枠のすぐ手前まで観測)は採る
        span = xs[0] - xs[-1]
        emerged = tr['color'] in ('don', 'ka') and len(pts) >= 6 and span >= 90 and xs.min() <= X0 + 2.0 * abs(a)
        if span < 0.33 * (X1 - X0) and emerged and len(pts) < 15:
            # 短い「現れた音符」は、直線にきれいに乗るものだけ(残差 sd ≤ 5px)。重なった音符(ドコドン)の脇から出る
            # 幽霊は 11〜12 点で残差 7px(夢と希望 おに 128.25s / 128.95s)
            r_ = xs - np.polyval(np.polyfit(fs, xs, 1), fs)
            if r_.std() > 5.0: resc.append((tr, 'short')); reject(tr, 'short-loose'); continue
            # 途中から現れた音符は、現れる直前(1〜4 フレーム前)にその予想位置を別の音符が覆っていたはず。覆うものが無いのは
            # レーンの光(カッを叩くと判定枠側が青く光る: Emma 裏 玄人 40.93s / 80.24s)などを音符と見たもの
            if os.environ.get('V2T_EMERGE_OCC', '1') != '0':
                f_first = int(fs.min()) if FR_ is None else int(pts[0][0])

                def _cov(fk, xp):
                    cs_ = frames_c[fk]
                    if any(abs(c['cx'] - xp) < 70 and c['color'] in ('don', 'ka', 'renda', 'balloon', 'roll_end') for c in cs_): return True
                    # 連打の棒の下(頭と終わりの丸の間)
                    return any(c['color'] in ('renda', 'balloon') and c['cx'] < xp and any(e['color'] == 'roll_end' and e['cx'] > xp for e in cs_) for c in cs_)
                occ = False
                for k_ in range(1, 5):
                    fk = int(pts[0][0]) - k_
                    if fk < 0: break
                    if _cov(fk, xs[0] + abs(a) * k_): occ = True; break
                if not occ: resc.append((tr, 'short')); reject(tr, 'short-noocc'); continue
        if span < 0.33 * (X1 - X0) and not emerged: resc.append((tr, 'short')); reject(tr, 'short'); continue
        fcross = (JUDGE_X - b) / a
        # 最終観測から判定枠までの所要フレーム(速度依存)+余裕 を超える外挿は捨てる
        # 連打のはしごの頭(15 点以上)も遠くから外挿してよい: 前の連打が叩かれ始める x≈900 で三日月が見えなくなる(resume:story)
        reach = (1000 - JUDGE_X) if (long_track or ladder_head or span_ok_) else (X0 + 60 - JUDGE_X)
        if (fcross < fs[-1] - 1 or fcross > fs[-1] + reach / abs(a) + 2) and not bridge_: reject(tr, 'extrap'); continue
        big = sum(tr['big']) * 2 >= len(tr['big'])
        note = ''
        if tr['color'] in ('don', 'ka') and not big:
            ws_ = np.array(tr.get('ws', [0] * len(pts)), float); cl = (ws_ >= 72) & (ws_ <= 80)
            if 12 <= cl.sum() < len(pts) and xs[cl].max() - xs[cl].min() >= 300:
                a_c, b_c = np.polyfit(fs[cl], xs[cl], 1)
                if a_c < -4:
                    fc_c = (JUDGE_X - b_c) / a_c
                    if abs(fc_c - fcross) >= 1.0 and fc_c >= fs[-1] - 1:
                        # 追い越した音符に右側を隠された点(幅 66-71 の部分顔・三日月)が線を傾けていた。素の顔の点だけで決める(彁 裏 54.4s のカッ)
                        note = ' clean-fit(%+.0fms, %d/%d pts)' % ((fc_c - fcross) / fps * 1000, int(cl.sum()), len(pts))
                        a, b, fcross = a_c, b_c, fc_c
        keep_ = None
        if tr['color'] in ('don', 'ka') and os.environ.get('V2T_REFINE', '1') != '0':
            ws_ = np.array(tr.get('ws', [0] * len(pts)), float)
            r2 = _refine_fit(fs, xs, ws_, big, a, b)
            if r2 is not None:
                fc2 = (JUDGE_X - r2[1]) / r2[0]
                # 除いた点(判定枠で消えていく大音符の細い点)の中心のずれは大小の半径の差(RIN_B − RIN_S = 19px)まで。直し直した直線が判定枠で
                # それより動くなら、除いた点と残した点は別の物(2 つの音符の点が混ざった軌跡)で、直しではない(一括試験 SAVAGE DELIGHT 43.66s:
                # 残差 5.9px の大ドンの軌跡が +69.7ms = 61px 動いて、本物の大ドンの 84ms 前の偽物として残った)
                if fc2 >= fs[-1] - 1 and abs(fc2 - fcross) * abs(r2[0]) <= RIN_B - RIN_S:
                    note += ' refine(%s %+.1fms)' % (r2[2], (fc2 - fcross) / fps * 1000); a, b, fcross = r2[0], r2[1], fc2; keep_ = r2[3]
        res_ = xs - (a * fs + b)
        # 直線に使わなかった点(判定枠で消えていく所・古い絵の線)は荒さに入れない(段位 resume:story 73.04s の大カッ: 入れると res_sd 5.6 で
        # 「連打の演出」として捨てられた)
        if keep_ is not None: res_ = res_[keep_]
        kink_tol = max(12.0, _JITTER_FR * abs(a))            # 速い音符は描画間隔の揺れ(±0.15 フレーム)だけで位置が 速さ×0.15 px 揺れる
        if GD_ is not None and os.environ.get('V2T_GC_KINK', '1') != '0':
            # 描画間隔の揺れをゲームの時刻で直した動画では、揺れの分の余地は要らない。残るのは候補の中心のぶれ(±5px)だけなので NEAR(6px)
            # (検出テスト ファイナル 53.8〜55.1s: 22px 間隔の同じ色の 64 分の壁で、本物の軌跡は残差 0.3〜3.7px、隣の音符を乗り継いだ偽の軌跡は 6.3〜11.3px
            # で速さ 16〜28。12px の許容では偽の軌跡が残り、その円を説明済みにして本物の直線を拾えなかった)
            kink_tol = max(6.0, 0.05 * abs(a))
        if tr['color'] in ('don', 'ka') and not big and len(pts) >= 8 and res_.std() > kink_tol:
            # 端の 1 点だけが外れている(判定枠の近くで別の物の点を最後に拾った: 〆ドレー2000 106.66s の 132px/フレームの小ドン、最後の点だけ +35px)
            # なら、その点を外して当て直す。2 つの音符を乗り継いだ幽霊は途中に段差ができるので、端を外しても直線に乗らない
            fixed = False
            for cut in (slice(0, -1), slice(1, None)):
                fs2, xs2 = fs[cut], xs[cut]
                a2, b2 = np.polyfit(fs2, xs2, 1)
                if a2 < -4 and (xs2 - (a2 * fs2 + b2)).std() <= kink_tol:
                    a, b = a2, b2; fcross = (JUDGE_X - b) / a; fixed = True
                    note += ' trim-end'
                    break
            if not fixed and os.environ.get('V2T_KINK_OUTL', '1') != '0' and len(fs) >= 8:
                # 離れた数点だけが外れている(隣の音符の候補を 1〜2 点拾った。同じフレームに 2 点あることも)なら、それを除いて当てはめ直す。
                # 2 つの音符を乗り継いだ幽霊は多くの点がまとまって段差を作るので、外れる点が 2 割を超える
                # (検出テスト ファイナル 121.2s: SCROLL 4 の 95px/フレームの小カッの 10 点のうち 1〜2 点が +23〜−27px、残りは 0〜6px)
                r0_ = xs - (a * fs + b); med_ = np.median(r0_); mad_ = 1.4826 * np.median(np.abs(r0_ - med_))
                ok_ = np.abs(r0_ - med_) <= max(3 * mad_, 6.0)
                if ok_.sum() >= max(6, int(np.ceil(0.8 * len(fs)))) and not ok_.all():
                    a4, b4 = np.polyfit(fs[ok_], xs[ok_], 1)
                    if a4 < -4 and (xs[ok_] - (a4 * fs[ok_] + b4)).std() <= kink_tol:
                        a, b = a4, b4; fcross = (JUDGE_X - b) / a; fixed = True; note += ' outlier(%d)' % int((~ok_).sum())
                        res_ = xs[ok_] - (a * fs[ok_] + b); cut = None
            FA_ = getattr(track, 'frac_alt', None)
            if not fixed and FA_ is not None and FR_ is None:
                # ゲームの描画間隔が乱れた所(重い譜面で処理落ちが続く: 検出テスト ファイナル 82.4s は流れが 60・46・75px と揺れ、同じ絵が 25ms 続いた
                # 次の絵は 64px しか進まない)では、音符の位置はフレーム番号ではなくゲームの時刻どおりに進む。レーンの流れから測ったフレームごとの
                # ゲームの時刻(prep60.frac_offsets)で当てはめ直して直線に乗るなら本物。描画の乱れはそのフレームの全部の音符に同じだけ効くので、
                # 2 つの音符を乗り継いだ幽霊の段差はこれでは消えない。普通の動画では流れの測りのぶれの方が大きい(全体に使うと 56 曲悪化: 2026-09-23)
                # ので、整数のフレームで直線に乗らないときだけ使う
                fsg_ = fs + FA_[np.clip(fs.astype(int), 0, len(FA_) - 1)]
                a3, b3 = np.polyfit(fsg_, xs, 1)
                r3_ = xs - (a3 * fsg_ + b3)
                if a3 < -4 and r3_.std() <= kink_tol:
                    a, b = a3, b3; fcross = (JUDGE_X - b) / a; fixed = True; note += ' game-clock'
                    res_ = r3_; cut = None
            if not fixed and os.environ.get('V2T_KINK_ISO', '1') != '0':
                # 乗り継ぎは近くに同じ色の別の候補があるときしか起きない。軌跡の全部のフレームで、その点から小音符の直径(2R = 100px)以内に同じ色の
                # 別の候補が無ければ、1 つの音符が描画の乱れで直線から外れたもの(検出テスト ファイナル 79.60s: 処理落ちで 1 フレームの進みが 62〜99px と揺れる
                # 76px/フレームの小カッ 14 点、残差 −45〜+32px。前後の音符は 150ms 離れていて、音符全部から測るゲームの時刻もここでは直せない)。
                # 64 分の壁の乗り継ぎ(隣が 22px)は近くに同じ色の候補があるので、これには当たらない
                iso_ = len(pts) >= 8 and all(not any(c_['color'] == tr['color'] and 2 < abs(c_['cx'] - p_[1]) < 100 for c_ in frames_c[int(p_[0])]) for p_ in pts)
                if iso_: fixed = True; note += ' isolated'
            if not fixed:
                if dbg: print('  kink 残差 ' + ' '.join('%d:%+.0f' % (int(f_), r_) for f_, r_ in zip(fs, xs - (a * fs + b))))
                reject(tr, 'kink'); continue          # 直線に乗らない(検証済み 6 曲では sd ≤ 9.2px): 2 つの音符を乗り継いだ幽霊
            if cut is not None: res_ = xs[cut] - (a * fs[cut] + b)
        onsets.append({'t': round((f0 + fcross) / fps, 4), 'color': tr['color'], 'big': bool(big),
                       'npts': len(pts), 'span': int(pts[-1][0] - pts[0][0]), 'vel': round(-a, 2), 'hidden': False, '_tr': tr,
                       'res_sd': round(float(res_.std()), 2)})     # 当てはめの荒さ(演出の偽トラックの判定に使う)
        if emerged and span < 0.33 * (X1 - X0): onsets[-1]['emerged'] = True   # 途中から現れた短い軌跡
        onsets[-1]['xr'] = (round(float(xs.max()), 1), round(float(xs.min()), 1))     # 軌跡が見えた x の範囲(右端, 左端)
        if tr.get('slow'): onsets[-1]['slow'] = True        # 時刻は後で「判定枠で叩かれて消えたフレーム」に直す(synth.slow_hits)
        if tr['color'] == 'balloon' and sum(tr.get('ku', [])) * 2 > len(pts): onsets[-1]['kusu'] = True     # くす玉(TJA の 9)
        _dbg_add(tr, 'accept', note.strip(), onsets[-1])
        if dbg: print('  accept %s t=%.3f vel=%.1f n=%d x%.0f-%.0f f%d-%d%s res_sd=%.1f res_max=%.1f%s' % (tr['color'], onsets[-1]['t'], -a, len(pts), xs[0], xs[-1], f0 + pts[0][0], f0 + pts[-1][0], ' B' if big else '', res_.std(), np.abs(res_).max(), note))
        if dbg and res_.std() > float(os.environ.get('DBG_PTS_SD', '8')): print('    pts(f,x,w,res): ' + ' '.join('%d,%.0f,%d,%+.0f' % (f0 + f, x, w_, rr) for (f, x), w_, rr in zip(pts, tr.get('ws', [0] * len(pts)), res_)))
    # 頭の無い連打の終わり: 連打の終わりの前には必ず頭がある。頭の軌跡が途中で切れて(棒に音符が重なると色の塊が割れ、頭の位置の測り方が跳ぶ)
    # 「短い」として捨てた頭のうち、終わりと同じ速さ(10% 以内)で、終わりより前(前の連打の終わりより後)に判定枠に着くものを、その連打の頭に戻す
    # (大勇者タツドン 108.39s: 10.4px/フレームの遅い連打の頭の最後の 31 点が短いとして捨てられ、頭の白い縁の円を ringtrack がドンとして足した)
    ends_ = sorted((o for o in onsets if o['color'] == 'roll_end' and o.get('vel')), key=lambda o: o['t'])
    # 終わりの頭とみなすのは、終わりの前にある連打の頭で、その間にドン・カッが無いもの(連打の途中に音符は置けない。風船には終わりの丸が無い)
    heads_t = [o['t'] for o in onsets if o['color'] == 'renda']
    dk_t = np.array(sorted(o['t'] for o in onsets if o['color'] in ('don', 'ka')))
    for k_, e_ in enumerate(ends_):
        pe_ = ends_[k_ - 1]['t'] if k_ > 0 else -1e9
        if any(pe_ < h < e_['t'] and not np.any((dk_t > h + 0.03) & (dk_t < e_['t'] - 0.03)) for h in heads_t): continue
        pe_ = max([pe_] + [float(x) for x in dk_t[dk_t < e_['t'] - 0.03]])      # 頭はこの終わりの直前の音符より後
        best_ = None
        for tr, why in resc:
            if why != 'short' or tr['color'] not in ('renda', 'balloon') or len(tr['pts']) < 5: continue
            fs_ = np.array([p[0] for p in tr['pts']], float); xs_ = np.array([p[1] for p in tr['pts']], float)
            a_, b_ = np.polyfit(fs_, xs_, 1)
            if a_ >= 0 or abs(-a_ / abs(e_['vel']) - 1) > 0.10: continue
            th_ = (f0 + (JUDGE_X - b_) / a_) / fps
            if not (pe_ < th_ < e_['t']): continue
            if best_ is None or th_ > best_[0]: best_ = (th_, tr, -a_)
        if best_ is not None:
            th_, tr, v_ = best_
            onsets.append({'t': round(th_, 4), 'color': tr['color'], 'big': bool(sum(tr.get('big', [])) * 2 > len(tr['pts'])), 'npts': len(tr['pts']),
                           'vel': round(v_, 2), 'hidden': False, '_tr': tr, 'head_for_end': True})
            heads_t.append(th_)
            if dbg: print('  headless-end %.3f → 捨てた頭 %s t=%.3f vel=%.1f n=%d を戻す' % (e_['t'], tr['color'], th_, v_, len(tr['pts'])))
    # 隠れて途中で消えた/途中から見えた音符の救済(彁 裏: 速いドンの流れの裏を流れる v11 のカッ、速いドンに追い越されるカッ)。
    # 条件: ドン/カッの小音符、12 点以上、素の顔(幅 ≥ 66)の点だけで当てた直線に乗る(残差 sd ≤ 4px)、
    # 見えていない区間(最終観測の後〜帯の左端、'short' なら最初の観測の前も)の予測位置に他の音符の候補が 7 割以上のフレームで重なる(=隠されていた)
    for tr, why in resc:
        pts = tr['pts']
        if len(pts) < 12 or tr['color'] not in ('don', 'ka'): continue
        if sum(tr['big']) * 2 >= len(tr['big']): continue          # 大音符は縁の行の推定に任せる
        fs = np.array([p[0] for p in pts], float); xs = np.array([p[1] for p in pts], float)
        ws = np.array(tr.get('ws', [0] * len(pts)), float); clean = (ws >= 72) & (ws <= 80)
        fs_c, xs_c = (fs[clean], xs[clean]) if clean.sum() >= 8 else (fs, xs)
        a, b = np.polyfit(fs_c, xs_c, 1); sd = float((xs_c - (a * fs_c + b)).std())
        if a > -4 or sd > 4.0:
            if dbg: print('  rescue-fail(%s) %s n=%d clean=%d vel=%.1f sd=%.1f' % (why, tr['color'], len(pts), int(clean.sum()), -a, sd))
            continue
        def covered(f_a, f_b):
            n = 0; hit = 0
            for f in range(int(np.ceil(f_a)), int(np.floor(f_b)) + 1):
                if f < 0 or f >= len(frames_c): continue
                xp = a * f + b; n += 1
                if any(abs(c['cx'] - xp) < 70 for c in frames_c[f] if not c.get('hidden') and not c.get('emerge')): hit += 1
            return (hit / n) if n else 1.0
        f_end = (X0 + 40 - b) / a; f_beg = (X1 - 100 - b) / a
        cov_after = covered(fs[-1] + 1, f_end) if f_end > fs[-1] + 1 else 1.0
        cov_before = covered(f_beg, fs[0] - 1) if (why == 'short' and fs[0] - 1 > f_beg) else 1.0
        fcross = (JUDGE_X - b) / a
        # 隠した物が音符の候補でなくても(割られる風船の大きな演出・画面の光)、20 点以上がきれいな直線(sd ≤ 2.5px)に乗り、前後 1 秒の採った音符と
        # 同じ速さ(3% 以内)なら本物の音符の軌跡(一括試験 束ね糸 96.19s: 30 点・x1851〜1073 で、判定枠で割られている風船の演出の下に消えたドン)
        t_x_ = (f0 + fcross) / fps
        vv_ = [abs(o['vel']) for o in onsets if o['color'] in ('don', 'ka') and o.get('vel') and abs(o['t'] - t_x_) < 1.0]
        strong_ = (os.environ.get('V2T_RESC_STRONG', '1') != '0' and len(pts) >= 20 and sd <= 2.5 and fcross >= fs[-1] - 1 and len(vv_) >= 2
                   and abs(-a / float(np.median(vv_)) - 1) <= 0.03
                   and not any(o['color'] == tr['color'] and abs(o['t'] - t_x_) * fps < 2.0 for o in onsets))
        if (cov_after < 0.7 or cov_before < 0.7 or fcross < fs[-1] - 1) and not strong_:
            if dbg: print('  rescue-fail(%s) %s n=%d vel=%.1f cov %.2f/%.2f' % (why, tr['color'], len(pts), -a, cov_before, cov_after))
            continue
        t_r = (f0 + fcross) / fps; win_r = 60.0 / abs(a) / fps
        # 重複の窓は「近くの同色の音符の最小間隔の半分」を超えない(Xa: BPM290 の 16 分は 52ms = 57px 間隔なので、
        # 60px 固定の窓だと隣の本物の音符を「同じ音符の断片」と誤って捨てる)
        # 間隔は速さが近い(±30%)音符だけで測る。別の層(ドンカマ 121.5s の 47px/frame の速いドン、39ms 間隔)を混ぜると
        # 窓が狭くなりすぎ、遅い層(6px/frame)の同じ音符の断片を別の音符として採ってしまう
        near = sorted(o['t'] for o in onsets if not o.get('rescued') and o['color'] == tr['color'] and abs(o['t'] - t_r) < 1.5
                      and abs(abs(o['vel']) / abs(a) - 1) < 0.3)
        # 25ms 未満の間隔はそれ自体が重複(同じ音符の 2 本のトラック。ドンカマ 120.989 と 121.001)なので使わない
        gaps = [b_ - a_ for a_, b_ in zip(near, near[1:]) if b_ - a_ > 0.025]
        if gaps: win_r = min(win_r, 0.55 * min(gaps))
        dup = [o for o in onsets if not o.get('rescued') and o['color'] == tr['color'] and abs(o['t'] - t_r) < win_r]
        if dup:      # 同色の通常音符が近く(60px ぶんの時刻)にある = 追い越されて偏った同じ音符の断片
            d0 = min(dup, key=lambda o: abs(o['t'] - t_r))
            if d0.get('res_sd', 0) > 5.0 and sd <= 3.0:
                # 既存の方が荒く(断片をつないだ長いトラック)、救済の方が綺麗なら入れ替える
                # (ドンカマ 120.19s: 既存は 187 フレームにわたる断片のつなぎで res_sd 6.7、救済は判定枠の手前の 39 点で sd 2.4。
                #  本家の音符は救済の時刻 120.244 と 2ms で一致、既存は 53ms 早い偽物だった)
                if dbg: print('  rescue-replace(%s) %s t=%.3f (sd %.1f) が %.3f (res_sd %.1f) を置き換え' % (why, tr['color'], t_r, sd, d0['t'], d0['res_sd']))
                _dbg_mark(d0, 'replaced-by-rescue'); onsets.remove(d0)
            else:
                if dbg: print('  rescue-fail(%s) %s n=%d vel=%.1f t=%.3f dup-of %.3f' % (why, tr['color'], len(pts), -a, t_r, dup[0]['t']))
                continue
        # 速度の階級: ±4 秒の同色の通常音符(20 点以上)の速度のどれかと 6% 以内で合うこと(追い越しで偏った断片は速度が 8-10% ずれる)
        cls = [abs(o['vel']) for o in onsets if not o.get('rescued') and o['color'] == tr['color'] and o['npts'] >= 20 and abs(o['t'] - t_r) < 4.0]
        if cls and not any(abs(abs(a) / v - 1) <= 0.06 for v in cls):
            if dbg: print('  rescue-fail(%s) %s n=%d vel=%.1f t=%.3f off-class (%s)' % (why, tr['color'], len(pts), -a, t_r, ' '.join('%.1f' % v for v in sorted(set(round(v, 1) for v in cls))[:8])))
            continue
        onsets.append({'t': round((f0 + fcross) / fps, 4), 'color': tr['color'], 'big': False, 'npts': len(pts), 'vel': round(-a, 2), 'hidden': False, 'rescued': True, '_tr': tr,
                       'xr': (round(float(max(p[1] for p in pts)), 1), round(float(min(p[1] for p in pts)), 1))})
        _dbg_add(tr, 'rescue', why, onsets[-1])
        if dbg:
            print('  rescue(%s) %s t=%.3f vel=%.1f n=%d clean=%d sd=%.1f cov %.2f/%.2f x%.0f-%.0f' % (why, tr['color'], onsets[-1]['t'], -a, len(pts), int(clean.sum()), sd, cov_before, cov_after, xs[0], xs[-1]))
            print('    pts(f,x,w): ' + ' '.join('%d,%.0f,%d' % (f0 + f, x, w_) for (f, x), w_ in zip(pts, ws)))
    onsets.sort(key=lambda o: o['t'])
    # ジグザグ偽トラックの除去: 等間隔に並んだ音符の「隣の音符の次の位置」を辿った偽トラック(速度=V+間隔)は、
    # 各点が他の(本物の)トラックの予測位置と一致する。本物の速い音符(SCROLL大)は他トラックの予測とは重ならない
    def predicted_at(tr, f):
        if '_ab' not in tr:                      # トラックごとの直線は 1 回だけ当てる
            fs = np.array([p[0] for p in tr['pts']], float); xs = np.array([p[1] for p in tr['pts']], float)
            tr['_ab'] = tuple(np.polyfit(fs, xs, 1))
        a, b = tr['_ab']; return a * f + b
    keep_on = []
    O_t = np.array([p['t'] for p in onsets], float); O_c = np.array([p['color'] for p in onsets]); O_v = np.abs(np.array([p['vel'] for p in onsets], float))
    for i_o, o in enumerate(onsets):
        tr = o['_tr']
        vel = abs(o['vel'])
        m_o = (O_c == o['color']) & (np.abs(O_t - o['t']) < 1.5) & (O_v < 0.6 * vel); m_o[i_o] = False     # 1 つずつ全部と比べると 1 曲 2 秒
        others = [onsets[k] for k in np.flatnonzero(m_o)]
        if others:
            hit = 0
            for f, x in tr['pts']:
                if any(abs(predicted_at(p['_tr'], f) - x) < 8 for p in others): hit += 1
            if hit >= 0.8 * len(tr['pts']):
                if dbg: print('  reject(zigzag) %s t=%.3f vel=%.1f n=%d' % (o['color'], o['t'], vel, len(tr['pts'])))
                _dbg_mark(o, 'zigzag'); continue
        keep_on.append(o)
    onsets = keep_on
    # 点の帰属: 1 つの候補点は 1 つの物体にしか属さない。ドン/カッの軌跡の点の 7 割以上が、先に採った別の軌跡(色は問わない)の
    # その時刻の予測位置(当てはめた直線)から 12px 以内にあるなら、それは別の物体の点を拾い直した重複(隣り合う 2 つのカッを乗り継いだ幽霊:
    # POLARiSNAUT 44.14s、叩いて赤くなった連打の終わりの丸をドンとして追ったもの: Emma 裏 17.37s)。当てはめのきれいな軌跡(残差の小さい順)から採る。
    # 上のジグザグの除去(遅い音符の予測位置に限る)を一般にしたもの
    OWN_TOL = 12.0      # 同じ物体とみなす位置の差(部分的に隠れた音符の中心の推定は ±10px ほどぶれる)
    if os.environ.get('V2T_OWN', '1') != '0':
        lines_ = []; drop_ = []
        LA_ = np.empty(2 * len(onsets) + 2); LB_ = np.empty(2 * len(onsets) + 2)     # lines_ の傾き・切片(配列で持つ)
        LK_ = np.zeros(2 * len(onsets) + 2, bool)          # 連打の終わりの丸の直線(ドンの点だけを説明する)
        for o in sorted((o for o in onsets if o.get('_tr')), key=lambda o: (o.get('res_sd', 99.0), -o['npts'])):
            pts_ = o['_tr']['pts']
            fs_ = np.array([p[0] for p in pts_], float); xs_ = np.array([p[1] for p in pts_], float)
            if o['color'] in ('don', 'ka') and not o.get('stationary') and len(pts_) >= 3 and lines_:
                LA = LA_[:len(lines_)]; LB = LB_[:len(lines_)]
                if o['color'] == 'ka':
                    # 連打の終わりの丸は叩くと赤くなってドンに見えるが、カッの青には見えない(バブル・リップ・レイヴ・マシン 67.80s: 連打の終わりの直後の
                    # カッを丸の重複として消した)
                    keep_ = ~LK_[:len(lines_)]; LA = LA[keep_]; LB = LB[keep_]
                pred_ = LA[:, None] * fs_[None, :] + LB[:, None]     # 音符は画面に入ってから判定枠まで同じ速さで動くので、予測位置は画面の中ならどの時刻でも使える
                hit = ((pred_ >= JUDGE_X - 60) & (pred_ <= X1 + 60) & (np.abs(pred_ - xs_[None, :]) <= OWN_TOL)).any(0)   # 全部の直線をまとめて
                if hit.mean() >= 0.7:
                    drop_.append(o); continue
            if len(pts_) >= 3:
                a_, b_ = np.polyfit(fs_, xs_, 1); LA_[len(lines_)] = a_; LB_[len(lines_)] = b_; lines_.append((a_, b_, fs_.min(), fs_.max()))
                # 連打の終わりの基準点は「棒の右端 − 2 半径」、その丸の中心は 1 半径右(叩くと赤くなりドンの候補になる)
                if o['color'] == 'roll_end': LA_[len(lines_)] = a_; LB_[len(lines_)] = b_ + RIN_S; LK_[len(lines_)] = True; lines_.append((a_, b_ + RIN_S, fs_.min(), fs_.max()))
        # 覗きの幽霊: 大音符が遅い音符の列を追い越すとき、大音符のすぐ後ろから下の遅い音符の端が次々に覗く。覗く三日月は大音符の縁に沿って動くので、
        # 別々の遅い音符の端がつながって、大音符の後ろに接した軌跡になる(Calculator 47.35・47.74・47.90・48.10s: 大音符の 35〜75px 後ろの小ドン)。
        # 欠けて細い(幅の中央値が小音符の内円の半径未満。丸ごと見える小音符は約 75px)
        if os.environ.get('V2T_PEEK', '1') != '0':
            for o in onsets:
                if o.get('big') or o['color'] not in ('don', 'ka') or not o.get('vel') or not o.get('_tr') or any(o is d_ for d_ in drop_): continue
                tr_ = o['_tr']; pts_ = tr_['pts']; ws_ = tr_.get('ws') or []
                if len(pts_) < 3 or len(ws_) != len(pts_): continue
                # 覗きは追い越しの間だけ起きる別々の音符の端のつながり。帯のほぼ全体(8 割以上)を横切って直線にきれいに乗る(残差 ≤ NEAR/2)軌跡は、
                # 大音符の後ろを同じ速さで流れる本物の小音符(検出テスト ファイナル 116.29s: 大ドンの 53px 後ろの小ドン、115 点・残差 0.8px)
                xs_all_ = [p[1] for p in pts_]
                if (o.get('res_sd', 99.0) <= __import__('ringtrack').NEAR / 2 and max(xs_all_) - min(xs_all_) >= 0.8 * (X1 - X0)
                        and os.environ.get('V2T_PEEK_FULL', '1') != '0'): continue
                # 点ごとに見る(軌跡の速さでは比べない。別々の三日月をつないだ軌跡の速さは混ざってずれる: Calculator 48.08s は大音符 16.4 に対して 14.1)
                pk_, pf_ = _peek_behind([p[0] for p in pts_], [p[1] for p in pts_], ws_, [q for q in onsets if not any(q is d_ for d_ in drop_)], fps, f0)
                if pk_:
                    if dbg: print('  reject(peek) %s t=%.3f n=%d w %.0f(大音符の後ろ %.0f%%)' % (o['color'], o['t'], len(pts_), np.median(ws_), 100 * pf_))
                    drop_.append(o); continue
                if os.environ.get('V2T_PEEK_SMALL', '1') != '0':
                    pk_, pf_ = _peek_small([p[0] for p in pts_], [p[1] for p in pts_], ws_, o, [q for q in onsets if not any(q is d_ for d_ in drop_)], fps, f0)
                    # 捨てずに疑いの印を付け、ringtrack で自分の縁が見えたときだけ残す(捨てるとその点が空き、下の遅い音符の縁と合わせて偽の直線ができた:
                    # チャーリーダッシュ! 17.59s)
                    if pk_:
                        if dbg: print('  suspect(peek-small) %s t=%.3f vel=%.1f n=%d w %.0f(速い音符の後ろで遅い %s の上 %.0f%%)' % (
                            o['color'], o['t'], abs(o['vel']), len(pts_), np.median(ws_), o['color'], 100 * pf_))
                        o['ghost_suspect'] = True; o['peek_suspect'] = True
        # 重なりの重心の幽霊: 速さの違う同じ色の音符が重なって追い越すとき、つながった色の塊の重心はどちらとも違う速さで動く(Turquoise Tachometer
        # 24.20s: 速さ 27.6 のドンが 12.9 のドンの列を追い越す所で、速さ 15.9 のドン)。点の 7 割以上が同じ色で速さの違う採った音符と重なっていて
        # (中心の差が小音符の内円の直径未満)、軌跡の速さが重なっているどの音符とも 10% 以上違い、点が直線にきれいに乗らないなら、その重心
        if os.environ.get('V2T_BLEND', '1') != '0':
            live_ = [q for q in onsets if q['color'] in ('don', 'ka') and q.get('vel') and not any(q is d_ for d_ in drop_)]
            for o in onsets:
                if o.get('big') or o['color'] not in ('don', 'ka') or not o.get('vel') or not o.get('_tr') or any(o is d_ for d_ in drop_): continue
                # 重なり方が変わるので重心は直線に乗りきらない。当てはめのきれいな軌跡(残差 NEAR/2 以下)は 1 つの円(λ7708 73.21s: 残差 0.3px の本物のドンを、
                # 重なっていた幽霊の速さが違うという理由で外した)
                if o.get('res_sd', 0.0) <= __import__('ringtrack').NEAR / 2: continue
                pts_ = o['_tr']['pts']
                if len(pts_) < 3: continue
                v_ = abs(o['vel'])
                oth_ = [q for q in live_ if q is not o and q['color'] == o['color'] and not q.get('big') and abs(abs(q['vel']) / v_ - 1) > 0.10]
                if not oth_: continue
                fs_ = np.array([p[0] for p in pts_], float); xs_ = np.array([p[1] for p in pts_], float)
                A_ = np.array([-abs(q['vel']) for q in oth_]); B_ = np.array([JUDGE_X + abs(q['vel']) * (q['t'] * fps - f0) for q in oth_])
                ov_ = (np.abs(A_[:, None] * fs_[None, :] + B_[:, None] - xs_[None, :]) < 2 * RIN_S)
                if ov_.any(0).mean() < 0.7: continue
                if dbg: print('  reject(blend) %s t=%.3f vel=%.1f n=%d(速さの違う %s と重なる点 %.0f%%)' % (o['color'], o['t'], v_, len(pts_),
                              ' '.join('%.3f/%.1f' % (oth_[k]['t'], abs(oth_[k]['vel'])) for k in np.flatnonzero(ov_.any(1))[:3]), 100 * ov_.any(0).mean()))
                drop_.append(o)
        if drop_:
            if dbg: print('  owned-dup ' + ' '.join('%s %.3f' % (o['color'], o['t']) for o in drop_))
            onsets = [o for o in onsets if not any(o is d_ for d_ in drop_)]
    # 軌跡の色の塊の幅の中央値(ringtrack が「同じ速さで重なった組」を探す手がかり: 重なった組の後ろの音符は細い三日月の軌跡になる)
    for o in onsets:
        ws_ = (o.get('_tr') or {}).get('ws') or []
        if ws_: o['wmed'] = float(np.median(ws_))
    for o in onsets: o.pop('_tr', None)
    onsets.sort(key=lambda o: o['t'])
    merged = list(onsets)      # 同色の重複の間引きは隠れ大音符の置き換えの後で行う(偏った大音符候補が 20ms 以内の小音符を間引いていた: 彁 裏 138.637)
    # 隠れ大音符(速い小音符の裏の遅い大音符)。縁の行の候補は上を流れる小音符の隙間から覗く部分なので、
    # 隙間と一緒に動いては跳び直す鋸歯(±30px)になる。真の軌跡は鋸歯の中心を通る直線なので、
    # 直線ハフ変換(速度 3-40px/frame × 切片)で点をまとめてから最小二乗で当てはめる(点が多いので中心は数 px で決まる)
    for color, pts_h in hidden_pts.items():
        pts_h = sorted(pts_h)
        FS_h = np.array([p[0] for p in pts_h], float); XS_h = np.array([p[1] for p in pts_h], float)     # 残りの点(配列で持って間引く)
        v_hi_ = None
        while len(FS_h) >= 12:
            fs_h = FS_h; xs_h = XS_h
            f_ref = fs_h.min(); best = (0, None, None)
            # 速さの範囲: 上限は曲で測れた音符の最速の 1.2 倍(40px/フレーム固定だと、それより速い隠れた大音符が一切出なかった)
            if v_hi_ is None: v_hi_ = max(40.0, 1.2 * max([abs(o['vel']) for o in merged if o['color'] in ('don', 'ka') and o.get('vel')] or [0.0]))
            v_hi = v_hi_                 # merged が変わったときだけ数え直す(下)
            for s in np.arange(0.5, v_hi, 0.5):      # 下限は決めない(止まっている物は下の「300px 以上・15 フレーム以上」で落ちる。Doppelgangers 裏 95.40s の BPM 52 の大ドンは 2.6px/フレーム)
                x0 = xs_h + s * (fs_h - f_ref)
                edges = np.arange(x0.min() - 16, x0.max() + 33, 16)
                # np.histogram(x0, bins=edges) と同じ数え方(edges[i] <= x < edges[i+1]、最後の箱は右端を含む)を、箱の番号の計算で
                # (配列の edges を渡すと二分探索になって 1 曲 6 秒)
                bi = np.clip(((x0 - edges[0]) // 16).astype(int), 0, len(edges) - 2)
                bi -= (x0 < edges[bi]); bi += (bi < len(edges) - 2) & (x0 >= edges[np.minimum(bi + 1, len(edges) - 1)])
                hist = np.bincount(bi, minlength=len(edges) - 1)
                if hist.size == 0: continue
                k = int(np.argmax(hist)); v = int(hist[k])
                if v > best[0]: best = (v, s, (edges[k] + edges[k + 1]) / 2)
            if best[0] < 12: break
            s, x0c = best[1], best[2]
            inl = np.abs(xs_h + s * (fs_h - f_ref) - x0c) <= 45
            for _ in range(2):
                a, b = np.polyfit(fs_h[inl], xs_h[inl], 1)
                inl = np.abs(a * fs_h + b - xs_h) <= 45
            n_in = int(inl.sum())
            if n_in < 12: break
            for _ in range(2):                                       # 縁の行の点は直線に ±2px で乗る: 内点を ±10px に絞って当て直す
                a, b = np.polyfit(fs_h[inl], xs_h[inl], 1)
                tight = inl & (np.abs(a * fs_h + b - xs_h) <= 10)
                if tight.sum() < 12: break
                inl = tight
            n_in = int(inl.sum())
            fi_in, xi_in = fs_h[inl], xs_h[inl]
            span_ok = (xi_in.max() - xi_in.min()) >= 300 and (fi_in.max() - fi_in.min()) >= 15
            a, b = np.polyfit(fi_in, xi_in, 1)
            fcross = (JUDGE_X - b) / a if a < 0 else None
            reach_h = (1000 - JUDGE_X) if n_in >= 20 else (X0 + 60 - JUDGE_X)
            ok = span_ok and a < 0 and fcross is not None and fi_in.max() - 1 <= fcross <= fi_in.max() + reach_h / abs(a) + 2
            if ok:
                # 上に何も重なっていないフレーム(譜面で先の音符・連打の棒が離れている)では、大音符は普通に見えて大音符の候補になるはず。
                # 見えるはずのフレームで見えなかった数が五分五分では 5% 未満でしか起きないほど多ければ、散らばった縁の点に引けた偽の直線
                # (速さの下限を外したら 0.2〜0.9px/フレームの偽の直線ができた: ドンカマ2000 138.96s・笑える 102.10s)
                t_c = (f0 + fcross) / fps
                cov_ = [(q['t'], abs(q['vel']), RIN_B + 25 if q.get('big') else RIN_S + 10) for q in merged
                        if q['color'] in ('don', 'ka') and q.get('vel') and q['t'] < t_c - 1e-6]
                # フレームごとに全部の音符と比べるので配列にしておく(1 つずつ比べると 1 曲 2200 万回・8 秒。計算の順は同じなので結果も同じ)
                cT_ = np.array([c_[0] for c_ in cov_], float); cV_ = np.array([c_[1] for c_ in cov_], float); cR_ = np.array([c_[2] for c_ in cov_], float)
                rl_ = []; h_ = None
                for q in sorted(merged, key=lambda q: q['t']):
                    if q['color'] in ('renda', 'balloon') and q.get('vel'): h_ = q
                    elif q['color'] == 'roll_end' and h_ is not None: rl_.append((h_['t'], q['t'], abs(h_['vel']))); h_ = None
                n_exp = n_hit = 0
                for f_ in range(int(np.ceil(fi_in.min())), int(np.floor(fcross)) + 1):
                    if not (0 <= f_ < len(frames_c)): continue
                    x_ = a * f_ + b
                    if not (X0 + RIN_B <= x_ <= X1 - RIN_B): continue
                    if len(cov_) and np.any(np.abs(JUDGE_X + (cT_ * fps - f0 - f_) * cV_ - x_) < cR_ + RIN_B + 25): continue
                    if any(ta < t_c and JUDGE_X + (ta * fps - f0 - f_) * vr - RIN_B <= x_ <= JUDGE_X + (tb * fps - f0 - f_) * vr + 2 * RIN_B for ta, tb, vr in rl_): continue
                    n_exp += 1
                    if any(c['color'] == color and c.get('big') and not c.get('hidden') and not c.get('emerge') and abs(c['cx'] - x_) <= 12
                           for c in frames_c[f_]): n_hit += 1
                from math import comb as _comb
                # 五分五分で n_hit 以下になる確率 < 5%(整数で比べる: 2 の n 乗は float にすると桁あふれする)
                if n_exp >= 5 and 20 * sum(_comb(n_exp, i) for i in range(0, n_hit + 1)) < 2 ** n_exp:
                    if dbg: print('  reject(hidden-unseen) %s t=%.3f vel=%.1f n=%d: 見えるはずのフレーム %d、見えた %d' % (color, t_c, -a, n_in, n_exp, n_hit))
                    FS_h = FS_h[~inl]; XS_h = XS_h[~inl]; continue
                t = round((f0 + fcross) / fps, 4)
                slow = -a < 12.0
                win = 0.25 if slow else 0.08
                # 同じ音符は同じ速さで動く: 速さが 10% 以上違うものは、時刻が近くても別の音符(六兆年と一夜物語 132.28s: 速さ 12.4 の大ドンを、0.23 秒後の
                # 速さ 10.8 の遅い大ドンの縁の直線が置き換えて消していた)
                dups = [o for o in merged if o['color'] == color and o.get('big') and abs(o['t'] - t) < win
                        and not (o.get('vel') and abs(abs(o['vel']) / -a - 1) > 0.10)
                        # 縁の行からの推定どうし(同じ測り方で偏りも同じ)で、どちらもよく測れた(20 点以上・速さ 3% 以内)なら、判定枠に着く位置が 24px 以内の
                        # ものだけが同じ音符(遅い大音符の窓 0.25 秒は速さ 11.8 で 177px になり、120px 離れた別の大ドンを置き換えて消した: 検出テスト ファイナル
                        # 121.43s)。通常の追跡との比べは窓のまま(縁の行の推定は通常の追跡と 25px ほどずれることがある: 段位 活声 48.79s。窓が広いのは
                        # 判定枠の近くの数点だけの粗い推定のため: VIGVANGS 裏 76.39s)
                        # 通常の追跡も、画面にいるフレームの 6 割以上で点が取れていればよく測れた推定(検出テスト ファイナル 114.13s: 148 点 / 184 フレームの
                        # 大ドンを、186ms 前の大ドンの縁の行の推定が置き換えた。段位 活声 48.79s の 26 点 / 48 フレームの追跡は偏った粗い推定で、窓でまとめる)
                        # 幅は、縁の行どうし(同じ測り方で偏りも同じ)は 24px、縁の行と通常の追跡は大音符の半径の半分(42px): 縁の行の推定は通常の追跡と
                        # 25px ほどずれることがある(亜空間遊泳 70.65s: 23 点の縁の行の推定と 51 点の追跡が 25px ずれて二重になった)。別の大音符は 70px 以上離れていた
                        and not ((o.get('hidden') and (o.get('npts') or 0) >= 20 or (o.get('npts') or 0) >= 0.6 * (X1 - X0) / max(abs(o.get('vel') or 1.0), 1.0))
                                 and n_in >= 20 and o.get('vel') and abs(abs(o['vel']) / -a - 1) <= 0.03
                                 and abs(o['t'] - t) * fps * (-a) >= (24.0 if o.get('hidden') else R_B_OUT / 2))]
                # 1 フレームの動きが候補の位置のぶれ(±5px)より小さい遅さでは、位置の小さな誤差が大きな時刻の誤差になる(0.2px/フレームなら 3px で 0.28 秒:
                # ドンカマ2000 138.96s、笑える 102.10s)。追跡の 1 点目と 2 点目の下限と同じ 5px/フレーム未満の直線は、ほかの推定(止まっている音符・
                # 通常追跡など)が無いときだけ足し、置き換えない(Doppelgangers 裏 95.40s の BPM 52 の大ドン 2.6px/フレームは、ほかの推定が無い)
                if -a < 5.0:
                    same_ = [o for o in merged if o['color'] == color and o['t'] < 1e9 and abs(o['t'] - t) * max(-a, abs(o.get('vel') or 0.0)) * fps < 84.0]
                    if same_:
                        if dbg: print('  hidden-slow-skip %s t=%.3f vel=%.1f(ほかの推定 %s)' % (color, t, -a, ' '.join('%.3f' % o['t'] for o in same_)))
                        FS_h = FS_h[~inl]; XS_h = XS_h[~inl]; continue
                    dups = []
                if dbg: print('  hidden %s t=%.3f vel=%.1f n=%d span=%.0f 見えるはず %d 見えた %d%s' % (color, t, -a, n_in, xi_in.max() - xi_in.min(), n_exp, n_hit, (' (replaces %d: %s)' % (len(dups), ' '.join('%.3f/v%.1f/n%d' % (o['t'], abs(o.get('vel') or 0), o.get('npts', 0)) for o in dups))) if dups else ''))
                # 同じ音符に 2 つの推定があるなら、点の多いほうを採る(VIGVANGS 裏 76.39s: 通常の軌跡は判定枠の近くの 9 点で速さ 16.3、
                # 縁の行からの推定は 58 点・1004px で 17.8。通常の方を残すと判定枠から離れた所の位置が 50px ずれる)
                if slow or not dups or all(n_in > o.get('npts', 0) for o in dups):
                    # 遅い大音符は通常追跡の候補が小音符の重なりで 50-150ms 偏るので、縁の行からの推定で置き換える
                    for o in dups: merged.remove(o); _dbg_mark(o, 'replaced-by-hidden')
                    v_hi_ = None
                    # 通常追跡の音符を置き換えたものは「通常の音符があった場所」なので、下の範囲の判定では通常扱いにする
                    merged.append({'t': t, 'color': color, 'big': True, 'npts': n_in, 'vel': round(-a, 2), 'hidden': True, 'replaced': bool(dups)})
                    _dbg_add(None, 'hidden', '', merged[-1], pts=[(int(f_), float(x_)) for f_, x_ in zip(fi_in, xi_in)])
            elif dbg: print('  reject(hidden) %s n=%d span=%.0f vel=%.1f' % (color, n_in, xi_in.max() - xi_in.min(), -a))
            FS_h = FS_h[~inl]; XS_h = XS_h[~inl]
    # 遅い(v<12)大音符の通常追跡で、縁の行の推定(hidden)が ±0.25s に無いのに ±3s には有るもの = 小音符と合体した run から出た幽霊
    # (本物の遅い大音符なら縁の行の推定が必ず付く)
    hid_ts = {}
    for o in merged:
        if o.get('hidden'): hid_ts.setdefault(o['color'], []).append(o['t'])

    def _slower_than_near(o, ms):
        """周りの音符(前後 3 秒の通常追跡のドン・カッ)の速さの中央値より 5% 以上遅い(上を追い越される遅い層)"""
        vs_ = [abs(q['vel']) for q in ms if q is not o and q['color'] in ('don', 'ka') and not q.get('hidden') and q.get('vel') and abs(q['t'] - o['t']) < 3.0]
        return bool(vs_) and abs(o['vel']) < 0.95 * float(np.median(vs_))
    kept = []
    for o in merged:
        if not o.get('hidden') and o['big'] and o['color'] in ('don', 'ka') and abs(o['vel']) < 12.0:
            hs = [t for t in hid_ts.get(o['color'], []) if abs(t - o['t']) < 3.0]
            v_h = [abs(q['vel']) for q in merged if q.get('hidden') and q['color'] == o['color'] and abs(q['t'] - o['t']) < 3.0]
            off_class = bool(v_h) and abs(abs(o['vel']) / float(np.median(v_h)) - 1) > 0.05
            if hs and off_class and not any(abs(t - o['t']) < 0.25 for t in hs):
                # 捨てずに疑いの印を付け、ringtrack の縁の確かめ(弱い音符と同じ。大音符の大きさの白い縁が道筋に沿って見えるか)に任せる。
                # 速さの層が 3 秒の中に 3 つ並ぶ譜面では、別の層の縁の推定の速さと比べて本物を捨てた(検出テスト ファイナル 114.03〜114.22s: 6.3px/フレームの
                # 大音符 143〜161 点・残差 0.6px を、15.7 と 9.5 の層の縁の推定と比べて幽霊とした。ringtrack は縁を 85 / 85 フレームで確かめていた)
                if dbg: print('  suspect(slow-big-ghost) %s t=%.3f vel=%.1f n=%d' % (o['color'], o['t'], abs(o['vel']), o['npts']))
                o['ghost_suspect'] = True
        kept.append(o)
    merged = kept
    # 同色の重複(同じ音符に 2 本の追跡)を間引く。隠れ大音符(縁の行の推定)は対象外
    ded = []; last_i = -1
    for o in sorted(merged, key=lambda o: o['t']):
        if o.get('hidden'): ded.append(o); continue
        win = max(0.02, 16.0 / max(abs(o['vel']), 1.0) / fps)      # 16px ぶん(遅い音符ほど広い): v6 → 44ms, v25 → 20ms
        if last_i >= 0 and abs(o['t'] - ded[last_i]['t']) < win and o['color'] == ded[last_i]['color']:
            if o['npts'] > ded[last_i]['npts']: _dbg_mark(ded[last_i], 'dup'); ded[last_i] = o
            else: _dbg_mark(o, 'dup')
            continue
        ded.append(o); last_i = len(ded) - 1
    merged = ded
    # 判定枠の右に現れた音符(裏に隠れていた音符): 連続するフレームの点を左向きの短い追跡にまとめ、
    # 3 点以上(移動 20px 以上)・左向き(追跡のつなぎの上限まで)・直線に乗る(残差 ≤ 4px)ものだけ。同色の音符が近く(60px ぶんの時刻)に無ければ足す
    for color, pts_e in emerge_pts.items():
        pts_e = sorted(pts_e)
        # 止まっている音符: 判定枠 ±30px に 20 フレーム以上連続(1 フレームの移動 2px 未満)。ドンカマ2000 の最後の大ドン(BPM 5 で判定枠に居座り、
        # 叩かれて消える)。同色の音符が ±0.3s に無いときだけ足す。時刻 = 最後に見えたフレームの次(消えた=叩かれた)
        run = []
        for f, x, _big in pts_e + [(10 ** 9, 0, False)]:
            # 判定枠の上の音符は点滅する(12 フレーム見えて 11 フレーム消える)ので、空白は 12 フレームまで許す。時刻 = 最後に見えたフレーム(叩かれて消える直前)
            if run and (f - run[-1][0] > 12 or abs(x - run[-1][1]) > 2.5 * max(1, f - run[-1][0]) or abs(x - JUDGE_X) > 45):
                if len(run) >= 12 and run[-1][0] - run[0][0] >= 40:
                    t = round((f0 + run[-1][0]) / fps, 4)
                    isbig = bool(run[-1][2])
                    # 縁の行の遅い直線(5px/フレーム未満)で同じ音符が足してあれば、時刻は止まっている音符の「叩かれて消えたフレーム」を採る(上の注)
                    slow_h = [o for o in merged if o['color'] == color and o.get('hidden') and abs(o.get('vel') or 99) < 5.0
                              and abs(o['t'] - t) * abs(o['vel']) * fps < 84.0]
                    for o in slow_h: merged.remove(o)
                    # 連打を叩いている間は、連打の棒が判定枠の上にあって赤く(ドンの色に)見える。頭から終わりまでの間に判定枠で止まって見える物は棒
                    # (大打音 71.47s: 大連打の終わりの直前にドンとして足していた)
                    # (頭が取れていない連打もあるので、止まって見えていた間〜直後 0.1 秒に連打の終わりが来るかで見る)
                    t_first = round((f0 + run[0][0]) / fps, 4)
                    in_roll_ = any(e['color'] == 'roll_end' and t_first - 0.05 <= e['t'] <= t + 0.1 for e in merged)
                    if in_roll_:
                        if dbg: print('  stationary-in-roll %s t=%.3f(連打の棒)' % (color, t))
                    elif not any(o['color'] == color and abs(o['t'] - t) < 0.3 for o in merged):
                        if dbg: print('  stationary %s t=%.3f n=%d x=%.0f big=%s' % (color, t, len(run), run[-1][1], isbig))
                        merged.append({'t': t, 'color': color, 'big': isbig, 'npts': len(run), 'vel': 0.0, 'hidden': False, 'emerge': True, 'stationary': True})
                run = []
            if abs(x - JUDGE_X) <= 45: run.append((f, x, _big))
        # 大小は分けてまとめる(混ぜると小音符が重なった所で重心が跳ぶ)。大音符の点だけの組も、直線に 4px 以内で乗るなら動いてくる大音符
        # (dance storm 137.79s: 連打の棒の下を流れてきた大ドンが、棒が終わった後の判定枠の直前でだけ見えていた)
        groups = []
        for f, x, _big in pts_e:
            g = next((g for g in groups if g[0] == _big and 1 <= f - g[-1][0] <= 3 and 0 < (g[-1][1] - x) / (f - g[-1][0]) <= d_max), None)
            if g is None: groups.append([_big, (f, x)])
            else: g.append((f, x))
        for g in groups:
            g_big = g[0]; g = g[1:]
            # 大音符は顔の絵(怒り顔など)が変わると色の塊の重心が数十 px 動くので、内円の半径以上動いたものだけ(Doppelgangers 裏 95.13s: 判定枠の近くで
            # ほぼ止まっている遅い大ドンの重心が 38px 動いて見え、偽の大ドンになった。dance storm 137.79s は 105px)
            if len(g) < 3 or g[0][1] - g[-1][1] < (RIN_B if g_big else 20): continue
            fs_e = np.array([p[0] for p in g], float); xs_e = np.array([p[1] for p in g], float)
            a, b = np.polyfit(fs_e, xs_e, 1)
            if not (-d_max <= a < 0) or np.abs(a * fs_e + b - xs_e).max() > 4: continue      # 左向き(移動 20px 以上は上で確かめた)
            t = round((f0 + (JUDGE_X - b) / a) / fps, 4)
            # 同じ音符かは、組の見えていたフレームでの位置の差で見る(ringtrack の「同じ軌跡」と同じ 12px 未満)。以前の「60px 動く時間以内」は遅い音符で
            # 0.2 秒になり、密に積み重なった列(乱数調整 116.68s: 4.5px/フレームのドンが 24px 間隔)の中で現れた本物をいつも捨てていた
            # 時刻の差が一致とみなす幅(35ms。評価・比較動画と同じ)未満でも同じ音符(追跡した時刻と 20〜27ms ずれた見積もり: 最果の魔法使い 120.07s・Mood Swing 65.29s)
            # 時刻の差だけで同じとみなした相手には、この組を「控え」として持たせる。相手が後の段(ringtrack)で偽物として外されたら控えを戻す
            # (最果の魔法使い 119.06s: 後で偽物として外す 6 点・残差 5px の軌跡と同じとみなして、本物を落とした)
            dup_ = False
            alt_ = {'t': t, 'color': color, 'big': g_big, 'npts': len(g), 'vel': round(-a, 2), 'hidden': False, 'emerge': True, 'restored_emerge': True}
            for o in merged:
                if o['color'] != color or not o.get('vel') or abs(o['t'] - t) > 1.0: continue
                xo_ = JUDGE_X + abs(o['vel']) * (o['t'] * fps - f0 - fs_e)
                if np.median(np.abs(xo_ - xs_e)) < 12.0: dup_ = True; break
                if abs(o['t'] - t) < 0.035: dup_ = True; o['_emerge_alt'] = alt_; break
            # 音符は自分の層の速さで流れる。組の速さが近く(±1 秒)のどの音符の速さの 0.7〜1.3 倍にも入らなければ、判定枠の上の顔を拾った偽物
            # (トロいか2000 61.43s: 周りは 46px/フレームなのに 1.6px/フレーム、POLARiSNAUT 134.96s: 22px に対して 2.5px)
            nv_ = [abs(o['vel']) for o in merged if o['color'] in ('don', 'ka') and o.get('vel') and abs(o['t'] - t) < 1.0 and not o.get('stationary')]
            if nv_ and not any(0.7 <= -a / v_ <= 1.3 for v_ in nv_): dup_ = True
            if dup_ or any(o['color'] == color and not o.get('vel') and abs(o['t'] - t) < 60.0 / abs(a) / fps for o in merged): continue
            if dbg: print('  emerge %s%s t=%.3f vel=%.1f n=%d x%.0f-%.0f' % (color, ' B' if g_big else '', t, -a, len(g), xs_e.max(), xs_e.min()))
            merged.append({'t': t, 'color': color, 'big': g_big, 'npts': len(g), 'vel': round(-a, 2), 'hidden': False, 'emerge': True, '_g': [(int(f_), float(x_)) for f_, x_ in g]})
            _dbg_add(None, 'emerge', '', merged[-1], pts=[(int(f_), float(x_)) for f_, x_ in g])
    # 判定枠での目撃: 速い音符の列の下にずっと隠れていた音符は、上の音符が叩かれて消えた瞬間に、判定枠の真上で丸ごと 1〜2 枚だけ写る
    # (λ7708 61.21s: 速さ 44.5 のドンの列の最後のカッ)。判定枠の真上(±_SIGHT_PX)の小音符の丸ごとの顔を目撃として集め、追跡した音符と対応する目撃から
    # 「目撃の時刻 − 音符の時刻」の偏り(叩かれる直前に写る分)を測り、同じ色のどの音符とも対応しない目撃をその偏りを引いた時刻の音符として足す。
    # 判定枠に居座る音符(3 枚以上続けて写る: 止まっている音符・連打の棒)は目撃にしない
    if os.environ.get('V2T_SIGHT', '1') != '0':
        sg_ = []
        # 周りの音符の速さ(見えた位置から判定枠に着く時刻を出すのに使う)
        vt_ = np.array(sorted((o['t'], abs(o['vel'])) for o in merged if o['color'] in ('don', 'ka') and o.get('vel') and not o.get('stationary')))
        for color, pts_e in emerge_pts.items():
            # 顔(黒い目が 2 つ以上)だけ。色づいた判定枠の円は外す。判定枠の真上から小音符の半径ぶん手前まで(叩くのが少し早いと、真上に来る前に消える:
            # INSPION 94.38s は 13px 手前で消えた)
            fx_ = sorted((f, x) for f, x, big_ in pts_e if not big_ and -_SIGHT_PX <= x - JUDGE_X <= RIN_S and emerge_fill.get((f, color, round(x)), 0) >= 2)
            run_ = []
            for q in fx_ + [None]:
                if run_ and (q is None or q[0] - run_[-1][0] > 1):
                    # 居座り(位置が減っていかない: 止まっている音符・連打の頭)は外す。時刻は判定枠に一番近い点から、周りの音符の速さで判定枠に着く時刻
                    moving_ = len(run_) == 1 or run_[-1][1] < run_[0][1] - 1.0
                    if moving_ and len(vt_):
                        f_l, x_l = run_[-1]; t_l = (f0 + f_l) / fps
                        near_v = vt_[np.abs(vt_[:, 0] - t_l) < 1.0, 1]
                        if len(near_v): sg_.append((t_l + (x_l - JUDGE_X) / float(np.median(near_v)) / fps, color, t_l, x_l))
                    run_ = []
                if q is not None: run_.append(q)
        # 連打・風船の頭は、叩いている間ずっと判定枠の上に居座り、叩くたびに点滅して写る(トロいか2000 28.6〜30.1s)。その間の目撃は使わない。
        # 範囲は下の「演出の範囲」と同じ: 連打は頭から終わりまで、風船は頭から 1.6 秒か終わりまでの長い方
        hs_ = sorted((o['t'], o['color']) for o in merged if o['color'] in ('renda', 'balloon'))
        es_ = sorted(o['t'] for o in merged if o['color'] == 'roll_end')
        rs_ = []
        for k_, (h_, hc_) in enumerate(hs_):
            nh_ = hs_[k_ + 1][0] if k_ + 1 < len(hs_) else None
            e_ = next((e for e in es_ if h_ < e and (nh_ is None or e < nh_)), None)
            if e_ is None:
                # 終わりがまだ見つかっていない連打は、次のドン・カッまで(連打の途中に音符は置けない: カノン(シンクロニカ Remix)96.02s は頭の 42ms 後に
                # 叩かれて赤くなった頭を目撃していた)
                e_ = min([o['t'] for o in merged if o['color'] in ('don', 'ka') and o['t'] > h_ + 1.0 / fps] + [nh_ if nh_ is not None else h_])
            b_ = max(e_, h_ + (1.6 if hc_ == 'balloon' else 0.0))
            rs_.append((h_ - 1.0 / fps, b_ + 1.0 / fps))
        sg_ = [g_ for g_ in sg_ if not any(a_ <= g_[0] <= b_ for a_, b_ in rs_)]
        dk2_ = [o for o in merged if o['color'] in ('don', 'ka') and not o.get('stationary')]

        def _dt(g_, o):
            # 目撃 − 音符: 目撃した位置から、その音符自身の速さで判定枠に着く時刻を出して比べる(周りの速さで換算すると、速さの違う音符の混ざる所で
            # 数十 ms ずれた: 彁 裏 131.78s は速さ 30.9 の列の中の 10.8 のカッ)
            v_ = abs(o['vel']) if o.get('vel') else None
            return (g_[2] + (g_[3] - JUDGE_X) / v_ / fps if v_ else g_[0]) - o['t']
        near_ = []
        for g_ in sg_:
            d_ = [_dt(g_, o) for o in dk2_ if o['color'] == g_[1] and abs(g_[0] - o['t']) < 0.2]
            d_ = [d for d in d_ if abs(d) < 0.06]
            near_.append(min(d_, key=abs) if d_ else None)
        m_ = [d for d in near_ if d is not None]
        if len(m_) >= 5:
            # 目撃は音符の時刻から −1〜+2 フレームに散らばる(判定枠に着いてから叩かれて消えるまで残る)。同じ音符とみなす範囲は、この曲で対応した目撃の
            # 「目撃 − 音符」の 1%〜99% 点 ±半フレーム(決め打ちの幅だと、32ms 間隔のドンの 2 つ目の +2 フレームの目撃を別の音符にし: トロいか2000 119.84s、
            # 広げると 64ms おきの密な遅いドンの隠れた 1 つを隣と同じとみなした: 104.90s)
            bias_ = float(np.median(m_))
            # 前側は目撃のフレームの刻みと、重なった音符の追跡の時刻のずれで 1 フレームぶん広げる(夏祭り 52.00s・食べるな 73.60s: 音符の 17〜23ms 前の目撃)
            lo_ = float(np.quantile(m_, 0.01)) - 1.0 / fps; hi_ = float(np.quantile(m_, 0.99)) + 0.5 / fps
            # 叩かれた音符は消えたあと、上へ飛んでいく絵が判定枠の真上から出る。叩くのが遅れると判定「可」の幅(75ms)まで遅れるので、同じ色の音符の後ろ
            # 75ms + 1 フレームまでの目撃はその飛んでいく絵(Xa 77.12s・両玄ノ舞 25.38s・カオスタイム 48.68s など 27 曲で偽の音符になった)
            hi_ = max(hi_, __import__('synth')._KA_WINDOW + 1.0 / fps)
            for g_, d_ in zip(sg_, near_):
                ts_, col_ = g_[0], g_[1]
                t_ = round(ts_ - bias_, 4)
                # 等間隔の列の続き: 直前の 2 つの間隔 u(±1 フレーム)のちょうど次(±1.5 フレーム)の位置にあり、u が飛んでいく絵の出る範囲(hi_)より長い。
                # 列の中に隠れていた音符はここに現れる(トロいか2000 104.91s: 64ms おきの遅いドンが 29px 間隔で詰まった列、Parousia 64.69s: 95ms おきのドンの列の最後)。
                # このときは同じ音符かどうかを時刻の差(u の半分未満)だけで見る(位置の半径で比べると、詰まった列の隣の音符と同じとみなした)
                pv_ = sorted(o['t'] for o in merged if o['color'] in ('don', 'ka') and o['t'] <= ts_ - 0.5 / fps)[-2:]
                stream_ = False
                if len(pv_) == 2:
                    u_ = pv_[1] - pv_[0]
                    pp_ = max([o['t'] for o in merged if o['color'] in ('don', 'ka') and o['t'] < pv_[0] - 0.5 / fps] or [-1e9])
                    stream_ = u_ > hi_ and abs((pv_[0] - pp_) - u_) <= 1.0 / fps and abs((t_ - pv_[1]) - u_) <= 1.5 / fps
                if stream_:
                    if any((o['color'] == col_ or (col_ == 'don' and o['color'] == 'roll_end')) and abs(o['t'] - t_) < u_ / 2 for o in merged): continue
                else:
                    # 連打の終わりの丸は叩かれると赤くなってドンに見える(ENDCAP と同じ)。ドンの目撃は連打の終わりでも説明する(夏祭り 128.81s・贈王 69.06s など)
                    if any((o['color'] == col_ or (col_ == 'don' and o['color'] == 'roll_end')) and abs(g_[0] - o['t']) < 0.3 and lo_ <= _dt(g_, o) <= hi_ for o in merged): continue
                    # 一部が隠れた大音符の顔は小音符の大きさに見える。目撃した位置が同じ色の大音符の予想位置から大音符の内円の半径以内なら、その大音符
                    # (ウルトラマンX 130.10s・resume:story 112.96s・弾丸ノーツ 裏 112.90s: 大カッの 20〜30ms 前の目撃)
                    # 小音符も同じ: 目撃した位置が同じ色の音符の予想位置から内円の半径以内なら、その音符の目撃(時刻がずれて追跡された音符: いろはすてっぷ! 33.32s)
                    if any(o['color'] == col_ and o.get('vel') and abs(JUDGE_X + abs(o['vel']) * (o['t'] - g_[2]) * fps - g_[3]) < (RIN_B if o.get('big') else RIN_S) for o in merged): continue
                    # 叩かれた後の判定枠の光(両玄ノ舞 25.38s: カッを叩くと判定枠の円が青く光り、顔の大きさの青い円に見える)と飛んでいく絵は、叩かれた音符と
                    # 同じ色。等間隔の列の続きでない目撃は、直前の音符と同じ色なら使わない
                    prev_ = max((o for o in merged if o['color'] in ('don', 'ka') and o['t'] <= ts_), key=lambda o: o['t'], default=None)
                    if prev_ is not None and prev_['color'] == col_: continue
                vv_ = [abs(o['vel']) for o in dk2_ if o.get('vel') and abs(o['t'] - t_) < 1.0]
                if not vv_: continue
                if dbg: print('  sighted %s t=%.3f(判定枠で目撃 %.3f − 偏り %.3f、対応した目撃 %d)' % (col_, t_, ts_, bias_, len(m_)))
                merged.append({'t': t_, 'color': col_, 'big': False, 'npts': 1, 'vel': round(float(np.median(vv_)), 2), 'hidden': False, 'emerge': True, 'sighted': True,
                               'sight_ts': ts_, 'sight_lo': lo_, 'sight_hi': hi_})
    # 判定枠での最終確認(一番前の顔の列): びっしり重なった列では、一番前の音符が叩かれるまで判定枠の所で顔ごと見え、叩かれると次の音符が約 20px 後ろから
    # 一番前に出てきて判定枠に近づく。各フレームの判定枠の近く(−30〜+45px)の目のある顔のうち一番前のものをたどり、同じ色で近づき続ける間を 1 つの音符とする
    # (後ろへ戻る・色が変わる・3 フレーム以上空くと次の音符)。2 フレーム以上近づいたものだけ(叩かれた瞬間の描き直しと飛んでいく絵は 1 フレーム)。
    # 乱数調整 115〜123.5s で 参照の 66 個と色まで 1 対 1 に対応した。既存の音符との時刻の偏りを測り、どの音符とも対応しないものを足す
    if os.environ.get('V2T_FRONT', '1') != '0' and os.environ.get('V2T_SIGHT', '1') != '0':
        # 顔ごとに動きで列をつなぐ(目の数は同じ顔でもフレームごとに 0〜2 とぶれる: トロいか2000 122.85s は 1・0・2。いちばん前の顔だけを拾うと、
        # 目の数えられないフレームで後ろの顔に乗り移って列が切れた: 104.84s)。列の中に目が 2 つ見えたフレームが 1 つでもあれば顔(色づいた判定枠の円の穴は明るい)
        runs_ = []; eyes2_ = []; open_ = []
        for fi_ in range(len(frames_c)):
            cs_ = sorted((c['cx'], c['color'], emerge_fill.get((fi_, c['color'], round(c['cx'])), 0)) for c in frames_c[fi_]
                         if c.get('emerge') and not c.get('big') and c['color'] in ('don', 'ka') and -30 <= c['cx'] - JUDGE_X <= 45)
            open_ = [k_ for k_ in open_ if fi_ - runs_[k_][-1][0] <= 2]
            taken_ = set()
            for x_, c_, e_ in cs_:
                best_ = None
                for k_ in open_:
                    lf_, lc_, lx_ = runs_[k_][-1]
                    if k_ in taken_ or lc_ != c_ or not (1 <= fi_ - lf_ <= 2) or x_ > lx_ + 3.0: continue
                    if best_ is None or lx_ - x_ < best_[1]: best_ = (k_, lx_ - x_)
                if best_ is not None:
                    k_ = best_[0]; runs_[k_].append((fi_, c_, x_)); eyes2_[k_] |= e_ >= 2
                else:
                    runs_.append([(fi_, c_, x_)]); eyes2_.append(e_ >= 2); k_ = len(runs_) - 1; open_.append(k_)
                taken_.add(k_)
        runs_ = [r_ for r_, e_ in zip(runs_, eyes2_) if e_]
        cand_ = []; one_ = []
        if dbg and os.environ.get('V2T_FRDBG'):
            tq_ = float(os.environ['V2T_FRDBG'])
            print('  frdbg runs ' + ' | '.join(','.join('%d:%s%.0f' % (f0 + a, b[0], c) for a, b, c in r_) for r_ in runs_ if abs((f0 + r_[-1][0]) / fps - tq_) < float(os.environ.get('V2T_FRDBG_W', '0.1'))))
        for r_ in runs_:
            f_l, c_, x_l = r_[-1]; t_l = (f0 + f_l) / fps
            vv_ = [abs(o['vel']) for o in merged if o['color'] in ('don', 'ka') and o.get('vel') and abs(o['t'] - t_l) < 1.0 and not o.get('stationary')]
            if not vv_: continue
            if len(r_) == 1:
                # 1 フレームだけの顔は下で別に扱う(叩かれた瞬間の描き直し・飛んでいく絵と区別するため)
                if x_l - JUDGE_X >= RIN_S / 2: one_.append((f0 + f_l, c_, x_l))
                continue
            if r_[0][2] - r_[-1][2] < 2.0: continue            # 近づいていない(止まっている)
            # 本物の一番前の顔は周りの音符と同じ速さで近づく: 速さが 0.6〜1.5 倍で、3 フレーム以上(速くて判定枠の近く 75px に 3 フレーム入らないなら 2)。
            # 叩かれた瞬間の描き直しと飛んでいく最初のコマは、2 フレームで少ししか動かない(夏祭り 44.19s: 既存のドンの 35ms 後に別の音符として足した)
            # 比べる速さは近くの音符のうち最も近いもの(速さの違う音符の混ざる曲で中央値と比べると、遅い列の顔を捨てた: トロいか2000 119.39s)
            sp_ = (r_[0][2] - r_[-1][2]) / max(r_[-1][0] - r_[0][0], 1)
            v_ = float(min(vv_, key=lambda v: abs(v / sp_ - 1)))
            if not (0.6 * v_ <= sp_ <= 1.5 * v_) or len(r_) < 2: continue
            ta_ = t_l + (x_l - JUDGE_X) / v_ / fps
            if any(a_ <= ta_ <= b_ for a_, b_ in rs_): continue
            cand_.append((ta_, c_, len(r_)))
        dk3_ = [o for o in merged if o['color'] in ('don', 'ka') and not o.get('stationary')]
        mm_ = []
        for ta_, c_, n_ in cand_:
            d_ = [ta_ - o['t'] for o in dk3_ if o['color'] == c_ and abs(ta_ - o['t']) < 0.06]
            if d_: mm_.append(min(d_, key=abs))
        if len(mm_) >= 10:
            fb_ = float(np.median(mm_)); ftol_ = max(1.5 / fps, 3 * float(np.median(np.abs(np.array(mm_) - fb_))))
            if dbg: print('  front: 一番前の顔の列 %d、既存の音符と対応 %d、偏り %+.1fms、幅 %.1fms' % (len(cand_), len(mm_), fb_ * 1000, ftol_ * 1000))
            # 1 つの音符が説明できる列は 1 つ(最も近い列)。32ms 間隔の同じ色の 2 つ(トロいか2000 119.39s)では、2 つの列が 1 つの音符の 50ms 以内に入り、
            # 両方をその音符で説明して 1 つ目を落としていた
            tol_ok_ = max(ftol_ + 0.5 / fps, 3.0 / fps)
            claim_ = {}
            for k_, (ta_, c_, n_) in enumerate(cand_):
                t_ = ta_ - fb_
                ok_ = [o for o in merged if o['color'] == c_ and abs(o['t'] - t_) <= tol_ok_]
                if ok_:
                    o_ = min(ok_, key=lambda o: abs(o['t'] - t_))
                    if id(o_) not in claim_ or abs(o_['t'] - t_) < claim_[id(o_)][1]: claim_[id(o_)] = (k_, abs(o_['t'] - t_))
            won_ = {k_ for k_, _ in claim_.values()}
            claimed_ = set(claim_)
            for k_, (ta_, c_, n_) in enumerate(cand_):
                t_ = round(ta_ - fb_, 4)
                if dbg and os.environ.get('V2T_FRDBG') and abs(t_ - float(os.environ['V2T_FRDBG'])) < 0.15:
                    print('  frdbg %s t=%.3f 近くの音符 %s' % (c_, t_, [(round(o['t'], 3), o['color'][:2], bool(o.get('emerged')), bool(o.get('rescued'))) for o in merged if abs(o['t'] - t_) < 0.1]))
                # 一番前の顔が 3 フレーム以上続いたなら、その音符は前の音符より 3 フレーム以上後ろ。既存の同じ色の音符から 3 フレーム以内はその音符自身
                # (夏祭り 105.55s・チャーリー 56.23s など: 既存の音符の 33〜36ms 後に足していた)
                ok_ = [o for o in merged if o['color'] == c_ and abs(o['t'] - t_) <= tol_ok_ and (k_ in won_ or id(o) not in claimed_)]
                if ok_:
                    # 判定枠で顔が丸ごと見えて時刻を確かめた音符。後段(ringtrack)で一部が隠れた円の当て直しに時刻を置き換えさせない
                    # (乱数調整 115.87s: 当て直しで 115.932 に 40ms 遅れていた)
                    o_ = min(ok_, key=lambda o: abs(o['t'] - t_))
                    # 顔の時刻と照合の幅(ftol_)以内で合った音符だけ「判定枠で確かめた」印を付ける(3 フレームの幅でしか合わない音符に付けると、
                    # 円で追い直した正しい時刻への置き換えを止めた: 彁 21.01s は顔と 40ms ずれた救済の音符 21.052 が、円 84 個の 21.012 に直らなかった)。
                    # ずれている音符には判定枠で見えた時刻と幅を持たせる(ringtrack が円で追い直した時刻と比べ、別の音符か確かめる)
                    if abs(o_['t'] - t_) <= ftol_: o_['front_ok'] = True
                    else: o_['_front_t'] = t_; o_['_front_tol'] = ftol_
                    continue
                # 途中から現れた短い軌跡(emerged)は、見えている部分の中心が遅れ側にずれて時刻が 40ms ほど遅れる(乱数調整 115.93s)。60ms 以内の同じ色の
                # そういう音符は、判定枠で顔が丸ごと見えた時刻に直す(足さない)
                em_ = [o for o in merged if o['color'] == c_ and abs(o['t'] - t_) < 0.06 and (o.get('emerged') or o.get('rescued')) and id(o) not in claimed_]
                if em_:
                    o_ = min(em_, key=lambda o: abs(o['t'] - t_))
                    if dbg: print('  front-retime %s %.3f → %.3f' % (c_, o_['t'], t_))
                    o_['t'] = t_; continue
                # 別の色の音符とほぼ同じ時刻(1 フレーム未満)なら、その音符の色の取り違え(足さない)
                if any(o['color'] in ('don', 'ka') and o['color'] != c_ and abs(o['t'] - t_) < 1.0 / fps for o in merged): continue
                vv_ = [abs(o['vel']) for o in dk3_ if o.get('vel') and abs(o['t'] - t_) < 1.0]
                if dbg: print('  front-add %s t=%.3f(一番前の顔 %d フレーム)' % (c_, t_, n_))
                merged.append({'t': t_, 'color': c_, 'big': False, 'npts': n_, 'vel': round(float(np.median(vv_)), 2) if vv_ else 0.0,
                               'hidden': False, 'emerge': True, 'front': True, '_front_tol': tol_ok_})
            # 1 フレームだけ一番前に見えた顔: 叩かれた瞬間の描き直しと飛んでいく絵は判定枠の所(中心が判定枠)に出るので、判定枠より半径の半分以上
            # 後ろに目の 2 つ見える顔だけ(上で選んだ)。そのフレームでの予想位置が照合の幅以内に来る同じ色の音符があれば、その音符の顔。どの音符の
            # 予想位置からも外れていれば、早めに叩かれて(可)判定枠に着く前に消えた音符(検出テスト 73.38s: 0.3 倍速で 26px ずつ重なったドンで、
            # 前のドンが叩かれた次のフレームに 21px 後ろで顔が見え、その次のフレームで叩かれた)。速さは予想位置の最も近い同じ色の音符のもの
            for F_, c_, x_ in one_:
                t_l = F_ / fps
                if any(a_ <= t_l <= b_ for a_, b_ in rs_): continue
                nb_ = [o for o in merged if o['color'] in ('don', 'ka') and o.get('vel') and not o.get('stationary') and abs(o['t'] - t_l) < 1.0]
                if not nb_: continue
                dx_ = [(abs(JUDGE_X + (o['t'] * fps - F_) * abs(o['vel']) - x_), o) for o in nb_]
                d_s = [d for d in dx_ if d[1]['color'] == c_]
                if not d_s: continue
                dm_, om_ = min(d_s, key=lambda d: d[0]); v1_ = abs(om_['vel'])
                if dm_ <= max(tol_ok_ * fps * v1_, RIN_S / 4): continue
                if any(d <= RIN_S / 4 for d, o in dx_ if o['color'] != c_): continue      # 別の色の音符の位置(色の取り違え)
                t_ = round(t_l + (x_ - JUDGE_X) / v1_ / fps - fb_, 4)
                if any(o['color'] == c_ and abs(o['t'] - t_) <= tol_ok_ for o in merged): continue
                if dbg: print('  front-add1 %s t=%.3f(1 フレームの顔 x%.0f、最も近い予想 %.0fpx)' % (c_, t_, x_, dm_))
                merged.append({'t': t_, 'color': c_, 'big': False, 'npts': 1, 'vel': round(v1_, 2),
                               'hidden': False, 'emerge': True, 'front': True, '_front_tol': tol_ok_})
    merged.sort(key=lambda o: o['t'])
    # 連打の途中に音符は置けない: 叩かれた棒の上の火花(赤の丸い塊)が短いトラックになって小ドンに化ける(resume:story 71.6s)。
    # 連打/風船の頭から終端までの間(両端 30ms は除く)にあるドン/カッは捨てる
    # 乗り継ぎの幽霊: 点が少なく(15 未満)当てはめが荒い(res_sd > 5)うえ、速さが前後 1.5 秒の音符の中央値から 8% 以上外れるドン/カッは、
    # 2 つの音符の点をつないだもの(活声 41.855s: 12 点で 52 フレームにまたがり vel 21.2、周りは 24.1)。点が途切れだらけのものに限る
    dk_ = [o for o in merged if o['color'] in ('don', 'ka') and not o.get('hidden') and o.get('vel')]
    ghost = set()
    for o in dk_:
        if o.get('npts', 99) >= 15 or o.get('res_sd', 0) <= 5.0: continue
        if o.get('span', 0) < 2.5 * o.get('npts', 99): continue      # 点が途切れだらけ(幅がフレーム数の 2.5 倍以上)のものだけ。
        # 速い音符(彁 裏 85.15s vel 61、段位リスドゥーンヴァーナ 129.8s vel 94)は点が少なくても途切れずに続く
        nb = [abs(q['vel']) for q in dk_ if q is not o and abs(q['t'] - o['t']) <= 1.5 and q.get('npts', 0) >= 20]
        if len(nb) >= 3 and abs(abs(o['vel']) / np.median(nb) - 1) > 0.08:
            if dbg: print('  reject(ghost-vel) %s t=%.3f n=%d vel=%.1f 周り %.1f res_sd=%.1f' % (o['color'], o['t'], o['npts'], o['vel'], np.median(nb), o['res_sd']))
            ghost.add(id(o))
    merged = [o for o in merged if id(o) not in ghost]
    # 終わりの無い連打: 頭の後に「途中で見えなくなった尻尾」(4 点以上、速さが頭の ±12%)があれば、判定枠まで延ばした時刻を終わりにする
    # (活声 144.9s の大連打: 尻尾が続く音符と重なって 5 点で見失われ、判定枠に居座る頭が「止まった大ドン」になり、
    #  赤い棒をゴーゴーとも誤認した。延ばした終わりは本家と 19ms)
    hd_ = sorted((o['t'], o) for o in merged if o['color'] in ('renda', 'balloon'))
    en_ = sorted(o['t'] for o in merged if o['color'] == 'roll_end')
    for k_, (h, ho) in enumerate(hd_):
        nxt = hd_[k_ + 1][0] if k_ + 1 < len(hd_) else 1e9
        if ho['color'] != 'renda': continue          # 風船には尻尾の棒が無い(もぺもぺ 55.4s の 2.1 秒の風船を 0.23 秒で終わらせていた)
        if any(h < e < nxt for e in en_): continue
        c_ = [le for le in lost_ends if h + 0.02 < le[0] < nxt and abs(le[1] / max(abs(ho.get('vel') or 0), 1e-6) - 1) <= 0.12]   # 0.02 秒: もぺもぺ 79.8s の 0.05 秒の連打の尻尾(0.1 秒だと弾いた)
        if c_:
            le = min(c_, key=lambda x: x[0])
            if dbg: print('  lost-end %s t=%.3f → 終わり %.3f (尻尾 %d 点 vel %.1f)' % (ho['color'], h, le[0], le[2], le[1]))
            merged.append({'t': le[0], 'color': 'roll_end', 'big': False, 'npts': le[2], 'vel': round(le[1], 2), 'hidden': False, 'lost_end': True})
    merged.sort(key=lambda o: o['t'])
    # 連打の終わりの丸は頭と同じ棒なので、頭と同じ速さで動く。速さが頭と 10% 以上違う終わりの軌跡は、終わりの丸と別の物(丸の下から出てくる遅い音符)の
    # 点が混ざったもの。終わりの候補の点ごとに頭の速さで判定枠に着く時刻を出し、0.02 秒(約 1 フレーム)の幅に最も多く集まる所の中央値を終わりにする
    # (Altale 56.93s: 速さ 17.8 の連打の終わりが、下から出てきた速さ 8.9 のドンとつながって 57.19s・速さ 10.6 になり、後ろのドン 2 つを連打の中として外した)
    end_pts_ = [(fi, c['cx']) for fi, cs in enumerate(frames_c) for c in cs if c['color'] == 'roll_end']
    hd2_ = sorted((o for o in merged if o['color'] == 'renda' and o.get('vel')), key=lambda o: o['t'])
    for k_, ho in enumerate(hd2_):
        nxt = min([q['t'] for q in merged if q['color'] in ('renda', 'balloon') and q['t'] > ho['t'] + 0.02] + [1e9])
        eo = next((q for q in merged if q['color'] == 'roll_end' and ho['t'] < q['t'] < nxt and q.get('vel')), None)
        v_h = abs(ho['vel'])
        if eo is None:
            # 終わりの見つからない連打: 棒の端に音符が重なると終わりの候補の位置が跳び、軌跡が細切れになって捨てられる(カノン(シンクロニカ Remix)97.63s)。
            # 同じく頭の速さで終わりの候補の点を判定枠に着く時刻にし、次の音符より前で 0.02 秒の幅に集まる所を終わりにする
            nn_ = min([q['t'] for q in merged if q['color'] in ('don', 'ka') and q['t'] > ho['t'] + 0.05 and not q.get('stationary')] + [nxt])
            tt = sorted((f0 + fi + (x - JUDGE_X) / v_h) / fps for fi, x in end_pts_)
            tt = [t for t in tt if 0.05 < t - ho['t'] and t < nn_]
            best = max(([t for t in tt if t0_ <= t < t0_ + 0.02] for t0_ in tt), key=len, default=[])
            if len(best) >= _MIN_PTS_END:
                t_new = round(float(np.median(best)), 4)
                if dbg: print('  roll-end by head speed(終わり無し)%s %.3f → 終わり %.3f(頭の速さ %.1f、%d 点)' % (ho['color'], ho['t'], t_new, v_h, len(best)))
                merged.append({'t': t_new, 'color': 'roll_end', 'big': False, 'npts': len(best), 'vel': ho['vel'], 'hidden': False})
            continue
        if eo.get('fast_roll') or abs(abs(eo['vel']) / v_h - 1) <= 0.10: continue
        # 速さは当てはめのよい方を信じる。頭の方が荒いなら(欠けて見えた頭: resume:story 73.08s は頭の残差 10.7px・終わり 0.7px)、下の sliver-head が
        # 終わりの速さで頭を直す
        if ho.get('res_sd', 99.0) >= eo.get('res_sd', 99.0): continue
        tt = sorted((f0 + fi + (x - JUDGE_X) / v_h) / fps for fi, x in end_pts_)
        tt = [t for t in tt if 0.05 < t - ho['t'] and t < nxt]
        best = max(([t for t in tt if t0_ <= t < t0_ + 0.02] for t0_ in tt), key=len, default=[])
        if len(best) < _MIN_PTS_END: continue
        t_new = round(float(np.median(best)), 4)
        if dbg: print('  roll-end by head speed %.3f(vel %.1f)→ %.3f(頭の速さ %.1f、%d 点)' % (eo['t'], eo['vel'], t_new, v_h, len(best)))
        eo['t'] = t_new; eo['vel'] = ho['vel']; eo['npts'] = len(best)
    merged.sort(key=lambda o: o['t'])
    heads = sorted(o['t'] for o in merged if o['color'] in ('renda', 'balloon'))
    ends = sorted(o['t'] for o in merged if o['color'] == 'roll_end')
    spans = []
    for i_h, h in enumerate(heads):
        e = next((x for x in ends if x > h), None)
        nxt = heads[i_h + 1] if i_h + 1 < len(heads) else None
        # 終端は直前の頭のもの。次の頭より後ろの終端は別の連打(終端を取りこぼした風船の次の連打など)なので対応させない
        if e is not None and (nxt is None or e < nxt): spans.append((h + 0.03, e - 0.03))
    # 演出の範囲: 連打は棒の間+直後(叩いた火花が判定枠に残る)、風船は頭から 1.6 秒(膨らむ球が帯を覆う。
    # くす玉は音符の絵が短くても演出は長い)。この範囲の「弱い」音符(点が 25 未満、または止まっている)は演出の誤検出
    wide = []
    for i_h, h in enumerate(heads):
        e = next((x for x in ends if x > h), None)
        nxt = heads[i_h + 1] if i_h + 1 < len(heads) else None
        is_b = any(abs(o['t'] - h) < 1e-6 and o['color'] == 'balloon' for o in merged)
        b_end = max(e if (e is not None and (nxt is None or e < nxt)) else h, h + (1.6 if is_b else 0.0))
        wide.append((h - 0.05, b_end + 0.35))
    # 連打の終わりの丸の下から判定枠の直前に現れた音符(Emma 裏 75.23s: 大連打の直後のドンが終わりの丸に隠れて流れ、判定枠の 50px 手前で出てくる)。
    # 出てくる途中の中心は遅れて動くので速さが 12 と出て演出扱いされた。終わりの 0.3 秒以内に現れ、速さが前後と 10% 以上違うなら、
    # 最後の点(ほぼ全体が見える)と前後の音符の速さで時刻を出し直す
    for o in merged:
        if not (o.get('emerge') and o.get('_g') and o['color'] in ('don', 'ka')) or os.environ.get('V2T_EMERGE_END', '1') == '0': continue
        if not any(0 < o['t'] - e_ < 0.3 for e_ in ends): continue
        near = [abs(q['vel']) for q in merged if q['color'] in ('don', 'ka') and q['npts'] >= 25 and abs(q['t'] - o['t']) < 3.0]
        if not near: continue
        mv_ = float(np.median(near))
        if abs(abs(o['vel']) / mv_ - 1) <= 0.10: continue
        fl, xl = o['_g'][-1]
        t_new = round((f0 + fl + (xl - JUDGE_X) / mv_) / fps, 4)
        if any(q is not o and q['color'] in ('don', 'ka') and abs(q['t'] - t_new) < 0.03 for q in merged): continue
        if dbg: print('  emerge-after-roll %s t=%.3f vel=%.1f → t=%.3f vel=%.1f' % (o['color'], o['t'], o['vel'], t_new, mv_))
        o['t'] = t_new; o['vel'] = round(mv_, 2); o['npts'] = max(o['npts'], 25)
    # 連打の終わりの丸は、連打の終わりと同じ時刻・同じ速さで動く同じ物体(叩くと赤くなってドンの候補になる)。譜面でも連打の終わり(8)と同じ所に別の音符は
    # 置けない。連打の終わりと 30ms 以内・速さ 10% 以内で重なるドン/カッはその丸として捨てる(Emma 裏 17.37s、VERTeX 17.09s の 71px/フレームの速い連打)。
    # 「点の帰属」(予測位置での照合)の、連打の終わりを時刻と速さで照合する版
    merged = _endcap(merged, fps, dbg)
    kept = []
    for o in merged:
        if o['color'] in ('don', 'ka') and any(a < o['t'] < b for a, b in spans):
            if dbg: print('  reject(in-roll) %s t=%.3f n=%d' % (o['color'], o['t'], o['npts']))
            _dbg_mark(o, 'in-roll'); continue
        if o['color'] in ('don', 'ka') and any(a < o['t'] < b for a, b in wide):
            # 演出(火花・膨らむ球)は「止まっている」か「点が少なくて速さも前後と違う」。点が少ないだけの本物(隠れていた
            # 大音符など)は捨てない: TDM 265.4s の大カッは点 16 でも速さが前後と同じ(25.8 vs 26.3)なので残す
            near = [abs(q['vel']) for q in merged if q['color'] in ('don', 'ka') and q['npts'] >= 25 and abs(q['t'] - o['t']) < 3.0]
            med_v = float(np.median(near)) if near else 0.0        # 近くが全部「止まっている音符」だと 0 になる
            off = med_v > 1.0 and abs(abs(o['vel']) / med_v - 1) > 0.10
            # 点が少ないだけでは捨てない。速さが前後と違い、かつ当てはめが荒い(残差 5px 超)ものだけ演出とみなす
            # 見えていた間に内円の半径の半分も動いていない(止まっている演出)。動いた量は見えていた x の範囲(無ければ 速さ × 見えていたフレーム数)
            xr_ = o.get('xr'); disp_ = (xr_[0] - xr_[1]) if xr_ else (abs(o['vel']) * o['span'] if o.get('span') else None)
            not_moving = disp_ is not None and disp_ < RIN_S / 2
            # 演出(火花・膨らむ球)は判定枠の所で生まれて、帯を流れてこない。見えていた間に帯の半分以上を判定枠へ向かって動いたものは流れてきた音符
            # (Destr0yer 88.18s: 周りの 10 倍の速さ 92px/フレームの大カッ。積み重なった音符の上を通るので当てはめが荒く(残差 23px)、演出として捨てていた)
            travelled = disp_ is not None and disp_ >= (X1 - X0) / 2
            if not_moving or o.get('stationary') or (o['npts'] < 25 and off and o.get('res_sd', 99) > 5.0 and not travelled):
                if dbg: print('  reject(roll-effect) %s t=%.3f n=%d vel=%.1f res_sd=%.1f (近くの速さ %.1f)' % (
                    o['color'], o['t'], o['npts'], abs(o['vel']), o.get('res_sd', -1), med_v))
                _dbg_mark(o, 'roll-effect'); continue
        kept.append(o)
    merged = kept
    # 縁の行だけから出た隠れ大音符は、通常の音符の範囲の外にあれば捨てる(リザルト画面の偽の当てはめ)。
    # ただし通常追跡の音符を置き換えたものは範囲を決める側に入れる(魔宵月の最後の大カッは連打の 2.3 秒後に 1 個だけ
    # 遅い速度で来るので、通常の音符の範囲(連打まで)から外れて消えていた)
    # 音符に頭を隠された連打・風船: 頭は見えている左端から位置を出すので、直前の音符に重なっていると遅く出る(resume:story 裏 33.3s: カッの直後の小連打が
    # +30ms)。頭が直前の音符と重なる(100px 未満)とき、直前の 2 つの音符の間隔 step が詰まっていて(100px 未満)、測った間隔が step の 1〜2 倍なら、
    # 頭を「直前の音符 + step」に置く(本家は 3 つともカッと同じ間隔の次の位置)
    dk_all = sorted((o['t'], o) for o in merged if o['color'] in ('don', 'ka') and not o.get('hidden'))
    dk_t = [x[0] for x in dk_all]
    # 頭の速さが同じ連打の終わりの速さと 25% 以上違うのは、隠れた頭の代わりに「追い越す音符の隙間から覗く棒の切れ端」(音符と同じ速さで動く)を
    # 追ったもの(resume:story 裏 60.51s: 半分の速さの連打の上を 2 倍速のカッが流れ、頭 −104ms・速さ 44.8 vs 終わり 21.9)。
    # 頭は終わりの前の最後のドン/カッの次の刻み(隠れた頭と同じ)に置き、速さは終わりのものにする
    hd_all = sorted(o['t'] for o in merged if o['color'] in ('renda', 'balloon'))
    for o in merged:
        if o['color'] != 'renda' or not o.get('vel') or os.environ.get('V2T_SLIVER', '1') == '0': continue
        nh = next((t for t in hd_all if t > o['t'] + 1e-6), 1e9)
        e = next((q for q in sorted(merged, key=lambda q: q['t']) if q['color'] == 'roll_end' and o['t'] < q['t'] < nh), None)
        if e is None or not e.get('vel') or abs(abs(o['vel']) / abs(e['vel']) - 1) <= 0.25: continue
        j = int(np.searchsorted(dk_t, e['t'])) - 1
        if j < 1: continue
        t1, t2 = dk_t[j], dk_t[j - 1]; step = t1 - t2
        v_e = abs(e['vel']) * fps
        if not (0 < step and step * v_e < 100 and t1 + step < e['t'] - 0.02 and abs(t1 + step - o['t']) < 0.3): continue
        if dbg: print('  sliver-head t=%.3f vel=%.1f → %.3f vel=%.1f(終わり %.3f)' % (o['t'], o['vel'], t1 + step, e['vel'], e['t']))
        o['t'] = round(t1 + step, 4); o['vel'] = e['vel']; o['occluded'] = True
    # 同じ色・同じ速さ(5% 以内)の音符が 30ms 以内に 2 つあり、片方だけ当てはめが荒い(残差 > 5px、もう片方 < 2px)なら荒い方は
    # 隣り合う 2 つの音符を乗り継いだ幽霊(POLARiSNAUT 44.14s: 16 分のカッの 20ms 後に残差 7.3px のカッ)。録画の「複製 → 追いつき」の古い絵で
    # 1 フレームぶんずれた点を拾ったもので、点の帰属では照合できない(見直し予定: HANDOFF 44)
    if os.environ.get('V2T_TWIN', '1') != '0':
        dk_ = sorted((o for o in merged if o['color'] in ('don', 'ka') and not o.get('hidden') and o.get('vel')), key=lambda o: o['t'])
        drop_ = set()
        for p_, q_ in zip(dk_, dk_[1:]):
            if p_['color'] != q_['color'] or q_['t'] - p_['t'] >= 0.03 or abs(abs(p_['vel']) / abs(q_['vel']) - 1) > 0.05: continue
            sp, sq = p_.get('res_sd', 0), q_.get('res_sd', 0)
            if sp > 5 and sq < 2: drop_.add(id(p_))
            elif sq > 5 and sp < 2: drop_.add(id(q_))
        if drop_:
            if dbg: print('  twin-ghost ' + ' '.join('%.3f' % o['t'] for o in merged if id(o) in drop_))
            merged = [o for o in merged if id(o) not in drop_]
    # 大音符のすぐ後ろ(中心の差が R_B − RIN_S = 44px 未満)を同じ速さで流れる同じ色の小音符は、その大音符の一部。上に描かれる大音符(先に着く)の外周が
    # そこでは小音符の色の円を全部覆うので、本物の小音符は写らない。写っているのは、前の音符に左を隠された大音符の右側の色を小音符の大きさで測ったもの
    # (検出テスト ファイナル 115.66s・116.04s: 大カッの 25px 後ろに 114 点・残差 1.1px の小カッ。大カッ自身も取れている)
    if os.environ.get('V2T_BIGSHADOW', '1') != '0':
        # 大音符は帯の上下の縁の行から推定したもの(hidden)も含める(115.66s の大カッは縁の行の推定 113 点。見えている色の塊は前の小ドンに左を隠された幅 77px)
        bg_ = [o for o in merged if o['color'] in ('don', 'ka') and o.get('big') and o.get('vel') and (not o.get('hidden') or (o.get('npts') or 0) >= 20)]
        drop_ = set()
        for o in merged:
            if o['color'] not in ('don', 'ka') or o.get('big') or not o.get('vel') or o.get('hidden'): continue
            v_ = abs(o['vel'])
            for b_ in bg_:
                if b_['color'] != o['color'] or abs(abs(b_['vel']) / v_ - 1) > 0.03: continue
                d_ = (o['t'] - b_['t']) * fps * v_
                if 0 < d_ < R_B_OUT - RIN_S:
                    if dbg: print('  big-shadow %s t=%.3f(大音符 %.3f の %.0fpx 後ろ)' % (o['color'], o['t'], b_['t'], d_))
                    drop_.add(id(o)); break
        if drop_: merged = [o for o in merged if id(o) not in drop_]
    for o in merged:
        if o['color'] not in ('renda', 'balloon') or not o.get('vel') or o.get('occluded'): continue
        v_ = abs(o['vel']) * fps
        j = int(np.searchsorted(dk_t, o['t'])) - 1
        if j < 1: continue
        t1, t2 = dk_t[j], dk_t[j - 1]
        gap, step = o['t'] - t1, t1 - t2
        if gap * v_ < 100 and 0 < step and step * v_ < 100 and step * 0.95 <= gap <= step * 2.0:
            if dbg: print('  occluded-head %s t=%.3f → %.3f(直前の音符 %.3f + 間隔 %.0fms)' % (o['color'], o['t'], t1 + step, t1, step * 1000))
            o['t'] = round(t1 + step, 4); o['occluded'] = True
    merged.sort(key=lambda o: o['t'])
    # 止まっている音符は、動く音符の範囲(最初の 1 秒前〜最後の 1 秒後)の外なら捨てる(憎悪と醜悪の花束 裏 139.4s: 曲の後にレーンが青くなり、
    # 空の判定枠の内側を止まったカッと誤認した)
    mv = [o['t'] for o in merged if o['color'] in ('don', 'ka') and not o.get('stationary') and not o.get('hidden')]
    if mv:
        # ただし長く止まっている音符(100 点以上)は残す(ドンカマ 2000 の最後の大ドン: 最後の動く音符の 3.8 秒後、2.7 秒止まる)
        merged = [o for o in merged if not o.get('stationary') or o.get('npts', 0) >= 100 or min(mv) - 1.0 <= o['t'] <= max(mv) + 1.0]
    normal = [o['t'] for o in merged if not o.get('hidden') or o.get('replaced')]
    if normal:
        lo, hi = min(normal) - 0.5, max(normal) + 0.5
        merged = [o for o in merged if not o.get('hidden') or o.get('replaced') or lo <= o['t'] <= hi]
    else:
        merged = [o for o in merged if not o.get('hidden')]
    if any(c['color'] in ('don', 'ka') for cs in frames_c[::50] for c in cs):     # 曲全体で見る(最初の 2000 フレームだけだと前奏の長い切り出しで飛ばした)     # 音符の追跡のときだけ(小節線の追跡では使わない)
        add = _line_recover(frames_c, fps, f0, merged, dbg)
        add += _fast_roll_recover(frames_c, fps, f0, merged, dbg)
        if add:
            merged = sorted(merged + add, key=lambda o: o['t'])
            merged = _endcap(merged, fps, dbg, mark_only=True)        # 拾い直した音符にも(終わりの丸を拾い直すことがある: そして勇者は眠りにつく 164.91s)
    if dbgT is not None:
        for o in merged:
            if id(o) in dbgT['byid']: dbgT['byid'][id(o)]['final'] = True
    if dbg: print('V(median) %.2f px/frame, tracks %d, onsets %d' % (V, len(tracks), len(merged)))
    track.V = V
    return merged

def run(video, start, end, dump=None, dbg=False, cal=None):
    """cal: calib.calibrate() の結果。基準配置(1080p アーケード)でなければ各フレームをアフィン変換してから検出する"""
    global X1
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(start * fps)); f1 = int(round(end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    M = None
    if cal is not None:
        import calib
        if not calib.is_canonical(cal):
            M, x1 = calib.affine(cal); X1 = x1
            if dbg: print('calibration: warp scale %.4f tx %.1f ty %.1f  X1=%d' % (M[0, 0], M[0, 2], M[1, 2], X1))
    frames_c = []; frames_b = []
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok: break
        if M is not None:
            frame = cv2.warpAffine(frame, M, (calib.CANON_W, calib.CANON_H), flags=cv2.INTER_LINEAR)
        frames_c.append(candidates(frame))
        frames_b.append(bar_candidates(frame))
        fi += 1
    cap.release()
    track.frac = None
    onsets = track(frames_c, fps, f0, dbg)
    bo = track(frames_b, fps, f0, False)
    _unbias(onsets, fps)
    bars = [o['t'] for o in bo]      # 小節線の判定枠通過時刻
    run.bars = bars; run.bar_vels = [o['vel'] for o in bo]
    if dump:
        json.dump({'fps': fps, 'start': start, 'end': end, 'onsets': onsets, 'bars': bars},
                  open(dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return fps, onsets

# ---------------------------------------------------------------------------------------------
# 並列版: 動画を時間で分割し、各プロセスが ffmpeg で正確に切り出した区間をデコード(レーン帯の行だけ)して
# フレームごとの候補を出す。追跡は結合後に一括(等速予測は時間方向の文脈が要るので分割しない)。
# ---------------------------------------------------------------------------------------------
import subprocess, os

def probe_pts(video):
    """全フレームの表示時刻(秒)を昇順で返す(パケット情報のみ、デコードしない)"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time',
                          '-of', 'csv=p=0', video], capture_output=True, text=True).stdout
    pts = np.array(sorted(float(x) for x in out.split() if x.strip() and x.strip() != 'N/A'))
    return pts

BAND_ROWS = (290, 480)     # 基準配置で検出に使う行(小節線の帯 296-478 と音符の帯 336-437 を含む)
EXT_ROWS = (250, 516)      # デコードして埋める行。ゴーゴーの特徴(gogo.py: 帯 295-325/450-476、縁の黒線 257-303/462-508)も同じデコードで取る

_CODEC = {}

def _gpu_decoder(video):
    """NVDEC(RTX)で使えるデコーダ名。無ければ None(V2T_GPU=0 でも None)"""
    if os.environ.get('V2T_GPU', '1') == '0': return None
    if video not in _CODEC:
        c = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height',
                            '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')
        _CODEC[video] = c
    # AV1 は GPU を使わない: -hwaccel cuda の AV1 は何曲も同時に動かすと、枚数はそろうのに一部の区切りの絵が壊れ、音符が数十個ずつ抜けた
    # (TDM・ヴリコラカス・HYPERNOVA・ココドコ。抜ける場所は実行ごとに変わる)。単独では正常
    name = {'h264': 'h264_cuvid', 'hevc': 'hevc_cuvid', 'vp9': 'vp9_cuvid'}.get(_CODEC[video][0])
    if name is None: return None
    # 1080p 以下は CPU の方が速い(ffmpeg を 8 本同時に動かすと NVDEC の取り合いで遅くなる: 高嶺の花子さん 並列の処理 15.7 → 13.7 秒)。
    # V2T_GPU=1 で常に使う。既定(auto)は 1080p を超える動画だけ
    if os.environ.get('V2T_GPU', 'auto') != '1' and int(_CODEC[video][2]) <= 1080: return None
    global _GPU_OK
    try: _GPU_OK
    except NameError:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-c:v', name, '-i', video, '-frames:v', '1', '-f', 'null', '-'], capture_output=True)
        _GPU_OK = r.returncode == 0
    return name if _GPU_OK else None


def _med_hist(h, lo=0):
    """度数分布 h(値 lo, lo+1, …)から中央値(np.median と同じ値: 偶数個なら真ん中 2 つの平均)"""
    c = np.cumsum(h.ravel()); n = int(c[-1])
    if n == 0: return float('nan')
    return (int(np.searchsorted(c, (n - 1) // 2 + 1)) + int(np.searchsorted(c, n // 2 + 1))) / 2.0 + lo


def _max3(a):
    """3 色の最大(a.max(2) と同じ値。OpenCV は numpy の 12 倍速い)"""
    return cv2.max(cv2.max(a[..., 0], a[..., 1]), a[..., 2])


def _med_u8(a, mask=None):
    """uint8 の画素の中央値(np.median と同じ値)。OpenCV の calcHist で度数分布を取る"""
    return _med_hist(cv2.calcHist([np.ascontiguousarray(a)], [0], mask, [256], [0, 256]))


def _gogo_feat(canvas, full=True):
    """ゴーゴーの特徴(生の値。指標の作り方は gogo.py 側で決める — ここを変えると全曲の再デコードになるので、
    窓ごとの「レーンの背景の色」をそのまま持たせる):
      v_  レーンの 赤−緑 の中央値、br_ 明るさ(V)の中央値、top/bot 縁の窓の行ごとの V の中央値(レーンが隠れた判定用)、
      wins 各窓 (RGB の中央値, 背景の画素の割合)。背景 = 連打の棒(叩く前の黄・叩いた後の真っ赤)・音符の白い縁・黒い輪郭を除いた画素"""
    import gogo
    # 3 色の最大は cv2.max、中央値は calcHist の度数分布で出す(np.median・max(2) と同じ値。numpy の max(2) が 1 回 0.6ms かかっていた)
    rows8 = np.vstack([canvas[a:b, gogo.X0:gogo.X1] for a, b in gogo.ROWS])
    v_ = float(np.median(rows8[:, :, 2].astype(np.int16) - rows8[:, :, 1])); br_ = _med_u8(_max3(rows8))
    # full=False: 全フレームの測り直し(gogo.right_rgb・brightness(step=1))に要る 赤−緑・明るさ・右端の窓だけ(残りは gogo.STEP おきのフレームで取る)
    top = np.median(_max3(canvas[257:304, gogo.X0:gogo.X1]), axis=1).astype(np.float32) if full else None
    bot = np.median(_max3(canvas[462:509, gogo.X0:gogo.X1]), axis=1).astype(np.float32) if full else None
    wins = []
    for i_w, (x0_, x1_) in enumerate(gogo.WINDOWS):
        if not full and i_w != gogo.W_RIGHT: wins.append(None); continue
        px = np.vstack([canvas[a:b, x0_:x1_] for a, b in gogo.ROWS]).reshape(-1, 1, 3)
        hv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h_, s_, v2 = hv[:, 0], hv[:, 1], hv[:, 2]
        bg = ~((s_ > 230) & (v2 > 220)) & ~((h_ >= 20) & (h_ <= 34) & (s_ > 120)) & ~((s_ < 40) & (v2 > 200)) & (v2 > 20)   # 黒い輪郭は V<20(暗い録画のレーンは V40 未満: Replay)
        frac = float(bg.mean())
        if frac > 0:
            mk = bg.astype(np.uint8).reshape(-1, 1)
            rgb = tuple(_med_hist(cv2.calcHist([px], [ch], mk, [256], [0, 256])) for ch in (2, 1, 0))
        else:
            rgb = (np.nan, np.nan, np.nan)
        wins.append((rgb, frac))
    return v_, br_, top, bot, wins


RT_FEAT = int(os.environ.get('V2T_RT_FEAT', '1'))          # 隠れた音符(ringtrack)の白い縁の円と画素の種類も、検出のデコードで全フレーム取る(ringtrack が動画を読み直すと 1 曲 20 秒)


def _rt_feat(canvas):
    """ringtrack 用: (白い縁の円 [(x, 色)], 画素の種類を zlib で縮めたもの)"""
    import zlib, synth, ringtrack
    return synth.rings(canvas), zlib.compress(ringtrack._labels(canvas).tobytes(), 1)


def _run_ff(cmd, timeout):
    """ffmpeg を時間制限つきで動かす。GPU のデコードがまれに止まったまま返らない(闇の魔法少女: 30 分止まった)ので、区切りの長さの 10 倍で打ち切り、
    失敗として返す(GPU なら CPU でやり直す)"""
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as ex:
        return subprocess.CompletedProcess(cmd, -9, ex.stdout or b'', ex.stderr or b'')


def _decode_chunk_worker(args):
    """ffmpeg でデコードする版。GPU(NVDEC)があれば帯の行だけデコーダで切り出して受け取る(全面を BGR にしない)。
    1080p60 で 1 プロセス 2300fps(OpenCV は全面の変換で並列にしても 550fps 止まり)。戻り値は _decode_chunk_cv2 と同じ"""
    video, i0, i1, t_start, src_rows, M, canon_w, canon_h, x1, pts_all, fps_hint = args[:11]
    gpu = args[11] if len(args) > 11 else None
    gf = args[12] if len(args) > 12 else None          # (ゴーゴーの最初のフレーム, 最後, 間隔): その間隔のフレームでゴーゴーの特徴も取る
    global X1
    X1 = x1
    cv2.setNumThreads(1)
    y0, y1 = src_rows; h = y1 - y0
    if video not in _CODEC:
        _CODEC[video] = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height',
                                        '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')
    W, H = int(_CODEC[video][1]), int(_CODEC[video][2])
    if M is not None:            # 色差は 2 行ごとなので、切り出しの行は偶数にそろえる(変換の位置を OpenCV の全面変換と同じにする)
        y0 -= y0 % 2; y1 = min(H, y1 + y1 % 2); h = y1 - y0
    n = i1 - i0
    # 較正で変形しない動画(1080p の基準配置)は、使う列(x ≥ EM_X0 = 535)だけを切り出してデコードする(変換・書き出し・読み込みが 27% 減る)。
    # 切り出しの左端は色差の 2 列ごとにそろえ、色差の補間の端の影響(数列)が使う列に届かないよう 15 列の余裕を取る
    xc = 520 if (M is None and W == canon_w) else 0; Wc = W - xc
    # 各フレームの実際の表示時刻は showinfo(stderr)から読む(-ss の着地が ±1 フレームずれても正しい番号を付けられる)
    import re as _re
    # -copyts で元の表示時刻を保つ。-t は入力側(‐i の前)に置く(出力側だと copyts 後の時刻と比べられて何も出ない)
    ss = ['-ss', '%.4f' % max(0.0, t_start - 0.5), '-t', '%.4f' % ((n / fps_hint) + 1.0), '-copyts']
    # 色の変換は OpenCV と同じにする(OpenCV は動画のタグ(BT.709)を見ずに BT.601・bicubic で BGR にする。ffmpeg の既定は G/R が +4〜5 ずれ、
    # VERTeX で偽の音符が 13 個増えた)。NVDEC の出力(NV12)は一度 yuv420p にしてから変換すると画素まで一致する
    conv = 'format=yuv420p,scale=in_color_matrix=bt601:flags=bicubic'
    if gpu:
        # -hwaccel cuda(ffmpeg のデコーダの上で NVDEC を使う)。h264_cuvid は B フレームの並べ替えで 5 枚に 1 枚ほど別の時刻の絵を出し
        # (段位エンドリ 23.55s など、余分な音符が出た)、av1_cuvid は -crop が効かなかった。-hwaccel cuda は CPU と画素まで一致
        cmd = ['ffmpeg', '-v', 'info', '-nostats', '-hwaccel', 'cuda'] + ss + ['-i', video, '-vf', 'crop=%d:%d:%d:%d,%s,showinfo' % (Wc, h, xc, y0, conv)]
    else:
        cmd = ['ffmpeg', '-v', 'info', '-nostats', '-threads', '2'] + ss + ['-i', video, '-vf', 'crop=%d:%d:%d:%d,%s,showinfo' % (Wc, h, xc, y0, conv)]
    # Windows の匿名パイプは 120MB/s しか出ない(1 枚 1.1MB を 9ms)ので、一時ファイルに書かせて読む(メモリのキャッシュに乗る: 10 倍速い)
    import tempfile
    tmp = os.path.join(os.environ.get('V2T_TMP') or tempfile.gettempdir(), 'v2t_%d_%d.raw' % (os.getpid(), i0))
    cmd += ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-y', tmp]
    import time as _tw; _tw0 = _tw.time()
    r = _run_ff(cmd, 60 + 10 * n / fps_hint)
    _tw1 = _tw.time()
    pts_list = [float(m) for m in _re.findall(rb'pts_time:\s*([-\d.]+)', r.stderr)]
    if gpu and (r.returncode != 0 or len(pts_list) < 0.95 * n):
        # GPU のデコードが失敗・途中で切れた(何曲も同時にデコードすると NVDEC が足りなくなり、区切りごと空で返って音符が数百個抜けた)。CPU でやり直す
        cmd = ['ffmpeg', '-v', 'info', '-nostats', '-threads', '2'] + ss + ['-i', video, '-vf', 'crop=%d:%d:%d:%d,%s,showinfo' % (Wc, h, xc, y0, conv)] +               ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-y', tmp]
        r = _run_ff(cmd, 60 + 20 * n / fps_hint)
        pts_list = [float(m) for m in _re.findall(rb'pts_time:\s*([-\d.]+)', r.stderr)]
    # 一時ファイルを 1 フレームずつ読んで、その場で候補・特徴を出す(区切り全部を読み込んでから処理すると、帯 600MB + ゴーゴー用の写し 540MB で
    # 1 プロセス 1GB になり、CPU 全部(29 プロセス)で 1 曲 27GB、何曲も同時に回すと 128GB でも足りなかった)
    nbytes = Wc * h * 3
    canvas = np.zeros((canon_h, canon_w, 3), np.uint8)
    b0, b1 = EXT_ROWS
    Mw = None
    if M is not None:
        s_ = float(M[0, 0]); Mw = np.array([[s_, 0, M[0, 2]], [0, s_, s_ * y0 + M[1, 2] - b0]], np.float32)
    out = []; cv = np.zeros((1080, 1920, 3), np.uint8); prev_band = None
    try:
        with open(tmp, 'rb') as fh:
            k = -1
            while True:
                buf = fh.read(nbytes)
                if len(buf) < nbytes: break
                k += 1
                crop = np.frombuffer(buf, np.uint8).reshape(h, Wc, 3)
                if Mw is not None:
                    canvas[b0:b1] = cv2.warpAffine(crop, Mw, (canon_w, b1 - b0), flags=cv2.INTER_LINEAR)
                else:
                    canvas[b0:b1, xc:] = crop[:b1 - b0]
                band = cv2.cvtColor(canvas[350:420, 760:1880], cv2.COLOR_BGR2GRAY).astype(np.float32)
                dif = float(np.abs(band - prev_band).mean()) if prev_band is not None else 99.0
                prev_band = band
                if k >= len(pts_list): continue
                # 表示時刻 → 全体のフレーム番号(pts 一覧の中の位置)。担当範囲 [i0, i1) だけ返す
                # showinfo の pts_time は 3 桁程度に丸められるので、最も近い pts の番号を採る(切り捨て側だと 1 フレームずれる)
                j = int(np.searchsorted(pts_all, pts_list[k]))
                cand = [jj for jj in (j - 1, j) if 0 <= jj < len(pts_all)]
                idx = min(cand, key=lambda jj: abs(pts_all[jj] - pts_list[k]))
                if not (i0 <= idx < i1): continue
                c = candidates(canvas); b = bar_candidates(canvas); row = band.mean(0)
                g = None; rt = None
                if gf:
                    cv[250:516] = canvas[250:516]
                    if gf[0] <= idx < gf[1] and (idx - gf[0]) % gf[2] == 0:
                        import gogo
                        g = _gogo_feat(cv, full=idx % gogo.STEP == 0)     # 間引きの位相は動画のフレーム番号で決める(区間の始まり = Δ で決めると、Δ が
                        #   27ms 変わっただけで見本のフレームが変わり、点滅するレーンで偽のゴーゴーが出た: 倍倍FIGHT! 61.5s)
                    if RT_FEAT: rt = _rt_feat(cv)
                out.append((idx, c, b, row, dif, g, rt))
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if os.environ.get('V2T_WTIME'):
        with open(os.environ['V2T_WTIME'], 'a') as fh_: fh_.write('%d ff %.2f loop %.2f n %d start %.2f' % (i0, _tw1 - _tw0, _tw.time() - _tw1, len(out), _tw0 % 1000) + chr(10))
    if len(out) < 0.95 * n and len(args) > 11 and args[11] is not None:
        # 受け取れた枚数が足りない(何曲も同時にデコードすると GPU 側で区切りの一部が欠け、曲の頭の 4 秒ぶんの音符が抜けた: TDM 172〜178s)。CPU でやり直す
        return _decode_chunk_worker(tuple(args[:11]) + (None,) + tuple(args[12:]))
    return out

def _decode_chunk_cv2(args):
    video, i0, i1, t_start, src_rows, M, canon_w, canon_h, x1, pts_all, fps_hint = args
    global X1
    X1 = x1
    y0, y1 = src_rows
    cv2.setNumThreads(1)          # 多数のプロセスで並べるので、各プロセスの OpenCV は 1 スレッド(既定の 32 スレッドずつだと食い合って 1/10 の速さ)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i0 - 30))      # 少し手前から読み、時刻で先頭を合わせる
    canvas = np.zeros((canon_h, canon_w, 3), np.uint8)
    b0, b1 = EXT_ROWS
    Mw = None
    if M is not None:
        s_ = float(M[0, 0]); Mw = np.array([[s_, 0, M[0, 2]], [0, s_, s_ * y0 + M[1, 2] - b0]], np.float32)
    out = []; idx = None; prev_band = None
    while True:
        ok, fr = cap.read()
        if not ok: break
        if idx is None:
            ms = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0        # 読んだフレームの表示時刻 → 全体のフレーム番号
            j = int(np.searchsorted(pts_all, ms)); cand = [jj for jj in (j - 1, j) if 0 <= jj < len(pts_all)]
            idx = min(cand, key=lambda jj: abs(pts_all[jj] - ms))
        else:
            idx += 1
        if idx >= i1: break
        if idx < i0 - 1: continue
        crop = fr[y0:y1]
        if Mw is not None:
            canvas[b0:b1] = cv2.warpAffine(crop, Mw, (canon_w, b1 - b0), flags=cv2.INTER_LINEAR)
        else:
            canvas[b0:b1] = crop[:b1 - b0]
        # 複製フレーム・ゲームの時計用(prep60): レーンの行プロファイルと直前との画素差(y350-420 × x760-1880)
        band = cv2.cvtColor(canvas[350:420, 760:1880], cv2.COLOR_BGR2GRAY).astype(np.float32)
        dif = float(np.abs(band - prev_band).mean()) if prev_band is not None else 99.0
        prev_band = band
        if idx < i0: continue                                  # 直前の 1 枚は画素差のためだけに読む
        out.append((idx, candidates(canvas), bar_candidates(canvas), band.mean(0), dif))
    cap.release()
    return out

# 音符の種類ごとの位置の偏り[px](検出した中心 − 本当の中心、流れる向きに正)。試験スイート 63 曲・4.4 万音符で、
# 参照譜面との時刻差 × 速さ を種類ごとに集めた中央値(速さによらずほぼ一定なので時間ではなく位置の偏り):
# 当初 ドン −0.45、カッ +0.55、大ドン +3.2、大カッ +4.7px(大音符の多くは判定枠で消えていく点のずれで、_refine_fit で外すと 1.6 / 3.1px)。大音符は 3〜5ms 遅く出て(遅い曲ほど大きい)、大音符の多い小節が
# グリッドから 3ms 以上外れて音符ごとの BPM 指定になっていた(VERTeX 25 小節)。V2T_TYPEBIAS=0 で無効
TYPE_BIAS_PX = {('don', False): -0.43, ('ka', False): 0.58, ('don', True): 1.60, ('ka', True): 3.08}   # 2026-09-22 大音符の判定枠付近の点を外した後(_refine_fit)に測り直し(regress/type_bias.py、60 曲)


def _unbias(onsets, fps):
    if os.environ.get('V2T_TYPEBIAS', '1') == '0': return
    for o in onsets:
        d = TYPE_BIAS_PX.get((o.get('color'), bool(o.get('big'))))
        v = abs(o.get('vel') or 0)
        if d and v > 1: o['t'] = round(o['t'] - d / (v * fps), 4)


def band_start(video, t0, t1, rows, step=1, cal=None, parts=None, tag=0):
    """iter_band の前半: ffmpeg(GPU があれば NVDEC)を parts 本同時に起動して、帯だけを一時ファイルに書かせる。
    戻り値のハンドルを band_iter に渡す(その間に別の処理をしてよい)"""
    import calib, tempfile
    M = None
    if cal is not None and not calib.is_canonical(cal): M, _ = calib.affine(cal)
    if video not in _CODEC:
        _CODEC[video] = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height',
                                        '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')
    W, H = int(_CODEC[video][1]), int(_CODEC[video][2])
    b0, b1 = rows
    if M is not None:
        s_ = float(M[0, 0]); ty = float(M[1, 2])
        y0 = max(0, int(np.floor((b0 - ty) / s_)) - 2); y1 = min(H, int(np.ceil((b1 - ty) / s_)) + 3)
    else:
        y0, y1 = b0, b1
    y0 -= y0 % 2; y1 = min(H, y1 + y1 % 2); h = y1 - y0
    if (video, 'rate') not in _CODEC:
        _CODEC[(video, 'rate')] = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video],
                                                 capture_output=True, text=True).stdout.strip().split(',')[0]
    rate = _CODEC[(video, 'rate')]
    try: num, den = rate.split('/'); fps = float(num) / float(den)
    except Exception: fps = 60.0
    gpu = _gpu_decoder(video)
    if parts is None: parts = min(4 if gpu else 8, cpu_budget())
    conv = 'format=yuv420p,scale=in_color_matrix=bt601:flags=bicubic'
    # 間引きは表示時刻で決める(区間ごとに n が 0 から数え直しになるので、フレーム番号 = round(t×fps) の剰余で選ぶ)
    f_first = int(round(t0 * fps))
    sel = r"select='eq(mod(round(t*%.6f)-%d\,%d)\,0)'," % (fps, f_first, step) if step > 1 else ''
    procs = []
    edges = np.linspace(t0, t1, parts + 1)
    for k in range(parts):
        a_, b_ = edges[k], edges[k + 1]
        tmp = os.path.join(os.environ.get('V2T_TMP') or tempfile.gettempdir(), 'v2t_band_%d_%d_%s_%d.raw' % (os.getpid(), __import__('threading').get_ident(), tag, k))   # スレッドごとに別名(gogo の測り直しを並列で呼ぶ)
        # 区間の境目で取りこぼし・重複が出ないよう、少し広めに読んで表示時刻で [a_, b_) に切る
        ss = ['-ss', '%.4f' % max(0.0, a_ - 0.5), '-t', '%.4f' % (b_ - a_ + 1.0), '-copyts']
        if gpu:
            cmd = ['ffmpeg', '-v', 'info', '-nostats', '-hwaccel', 'cuda'] + ss + ['-i', video, '-vf', 'crop=%d:%d:0:%d,%s%s,showinfo' % (W, h, y0, sel, conv)]   # h264_cuvid は絵を取り違える(上記)
        else:
            cmd = ['ffmpeg', '-v', 'info', '-nostats', '-threads', '2'] + ss + ['-i', video, '-vf', 'crop=%d:%d:0:%d,%s%s,showinfo' % (W, h, y0, sel, conv)]
        cmd += ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-y', tmp]
        errf = open(tmp + '.log', 'wb')
        procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf), errf, tmp, a_, b_))
    return {'procs': procs, 'W': W, 'h': h, 'y0': y0, 'M': M, 'rows': rows, 'fps': fps}


def band_iter(hd):
    """band_start のハンドルから (フレーム番号, キャンバス) を時刻順に返す。キャンバスは使い回し"""
    import re as _re
    W, h, y0, M, fps = hd['W'], hd['h'], hd['y0'], hd['M'], hd['fps']; b0, b1 = hd['rows']
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    Mw = None
    if M is not None:
        s_ = float(M[0, 0]); Mw = np.array([[s_, 0, M[0, 2]], [0, s_, s_ * y0 + M[1, 2] - b0]], np.float32)
    nbytes = W * h * 3; seen = set()
    try:
        for p, errf, tmp, a_, b_ in hd['procs']:
            p.wait(); errf.close()
            pts = [float(m) for m in _re.findall(rb'pts_time:\s*([-\d.]+)', open(tmp + '.log', 'rb').read())]
            n = os.path.getsize(tmp) // nbytes if os.path.exists(tmp) else 0
            if not n: continue
            raw = np.memmap(tmp, np.uint8, 'r', shape=(n, h, W, 3))
            for k in range(min(n, len(pts))):
                if not (a_ - 1e-6 <= pts[k] < b_ - 1e-6): continue
                f = int(round(pts[k] * fps))
                if f in seen: continue
                seen.add(f)
                crop = raw[k]                 # キャンバスへ写す(下)ので複製しない。参照は下の finally で切る(残ると Windows では一時ファイルを消せず、1 日で 635GB 溜まった)
                if Mw is not None: canvas[b0:b1] = cv2.warpAffine(crop, Mw, (1920, b1 - b0), flags=cv2.INTER_LINEAR)
                else: canvas[b0:b1] = crop[:b1 - b0]
                yield f, canvas
            del raw
    finally:
        raw = crop = None
        import gc; gc.collect()
        for p, errf, tmp, a_, b_ in hd['procs']:
            try: p.kill()
            except Exception: pass
            try: errf.close()
            except Exception: pass
            for q in (tmp, tmp + '.log'):
                try: os.remove(q)
                except OSError: pass


def iter_band(video, t0, t1, rows, step=1, cal=None):
    """動画の [t0, t1) を step 枚おきに読み、基準配置のキャンバス(1080×1920)の rows=(y0, y1) の行だけを埋めて返す。
    yield (フレーム番号, キャンバス)。ffmpeg(GPU があれば NVDEC)で帯だけ切り出し、何本か同時に一時ファイルへ書かせて読む
    (OpenCV で全フレームを読むより 5〜10 倍速い)"""
    return band_iter(band_start(video, t0, t1, rows, step, cal))


def cache_dir():
    """キャッシュの置き場所(V2T_CACHE。0 で無効)。既定は V2T_WORK/cache か一時フォルダ"""
    c = os.environ.get('V2T_CACHE')
    if c == '0': return None
    if not c:
        import tempfile
        c = os.path.join(os.environ.get('V2T_WORK') or os.path.join(tempfile.gettempdir(), 'video2tja'), 'cache')
    os.makedirs(c, exist_ok=True)
    return c


def src_hash(*parts):
    import hashlib
    h = hashlib.sha1()
    for x in parts: h.update(repr(x).encode('utf-8', 'replace'))
    return h.hexdigest()[:20]


def video_id(video):
    st = os.stat(video)
    return (os.path.realpath(video), st.st_size, int(st.st_mtime))     # 実体のパス(開発用コピーの testvideos はジャンクションなので本体とキャッシュを共有する)


def _cand_code():
    """候補抽出の段階のコード(ファイル冒頭〜追跡の前、デコード関連の関数、全部の定数行、gogo の定数)。これが変われば候補のキャッシュは無効"""
    import inspect, re as _re, gogo
    txt = open(__file__, encoding='utf-8').read()
    head = txt[:txt.index('def _refine_fit(')]
    consts = '\n'.join(x for x in _re.findall(r'^[A-Z][A-Z0-9_]*\s*=.*$', txt, _re.M) if not x.startswith('TYPE_BIAS_PX'))   # 偏りの補正は追跡の後なので候補に関係しない
    import synth, ringtrack
    fns = ''.join(inspect.getsource(f) for f in (_decode_chunk_worker, _decode_chunk_cv2, _gpu_decoder, _gogo_feat, _rt_feat, synth.rings, ringtrack._labels))
    fns += repr((synth.CY, ringtrack.CY, ringtrack.HB, ringtrack.LX0))
    return src_hash(head, consts, fns, gogo.ROWS, gogo.X0, gogo.X1, gogo.STEP)


def _live_write(path, data):
    """画面(web)の途中経過を書く(変換の結果には使わない)。書きかけを読まれないよう一時ファイルから置き換える"""
    try:
        tmp = path + '.%d.tmp' % os.getpid()
        with open(tmp, 'w', encoding='utf-8') as fh: json.dump(data, fh, ensure_ascii=False, default=float)
        os.replace(tmp, path)
    except Exception:
        pass


class _LiveProv:
    """読み終えた区切りの候補から仮の音符を見積もり、画面(web)に出す(途中経過だけ。検出の結果には使わない)。
    区切りごとのレーンの流れ(行の相互相関の中央値 px/フレーム)でドン・カッの候補を判定枠まで延ばし、読み終えた全部の区切りの点を
    まとめて、同じ色で到着が近い点の集まりを 1 つの音符とする。区切りは読み終えた順(前後する)に来るので、来るたびに全部から数え直す。
    速さの違う音符(SCROLL の仕掛け)は時刻がずれるが、追跡が終われば本物に置き換わる"""
    min_pts = 6      # 1 つの音符とみなす点の数(音符は判定枠に着くまで数十フレーム写る。それより少ないのは欠片・幽霊)
    gap = 0.8        # 同じ音符の点の到着のばらつき(フレーム)。流れの速さの誤差と円の中心のぶれで ±0.5 程度

    def __init__(self, path, pts, f_first, f_last):
        self.path = path; self.pts = np.asarray(pts, float); self.f_first = f_first; self.f_last = f_last; self.chunks = {}; self.ranges = []

    def _t(self, f):
        return float(np.interp(f, np.arange(len(self.pts)), self.pts))

    def add(self, items, shifts=None):
        """items: [(フレーム, 候補, …, 行, …)] 読み終えた区切り 1 つ(か、キャッシュから全部)。shifts: 流れ(無ければ行から求める)"""
        try:
            items = sorted(items, key=lambda it: it[0])
            if not items: return
            if shifts is None:
                rows = [it[3] for it in items if len(it) > 3 and it[3] is not None]
                import prep60
                shifts = prep60.shifts_from_rows(rows) if len(rows) >= 3 else []
            sh = np.asarray(shifts, float); sh = sh[sh > 0.5]
            fa, ka, bg, vv = [], [], [], []
            if len(sh):
                v = float(np.median(sh))
                for it in items:
                    for c in it[1]:
                        if c.get('color') in ('don', 'ka') and c.get('w', 0) >= 20 and JUDGE_X + 40 <= c['cx'] <= X1 - 20:
                            fa.append(it[0] + (c['cx'] - JUDGE_X) / v); ka.append(c['color'] == 'ka'); bg.append(bool(c.get('big'))); vv.append(v)
            self.chunks[items[0][0]] = (np.array(fa, float), np.array(ka, bool), np.array(bg, bool), np.array(vv, float))
            self.ranges.append((items[0][0], items[-1][0]))
            notes = []
            ch = [x for x in self.chunks.values() if len(x[0])]
            if ch:
                FA = np.concatenate([x[0] for x in ch]); KA = np.concatenate([x[1] for x in ch])
                BG = np.concatenate([x[2] for x in ch]); VV = np.concatenate([x[3] for x in ch])
                for k_ in (False, True):
                    m = KA == k_; o = np.argsort(FA[m]); f = FA[m][o]; b = BG[m][o]; v = VV[m][o]
                    if not len(f): continue
                    cut = np.flatnonzero(np.diff(f) > self.gap) + 1
                    for s_, e_ in zip(np.r_[0, cut], np.r_[cut, len(f)]):
                        if e_ - s_ >= self.min_pts:
                            notes.append([round(self._t(float(np.median(f[s_:e_]))), 4), 'ka' if k_ else 'don', bool(b[s_:e_].sum() * 2 > e_ - s_),
                                          round(float(np.median(v[s_:e_])), 2)])
            notes.sort()
            mg = []
            for a_, b_ in sorted(self.ranges):
                if mg and a_ <= mg[-1][1] + 1: mg[-1][1] = max(mg[-1][1], b_)
                else: mg.append([a_, b_])
            upto = self._t(mg[0][1]) if mg and mg[0][0] <= self.f_first + 1 else None      # 頭から続けて読み終えた所
            _live_write(self.path, {'stage': '検出(読み込み中・仮の音符)', 'upto': upto, 'ranges': [[round(self._t(a_), 3), round(self._t(b_), 3)] for a_, b_ in mg],
                                    'span': [round(self._t(self.f_first), 3), round(self._t(self.f_last), 3)], 'prov': notes})
        except Exception:
            pass


GOGO_FEAT_STEP = 1      # ゴーゴーの特徴は全フレームで取る(gogo.py は STEP おきに使い、境目・短い区間の全フレームの測り直しもここから引く。
                        # 動画を読み直すとキーフレームからのデコードが毎回 0.5 秒かかり、1 曲 10 秒)


def run_parallel(video, start, end, dump=None, dbg=False, cal=None, workers=None, decoder='cv2', skip_frames=None, clock=None, auto_dups=False, gogo_range=None):
    """run() と同じ結果を並列で。戻り値も同じ (fps, onsets)。run_parallel.bars に小節線。
    skip_frames: 候補を空にするフレーム番号の集合(120fps 録画を 60fps に直したとき、新しい絵が無くて直前の絵を入れたスロット。
    古い絵の点(1 フレームぶん後ろ)を追跡に混ぜると時刻が 3ms 偏り幽霊も出るので、そのフレームは「無い」ことにする)
    clock: {フレーム番号: ゲームの描画フレーム番号}(prep60.game_clock)。録画が挟んだ複製は捨て、落ちたフレームは空けて、
    ゲームのフレーム順に並べて追跡する。時刻は「ゲームのフレーム → 動画のフレーム」の直線で戻す(全体の長さは変えない)"""
    import calib
    from concurrent.futures import ProcessPoolExecutor
    import time as _tm; _t0 = _tm.time()
    pts = probe_pts(video)
    if dbg: print('[検出内訳] pts %.1fs' % (_tm.time() - _t0))
    rate = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
                           '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')[0]
    try:
        num, den = rate.split('/'); fps = float(num) / float(den)
    except Exception:
        fps = (len(pts) - 1) / (pts[-1] - pts[0])
    i0 = int(np.searchsorted(pts, start)); i1 = int(np.searchsorted(pts, end))
    M = None; x1 = 1910
    if cal is not None and not calib.is_canonical(cal):
        M, x1 = calib.affine(cal)
        s_ = float(M[0, 0]); ty = float(M[1, 2])
        y0 = max(0, int(np.floor((EXT_ROWS[0] - ty) / s_))); y1 = min(cal['height'], int(np.ceil((EXT_ROWS[1] - ty) / s_)) + 1)
        if dbg: print('calibration: warp scale %.4f tx %.1f ty %.1f  X1=%d  src rows %d-%d' % (s_, M[0, 2], ty, x1, y0, y1))
    else:
        y0, y1 = EXT_ROWS
    if workers is None:
        workers = int(os.environ.get('V2T_WORKERS') or min(28, cpu_budget()))
    n = i1 - i0
    # チャンクは 300 枚以下(ffmpeg の一時ファイルが 1 枚 1.1MB なので、大きいと同時に数十 GB になる)。プロセスはワーカー数ぶん
    nch = max(workers, int(np.ceil(n / float(os.environ.get('V2T_CHUNK', '300')))))
    per = int(np.ceil(n / nch))
    jobs = []
    for w in range(nch):
        a = i0 + w * per; b = min(i1, a + per)
        if a >= b: break
        jobs.append((video, a, b, float(pts[a]), (y0, y1), M, calib.CANON_W, calib.CANON_H, x1, pts, fps))
    results = {}; rows = {}; difs = {}; gfeat = {}; rtfeat = {}
    gpu = _gpu_decoder(video) if decoder == 'cv2' else None
    gf = None
    if gogo_range:                              # ゴーゴーの特徴も同じデコードで取る(別に読むと GPU の奪い合いで 1 曲 25 秒)
        import gogo
        gf = (int(round(gogo_range[0] * fps)), int(round(gogo_range[1] * fps)), GOGO_FEAT_STEP)
    if gpu or gf:                               # GPU があれば ffmpeg(NVDEC)で帯だけデコードする(OpenCV の 5〜10 倍速い)
        decoder = 'ffmpeg'; jobs = [j + (gpu, gf) for j in jobs]
    worker = _decode_chunk_cv2 if decoder == 'cv2' else _decode_chunk_worker
    if dbg: print('デコード: %s' % (gpu or decoder))
    # 候補のキャッシュ: 動画・範囲・較正・候補抽出のコードが同じなら、前回のフレームごとの候補を使う(検証で追跡やビルダーだけ直したとき
    # デコードと候補抽出(1 曲 CPU 50 秒)を省く)。候補抽出のコードや定数を変えるとキーが変わって自動でやり直す
    import pickle
    cdir = cache_dir(); cpath = None; shp = None
    key = src_hash(video_id(video), i0, i1, None if M is None else np.round(M, 6).tolist(), x1, y0, y1, gf, _cand_code(), 'dec-v4')   # dec-v2: h264_cuvid をやめた。dec-v3: GPU のデコードが途中で切れたら CPU でやり直す(欠けたキャッシュは読み込み時に枚数で弾く)。dec-v4: 流れを小数で持つ
    if cdir: cpath = os.path.join(cdir, 'cand_%s.pkl' % key)
    # 同じプロセスの 2 回目(録画の落ちを直した検出のやり直し: 候補は同じ)は、前回の候補をメモリから使う。キャッシュを切った実行(一括試験 V2T_CACHE=0)では
    # やり直しのたびにデコードし直していた(1100 本中 102 本、1 本 45 秒)
    # (ディスクのキャッシュと同じく、後の処理が書き換えない複製を渡す: 固めたものを持っておき、使うたびに戻す)
    # 画面(web)の途中経過(V2T_LIVE): 読み終えた区切りから仮の音符を出す。検出の結果には使わない
    live_ = _LiveProv(os.environ['V2T_LIVE'], pts, i0, i1 - 1) if os.environ.get('V2T_LIVE') else None
    mem_ = getattr(run_parallel, '_cand_mem', None)
    if mem_ is not None and mem_[0] == key:
        results, shp, difs, gfeat, rtfeat = pickle.loads(mem_[1])
        if dbg: print('候補は同じ実行の前回のものを使う')
    elif cpath and os.path.exists(cpath):
        with open(cpath, 'rb') as fh: results, shp, difs, gfeat, *rest_ = pickle.load(fh)
        rtfeat = rest_[0] if rest_ else {}
        if len(results) < 0.98 * (i1 - i0):          # 枚数が足りない(GPU のデコードの欠け)キャッシュは使わない
            if dbg: print('候補のキャッシュの枚数が足りない(%d / %d)ので作り直す' % (len(results), i1 - i0))
            results = {}; shp = None; difs = {}; gfeat = {}; rtfeat = {}; cpath_bad = True
        elif dbg: print('候補のキャッシュを使う: %s' % os.path.basename(cpath))
    if not results:
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'threads;2')     # デコーダのスレッドも絞る(子プロセスに引き継がれる)
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            # 読み終えた区切りから受け取る(頭から順に待つと、遅い区切り 1 つの後ろが全部待たされ、画面の途中経過がまとめて増えた)
            futs_ = [ex.submit(worker, jb) for jb in jobs]
            for k_, fu_ in enumerate(as_completed(futs_)):
                chunk = fu_.result()
                progress('検出', k_ + 1, len(jobs), '動画の区切りを読んで丸を探した(%d 本で並べて)' % min(workers, len(jobs)))
                for it in chunk:
                    results[it[0]] = (it[1], it[2])
                    if len(it) > 3: rows[it[0]] = it[3]; difs[it[0]] = it[4]
                    if len(it) > 5 and it[5] is not None: gfeat[it[0]] = it[5]
                    if len(it) > 6 and it[6] is not None: rtfeat[it[0]] = it[6]
                if live_: live_.add(chunk)
        # 読み終えた順に集めたので、フレームの順に並べ直す(以前の ex.map は頭から順に入った。後の処理が辞書の順を使っても同じになるように)
        results = dict(sorted(results.items())); rows = dict(sorted(rows.items())); difs = dict(sorted(difs.items()))
        gfeat = dict(sorted(gfeat.items())); rtfeat = dict(sorted(rtfeat.items()))
        if dbg: print('[検出内訳] 並列の処理 %.1fs' % (_tm.time() - _t0))
        if rows:
            import prep60
            ii = [i for i in sorted(results) if i in rows]
            shp = (ii, prep60.shifts_from_rows([rows[i] for i in ii]))
        rows = None
        if dbg: print('[検出内訳] 流れの計算 %.1fs' % (_tm.time() - _t0))
        if cpath:
            tmpc = cpath + '.%d.tmp' % os.getpid()
            with open(tmpc, 'wb') as fh: pickle.dump((results, shp, difs, gfeat, rtfeat), fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmpc, cpath)
        if not cpath: run_parallel._cand_mem = (key, pickle.dumps((results, shp, difs, gfeat, rtfeat), protocol=pickle.HIGHEST_PROTOCOL))
    elif live_ and shp and not os.path.exists(live_.path):
        # 候補をキャッシュから使ったとき(読み込みが無い)は、全部の候補から一度に仮の音符を出す(やり直しの 2 回目は出さない: 本物の音符を上書きしない)
        live_.add([(i, results[i][0]) for i in sorted(results)], shifts=shp[1])
    idxs = sorted(results)
    # デコードで欠けたフレーム(区切りの ffmpeg が途中で切れた・キャッシュは 98% 以上で使う)は、空のフレームとして番号の位置を保つ。
    # 追跡はリストの位置をフレーム番号にするので、欠けを詰めると以降が 1 枚ずつ前にずれ、音符の時刻が遅れていく
    # (一括試験 オペレート・ミー: 162s で 3 フレーム = 50ms 遅れ、本家との時刻差が曲の中で −82 → −116ms と流れていた)
    if idxs:
        miss_ = [i for i in range(idxs[0], idxs[-1] + 1) if i not in results]
        if miss_:
            if dbg: print('デコードで欠けたフレーム %d 枚: 空のフレームとして番号を保つ(%s)' % (len(miss_), ' '.join(str(i) for i in miss_[:10])))
            for i in miss_: results[i] = ([], [])
            idxs = sorted(results)
    if dbg: print('[検出内訳] デコード完了 %.1fs' % (_tm.time() - _t0))
    if auto_dups and shp and not skip_frames and not clock:
        # 複製フレーム(30fps 相当の動画)とゲームの時計(録画が挟んだ複製・落としたフレーム)をデコードしたフレームから決める
        import prep60
        ii, sh = shp
        dups, frac = prep60.dups_from_shifts(ii, sh, dbg)
        if frac >= 0.2:
            skip_frames = dups
            if dbg: print('複製フレームが %.0f%% あるので外して検出する(30fps 相当の動画)' % (frac * 100))
        elif os.environ.get('V2T_CLOCK', '1') != '0':
            clock = prep60.game_clock(ii, sh, dbg, diff=np.array([difs[i] for i in ii]),
                                      cands={i: [(c['color'], c['cx']) for c in results[i][0] if c['color'] in ('don', 'ka') and 800 <= c['cx'] <= 1800] for i in ii})
            if prep60.game_clock.blank: skip_frames = set(prep60.game_clock.blank)     # 処理落ちの組の間の、予定とずれた絵
    if skip_frames:
        ns = 0
        for i in idxs:
            if i in skip_frames: results[i] = ([], []); ns += 1
        if dbg: print('skip_frames: %d frames の候補を空にした(古い絵)' % ns)
    if dbg: print('parallel decode: %d frames in %d chunks (%d workers), fps %.2f' % (len(idxs), len(jobs), min(workers, len(jobs)), fps))
    frames_c = [results[i][0] for i in idxs]; frames_b = [results[i][1] for i in idxs]
    f0 = idxs[0] if idxs else i0
    lin = None
    if clock and idxs:
        ks = np.array(sorted(clock)); gs = np.array([clock[k] for k in ks]); slip = gs - ks
        def g_of(i):
            j = int(np.clip(np.searchsorted(ks, i), 0, len(ks) - 1)); return int(i + slip[j])
        gmap = {}
        for i in idxs: gmap.setdefault(g_of(i), i)              # 同じゲームフレームが 2 枚あれば先の 1 枚
        g0, g1 = min(gmap), max(gmap)
        frames_c = [results[gmap[g]][0] if g in gmap else [] for g in range(g0, g1 + 1)]
        frames_b = [results[gmap[g]][1] if g in gmap else [] for g in range(g0, g1 + 1)]
        f0 = g0
        lin = np.polyfit(gs.astype(float), ks.astype(float), 1)    # ゲームフレーム → 動画フレーム
        if dbg: print('clock: ゲームのフレーム順に並べ直した(%d 枚 → %d 枚、動画フレーム = %.6f × ゲーム %+.2f)' % (len(idxs), len(frames_c), lin[0], lin[1]))
    if dbg: print('[検出内訳] 追跡前 %.1fs' % (_tm.time() - _t0))
    if os.environ.get('V2T_HASH'):
        import hashlib
        fa_, fb_ = [int(x) for x in os.environ['V2T_HASH'].split(':')]
        for g_ in range(fa_, fb_):
            k_ = g_ - f0
            if 0 <= k_ < len(frames_c): print('HASH %d %d %s  %s' % (g_, len(frames_c[k_]), hashlib.md5(repr(sorted((c['color'], round(c['cx'], 1)) for c in frames_c[k_])).encode()).hexdigest()[:8], ' '.join('%s@%.0f(w%s)' % (c['color'][:2], c['cx'], c.get('w')) for c in sorted(frames_c[k_], key=lambda c: c['cx']))))
    track.frac = None; track.frac_alt = None
    if shp and os.environ.get('V2T_FRAC', '0') != '1' and os.environ.get('V2T_KINK_CLOCK', '1') != '0':
        # 折れ曲がって捨てる直前の軌跡だけを当てはめ直す小数のゲーム時刻(上の kink)
        import prep60
        fo_ = prep60.frac_offsets(shp[0], shp[1], clock)
        if clock and idxs: track.frac_alt = np.array([fo_.get(gmap[g], 0.0) if g in gmap else 0.0 for g in range(g0, g1 + 1)])
        else: track.frac_alt = np.array([fo_.get(i, 0.0) for i in idxs])
    if os.environ.get('V2T_FRAC', '0') == '1' and shp:
        # 小数のゲーム時刻: 当てはめに使うフレーム時刻 = フレーム番号 + ずれ(prep60.frac_offsets)
        import prep60
        fo = prep60.frac_offsets(shp[0], shp[1], clock)
        if clock and idxs:
            track.frac = np.array([fo.get(gmap[g], 0.0) if g in gmap else 0.0 for g in range(g0, g1 + 1)])
        else:
            track.frac = np.array([fo.get(i, 0.0) for i in idxs])
        if dbg: print('小数のゲーム時刻: ずれ sd %.3f フレーム、|ずれ| > 0.3 は %d 枚' % (float(np.std(track.frac)), int((np.abs(track.frac) > 0.3).sum())))
    track.relink = False
    progress('検出', what='丸をフレームをまたいで追いかける')
    onsets = track(frames_c, fps, f0, dbg)
    if track.relink:
        gd1_ = track.gdelta
        if dbg: print('ゲームの時刻でつなぎ直す(ずれ最大 %.2f フレーム)' % float(np.max(np.abs(gd1_))))
        if dbg and os.environ.get('V2T_GDPRINT'):
            _a, _b = [int(x) for x in os.environ['V2T_GDPRINT'].split(':')]
            print('GD ' + ' '.join('%d:%+.2f%s' % (g_, gd1_[g_ - f0], '' if _game_delta.has[g_ - f0] else '?') for g_ in range(_a, _b) if 0 <= g_ - f0 < len(gd1_)))
        progress('検出', what='ゲームの時刻で追いかけ直す(録画の描画の揺れを直す)')
        onsets = track(frames_c, fps, f0, dbg, tclock=gd1_)
    gdelta_ = track.gdelta
    # 後段(ringtrack)が円の位置を計算するときのゲームの時刻(動画のフレーム f → f + ずれ)。時計で並べ直した動画(clock)はフレーム番号が別なので使わない
    run_parallel.gdelta = (f0, gdelta_) if (gdelta_ is not None and not clock and os.environ.get('V2T_GCLOCK_RT', '1') != '0') else None
    if dbg: print('[検出内訳] 音符の追跡 %.1fs' % (_tm.time() - _t0))
    bo = track(frames_b, fps, f0, False, tclock=gdelta_, gd_fit=gdelta_)
    if dbg: print('[検出内訳] 小節線の追跡 %.1fs' % (_tm.time() - _t0))
    if lin is not None:
        for o in onsets + bo: o['t'] = round(float(np.polyval(lin, o['t'] * fps)) / fps, 4)
    # 動画のフレーム → 音符の時刻と同じ時計のフレーム(ゲームの時計で並べ直して直線で戻したもの)。後段(ringtrack)が位置を計算するのに使う
    run_parallel.skip = set(skip_frames) if skip_frames else set()     # 古い絵のフレーム(30fps の中身の複製など)。後段も使わない
    run_parallel.clk = (np.array(sorted(clock)), np.array([clock[k] for k in sorted(clock)]) - np.array(sorted(clock)), lin) if (clock and lin is not None) else None
    _unbias(onsets, fps)
    bars = [o['t'] for o in bo]
    run_parallel.bars = bars; run_parallel.bar_vels = [o['vel'] for o in bo]
    run_parallel.gogo_feats = (gfeat, fps) if gogo_range else None
    run_parallel.rt_feats = rtfeat or None      # {動画のフレーム番号: (円, 画素の種類)}: ringtrack が動画を読み直さずに使う
    if dump:
        json.dump({'fps': fps, 'start': start, 'end': end, 'onsets': onsets, 'bars': bars, 'bar_vels': run_parallel.bar_vels},
                  open(dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return fps, onsets

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    video = args[0]; start = float(args[1]); end = float(args[2])
    dump = args[3] if len(args) > 3 else None
    import time; t0 = time.time()
    cal = None
    if '--cal' in sys.argv:                          # 配置を自動推定してから検出
        import calib
        cal = calib.calibrate(video, start, end)
    if '--slow' in sys.argv:
        fps, onsets = run(video, start, end, dump, dbg=True, cal=cal)
    else:
        fps, onsets = run_parallel(video, start, end, dump, dbg=True, cal=cal)
        run.bars = run_parallel.bars; run.bar_vels = run_parallel.bar_vels
    print('fps', fps, 'range', start, '-', end, 's  notes', len(onsets), ' bars', len(run.bars), ' (%.0fs)' % (time.time() - t0))
    from collections import Counter
    print('breakdown', dict(Counter((o['color'], o['big']) for o in onsets)))
    for o in onsets[:30]:
        print('  %8.3fs %-7s %s v=%.1f n=%d' % (o['t'], o['color'], 'BIG' if o['big'] else '   ', o['vel'], o['npts']))
