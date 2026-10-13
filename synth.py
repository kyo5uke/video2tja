"""隠れて流れる音符の拾い直し(描いて比べる)。

速さの違う音符が追い越し合う譜面(カオスタイム おに: 速さ 120 の遅いカッが速さ 160 の音符の流れに混ざり、譜面で先の速い音符に上から追い越されて
隠れながら流れる)では、音符の色の塊からは隠れた音符を拾えない。
 1. 白い縁の円(ハフ変換)の中心のうち、見つかっている音符の予想位置で説明できないものを集める
 2. 同じ色で一定の速さの直線に並ぶ組を探す(速さはその曲に出てくる音符の速さの群れに限る)
 3. 直線ごとに、見つかっている音符を見本の形で描き(譜面で先の音符を上に重ねる)、その音符を足すと画面の画素の種類
    (背景・ドンの色・カッの色・白い縁・黒い輪郭)の食い違いがどれだけ減るかを、画面にいる間のフレームで平均する。減る量が大きいものだけ足す
"""
import os, numpy as np, cv2

JX = 615.0          # 判定枠(基準配置)。detect2.JUDGE_X と同じ(音符の時刻はこの位置に着く時刻。以前は 618 で 3px ずれていた)
CY = 386            # レーンの中心の行
HALF = 48
HALF_B = 80         # 大音符の見本の横の半分(白い縁の外まで直径およそ 150px)
R_OUT_B = 76
R_OUT = 46          # 音符の外周(黒い縁まで)の半径
GAIN_MIN = float(os.environ.get('V2T_SYNTH_GAIN', '0.08'))    # 足すのに要る「食い違いの減り」(見えている画素の割合)。カオスタイム 59.05s の遅いカッは 0.13


def labels(img):
    """画素の種類: 0 背景 / 1 ドンの色 / 2 カッの色 / 3 白い縁 / 4 黒い輪郭・顔"""
    h = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    H, S, Vv = h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
    L = np.zeros(H.shape, np.int8)
    L[((H <= 10) | (H >= 177)) & (S >= 150) & (Vv >= 160)] = 1
    L[(H >= 82) & (H <= 92) & (S >= 60) & (Vv >= 150)] = 2
    L[(S < 50) & (Vv > 190)] = 3
    L[Vv < 50] = 4
    return L


def rings(canvas, y_off=0):
    """白い縁の円の中心 [(x, 色)](小音符の大きさ、レーンの高さのものだけ)。canvas が y_off 行目からの切り出しなら y_off を渡す"""
    y0, y1 = CY - 70, CY + 70
    band = canvas[y0 - y_off:y1 - y_off, 700:1910]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    w = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    w = cv2.GaussianBlur(w, (5, 5), 1.5)
    cs = cv2.HoughCircles(w, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=100, param2=14, minRadius=36, maxRadius=50)
    out = []
    if cs is None: return out
    for x, y, r in cs[0]:
        if abs(y + y0 - CY) > 12: continue
        xi, yi = int(round(x)), int(round(y))
        inner = hsv[max(0, yi - 18):yi + 19, max(0, xi - 18):xi + 19].reshape(-1, 3)
        if not len(inner): continue
        don = ((((inner[:, 0] <= 10) | (inner[:, 0] >= 177)) & (inner[:, 1] >= 150) & (inner[:, 2] >= 160))).mean()
        ka = ((inner[:, 0] >= 82) & (inner[:, 0] <= 92) & (inner[:, 1] >= 60) & (inner[:, 2] >= 150)).mean()
        if max(don, ka) < 0.3: continue
        out.append((float(700 + x), 'don' if don > ka else 'ka'))
    return out


def recover(video, onsets, fps, t0, t1, cal=None, dbg=False):
    """隠れて流れる音符を探して [onset] を返す(onsets は見つかっている音符、動画秒)"""
    import detect2
    notes = [o for o in onsets if o['color'] in ('don', 'ka') and o.get('vel') and not o.get('stationary')]
    _set_others(onsets)
    if len(notes) < 20: return [], []
    # 速さの群れ(その曲に出てくる音符の速さ)
    vs = sorted(abs(o['vel']) for o in notes)
    vc = []
    for v in vs:
        if vc and abs(v / vc[-1][-1] - 1) <= 0.03: vc[-1].append(v)
        else: vc.append([v])
    speeds = [float(np.median(g)) for g in vc if len(g) >= 5]
    if len(speeds) < 2: return [], []                      # 速さが 1 つだけの曲は追い越しが起きない
    nt = np.array([o['t'] for o in notes]); nv = np.array([abs(o['vel']) for o in notes])
    # 追い越しのある区間だけ調べる: 譜面で先の音符 e が、後の音符 s より後から画面に入る(速さが 10% 以上違う)とき、s は e に上から追い越されて隠れうる
    enter = nt - (1880 - JX) / (nv * fps)
    wins = []
    order = np.argsort(nt)
    for a_ in range(len(order)):
        i = order[a_]
        for b_ in range(a_ + 1, len(order)):
            j = order[b_]                                   # nt[j] >= nt[i](j が後)
            if nt[j] - nt[i] > 4.0: break
            if enter[i] > enter[j] and max(nv[i], nv[j]) / min(nv[i], nv[j]) > 1.1:
                wins.append((min(enter[j], enter[i]) - 1.0, max(nt[i], nt[j]) + 1.0))
    if not wins: return [], []
    wins.sort(); merged_w = [list(wins[0])]
    for a, b in wins[1:]:
        if a <= merged_w[-1][1]: merged_w[-1][1] = max(merged_w[-1][1], b)
        else: merged_w.append([a, b])
    merged_w = [(max(t0, a), min(t1, b)) for a, b in merged_w if b > t0 and a < t1]
    if dbg: print('  synth: 追い越しのある区間 ' + ' '.join('%.1f-%.1f' % w for w in merged_w))

    pts, store, crops = _read(video, merged_w, notes, onsets, fps, cal)
    if dbg: print('  synth: 説明できない円 %d 点、速さ %s' % (len(pts), ['%.1f' % v for v in speeds]))
    if len(pts) < 8: return [], []
    # 2. 直線(判定枠に着くフレーム, 速さ)に投票
    votes = {}
    F = np.array([p[0] for p in pts], float); X = np.array([p[1] for p in pts]); C = np.array([p[2] for p in pts])
    for v in speeds:
        fc = F + (X - JX) / v                          # その速さなら判定枠に着くフレーム
        for col in ('don', 'ka'):
            m = C == col
            for k in np.round(fc[m]).astype(int):
                for kk in (k - 1, k, k + 1): votes[(kk, v, col)] = votes.get((kk, v, col), 0) + 1
    cands = sorted(((n, fc, v, col) for (fc, v, col), n in votes.items() if n >= int(os.environ.get('V2T_SYNTH_VOTES', '5'))), reverse=True)
    # 同じ直線の重複を除く
    chosen = []
    for n, fc, v, col in cands:
        if any(abs(fc - c[1]) <= 3 and v == c[2] for c in chosen): continue
        if any(_same_path(o, {'t': fc / fps, 'vel': v}, fps) for o in notes): continue     # 1 つの軌跡に音符は 1 つ(色を問わない)
        chosen.append((n, fc, v, col))
    if dbg: print('  synth: 直線の候補 %d' % len(chosen))
    if not chosen: return [], []
    # 3. 描いて比べて確かめる
    T = _make_T(video, notes, crops, fps, cal, dbg)
    if T is None: return [], []
    out = []; gone = []

    def frames_of(*ns):
        a = min(o['t'] * fps - (1850 - JX) / abs(o['vel']) for o in ns); b = max(o['t'] * fps - 20 / abs(o['vel']) for o in ns)
        return [(f, bg, La) for f, (bg, La) in store.items() if a <= f <= b]

    # 3a. 追い越しのある区間で速さが群れから 1.5% 超外れた音符(短い軌跡から測った音符)は、いちばん近い群れの速さで時刻を ±4 フレーム振って
    #     描き直し、入れ替えたほうが画面に合えば入れ替える
    def best_swap(o, cand):
        best = None
        for h in cand:
            g = _swap(frames_of(o, h), notes, o, h, fps, T)
            if g is not None and (best is None or g > best[0]): best = (g, h)
        return best
    for o in list(notes):
        if not any(a <= o['t'] <= b + 1 for a, b in merged_w): continue
        v0 = abs(o['vel'])
        vs_ = min(speeds, key=lambda v: abs(v / v0 - 1))
        if abs(vs_ / v0 - 1) <= 0.015: continue           # 速さが群れに入っている音符はそのまま
        alt = [vs_]
        # 粗く(2 フレームおき)探してから、良い速さの周りを 1 フレームおきに
        b1 = best_swap(o, [dict(o, t=o['t'] + k / fps, vel=v) for v in alt for k in (-4, -2, 0, 2, 4)])
        if b1 is None: continue
        b2 = best_swap(o, [dict(o, t=b1[1]['t'] + k / fps, vel=b1[1]['vel']) for k in (-1, 1)])
        best = b2 if b2 and b2[0] > b1[0] else b1
        if best[0] >= GAIN_MIN:
            if dbg: print('  synth: 測り直し %s t=%.3f 速さ %.1f → t=%.3f 速さ %.1f(利得 %.3f)' % (o['color'], o['t'], v0, best[1]['t'], best[1]['vel'], best[0]))
            h = dict(best[1], t=round(best[1]['t'], 4), synth=True)
            notes[notes.index(o)] = h; gone.append(o); out.append(h)
    fitted = set()
    for n, fc, v, col in chosen:
        # 仮説を確かめる前に、その仮説と重なって流れる既存の音符の描く位置を画面に合わせ直す(検出の時刻の誤差は半フレーム以内なので
        # ±速さの半分 px)。描き方のずれを仮説の円が埋めて「合う」ように見えるのを防ぐ(ドンカマ2000 30.96s)
        f_in = fc - (1850 - JX) / v; f_out = fc - 20 / v
        fr_ = [(f, bg, La) for f, (bg, La) in store.items() if f_in <= f <= f_out]
        hh = {'t': fc / fps, 'vel': v}
        for q in notes:
            if id(q) in fitted: continue
            if min((abs(_x(q, f, fps) - _x(hh, f, fps)) for f, _, _ in fr_), default=1e9) < 2 * R_OUT:
                _fit_dx(frames_of(q), notes, q, fps, T); fitted.add(id(q))
        best = None
        f_in = fc - (1850 - JX) / v; f_out = fc - 20 / v
        fr_ = [(f, bg, La) for f, (bg, La) in store.items() if f_in <= f <= f_out]
        if len(fr_) < 0.8 * (f_out - f_in) / 4:
            continue                                   # 画面を横切る間の 8 割以上を読んでいない(区間の外から入ってくる遅い音符)は確かめられない
        for c2 in (col,):                               # 色は円の中身の色(証拠)のまま。別の色を試すと、隣の音符の描き方のずれを埋める偽物が出る
            for dfc in (-0.5, 0.0, 0.5):
                h = {'t': (fc + dfc) / fps, 'color': c2, 'vel': v}
                g = _gain(fr_, notes, h, fps, T)
                if g is not None and (best is None or g > best[0]): best = (g, h, _gain.t)
        if dbg and best: print('  synth: 候補 t=%.3f %s 速さ %.1f 票 %d → 利得 %.3f(%s t=%.3f) 確かさ %.1f' % (fc / fps, col, v, n, best[0], best[1]['color'], best[1]['t'], best[2]))
        if best and best[0] >= GAIN_MIN:
            h = best[1]
            if any(_same_path(o, h, fps) for o in notes + out): continue
            fr_ = [(f, bg, La) for f, (bg, La) in store.items() if f_in <= f <= f_out]
            dup = _replaces(fr_, notes + out, h, fps, T)
            if dup:
                # h と同じ音符とみなせる音符のうち、h に入れ替えたほうが画面に合うものは入れ替える(大きさは元の音符のまま)。1 つも無ければ h は足さない
                sw = []
                for o_ in dup:
                    g_s = _swap(frames_of(o_, h), notes + out, o_, dict(h, big=bool(o_.get('big'))), fps, T)
                    if dbg: print('  synth:   → %s t=%.3f と同じ音符(入れ替えの利得 %s)' % (o_['color'], o_['t'], g_s and round(g_s, 3)))
                    if g_s is not None and g_s >= GAIN_MIN: sw.append(o_)
                if not sw: continue
                for o_ in sw:
                    if any(o_ is q for q in notes): notes.remove(o_); gone.append(o_)
                    else: out.remove(o_)
                h = dict(h, big=any(o_.get('big') for o_ in sw))
            out.append({'t': round(h['t'], 4), 'color': h['color'], 'big': bool(h.get('big')), 'npts': int(n), 'vel': round(v, 2), 'hidden': False, 'synth': True, 'synth_gain': round(best[0], 3)})
    return out, gone


def _set_others(onsets):
    """風船・連打・くす玉は見本で描けないので、流れている範囲(頭〜終わり、風船はしっぽまで)の列は比べない"""
    OTHERS.clear()
    ends = sorted(o['t'] for o in onsets if o['color'] == 'roll_end')
    for o in onsets:
        if o['color'] in ('don', 'ka', 'roll_end') or not o.get('vel'): continue
        te = next((t for t in ends if t > o['t']), None)
        OTHERS.append((o['t'], te if te is not None and te - o['t'] < 20 else o['t'], abs(o['vel'])))


def _read(video, wins, notes, onsets, fps, cal):
    """区間を読み、(説明できない円の中心 [(f, x, 色)], 4 枚おきの (背景, 画素の種類), 見本の切り出し) を返す。
    音符ごとの描く位置のずれ(DX)も、その音符で説明できた円の中心と予想位置の差の中央値で決める
    (密な連打では検出の時刻が ~9ms ずれ、速い音符で 15px になる)"""
    nt = np.array([o['t'] for o in notes]); nv = np.array([abs(o['vel']) for o in notes])
    # 円(小音符の大きさ)が音符の中にあれば説明できる: 小音符は中心 15px 以内、大音符は半径の差(R_OUT_B − R_OUT)だけ広く
    # (カ 裏 106.1s: 遅い層の大ドンの中の顔の円が予想位置から 25px ずれて見え、別の小ドンとして足していた)
    tol = np.array([15 + (R_OUT_B - R_OUT if o.get('big') else 0) for o in notes], float)
    res = {}

    def explained(fi, x):
        px = JX + (nt * fps - fi) * nv
        d = px - x; k = int(np.argmin(np.abs(d)))
        if abs(d[k]) < 15:
            res.setdefault(k, []).append(-d[k]); return True
        return bool(np.any(np.abs(d) < tol))
    pts = []; store = {}
    # 見本を取る音符: 孤立した(左右 150px に他の音符がいない)音符。区間を読むついでに x ≈ 1300 で切り出す
    iso = [o for o in notes if not any(q is not o and abs(q['t'] - o['t']) * abs(o['vel']) * fps < 150 for q in onsets if q.get('vel'))]
    want = {}
    for o in iso:
        want.setdefault(int(round(o['t'] * fps - (1300 - JX) / abs(o['vel']))), []).append(o)
    crops = {}
    for f, canvas in _iter_windows(video, wins, cal):
        for o in want.get(f, []):
            key = (o['color'], bool(o.get('big'))); hb = HALF_B if key[1] else HALF
            x = int(round(JX + (o['t'] * fps - f) * abs(o['vel'])))
            m_ = hb + MARG
            if len(crops.get(key, [])) < 40 and m_ <= x < 1920 - m_:
                crops.setdefault(key, []).append((abs(o['vel']), canvas[CY - HALF:CY + HALF, x - m_:x + m_].astype(np.float32)))
        for x, col in rings(canvas):
            if JX + 40 < x < 1880 and not explained(f, x): pts.append((f, x, col))
        if f % 4 == 0:
            store[f] = (np.median(np.vstack([canvas[295:325], canvas[450:476]]).astype(np.float32), axis=0),
                        labels(canvas[CY - HALF:CY + HALF].astype(np.float32)))
    DX.clear()
    for k, r in res.items():
        if len(r) >= 3: DX[id(notes[k])] = float(np.median(r))
    return pts, store, crops


def _make_T(video, notes, crops, fps, cal, dbg):
    XO[:] = [0.0, 0.0]
    if len(crops.get(('don', False), [])) >= 5 and len(crops.get(('ka', False), [])) >= 5:
        return _center_templates(crops, dbg)
    return _templates(video, notes, fps, cal)


def _merge(wins):
    wins = sorted(wins); out = []
    for a, b in wins:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [tuple(w) for w in out]


def verify_emerged(video, onsets, fps, cal=None, dbg=False):
    """途中から現れた短い軌跡の音符(detect2 の emerged / emerge)を描いて確かめる。隠れていたなら、描いた画面でもその間は上の音符に隠れて
    食い違いは変わらない。その音符を消したほうが、描いた画面が変わる画素で GAIN_MIN 以上よく合うなら偽物として外す
    (TDM 裏 133.74s: 風船を叩いたときに判定枠から出るどんちゃんの顔。描くと右端からドンが流れてくるが画面には無い)。戻り値は外す音符"""
    notes = [o for o in onsets if o['color'] in ('don', 'ka') and o.get('vel') and not o.get('stationary')]
    tg = [o for o in notes if o.get('emerged') or o.get('emerge')]
    if not tg or len(notes) < 20: return []
    _set_others(onsets)
    wins = _merge([[o['t'] - (1850 - JX) / (abs(o['vel']) * fps) - 0.1, o['t'] + 0.05] for o in tg])
    _, store, crops = _read(video, wins, notes, onsets, fps, cal)
    T = _make_T(video, notes, crops, fps, cal, dbg)
    if T is None: return []
    gone = []
    for o in tg:
        a = o['t'] * fps - (1850 - JX) / abs(o['vel']); b = o['t'] * fps - 20 / abs(o['vel'])
        fr_ = [(f, bg, La) for f, (bg, La) in store.items() if a <= f <= b]
        rest = [q for q in notes if q is not o]
        g = []
        for f, bg, La in fr_:
            nsA = [q for q in notes if _on(q, f, fps)]; nsB = [q for q in rest if _on(q, f, fps)]
            xa, xb = _span(f, fps, bg.shape[0], o)
            if xb <= xa: continue
            iA, oA = _render(f, bg, nsA, fps, T, xa, xb); iB, oB = _render(f, bg, nsB, fps, T, xa, xb)
            La_ = La[:, xa:xb]; lA = labels(iA); lB = labels(iB); d = (lA != lB) & ~_ign(f, fps, xa, xb)[None, :]
            if d.sum() < 200: continue
            g.append(float((lA[d] != La_[d]).mean() - (lB[d] != La_[d]).mean()))
        gm = float(np.mean(g)) if len(g) >= 5 else None
        if dbg: print('  synth: 現れた音符 %s t=%.3f 速さ %.1f: 消すと食い違いが %s 減る(%d 枚)' % (o['color'], o['t'], abs(o['vel']), 'NA' if gm is None else '%.3f' % gm, len(g)))
        if gm is not None and gm >= GAIN_MIN: gone.append(o)
    return gone


def _iter_windows(video, wins, cal):
    import detect2
    for a, b in wins:
        hd = detect2.band_start(video, a, b, (290, 480), 1, cal)
        for f, canvas in detect2.band_iter(hd): yield f, canvas


def _same_path(o, h, fps):
    """同じ音符か: h が画面にいる間(判定枠〜x1850)、o の位置との差の中央値が 25px 未満(速さが連続的に変わる所で速さの群れが 10% 違っても同じ音符)"""
    vh = abs(h['vel']); vo = abs(o['vel'])
    fs = np.linspace(h['t'] * fps - (1850 - JX) / vh, h['t'] * fps, 20)
    d_ = np.abs((h['t'] * fps - fs) * vh - (o['t'] * fps - fs) * vo)
    return float(np.median(d_)) < 25


def _templates(video, notes, fps, cal):
    """見本: 孤立した(左右 110px に他の音符がいない)小ドン/小カッの見た目の中央値"""
    import detect2
    tpl = {}
    for big in (False, True):
        hb = HALF_B if big else HALF
        iso = [o for o in notes if bool(o.get('big')) == big and not any(q is not o and abs(q['t'] - o['t']) * abs(o['vel']) * fps < 150 for q in notes)]
        for o in iso[::max(1, len(iso) // 40)][:40]:
            fi = int(round(o['t'] * fps - (1300 - JX) / abs(o['vel'])))     # x ≈ 1300 のフレーム
            hd = detect2.band_start(video, fi / fps, (fi + 1) / fps, (290, 480), 1, cal, parts=1)
            for f, canvas in detect2.band_iter(hd):
                x = int(round(JX + (o['t'] * fps - f) * abs(o['vel'])))
                if hb <= x < 1920 - hb:
                    tpl.setdefault((o['color'], big), []).append(canvas[CY - HALF:CY + HALF, x - hb:x + hb].astype(np.float32))
                break
    if ('don', False) not in tpl or ('ka', False) not in tpl: return None
    return {k: np.median(np.array(v), 0) for k, v in tpl.items()}


def _frames(video, t_hit, v, fps, cal):
    """判定枠に t_hit に着く速さ v の音符が画面にいる間のフレーム(4 枚おき) [(f, 帯の画像, 背景, 画素の種類)]"""
    import detect2
    f_in = t_hit * fps - (1850 - JX) / v - 2; f_out = t_hit * fps - 20 / v + 2
    hd = detect2.band_start(video, max(0, f_in) / fps, f_out / fps, (290, 480), 4, cal)
    out = []
    for f, canvas in detect2.band_iter(hd):
        bg = np.median(np.vstack([canvas[295:325], canvas[450:476]]).astype(np.float32), axis=0)
        out.append((f, bg, labels(canvas[CY - HALF:CY + HALF].astype(np.float32))))
    return out


_AL = {}
OTHERS = []         # [(頭の時刻, 終わりの時刻, 速さ)]
XO = [0.0, 0.0]     # 描く位置のずれ(定数 px, 速さあたり px)
MARG = 30           # 見本の切り出しの余白(中心合わせ用)


DX = {}             # 音符ごとの描く位置のずれ px(id(音符) → 円の中心から測った値)


def _x(o, f, fps):
    v = abs(o['vel'])
    if id(o) in DX: return JX + (o['t'] * fps - f) * v + DX[id(o)]
    return JX + (o['t'] * fps - f) * v + XO[0] + XO[1] * v


TAIL = 160          # 風船のしっぽ(頭の中心から右へ)


def _ign(f, fps, xa, xb):
    """x ∈ [xa, xb) のうち風船・連打・くす玉が流れている列"""
    m = np.zeros(xb - xa, bool)
    for t0, t1, v in OTHERS:
        a = JX + (t0 * fps - f) * v - HALF_B; b = max(JX + (t1 * fps - f) * v, JX + (t0 * fps - f) * v + TAIL) + HALF_B
        a = max(xa, int(a)); b = min(xb, int(b) + 1)
        if b > a: m[a - xa:b - xa] = True
    return m


def _alpha(big):
    if big not in _AL:
        hb, r = (HALF_B, R_OUT_B) if big else (HALF, R_OUT)
        yy, xx = np.mgrid[-HALF:HALF, -hb:hb]; _AL[big] = np.hypot(xx, yy) <= r    # 大音符は帯の高さで上下が切れる
    return _AL[big]


def _render(f, bg, ns, fps, T, xa=0, xb=None):
    """帯の x ∈ [xa, xb) を描く: 背景 → 譜面で後の音符から順に上書き(先の音符が上)。(画像, 画素ごとの音符の番号)"""
    if xb is None: xb = bg.shape[0]
    W = xb - xa
    img = np.repeat(bg[None, xa:xb, :], 2 * HALF, axis=0).copy(); owner = np.full((2 * HALF, W), -1, int)
    for k, o in sorted(enumerate(ns), key=lambda ko: -ko[1]['t']):
        big = bool(o.get('big')) and (o['color'], True) in T
        hb = HALF_B if big else HALF; al = _alpha(big); tp = T[(o['color'], big)]
        xg = int(round(_x(o, f, fps)))
        if xg < JX - 30: continue
        x = xg - xa; a0 = max(0, x - hb); a1 = min(W, x + hb)
        if a1 <= a0: continue
        s0 = a0 - (x - hb); s1 = s0 + (a1 - a0); m = al[:, s0:s1]
        img[:, a0:a1][m] = tp[:, s0:s1][m]; owner[:, a0:a1][m] = k
    return img, owner


def _center_templates(crops, dbg):
    """孤立した音符の切り出しから、実際の中心と軌跡の予想位置とのずれを測る。ずれは「定数 + 係数×速さ」(検出の時刻の系統的な遅れ)に
    当てはめて XO に入れ、中心に合わせ直した切り出しの中央値を見本にする"""
    meas = []
    for (col, big), lst in crops.items():
        hb = HALF_B if big else HALF
        for v, c in lst:
            L = labels(c[HALF - 8:HALF + 8]); fg = ((L == 1) | (L == 2) | (L == 3)).mean(0) > 0.5
            mid = hb + MARG
            if not fg[mid - 10:mid + 11].any(): continue
            a = b = mid - 10 + int(np.argmax(fg[mid - 10:mid + 11]))
            while a > 0 and fg[a - 1]: a -= 1
            while b < len(fg) - 1 and fg[b + 1]: b += 1
            meas.append((v, (a + b) / 2 - mid, col, big))
    if len(meas) >= 10:
        V = np.array([m[0] for m in meas]); D = np.array([m[1] for m in meas])
        if V.max() / V.min() > 1.3: b_, a_ = np.polyfit(V, D, 1)
        else: a_, b_ = float(np.median(D)), 0.0
        r = D - (a_ + b_ * V); ok = np.abs(r) < 3 * max(1.0, np.median(np.abs(r)))
        if ok.sum() >= 10:
            if V[ok].max() / V[ok].min() > 1.3: b_, a_ = np.polyfit(V[ok], D[ok], 1)
            else: a_, b_ = float(np.median(D[ok])), 0.0
        XO[:] = [float(a_), float(b_)]
        if dbg: print('  synth: 描く位置のずれ %.1f + %.3f×速さ px(%d 個から)' % (a_, b_, len(meas)))
    T = {}
    for (col, big), lst in crops.items():
        hb = HALF_B if big else HALF
        cs = []
        for v, c in lst:
            dx = int(round(XO[0] + XO[1] * v)); s0 = MARG + dx
            if 0 <= s0 and s0 + 2 * hb <= c.shape[1]: cs.append(c[:, s0:s0 + 2 * hb])
        if len(cs) >= 3: T[(col, big)] = np.median(np.array(cs), 0)
    if ('don', False) not in T or ('ka', False) not in T: return None
    return T


def _mis(R, own, ns, La):
    """描いた画素の種類 R と実際 La の食い違い"""
    return R != La


def _span(f, fps, W, *ns):
    """音符たちが描かれる x の範囲(帯の幅に切る)"""
    xs = [_x(o, f, fps) for o in ns]
    return max(0, int(min(xs)) - HALF_B - 1), min(W, int(max(xs)) + HALF_B + 2)


def _on(o, f, fps, W=1920):
    return JX - 30 < JX + (o['t'] * fps - f) * abs(o['vel']) < W + HALF_B


def _gain(frames, notes, h, fps, T):
    """仮説 h を足したときの、見えている画素の種類の食い違いの減り(画面にいる間の平均)"""
    gains = []
    for f, bg, La in frames:
        if not (JX + 20 < JX + (h['t'] * fps - f) * h['vel'] < 1850): continue
        ns = [o for o in notes if _on(o, f, fps)]
        xa, xb = _span(f, fps, bg.shape[0], h)
        img0, own0 = _render(f, bg, ns, fps, T, xa, xb); img1, own = _render(f, bg, ns + [h], fps, T, xa, xb)
        vis = (own == len(ns)) & ~_ign(f, fps, xa, xb)[None, :]
        if vis.sum() < 200: continue
        La = La[:, xa:xb]; e0 = _mis(labels(img0), own0, ns, La); e1 = _mis(labels(img1), own, ns + [h], La)
        gains.append(float(e0[vis].mean() - e1[vis].mean()))
    _gain.t = float(np.mean(gains) / (np.std(gains) / np.sqrt(len(gains)) + 1e-9)) if len(gains) >= 5 else 0.0
    return float(np.mean(gains)) if len(gains) >= 5 else None


def _replaces(frames, notes, h, fps, T):
    """h が見つかっている音符の描き直し(同じ音符の速さ・時刻の測り違い)にすぎないか: h を足した上で、h と重なって流れる音符 o を
    1 つ抜いても、描いた画面が変わる画素での食い違いがほとんど増えない(GAIN_MIN 未満)なら、o は h と同じ音符。その o をすべて返す"""
    near = []
    for o in notes:
        fs = [f for f, _, _ in frames if JX + 20 < JX + (h['t'] * fps - f) * h['vel'] < 1850]
        if fs and min(abs((h['t'] * fps - f) * h['vel'] - (o['t'] * fps - f) * abs(o['vel'])) for f in fs) < 2 * R_OUT: near.append(o)
    found = []
    for o in near:
        rest = [q for q in notes if q is not o]
        cost = []
        for f, bg, La in frames:
            if not (JX + 20 < JX + (h['t'] * fps - f) * h['vel'] < 1850): continue
            nsA = [q for q in notes if _on(q, f, fps)] + [h]; nsB = [q for q in rest if _on(q, f, fps)] + [h]
            xa, xb = _span(f, fps, bg.shape[0], h, o)
            if xb <= xa: continue
            iA, oA = _render(f, bg, nsA, fps, T, xa, xb); iB, oB = _render(f, bg, nsB, fps, T, xa, xb)
            La_ = La[:, xa:xb]; lA = labels(iA); lB = labels(iB); d = (lA != lB) & ~_ign(f, fps, xa, xb)[None, :]
            if d.sum() < 200: continue
            cost.append(float(_mis(lB, oB, nsB, La_)[d].mean() - _mis(lA, oA, nsA, La_)[d].mean()))
        if len(cost) >= 5 and float(np.mean(cost)) < GAIN_MIN: found.append(o)
    return found


def _swap(frames, notes, o, h, fps, T):
    """音符 o を h に入れ替えたときの、描いた画面が変わる画素での食い違いの減り(フレームの平均)"""
    rest = [q for q in notes if q is not o]
    g = []
    for f, bg, La in frames:
        nsA = [q for q in notes if _on(q, f, fps)]; nsB = [q for q in rest if _on(q, f, fps)] + ([h] if _on(h, f, fps) else [])
        xa, xb = _span(f, fps, bg.shape[0], h, o)
        if xb <= xa: continue
        iA, oA = _render(f, bg, nsA, fps, T, xa, xb); iB, oB = _render(f, bg, nsB, fps, T, xa, xb)
        La_ = La[:, xa:xb]; lA = labels(iA); lB = labels(iB); d = (lA != lB) & ~_ign(f, fps, xa, xb)[None, :]
        if d.sum() < 200: continue
        g.append(float(_mis(lA, oA, nsA, La_)[d].mean() - _mis(lB, oB, nsB, La_)[d].mean()))
    return float(np.mean(g)) if len(g) >= 5 else None


_KA_WINDOW = 0.075    # 判定「可」の片側の幅(秒)


def slow_hits(video, onsets, fps, cal=None, dbg=False):
    """遅い音符(速さ 4px/フレーム未満)の時刻を「判定枠の近くで叩かれて消えた最後のフレーム」に直す。
    遅い音符は位置の数 px の誤差が数十 ms になる(0.9px/フレームなら 3.5px で 65ms)ので、軌跡から判定枠に着く時刻を出さない。
    同じ音符を 2 本の軌跡で拾っていたら、直した時刻が同じになるので 1 つにまとめる"""
    import detect2
    slow = [o for o in onsets if o.get('slow')]
    if not slow: return onsets
    for o in slow:
        v = abs(o['vel']); fp = o['t'] * fps
        # 叩かれて消えたフレームには、プレイヤーの叩くずれ(判定「可」の幅 ±75ms まで)が入る。軌跡の時刻の誤差(候補の位置の誤差 6px ÷ 速さ)がそれより
        # 小さい速さなら軌跡の時刻を使う(UFO Swingin' 19.61s: 3.06px/フレームの大カッを 78ms 早く叩いていて、消えたフレームで 19.533 にしていた。軌跡は 19.617)
        if 6.0 / v / fps < _KA_WINDOW: continue
        f_a = int(fp - 80 / v); f_b = int(fp + 60 / v)             # 判定枠の手前 80px 〜 通り過ぎて 60px
        col = 1 if o['color'] == 'don' else 2; r = 30
        last = None
        # フレームは検出と同じ時計に直し(ringtrack._eff)、古い絵(複製)は使わない(複製を「最後に見えた」と取ると 1 フレーム遅れる)
        import ringtrack
        clk = getattr(detect2.run_parallel, 'clk', None); skip = getattr(detect2.run_parallel, 'skip', set())
        hd = detect2.band_start(video, f_a / fps, f_b / fps, (290, 480), 1, cal)
        for f, canvas in detect2.band_iter(hd):
            if f in skip: continue
            fe = ringtrack._eff(f, clk)
            x = int(round(JX + (fp - fe) * v))
            L = labels(canvas[CY - r:CY + r, max(0, x - r):x + r].astype(np.float32))
            if L.size and (L == col).mean() >= 0.5: last = fe
        if last is not None:
            if dbg: print('  遅い音符 %s t=%.3f 速さ %.2f → 叩かれて消えたフレーム %.3f' % (o['color'], o['t'], v, last / fps))
            o['t'] = round(last / fps, 4)
    out = []
    for o in sorted(onsets, key=lambda o: o['t']):
        if o.get('slow') and any(q.get('slow') and q['color'] == o['color'] and abs(q['t'] - o['t']) < 0.05 for q in out):
            continue
        out.append(o)
    return out


def _fit_dx(frames, notes, q, fps, T):
    """既存の音符 q の描く位置を、±速さの半分 px(半フレーム)の範囲で画面に最も合うように合わせ直す(DX)"""
    v = abs(q['vel']); r = int(np.ceil(v / 2))
    if not frames or r < 1: return
    base = _x(q, 0, fps) - (JX + q['t'] * fps * v)            # 今のずれ
    frs = frames[::max(1, len(frames) // 12)]
    best = None
    for d in range(-r, r + 1):
        DX[id(q)] = base + d
        err = 0.0; n = 0
        for f, bg, La in frs:
            x = _x(q, f, fps)
            if not (JX + 20 < x < 1850): continue
            xa, xb = max(0, int(x) - HALF_B), min(bg.shape[0], int(x) + HALF_B)
            ns = [o for o in notes if _on(o, f, fps)]
            img, own = _render(f, bg, ns, fps, T, xa, xb)
            m = ~_ign(f, fps, xa, xb)[None, :] & np.ones((2 * HALF, xb - xa), bool)
            err += float((labels(img) != La[:, xa:xb])[m].mean()); n += 1
        if n and (best is None or err / n < best[0]): best = (err / n, d)
    DX[id(q)] = base + (best[1] if best else 0)
