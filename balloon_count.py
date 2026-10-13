"""風船の打数を動画から読む(ニジイロ版の表示が前提)。
風船の頭が判定枠に着くと、判定枠の上(基準配置で x≈610-850, y≈70-260)に白い吹き出しが出て「残り打数」が表示される。
検証済み動画 4 本・風船 18 個すべてで、吹き出しの最初の値は参照 TJA の打数より 1 小さかった(40→39, 10→9, 3→2, 24→23)。
つまり吹き出しは「最初の一打が入った後の残り」を出す(到着のフレームで一打目が入る)。よって 打数 = 最初の表示値 + 1。
数字は「吹き出しの中の数字の白い塗り(明るい連結成分)」ごとに区切り、その範囲の非背景画素を塗りつぶし形として切り出して、
実動画から採取したテンプレート(balloon_digits.npz)と照合する。隣の数字と縁取りが繋がっても塗りで区切れる。
使い方:
  balloon_count.py <video> <ons.json> [--cal] [-v]  json の風船(color=balloon)に count を書き込んで上書き保存
  balloon_count.py --harvest                        既知の系列(LABELS)からテンプレートを作り直し、精度を出す
"""
import sys, os, json, numpy as np, cv2

ROI = (560, 1000, 30, 300)          # 基準配置(1920x1080)での吹き出し探索範囲 x0, x1, y0, y1
TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'balloon_digits.npz')
GW, GH = 24, 32                      # グリフの正規化サイズ(高さを揃え、幅は比率を保って中央寄せ)
WIN = (-3, 18)                       # 到着時刻の何フレーム前から何フレーム後まで読むか
CONF = 0.90                          # 数字 1 桁の照合スコア(cos 類似度)の採用下限
PLUS = 1                             # 打数 = 最初の表示値 + PLUS(上記の観察による)
RATE = 14.5                          # 風船の窓の長さ = 打数 / RATE [打/秒]。本家 19 個の実測(中央値 15.3、大きい風船は 12.5〜15)


def _norm(mask):
    h, w = mask.shape
    s = GH / float(h); nw = max(1, min(GW, int(round(w * s))))
    g = cv2.resize(mask, (nw, GH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out = np.zeros((GH, GW), np.float32); x0 = (GW - nw) // 2
    out[:, x0:x0 + nw] = g
    return out


def glyphs(frame):
    """基準配置フレーム → 吹き出しの中の数字グリフを左から [(x, y, w, h, 正規化マスク)]。吹き出しが無ければ None"""
    x0, x1, y0, y1 = ROI
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 2] > 190) & (hsv[..., 1] < 70)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n < 2: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, ba = [int(v) for v in st[k]]
    if ba < 8000 or bh < 80 or bw < 100: return None
    body = (lab == k)
    # 数字の塗り: 吹き出しの枠内にある、それなりの高さの明るい成分(本体以外)
    fills = []
    for j in range(1, n):
        x, y, w, h, a = [int(v) for v in st[j]]
        if j == k or a < 150: continue
        if x < bx or x + w > bx + bw or y < by or y + h > by + bh: continue
        if h < 0.3 * bh or h > 0.75 * bh or w > 1.1 * h: continue
        fills.append((x, y, w, h))
    if not fills: return None
    fills.sort()
    # x が重なる塗りは同じ数字(8 の上下など)としてまとめる
    groups = []
    for f in fills:
        if groups and f[0] <= groups[-1][0] + groups[-1][2] + 2:
            g = groups[-1]; gx0 = min(g[0], f[0]); gx1 = max(g[0] + g[2], f[0] + f[2]); gy0 = min(g[1], f[1]); gy1 = max(g[1] + g[3], f[1] + f[3])
            groups[-1] = (gx0, gy0, gx1 - gx0, gy1 - gy0)
        else: groups.append(f)
    H, W = body.shape
    nonbg = ~body                                   # 吹き出しの背景以外(縁取り+塗り)
    out = []
    for i, (x, y, w, h) in enumerate(groups):
        left = (groups[i - 1][0] + groups[i - 1][2] + x) // 2 if i > 0 else x - 10
        right = (x + w + groups[i + 1][0]) // 2 if i + 1 < len(groups) else x + w + 10
        top, bot = max(0, y - 10), min(H, y + h + 10)
        left, right = max(0, left), min(W, right)
        m = nonbg[top:bot, left:right].astype(np.uint8) * 255
        ys, xs = np.where(m > 0)
        if len(xs) == 0: continue
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        gh, gw = m.shape
        if gh < 0.35 * bh or gw < 8: continue
        out.append((left + int(xs.min()), top + int(ys.min()), gw, gh, _norm(m)))
    return out or None


def load_tpl():
    if not os.path.exists(TPL): return {}
    z = np.load(TPL)
    return {k: z[k] for k in z.files}


def match(g, tpl):
    """グリフ 1 個をテンプレート群と照合 → (数字, スコア, 2 位スコア)。サンプルごとの cos 類似度の最大"""
    best = (None, 0.0); second = 0.0
    gn = np.sqrt((g ** 2).sum()) + 1e-6
    for d, arr in tpl.items():
        s = float(np.max(np.tensordot(arr, g, axes=([1, 2], [0, 1])) / (np.sqrt((arr ** 2).sum(axis=(1, 2))) * gn + 1e-6)))
        if s > best[1]: second = best[1]; best = (d, s)
        elif s > second: second = s
    return best[0], best[1], second


def read_frame(frame, tpl):
    """→ (読めた数, 最小スコア, グリフ列)。吹き出しが無い/読めないときは (None, 0, ...)"""
    gl = glyphs(frame)
    if not gl: return None, 0.0, []
    ds = [match(g[4], tpl) for g in gl]
    if any(d[0] is None for d in ds): return None, 0.0, ds
    return int(''.join(d[0] for d in ds)), min(d[1] for d in ds), ds


_CAL = {}     # annotate が渡された較正を覚えておく(frames_at を ffmpeg で読むのに使う)


def frames_at(video, t, M=None, win=WIN):
    if video in _CAL:
        # ffmpeg で頭出しして帯(吹き出しの行)だけ読む(OpenCV の頭出しはキーフレームから 1 本のスレッドでデコードし直して 1 回 1 秒)
        import detect2
        hd = detect2.band_start(video, max(0.0, t + (win[0] - 0.5) / 60.0), t + (win[1] - 0.5) / 60.0, (0, ROI[3] + 10), 1, _CAL[video], parts=1,
                                tag='bc%d' % int(t * 1000))
        fa = int(round(t * hd['fps']))
        for f, canvas in detect2.band_iter(hd):
            yield f - fa, canvas
        return
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + win[0] / fps) * 1000.0)
    for k in range(win[1] - win[0]):
        ok, fr = cap.read()
        if not ok: break
        if M is not None: fr = cv2.warpAffine(fr, M, (1920, 1080))
        yield k + win[0], fr
    cap.release()


def first_value(video, t, tpl, M=None, verbose=False):
    """風船の頭が判定枠に着いた時刻 t の周りを読み、吹き出しの最初の表示値を返す(読めなければ None)。
    最初に十分なスコアで読めた値。次のフレームがそれ以下の値なら整合(減るだけ)とみなす"""
    reads = []
    for k, fr in frames_at(video, t, M):
        if k < 0: continue                                   # 吹き出しは到着より前には出ない(前の演出の誤読を避ける)
        v, c, ds = read_frame(fr, tpl)
        if verbose: print('    f%+d: %s conf %.2f  %s' % (k, v, c, ' '.join('%s(%.2f/%.2f)' % d for d in ds)))
        if v is not None and c >= CONF: reads.append(v)
        if len(reads) >= 3: break
    if not reads or (len(reads) < 2 and reads[0] > 2): return None   # 2 フレーム以上で読めて減る方向に整合していること(1〜2 打の風船は 1 フレームで割れるので 1 回でよい)
    v0 = reads[0]
    if any(r > v0 for r in reads[1:]): v0 = max(reads)     # 最初のフレームが欠けて読めた(拡大中など)ときの保険
    if any(r < v0 - 3 for r in reads): return None
    return v0


def count_at(video, t, tpl, M=None, verbose=False):
    v0 = first_value(video, t, tpl, M, verbose)
    return None if v0 is None else v0 + PLUS


def _affine_for(video, cal):
    if cal is None: return None
    import calib
    return None if calib.is_canonical(cal) else calib.affine(cal)[0]     # affine は (M, x1) を返す


def bubble_end(video, t, M=None, max_s=6.0, step=4):
    """風船の吹き出しが消える時刻(= 割れた/終わった時刻)。頭の時刻 t から max_s 秒先まで step フレームごとに探す。
    くす玉のように叩く窓が長いものは、音符の絵(丸+小さな風船)の長さでは窓が判らないので、これで測る"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    f0 = int(round(t * fps)); last = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)                  # 1 回だけ飛び、あとは順に読む(step ごとに飛ぶとキーフレームから毎回デコードし直して 1 曲 200 秒)
    for f in range(f0, f0 + int(max_s * fps), step):
        if f > f0:
            for _ in range(step - 1): cap.grab()
        ok, img = cap.read()
        if not ok: break
        if M is not None: img = cv2.warpAffine(img, M, (1920, 1080))
        if glyphs(img) is not None: last = f / fps
        elif last is not None and f / fps - last > 0.12: break      # 2 回続けて無ければ終わり
    cap.release()
    return last


def annotate(video, onsets, cal=None, verbose=False):
    """onsets の風船に count を書き込む(読めなければ 5)。打数に対して終端(8)が近すぎる風船は終端も直す。
    戻り値: [(t, count, 自動で読めたか)]"""
    tpl = load_tpl(); M = _affine_for(video, cal); res = []
    if os.environ.get('V2T_BC_CV') != '1': _CAL[video] = cal
    ons = sorted(onsets, key=lambda o: o['t'])
    bl = [o for o in ons if o['color'] == 'balloon']
    from concurrent.futures import ThreadPoolExecutor     # 風船ごとに独立して読む(デコードは GIL を離す)
    with ThreadPoolExecutor(max_workers=min(8, __import__('detect2').cpu_budget())) as ex_:
        cs = list(ex_.map(lambda o: count_at(video, o['t'], tpl, M, False) if tpl else None, bl))
    cmap = {id(o): c for o, c in zip(bl, cs)}
    for i, o in enumerate(ons):
        if o['color'] != 'balloon': continue
        c = cmap[id(o)]
        o['count'] = int(c) if c else 5; o['count_auto'] = bool(c)
        res.append((o['t'], o['count'], bool(c)))
        # 終端の確認: 1 秒に 40 打を超える必要がある終端は短すぎる(風船/くす玉は音符の絵の長さ = 叩く窓ではない)。
        # 窓は「打数 ÷ RATE」にする(本家 19 個の実測: 打/秒の中央値 15.3、10 打以上の風船は 12.5〜15.0)。
        # 吹き出しが見えていた時間を下限、次の音符を上限にする
        end = next((q for q in ons[i + 1:] if q['color'] == 'roll_end'), None)
        if end is None or o['count'] / max(0.05, end['t'] - o['t']) <= 40.0: continue
        be = bubble_end(video, o['t'], M)
        nxt = next((q for q in ons[i + 1:] if q['color'] in ('don', 'ka', 'renda', 'balloon') and q['t'] > o['t']), None)
        t_new = o['t'] + o['count'] / RATE
        if be: t_new = max(t_new, be)
        if nxt: t_new = min(t_new, nxt['t'] - 0.05)
        if t_new > end['t'] + 0.1:
            print('風船の終端を %.3f → %.3f に直した(打数 %d → %.2f 秒 = %.1f 打/秒%s)' % (
                end['t'], t_new, o['count'], t_new - o['t'], o['count'] / (t_new - o['t']),
                '、吹き出しは %.2f 秒' % (be - o['t']) if be else '、吹き出しは読めず'))
            end['t'] = round(float(t_new), 4); end['from_bubble'] = True
    if res: print('風船の打数: ' + ' '.join('%.2fs→%s%s' % (t, c, '' if ok else '(読めず、仮)') for t, c, ok in res))
    return res


# ---- テンプレート採取 ----------------------------------------------------------------------------------------
OLD = ''      # テンプレートを採り直すときは、動画の置き場を入れる(公開版では空)
NEW = ''
# 吹き出しの表示系列(フレーム画像を目で読んだもの)。k は frames_at(WIN 起点 -3)のフレーム番号
def _seq(k0, vals): return {k0 + i: v for i, v in enumerate(vals)}
LABELS = [   # (video, 到着時刻, {k: 表示値})
    (OLD + 'XHcmLQ7vC5o.mp4', 441.56, _seq(2, [39, 39, 38, 37, 37, 36, 36, 35, 34, 33, 33, 32, 31, 30, 29, 29])),
    (OLD + 'XHcmLQ7vC5o.mp4', 244.64, _seq(2, [9, 9, 8, 7, 7, 6, 5, 5, 4, 3, 2, 1])),
    (OLD + 'ylaH01cmYPs.mp4', 91.00, _seq(3, [23, 23, 22, 21, 20, 19, 19, 18, 18, 17, 16, 15, 15, 14, 13])),
    (OLD + 'XHcmLQ7vC5o.mp4', 340.66, _seq(1, [2, 2, 1])),
    (NEW + '33h_4wd4q_E.mp4', 115.366, _seq(4, [5, 5, 4, 3, 2])),
    (NEW + '33h_4wd4q_E.mp4', 52.822, _seq(2, [2, 2, 1])),
]
# 精度確認用: (video, 検出 json, 参照 TJA の BALLOON(時刻順))。銀の黎明は吹き出し(2, 5)+1 の推定値
HARVEST = [
    (OLD + 'XHcmLQ7vC5o.mp4', 'hyper_fast_ons.json', [3, 3, 3, 3, 3, 3, 40]),
    (OLD + 'XHcmLQ7vC5o.mp4', 'tdm_fast_ons.json', [3, 10, 2, 3, 4, 2, 2]),
    (OLD + 'ylaH01cmYPs.mp4', 'vry_fast_ons.json', [4]),
    (OLD + 'ylaH01cmYPs.mp4', 'kok_fast_ons.json', [9, 24, 17]),
    (NEW + '33h_4wd4q_E.mp4', 'gin_ura_ons.json', [3, 6]),
]


def _balloons(js):
    d = json.load(open(js)); return sorted(o['t'] for o in d['onsets'] if o['color'] == 'balloon')


def harvest():
    samples = {}
    for video, t, seq in LABELS:
        n_ok = 0
        for k, fr in frames_at(video, t):
            if k not in seq: continue
            gl = glyphs(fr); e = str(seq[k])
            if not gl or len(gl) != len(e):
                print('  skip %s t=%.2f f%+d: 期待 %s, グリフ %s' % (os.path.basename(video), t, k, e, [(g[0], g[2], g[3]) for g in gl] if gl else None)); continue
            for g, ch in zip(gl, e): samples.setdefault(ch, []).append(g[4])
            n_ok += 1
        print('  %s t=%.2f: %d/%d frames' % (os.path.basename(video), t, n_ok, len(seq)))
    tpl = {d: np.stack(v) for d, v in samples.items()}
    np.savez(TPL, **tpl)
    print('saved', TPL, {d: len(v) for d, v in sorted(tpl.items())})
    # 系列の読み直し(自己整合)
    bad = 0; tot = 0
    for video, t, seq in LABELS:
        for k, fr in frames_at(video, t):
            if k not in seq: continue
            v, c, ds = read_frame(fr, tpl); tot += 1
            if v != seq[k]: bad += 1; print('  NG %s t=%.2f f%+d: 期待 %d 読み %s (conf %.2f)' % (os.path.basename(video), t, k, seq[k], v, c))
    print('系列の再読 %d/%d 一致' % (tot - bad, tot))
    # 既知の風船の打数
    ok = 0; tot = 0
    for video, js, counts in HARVEST:
        ts = _balloons(js)
        if len(ts) != len(counts): print('skip %s: 風船 %d 個 vs 参照 %d 個' % (js, len(ts), len(counts))); continue
        for t, N in zip(ts, counts):
            c = count_at(video, t, tpl); tot += 1; ok += (c == N)
            print('  %-18s t=%.2f 参照 %2d → 読み %s %s' % (os.path.basename(js), t, N, c, '' if c == N else '  <-- NG'))
    print('accuracy %d/%d' % (ok, tot))


def main():
    if '--harvest' in sys.argv: harvest(); return
    video, js = sys.argv[1], sys.argv[2]
    d = json.load(open(js, encoding='utf-8'))
    cal = None
    if '--cal' in sys.argv:
        import calib
        cal = calib.calibrate(video, d['start'], d['end'])
    annotate(video, d['onsets'], cal, verbose='-v' in sys.argv)
    json.dump(d, open(js, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
