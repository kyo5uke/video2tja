"""動画の音声から、Songs の中のどの曲(音源)かを探す。
使い方: python songfind.py <動画> [--top 5]   → 候補の曲フォルダ・音源・一致の強さ・Δ(動画の秒 − 音源の秒)
        python songfind.py --index            → 音源の索引だけ作る(初回は全音源を読むので数分。以後は増えた・変わった音源だけ)

曲フォルダを名前で選ぶと、同じ曲名の別の曲(虹: Aqua Timez / フィッシャーズ)や版違い(大好きな太鼓の音の MV 版、Shine と Shine!!、
スーパーマリオブラザーズの新旧)を選び間違える(一括試験の参照の誤り 14 曲はどれもこれ)。音源の音の高さの帯 8 つの立ち上がりの並び(1 秒 21.5 点)を動画の音声と相関で比べる。
比べ方は run_song.find_delta と同じ(ずらした位置ごとの正規化した相関)で、音源の半分以上が動画に重なる位置だけを見る。
音源の索引は検出のキャッシュ(detect2.cache_dir())に置く。"""
import os, sys, json, glob
import numpy as np
import audio_align as A

SONGS = os.environ.get('V2T_SONGS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Songs')
HOP = 512                                  # 11025 / 512 = 21.5 点/秒(曲を見分けるには十分。find_delta は 64 で時刻を詰める)


def _env(x):
    """音の高さの帯(60Hz〜5kHz を対数で 8 つ)ごとの、音の立ち上がりの強さ(帯の対数エネルギーの増加分)。点 × 8。
    1 本の音量の強弱だけではどの曲も似た形になり見分けられなかった(虹・Shine!! などで別の曲が 1 位)"""
    N = 1024
    n = (len(x) - N) // HOP
    if n < 10: return None
    idx = np.arange(N)[None, :] + HOP * np.arange(n)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(N)[None, :], axis=1)) ** 2
    fr = np.fft.rfftfreq(N, 1.0 / A.SR)
    edges = np.geomspace(60, 5000, 9)
    E = np.stack([S[:, (fr >= lo) & (fr < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    F = np.maximum(np.diff(np.log1p(E * 1e3), axis=0), 0)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    return F.astype(np.float32)


def _one(p):
    try: return p, _env(A.load(p))
    except Exception: return p, None


def _index_path():
    import detect2
    d = detect2.cache_dir() or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testvideos', 'work')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'songfind_on8_h%d.npz' % HOP)


def index(workers=None, verbose=True):
    """{音源のパス: 強弱の形}。音源の更新時刻と大きさが変わったもの・増えたものだけ読み直す"""
    from concurrent.futures import ProcessPoolExecutor
    import detect2
    oggs = sorted(glob.glob(os.path.join(SONGS, '**', '*.ogg'), recursive=True))
    ip = _index_path(); old = {}
    if os.path.exists(ip):
        # 読んだら閉じる(np.load の npz はファイルを開いたままにし、Windows では同じファイルを置き換えられず、音源が 1 個増えただけで
        # 曲探しが「アクセスが拒否されました」で止まった)
        with np.load(ip, allow_pickle=True) as z:
            old = {k: (tuple(s), e) for k, s, e in zip(z['paths'], z['stamps'], z['envs'])}
    stamp = {p: (os.path.getmtime(p), os.path.getsize(p)) for p in oggs}
    todo = [p for p in oggs if p not in old or tuple(old[p][0]) != stamp[p]]
    env = {p: old[p][1] for p in oggs if p in old and p not in todo}
    if todo:
        if verbose: print('音源の索引: %d 個を読む(全 %d 個)' % (len(todo), len(oggs)), flush=True)
        with ProcessPoolExecutor(max_workers=workers or detect2.cpu_budget()) as ex:
            for k, (p, e) in enumerate(ex.map(_one, todo, chunksize=8)):
                if e is not None: env[p] = e
                if verbose and (k + 1) % 200 == 0: print('  %d / %d' % (k + 1, len(todo)), flush=True)
        ps = sorted(env)
        np.savez(ip + '.tmp.npz', paths=np.array(ps, dtype=object), stamps=np.array([stamp[p] for p in ps]), envs=np.array([env[p] for p in ps], dtype=object))
        try: os.replace(ip + '.tmp.npz', ip)
        except PermissionError:
            # 別のプロセス(画面のサーバーなど)が索引を開いている: 今回は作った索引をメモリで使い、置き換えは次に任せる
            if verbose: print('音源の索引: ほかのプロセスが使っているので、置き換えは次の機会に')
    return env


def _score(ea, eb):
    """_score3 の (相関の最大, 位置)。山の上位は _score.peaks に置く(1 本ずつ呼ぶとき用)"""
    s, k, pk_ = _score3(ea, eb)
    _score.peaks = pk_
    return s, k


def _score3(ea, eb):
    """動画の特徴 ea の中の、音源の特徴 eb の位置ごとの相関(帯ごとの相互相関の和を、重なりのエネルギーで割ったもの)の最大と、その位置(ea の点)と、
    山の上位 5 つの位置(スレッドで並べて呼べるように戻り値で返す)"""
    L = len(eb); n = 1
    while n < len(ea) + L: n *= 2
    # 位置 k(音源の頭が動画の k 点目)。ゲームは曲を頭から最後まで流すので、音源の 95% 以上が動画に入る位置だけ
    # (半分の重なりまで許すと、短い重なりの偶然の高い相関で、短い音源がどの動画にも 1 位になった)
    m_ = int(0.05 * L)
    if len(ea) + m_ < L: return -1.0, 0, []
    cc = np.fft.irfft(np.fft.rfft(ea, n, axis=0) * np.conj(np.fft.rfft(eb, n, axis=0)), n, axis=0).sum(1)
    ks = np.arange(-m_, len(ea) - L + m_ + 1)
    c = cc[ks % n]
    cs_a = np.cumsum(np.r_[0, (ea.astype(np.float64) ** 2).sum(1)]); cs_b = np.cumsum(np.r_[0, (eb.astype(np.float64) ** 2).sum(1)])
    a0 = np.clip(ks, 0, len(ea)); a1 = np.clip(ks + L, 0, len(ea))
    b0 = a0 - ks; b1 = a1 - ks
    den = np.sqrt(np.maximum(cs_a[a1] - cs_a[a0], 1e-6) * np.maximum(cs_b[b1] - cs_b[b0], 1e-6))
    s = c / den
    i = int(np.argmax(s))
    # 山の上位(0.25 秒以上離れたもの)も覚えておく: 繰り返しの多い曲は数小節ずれた位置の方が高く出ることがある(Break Through the Dome:
    # 1 位の位置は本物から 2.5 秒ずれていて、そこで波形を比べて 0.067 になり別の曲に負けた)
    order_ = np.argsort(-s); pk_ = []
    for j in order_[:400]:
        if all(abs(int(ks[j]) - q) * HOP / A.SR >= 0.25 for q in pk_): pk_.append(int(ks[j]))
        if len(pk_) >= 5: break
    return float(s[i]), int(ks[i]), pk_


def _wave_score(a, b, coarse):
    """動画の音声 a と音源 b を Δ ≈ coarse 秒で重ね、曲の 10 か所(10 秒ずつ)で ±0.25 秒の範囲の生波形の正規化した相関の最大を取り、
    その中央値と、よく合ったか所のずれの中央値で直した Δ を返す(run_song.find_delta の細かい合わせと同じ)"""
    dur = len(b) / A.SR; m_ = int(0.25 * A.SR); seg_n = int(min(10.0, dur * 0.08) * A.SR); rows = []
    for p_ in np.linspace(0.05, 0.9, 10):
        vs = coarse + dur * p_; a0 = int(round(vs * A.SR)); a2 = a[a0:a0 + seg_n]
        o0 = int(round((vs - coarse) * A.SR)) - m_
        if a0 < 0 or len(a2) < seg_n or o0 < 0 or o0 + seg_n + 2 * m_ > len(b): continue
        # 要るずらしは 0〜2m だけで、w の長さ(seg_n + 2m)以上の FFT なら巡回の折り返しが届かない(以前は len(w) + seg_n で 2 倍の長さ、4.6 倍遅かった)
        w = b[o0:o0 + seg_n + 2 * m_]; nf = 1
        while nf < len(w): nf *= 2
        xc = np.fft.irfft(np.fft.rfft(w, nf) * np.conj(np.fft.rfft(a2, nf)), nf)[:2 * m_ + 1]
        c2 = np.cumsum(np.r_[0.0, w.astype(np.float64) * w]); nw = np.sqrt(np.maximum(c2[seg_n:seg_n + 2 * m_ + 1] - c2[:2 * m_ + 1], 1e-12))
        rows.append(xc / (nw * (np.linalg.norm(a2) + 1e-9)))
    if not rows: return -1.0, coarse
    R_ = np.array(rows); jb = np.argmax(R_, axis=1); cb = R_[np.arange(len(R_)), jb]
    good = cb >= 0.5 * cb.max()
    return float(np.median(cb)), coarse + (m_ - float(np.median(jb[good]))) / A.SR


def find(video, top=5, env=None, n2=40):
    """候補 [(一致の強さ, 音源のパス, Δ 秒, 並びの強さ, 音源の長さ 秒)] を強い順に。
    1 段目: 帯ごとの立ち上がりの並びの相関で上位 n2 曲に絞る。2 段目: その曲の音源を読み、生波形の相関(10 か所の中央値)で並べ直す。
    動画には叩いた音が入るので、密な譜面では立ち上がりの並びの差が縮み、本物が 1 段目で 25〜32 位になることがある(プチポチ・ラパスの虹)。
    生波形の相関は本物の曲だけが 0.13〜0.7、別の曲は 0.12 以下"""
    from concurrent.futures import ThreadPoolExecutor
    env = env if env is not None else index()
    a = A.load(video)
    ea = _env(a)
    def _s1(item):
        p, eb = item
        s, k, pk_ = _score3(ea, eb)
        return (s, p, [q * HOP / A.SR for q in pk_], (len(eb) + 1) * HOP / A.SR)
    with ThreadPoolExecutor(8) as ex:        # numpy の FFT は GIL を離すので、曲ごとの比べをスレッドで並べる
        c1 = list(ex.map(_s1, [(p, eb) for p, eb in env.items() if eb is not None and len(eb) >= 10], chunksize=64))
    c1.sort(key=lambda r: -r[0])
    c1 = c1[:n2]

    def _one(c):
        # 読み込み(ffmpeg は別プロセス)と波形の比べ(numpy の FFT は GIL を離す)を曲ごとにスレッドで並べる
        s, p, ds, _ = c
        try: b = A.load(p)
        except Exception: return None
        w, d2 = max((_wave_score(a, b, d) for d in ds), key=lambda r: r[0])     # 山の上位の位置のうち、生波形が最もよく合うもの
        return (w, p, d2, s, len(b) / A.SR)
    import detect2
    detect2.progress('曲探し', 1, 2, '全曲 %d 個の音源と音の並びを比べた' % len(env))
    with ThreadPoolExecutor(8) as ex: out = [r for r in ex.map(_one, c1) if r is not None]
    detect2.progress('曲探し', 2, 2, '候補 %d 曲の音源と波形を比べた' % len(c1))
    out.sort(key=lambda r: -r[0])
    return out[:top]


SUB_DIRS = ('12 Dan Dojo', '13 AI', '99 Others')       # 同じ音源を別の曲フォルダに置き直したもの(段位・自動採譜・ライブ)。元の曲フォルダを優先する


def choose(cands):
    """候補から曲を 1 つ選ぶ。戻り値 (選んだ候補 or None, 同じ音源ごとにまとめた候補の組)。
    同じ音源(生波形の相関と Δ が同じ)は 1 組にまとめ、組の中は元の曲フォルダを優先する。選ぶのは 1 位の組が 0.13 以上で、2 位の組の 1.5 倍以上のとき
    (一括試験の 200 本: 本物は 0.135〜0.70、別の曲は 0.12 以下。版違い(チルノ ⑨周年 / 通常 0.291 / 0.290、オレサマ / イカサマパイレーツ)は選ばない)"""
    groups = []
    for c in cands:
        g = next((g for g in groups if abs(g[0][0] - c[0]) < 0.003 and abs(g[0][2] - c[2]) < 0.02), None)
        if g is None: groups.append([c])
        else: g.append(c)
    for g in groups:
        g.sort(key=lambda c: any(x in c[1] for x in SUB_DIRS))
    # 切り出しの音源(練習用の曲フォルダの、曲の一部だけの音源など)は、動画の中で、同じ曲の長い音源の置かれた範囲 [Δ, Δ + 長さ] の中に収まる。
    # 長い方も本物の強さ(0.13 以上)で合っていれば、長い方が動画の曲をより多く説明するので切り出しの組を外す(エンジェルドリーム: 練習用の
    # 44 秒・60 秒の切り出しが 0.553・0.550、本家の 145 秒の曲が 0.534 で 3 位だった)。演奏が切り出しだけの動画では、長い音源は動画に収まらないか、
    # 曲の無い所で波形が合わず 0.13 に届かない。切り出しとみなすのは長い方より 10 秒(波形を比べる 1 区間)以上短いものだけ(版違いは長さがほぼ同じ:
    # チルノ ⑨周年 126.8 秒 / 通常 128.2 秒。これは選ばずに候補を出す)
    def _inside(a, b): return a[4] + 10.0 <= b[4] and b[2] - 1.0 <= a[2] and a[2] + a[4] <= b[2] + b[4] + 1.0
    # 長い方が同じ録音なら、切り出しの位置の波形は同じに合い、全体の相関も切り出しと同じくらいになる。切り出しの 1.5 分の 1 未満(自動で選ぶときの
    # 「はっきり上」の比)なら別の録音(IRIS OUT 0.334 の動画で、別の曲 ミルミル が 0.141 で範囲を覆った)
    groups = [g for g in groups if not any(h is not g and h[0][0] >= 0.13 and 1.5 * h[0][0] > g[0][0] and _inside(g[0], h[0]) for h in groups)]
    # 置き直しのフォルダだけの組(段位道場・自動採譜など)は、同じ曲名の元の曲フォルダが候補にあれば外す。譜面の答え(参照 TJA)は元の曲フォルダにあり、
    # 段位の版の音源は元の曲フォルダの音源と別の録音のことがある(達人の 3 曲 Nivalis*Anima・銀の黎明か、黒の晶華か。・魔宵月: 動画は段位の版の音源と
    # 0.50〜0.61、元の曲フォルダの音源と 0.25〜0.29 で合い、段位のフォルダを選んでいた)
    def _sub(c): return any(x in c[1] for x in SUB_DIRS)
    groups = [g for g in groups if not (all(_sub(c) for c in g) and any(h is not g and not _sub(h[0]) and _titles(g[0][1]) & _titles(h[0][1]) for h in groups))]
    if not groups: return None, groups
    # 2 位と比べるのは別の曲フォルダの組だけ(同じ曲フォルダの別の音源(元の音源とカバー版など)は、どちらでも曲フォルダは同じ: アイドル 0.199 / 0.173、
    # 魔法をかけて! 0.184 / 0.181)
    tdir = os.path.dirname(groups[0][0][1])
    rest = [g for g in groups[1:] if not any(os.path.dirname(c[1]) == tdir for c in g)]
    top = groups[0][0][0]; sec = rest[0][0][0] if rest else 0.0
    return (groups[0][0] if top >= 0.13 and top >= 1.5 * max(sec, 0.01) else None), groups


_TT = {}


def _titles(ogg):
    """音源を WAVE に持つ TJA の曲名(TITLE・TITLEJA)を、全角半角・大文字小文字・記号をそろえた集合で"""
    if ogg not in _TT:
        import re, unicodedata
        out = set()
        d = os.path.dirname(ogg)
        for t in sorted(glob.glob(os.path.join(d, '*.tja'))):
            try: txt = open(t, encoding='utf-8-sig', errors='replace').read(4000)
            except Exception: continue
            w = re.search(r'^WAVE:\s*(.+?)\s*$', txt, re.M)
            if w and w.group(1) != os.path.basename(ogg): continue
            for m in re.finditer(r'^TITLE(?:JA)?:\s*(.+?)\s*$', txt, re.M):
                n_ = re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', m.group(1)).lower())
                if n_: out.add(n_)
            break
        _TT[ogg] = out
    return _TT[ogg]


def _title(ogg):
    """音源を WAVE に持つ TJA(音源と同じ名前のものを優先: 裏譜面だけの別ファイルより本体)の (ファイル名, 曲名)"""
    d = os.path.dirname(ogg); stem = os.path.splitext(os.path.basename(ogg))[0]
    for t in sorted(glob.glob(os.path.join(d, '*.tja')), key=lambda t: os.path.splitext(os.path.basename(t))[0] != stem):
        try:
            txt = open(t, encoding='utf-8-sig', errors='replace').read(4000)
        except Exception:
            continue
        import re
        w = re.search(r'^WAVE:\s*(.+)$', txt, re.M)
        if w and w.group(1).strip() != os.path.basename(ogg): continue
        m = re.search(r'^TITLEJA:\s*(.+)$', txt, re.M) or re.search(r'^TITLE:\s*(.+)$', txt, re.M)
        return os.path.basename(t), (m.group(1).strip() if m else '')
    return None, ''


if __name__ == '__main__':
    if '--index' in sys.argv:
        e = index(); print('索引 %d 個: %s' % (len(e), _index_path())); sys.exit(0)
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 5
    ch, groups = choose(find(sys.argv[1], max(top, 8)))
    for g in groups[:top]:
        for j, (w, p, d, s, *_) in enumerate(g):
            tj, ti = _title(p)
            print('%s波形 %.3f(並び %.3f)  %s  (%s / %s)  Δ %.3fs' % ('   同じ音源 ' if j else '', w, s, os.path.relpath(os.path.dirname(p), SONGS).replace(os.sep, '/'), ti, tj or os.path.basename(p), d))
    print('選んだ曲: %s' % (os.path.relpath(os.path.dirname(ch[1]), SONGS).replace(os.sep, '/') if ch else 'なし(はっきり 1 つに決まらない)'))
