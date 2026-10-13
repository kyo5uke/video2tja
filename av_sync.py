"""動画の「映像 → 音声」の遅れを、叩いたカッの音から測る(参照譜面を使わない同期)。

なぜ必要か: 検出した音符の時刻は**映像**から測っている。一方 Δ(動画↔ogg)は**音声**の相関で出している。
動画の映像と音声にずれ(キャプチャ機材・音声出力の遅れ)があると、その分だけ OFFSET がずれる。
実測では動画によって +5〜+130ms もあり、これまでは参照譜面の音符と突き合わせて補正していた(=参照に頼っていた)。

どう測るか: カッ(縁)の叩き音は高域の鋭いクラック音で、音楽には同じ形の音がほとんど無い。
検出したカッの時刻のまわりで高域エネルギーを平均すると、叩き音の山が 1 個だけ立つ。その立ち上がりが遅れ。
・帯域: 移動平均を引いた高域(≒2kHz 以上)
・窓 2.9ms、1ms 刻み
・立ち上がりの位置 = 山の高さの 50% を横切る点

精度: 段位道場ほか 34 曲で参照由来の値と比べて 平均 +0.4ms、sd 6.8ms、最大 15ms(50% 点)。
参照が怪しい譜面(彁、魔宵月)では 20〜25ms 食い違い、こちら側が正しいと見ている
(彁はドンの音でも独立に測って +23ms、カッと 2.6ms で一致。参照は +50ms を要求する)。
"""
import numpy as np
import audio_align as A

MS = 200          # 前後に見る幅(ms)
FR = 32           # 高域エネルギーの窓(サンプル)= 2.9ms
MIN_KA = 40       # これ未満のカッ数では測らない
MIN_SNR = 3.2     # 山の高さ / 山の外のばらつき(紅は 4.0 ちょうどで、推定は −0.9ms と正しかった。
                  # Nivalis 裏は 2.7 で本当に山が無い(音源の版が違う動画)ので、その間に置く)


def _hi_energy(x):
    """高域(≒2kHz 以上)の短時間エネルギー、1ms 刻み"""
    k = np.ones(6) / 6
    h = x - np.convolve(x, k, 'same')
    c = np.cumsum(np.r_[0.0, h * h])
    step = A.SR / 1000.0
    idx = (np.arange(int((len(x) - FR) / step)) * step).astype(int)
    return np.sqrt((c[idx + FR] - c[idx]) / FR)


def _profile(e, times_ms):
    off = np.arange(-MS, MS + 1)
    ok = (times_ms > MS + 5) & (times_ms < len(e) - MS - 5)
    if ok.sum() < MIN_KA: return None, off, int(ok.sum())
    return e[times_ms[ok][:, None] + off[None, :]].mean(0), off, int(ok.sum())


def measure(video, ka_times, lo=None, hi=None, verbose=True):
    """ka_times(動画秒)から映像→音声の遅れ[秒]を返す。測れなければ None。
    戻り値: (lag秒, 山の高さ, SNR, 使ったカッ数) または (None, ...)"""
    ka = np.asarray(sorted(ka_times), float)
    if len(ka) < MIN_KA:
        if verbose: print('[av_sync] カッが %d 個しか無いので測らない' % len(ka))
        return None, 0.0, 0.0, len(ka)
    lo = (ka.min() - 1.0) if lo is None else lo
    hi = (ka.max() + 1.0) if hi is None else hi
    e = _hi_energy(A.load(video, lo, hi - lo))
    p, off, n = _profile(e, ((ka - lo) * 1000).round().astype(int))
    if p is None:
        if verbose: print('[av_sync] 区間内のカッが %d 個で足りない' % n)
        return None, 0.0, 0.0, n
    k = int(np.argmax(p)); base = float(np.median(p)); amp = float(p[k]) - base
    outside = p[np.abs(off - off[k]) > 60]
    snr = amp / (outside.std() + 1e-12)
    # 立ち上がりは山の高さの 50% を横切る点(2026-09-22 に 25% から変更)。段位道場など 34 曲で参照由来の値と比べると
    # 10%: 平均 −6.9 sd 7.2 / 25%: −3.7 sd 7.0 / 50%: +0.4 sd 6.8 / 傾き最大: −0.4 sd 7.5 / 頂点: +12.3 sd 6.2(ms)。
    # 50% は定数の調整なしで偏りがほぼ 0。どの取り方でも sd は 6〜7ms 残る(筐体ごとに叩き音と曲の音の出方が違うと見ている)
    lv = base + 0.50 * amp
    i = k
    while i > 0 and p[i] > lv: i -= 1
    dd = p[i + 1] - p[i]
    lag = (off[i] + ((lv - p[i]) / dd if dd else 0.0)) / 1000.0
    if verbose:
        print('[av_sync] 映像→音声の遅れ %+.1fms(カッ %d 個、山の高さ %.3f、SNR %.1f、頂点 %+dms)%s' % (
            lag * 1000, n, amp, snr, off[k], '' if snr >= MIN_SNR else '  ← 山が弱いので使わない'))
    if snr < MIN_SNR or not (-0.4 < lag < 0.4): return None, amp, snr, n
    return float(lag), amp, snr, n


def audio_stability(video, ogg, delta, seg=15.0, verbose=True):
    """Δ(動画音声↔ogg)の確かさを、区間ごとの生波形の遅れで測る。
    同じ音源なら区間をまたいでも遅れは一定(実測 sd 0.0〜0.7ms)。音源の版が違う/音が濁った録画では
    区間ごとにばらつき、相関も 0.1 を割る(ユーザーの Xa の OBS 録画: sd 41.7ms、相関 0.05〜0.31)。
    戻り値: (Δ の補正[秒], ばらつき[秒], 区間ごとの相関の中央値, 区間数, 使えるか)"""
    b = A.load(ogg); dur = len(b) / A.SR
    lags = []; cors = []
    for t0 in np.arange(2.0, dur - seg - 2.0, seg):
        a = A.load(video, delta + t0, seg)
        W = int(0.060 * A.SR); o0 = int(round(t0 * A.SR))
        if o0 - W < 0 or o0 + len(a) + W > len(b): continue
        s = b[o0 - W:o0 + len(a) + W]
        cc = np.fft.irfft(np.fft.rfft(s, len(s)) * np.conj(np.fft.rfft(a, len(s))), len(s))
        k = int(np.argmax(cc[:2 * W]))
        lags.append((k - W) / A.SR)
        cors.append(cc[k] / (np.linalg.norm(a) * np.linalg.norm(s[k:k + len(a)]) + 1e-9))
    if len(lags) < 3: return 0.0, 9.9, 0.0, len(lags), False
    lags = np.array(lags); cors = np.array(cors); med = float(np.median(lags))
    # 多数決: 中央値から ±3ms 以内の区間が 6 割以上あれば、その区間の値で Δ を直す。
    # 動画によっては一部の区間だけ遅れが違う(カーマイン: 5 区間中 4 区間が +40.6ms、1 区間だけ −0.7ms。
    # find_delta は曲の中ほど 60 秒だけで精密化するので、たまたまその 1 区間に合わせていた)
    inl = np.abs(lags - med) < 0.003
    fix = float(np.median(lags[inl])) if inl.any() else med
    cm = float(np.median(cors[inl])) if inl.any() else float(np.median(cors))
    ok = bool(inl.mean() >= 0.6 and cm >= 0.08 and abs(fix) < 0.080)
    if verbose:
        print('[av_sync] 音源の一致: 区間 %d 個 %s、そろった区間 %d 個の遅れ %+.1fms、相関 %.2f → %s' % (
            len(lags), ' '.join('%+.1f' % (x * 1000) for x in lags), int(inl.sum()), fix * 1000, cm, '確か' if ok else '当てにならない'))
    # 戻り値の fix は「Δ から引く量」(ogg が fix だけ後ろに見える = ogg_t = video_t − (Δ − fix))
    return fix, float(lags.std()), cm, len(lags), ok
