"""動画 + 曲フォルダ(tja と ogg)→ 音声でΔ(動画↔ogg)を出し、区間を決め、キャリブレーション付きで検出し、
全コースの参照と照合して、最も合うコースの結果を報告する。TJA も生成する。
使い方: run_song.py <video> <song_dir> <out_prefix> [--no-tja]
  出力: <out_prefix>_ons.json, <out_prefix>.tja, 標準出力に評価
"""
import sys, os, glob, re, json, numpy as np
import audio_align as A
import calib, detect2
from eval_timing import parse_tja_times, parse_tja_bars, bar_shift
from eval_ons import evaluate, load_ons

def find_delta(video, ogg):
    """動画全体 × ogg 全体のエンベロープ相関で Δ 候補(上位)→ 最良を音声サンプル精度で精密化"""
    a = A.load(video); b = A.load(ogg); hop = 64
    ea, eb = A.envelope(a, hop), A.envelope(b, hop)
    n = 1
    while n < len(ea) + len(eb): n *= 2
    cc = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)[:len(ea)]
    cs = np.cumsum(np.r_[0, ea ** 2]); L = len(eb)
    den = np.sqrt(np.maximum(cs[np.minimum(np.arange(len(cc)) + L, len(ea))] - cs[np.arange(len(cc))], 1e-6)) * np.linalg.norm(eb)
    score = cc / den
    dur = len(b) / A.SR
    # 包絡の相関の上位 5 つ(1 秒以上離れたもの)をそれぞれ生波形で精密化し、波形の相関が最も高いものを採る。
    # 最大の 1 つだけだと、繰り返しの多い曲で 2 小節ずれた位置に合うことがある(夏祭り 新音源: 3.4 秒 = 140BPM の 2 小節ずれ)
    # 包絡の相関の上位 20 個(1 秒以上離れたもの)をそれぞれ生波形で精密化し、曲の 10 か所(10 秒ずつ)の波形の相関の中央値で比べる。
    # 包絡は曲によってほとんど差が出ず(天国と地獄: 本物 0.538 / 偽 0.550)、上位 5 つに本物が入らなかった。ゲームの音声と音源は一部だけ
    # 違うことがある(前半の編曲が違う等)ので、1 か所でなく 10 か所の中央値。繰り返しの多い曲の数小節ずれ(夏祭り 新音源)も中央値で落ちる
    # 候補は包絡の相関の上位から、互いに 0.25 秒以上離れたもの 40 個(同じ山の中の細かい揺れで埋まらない間隔。1 秒離すと、0.74 秒隣の山に
    # 本物が消された: 天国と地獄)。1 つ 0.06 秒ほどで確かめられる
    order = np.argsort(-score); peaks = []
    for k in order:
        if len(peaks) >= 40: break
        if all(abs(k - p) * hop / A.SR > 0.25 for p in peaks): peaks.append(int(k))

    def corr_at(d, vs, seg_len):
        """動画の vs 秒から seg_len 秒と、音源の (vs − d) 秒からの生波形の相関(動画の音声は読み込み済みの a から切り出す)"""
        a0 = int(round(vs * A.SR)); a2 = a[a0:a0 + int(seg_len * A.SR)]
        o0 = int(round((vs - d) * A.SR))
        if a0 < 0 or len(a2) < A.SR or o0 < 0 or o0 + len(a2) > len(b): return None
        seg = b[o0:o0 + len(a2)]
        return float(np.dot(a2, seg) / (np.linalg.norm(a2) * np.linalg.norm(seg) + 1e-9))

    def refine(coarse, pos, seg_len):
        """曲の pos(0〜1)の位置 seg_len 秒で、生波形の相関が最大になる Δ と相関"""
        vs = coarse + max(3.0, dur * pos)
        cand = (coarse, -1.0)
        for d in np.arange(coarse - 3 * hop / A.SR, coarse + 3 * hop / A.SR, 1.0 / A.SR):
            c = corr_at(d, vs, seg_len)
            if c is not None and c > cand[1]: cand = (d, c)
        return cand
    m_ = int(0.25 * A.SR)                           # 細かい合わせの範囲 ±0.25 秒(候補の間隔と同じ。包絡の山は本物から 125ms ずれることがある: Xa の OBS 録画)

    def fine(coarse):
        """曲の 10 か所(10 秒ずつ)で、ずれ(±3 刻み、音声サンプル単位)ごとの生波形の相関を FFT でまとめて出し、10 か所の中央値が最大のずれ。
        1 か所で細かく合わせると、音源とゲームの音声の中身が違う部分(天国と地獄の前半)で位置が外れた"""
        seg_n = int(min(10.0, dur * 0.08) * A.SR); rows = []
        for p_ in np.linspace(0.05, 0.9, 10):
            vs = coarse + dur * p_; a0 = int(round(vs * A.SR)); a2 = a[a0:a0 + seg_n]
            o0 = int(round((vs - coarse) * A.SR)) - m_
            if a0 < 0 or len(a2) < seg_n or o0 < 0 or o0 + seg_n + 2 * m_ > len(b): continue
            w = b[o0:o0 + seg_n + 2 * m_]
            nf = 1
            while nf < len(w) + seg_n: nf *= 2
            xc = np.fft.irfft(np.fft.rfft(w, nf) * np.conj(np.fft.rfft(a2, nf)), nf)[:2 * m_ + 1]      # xc[j] = Σ a2[i]·w[i + j]
            c2 = np.cumsum(np.r_[0.0, w * w]); nw = np.sqrt(np.maximum(c2[seg_n:seg_n + 2 * m_ + 1] - c2[:2 * m_ + 1], 1e-12))
            rows.append(xc / (nw * (np.linalg.norm(a2) + 1e-9)))
        if not rows: return coarse, -1.0
        # か所ごとに最もよく合うずれを取り、その相関の中央値で候補を比べる(曲の中でずれが 20ms ほど変わる録画がある: 天国と地獄 前半 11.126 / 後半 11.145。
        # 同じずれでの中央値だと、どのずれでも一部のか所しか合わない)。Δ はよく合ったか所(最良の半分以上)のずれの中央値
        R_ = np.array(rows); jb = np.argmax(R_, axis=1); cb = R_[np.arange(len(R_)), jb]
        good = cb >= 0.5 * cb.max()
        return coarse + (m_ - float(np.median(jb[good]))) / A.SR, float(np.median(cb))     # 音源の位置が j − m_ サンプル後ろ = Δ はその分小さい
    best = None; first = None
    for k in peaks:
        coarse = k * hop / A.SR
        d_f, c_f = fine(coarse)
        cand = (d_f, c_f, float(score[k]), c_f)
        if first is None: first = cand
        if best is None or cand[1] > best[1]: best = cand
    # 包絡の最大の候補から乗り換えるのは、10 か所の相関の中央値がはっきり高いとき(1.5 倍以上かつ 0.15 以上)だけ
    if best is not first and not (best[1] >= 0.15 and best[1] >= 1.5 * max(first[1], 1e-6)): best = first
    best = (best[0], best[3], best[2])
    return best[0], best[1], best[2], dur

def find_delta_cached(video, ogg):
    """find_delta の結果を動画・音源・コードで引くキャッシュ(detect2.cache_dir)"""
    import pickle, inspect, audio_align
    cdir = detect2.cache_dir()
    if not cdir: return find_delta(video, ogg)
    key = detect2.src_hash(detect2.video_id(video), detect2.video_id(ogg), inspect.getsource(find_delta), open(audio_align.__file__, encoding='utf-8').read())
    cpath = os.path.join(cdir, 'delta_%s.pkl' % key)
    if os.path.exists(cpath):
        with open(cpath, 'rb') as fh: return pickle.load(fh)
    r = find_delta(video, ogg)
    with open(cpath + '.tmp%d' % os.getpid(), 'wb') as fh: pickle.dump(r, fh)
    os.replace(cpath + '.tmp%d' % os.getpid(), cpath)
    return r


def ensure_60fps(video):
    """120fps などの録画(OBS)は 60fps に間引いてから使う(検出器の速度・フレーム数の閾値は 60fps 基準)。
    間引きは fps フィルタ(120→60 なら 1 枚おき、再サンプリング無し)。出力は V2T_WORK(既定: 一時フォルダ)/<名前>_60.mp4。既にあれば再利用"""
    import subprocess, tempfile
    # 冒頭の時刻が 0 でない動画は、映像と音声を同じだけずらして 0 秒始まりに直す(再エンコードなしのコピー)。後段の読み直しはフレーム番号を
    # 「時刻 × fps」で付けるので、冒頭がずれていると検出(何枚目か)と食い違い、一致率が 48% に崩れた(試験用の varoff_shippuu: 冒頭 1.5 秒)
    st = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip()
    try: st = float(st)
    except ValueError: st = 0.0
    if abs(st) > 0.001:
        work = os.environ.get('V2T_WORK') or os.path.join(tempfile.gettempdir(), 'video2tja')
        os.makedirs(work, exist_ok=True)
        z = os.path.join(work, os.path.splitext(os.path.basename(video))[0] + '_z0.mp4')
        if not os.path.exists(z) or os.path.getmtime(z) < os.path.getmtime(video):
            print('動画の冒頭の時刻が %.3f 秒なので 0 秒始まりに直す: %s' % (st, z))
            # 時刻の数値だけを引き算する(-itsoffset でずらすと負の時刻の分が捨てられ、冒頭 1.5 秒が削れた)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-map', '0', '-c', 'copy', '-bsf:v', 'setts=ts=TS-STARTPTS', '-bsf:a', 'setts=ts=TS-STARTPTS', z], check=True)
        video = z
    ensure_60fps.base = video                  # 0 秒始まりに直した元の動画(音声や元の fps の処理もこれを使う)
    rate = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video],
                          capture_output=True, text=True).stdout.strip().split(',')[0]
    try:
        num, den = rate.split('/'); fps = float(num) / float(den)
    except Exception:
        return video, None
    if 59.0 <= fps <= 61.0: return video, None
    if fps < 20.0: return video, None
    if fps < 59.0:                         # 30fps・50fps などは 60fps のスロットに並べ直す(間は複製として skip_frames で空にする。
                                           # 50fps をそのまま通すと速さ px/フレームが 60fps 前提の SCROLL の換算と 1.2 倍ずれ、速い連打の拾い直しも働かない)
        work = os.environ.get('V2T_WORK') or os.path.join(tempfile.gettempdir(), 'video2tja')
        os.makedirs(work, exist_ok=True)
        out = os.path.join(work, os.path.splitext(os.path.basename(video))[0] + '_up60.mp4')
        if not os.path.exists(out + '.map.json') or os.path.getmtime(out) < os.path.getmtime(video):
            print('動画が %.2ffps なので 60fps のスロットに並べ直す(prep60.prepare_up): %s' % (fps, out))
            import prep60; prep60.prepare_up(video, out)
        return out, out + '.map.json'
    work = os.environ.get('V2T_WORK') or os.path.join(tempfile.gettempdir(), 'video2tja')
    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, os.path.splitext(os.path.basename(video))[0] + ('_60nv.mp4' if os.environ.get('V2T_PREP_ENC') == 'nvenc' else '_60.mp4'))
    import prep60
    def _rule_ok():
        try: return json.load(open(out + '.map.json')).get('rule') == prep60.RULE      # 新しい絵の決め方が変わったら作り直す
        except Exception: return False
    if not os.path.exists(out + '.map.json') or os.path.getmtime(out) < os.path.getmtime(video) or not _rule_ok():
        print('動画が %.0ffps なので新しい絵のフレームだけで 60fps に直す(prep60): %s' % (fps, out))
        import prep60; prep60.prepare(video, out)
    else:
        print('動画が %.0ffps: 変換済みの %s を使う' % (fps, out))
    return out, out + '.map.json'

def _live(stage, onsets, fps, bars=None, bvel=None, gogo=None):
    """画面(web)の途中経過(V2T_LIVE のファイル)に、その段階までに決まった音符を書く。変換の結果には使わない"""
    p = os.environ.get('V2T_LIVE')
    if not p: return
    keep = ('t', 'color', 'big', 'vel', 'count', 'ringtrack', 'front', 'sighted', 'line', 'rescued', 'emerged', 'emerge', 'hidden')
    ons = [{k: o[k] for k in keep if k in o} for o in onsets]
    detect2._live_write(p, {'stage': stage, 'ons': {'fps': fps, 'onsets': ons, 'bars': list(bars or []), 'bar_vels': list(bvel or []), 'gogo': list(gogo or [])}})

import time as _time
_T0 = [_time.time()]
def _lap(name):
    t = _time.time(); print('[時間] %-10s %6.1fs' % (name, t - _T0[0]), flush=True); _T0[0] = t

def main():
    video, song_dir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    video, timemap = ensure_60fps(video); video_raw = ensure_60fps.base
    if song_dir.lower() == 'auto':
        # 曲フォルダを動画の音声から探す(songfind.py)。同じ音源は 1 組にまとめて元の曲フォルダを優先し、1 位の組がはっきり高いときだけ使う
        import songfind
        ch_, groups_ = songfind.choose(songfind.find(video_raw, 8))
        for g_ in groups_[:5]:
            print('曲の候補: 波形 %.3f(並び %.3f)%s' % (g_[0][0], g_[0][3], ' / '.join(os.path.relpath(os.path.dirname(c_[1]), songfind.SONGS) for c_ in g_)))
        if ch_ is None: sys.exit('曲をはっきり 1 つに決められませんでした。曲フォルダを指定してください')
        song_dir = os.path.dirname(ch_[1])
        tj_, _ = songfind._title(ch_[1])
        if tj_ and not os.environ.get('REF_TJA'): os.environ['REF_TJA'] = tj_; os.environ['V2T_REF_AUTO'] = '1'
        print('曲フォルダ: %s(参照 TJA %s)' % (song_dir, tj_))
        # 曲探しは音源との時刻のずれ Δ を下の音声合わせと同じ方法(曲の 10 か所の生波形の相関)で測っているので、それを使って音声合わせを省く(7〜14 秒)
        if not os.environ.get('VID2OGG'): os.environ['VID2OGG'] = '%.4f' % ch_[2]
    tjas = sorted(glob.glob(os.path.join(song_dir, '*.tja')))
    # 曲フォルダに TJA が複数ある(双打版・別アーティスト版・裏だけの別ファイルなど)。フォルダ名(先頭の番号を除く)と
    # 同じ名前のものを優先する。REF_TJA=<ファイル名> で明示もできる
    want = os.environ.get('REF_TJA')
    if want:
        tja = want if os.path.isabs(want) else os.path.join(song_dir, want)
    else:
        base = re.sub(r'^[0-9]+\s+', '', os.path.basename(os.path.normpath(song_dir)))
        tja = next((t for t in tjas if os.path.splitext(os.path.basename(t))[0] == base), tjas[0])
    if len(tjas) > 1: print('参照 TJA: %s(このフォルダには %d 個: %s)' % (os.path.basename(tja), len(tjas), ', '.join(os.path.basename(t) for t in tjas)))
    oggs = sorted(glob.glob(os.path.join(song_dir, '*.ogg')))
    _w = re.search(r'^WAVE:\s*(.+)$', open(tja, encoding='utf-8-sig', errors='replace').read(), re.M)   # 選んだ TJA の音源を使う
    ogg = os.path.join(song_dir, _w.group(1).strip()) if (_w and os.path.exists(os.path.join(song_dir, _w.group(1).strip()))) else oggs[0]
    if len(oggs) > 1: print('音源: %s(%d 個)' % (os.path.basename(ogg), len(oggs)))
    txt = open(tja, encoding='utf-8-sig').read()
    offset = float(re.search(r'^OFFSET:\s*([-\d.]+)', txt, re.M).group(1))
    title = re.search(r'^TITLE:(.*)$', txt, re.M).group(1).strip()
    _lap('準備')
    if os.environ.get('VID2OGG'):
        # Δ が与えられていれば(曲探し・小節線から求めた値)音声合わせを飛ばす。音源の長さだけ測る
        import subprocess as _sp
        dur = float(_sp.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', ogg], capture_output=True, text=True).stdout.strip() or 0)
        delta, corr, escore = float(os.environ['VID2OGG']), float('nan'), float('nan')
    else:
        delta, corr, escore, dur = find_delta_cached(video_raw, ogg)
    _lap('音声合わせ')
    if os.environ.get('VID2OGG'):      # 音声の相関が失敗する動画用(小節線から求めた値を渡す: find_delta_bars.py)
        delta = float(os.environ['VID2OGG']); print('[%s] Δ(VID2OGG) を %.4f に上書き' % (title, delta))
    print('[%s] Δ(VID2OGG) = %.4f  (envelope score %.3f, waveform corr %.3f)  ogg %.1fs' % (title, delta, escore, corr, dur))
    start = max(0.0, delta - 3.0); end = delta + dur + 2.0
    # ゴーゴーは音源が終わった後の空の小節まで続くことがある(第六天魔王 裏: 本家の #GOGOEND は音源の終わりの約 7 秒後)。デコードは +12 秒まで広げ、
    # 音符・小節線は従来どおり音源の範囲(lo〜hi)だけ使う
    import cv2 as _cv2
    _cap = _cv2.VideoCapture(video); _vlen = (_cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0) / (_cap.get(_cv2.CAP_PROP_FPS) or 60.0); _cap.release()
    end_g = min(delta + dur + 12.0, _vlen - 0.1) if _vlen > 0 else end
    # キャリブレーション(音符が流れている区間の中ほど 40 秒)。曲フォルダの TJA に音符があれば、その範囲(全コースを合わせた最初〜最後の音符)の中ほど。
    # 音源の長さの中ほどだと、音源の後ろが演奏に使われない曲で結果画面を読んだ(Hurtling Beams: 音源 74.7 秒、音符は 0〜29.3 秒。判定枠 x=1109 と誤った)
    n0_, n1_ = start, end
    try:
        _nt = [t for crs in re.findall(r'^COURSE:\s*(\w+)', txt, re.M) for t, *_ in parse_tja_times(tja, crs)[1]]
        if _nt: n0_, n1_ = min(max(delta - offset + min(_nt), start), end), min(max(delta - offset + max(_nt), start), end)
    except Exception as _ex:
        print('[%s] 較正の区間: TJA の音符の範囲を読めなかった(%s)。音源の長さで決める' % (title, _ex))
    # 判定枠・レーンは動かない画面の部品なので、音符の範囲の 25%・50%・75% の 3 か所(6 秒ずつ)で測り、判定枠の位置が中央値の結果を採る
    # (1 か所だと長い連打の棒が判定枠を覆う区間に当たり、判定枠を 84px 手前と誤った: ドキドキ胸きゅん おまつりタイム)
    # 3 か所は同時に測る(OpenCV のデコードは GIL を離す。順に測ると 1 曲 5 秒)
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _cal1(q_):
        m_ = n0_ + (n1_ - n0_) * q_
        try: return calib.calibrate(video, max(start, m_ - 20), min(end, m_ + 20), window=6.0)
        except RuntimeError as _ex: print('[%s] 較正(%.0f%%)できず: %s' % (title, q_ * 100, _ex)); return None
    with _TPE(3) as _ex3: cals = [c for c in _ex3.map(_cal1, (0.25, 0.5, 0.75)) if c is not None]
    if not cals: cals = [calib.calibrate(video, max(start, (start + end) / 2 - 20), min(end, (start + end) / 2 + 20))]
    cal = sorted(cals, key=lambda c: c['judge_x'])[len(cals) // 2]
    if len(cals) > 1: print('[%s] 較正 3 か所: 判定枠 x %s・レーン y %s → 判定枠 x %.1f を採用' % (
        title, ' / '.join('%.1f' % c['judge_x'] for c in cals), ' / '.join('%.1f' % c['lane_y'] for c in cals), cal['judge_x']))
    _lap('較正')
    lo, hi = delta - 0.2, delta + dur + 0.2
    if os.environ.get('REUSE_ONS') and os.path.exists(prefix + '_ons.json'):
        # ビルダーだけ直したとき用: 前回の検出(JSON)をそのまま使う(検出 5 分 → 数秒)。検出器を変えたら付けないこと
        dj = json.load(open(prefix + '_ons.json', encoding='utf-8')); fps = dj['fps']; onsets = dj['onsets']; bars = dj['bars']; bvel = dj.get('bar_vels') or [None] * len(bars); gogo_iv = dj.get('gogo')
        print('[%s] REUSE_ONS: %s の検出 %d 音符・%d 小節線を再利用' % (title, prefix + '_ons.json', len(onsets), len(bars)))
    else:
        skip = None; clock = None
        if timemap:
            mp_ = json.load(open(timemap)); use_ = mp_['use']
            skip = {k for k in range(1, len(use_)) if use_[k] == use_[k - 1]}     # 新しい絵が無かったスロット
        # 60fps の入れ物に 30fps の中身の動画(複製フレーム)と、録画が挟んだ複製・落としたフレーム(ゲームの時計)は
        # 検出のデコード中に行プロファイルから決める(detect2.run_parallel の auto_dups)
        fps, onsets = detect2.run_parallel(video, start, max(end, end_g), dump=None, dbg=True, cal=cal, skip_frames=skip, clock=clock, auto_dups=not timemap,
                                             gogo_range=(max(start, lo), max(min(end, hi), end_g)))
        _lap('検出')
        bars = detect2.run_parallel.bars; bvel = detect2.run_parallel.bar_vels
        _live('検出', onsets, fps, bars, bvel)
        import prep60
        single = list(getattr(prep60.game_clock, 'single', []) or [])
        # 2〜3 枚の落ちとして時刻を直したものを、直したあとの小節線で確かめる。録画の落ちなら小節線は前の格子どおりに続き、ゲーム側の処理落ち(動画の時刻は
        # 正しく、絵だけ先へ跳んだ)なら直した分((k−1) フレーム)ちょうど遅れる。後者は直さずにやり直す(Synthesis. 51.03s: 直すと小節線が +33ms 遅れ、
        # 跳びの前後で同じ音符が 40ms ずれた 2 つになった。ほうかご☆マジシャン 117.5s は直したあと 0ms)
        nodrop_ = []
        if not timemap and 'V2T_NODROP' not in os.environ:
            for f_, k_ in (getattr(prep60.game_clock, 'drops', []) or []):
                if k_ < 3: continue
                med_, good_ = prep60.bar_offset(bars, f_ / fps, fps)
                if med_ is not None and good_ >= 4 and abs(med_ - (k_ - 1) * 1000.0 / fps) <= 4:
                    print('[%s] %d 枚落ち フレーム %d(%.2fs): 直したあとの小節線が %+.1fms 遅れる → ゲーム側の処理落ちとして直さない' % (title, k_ - 1, f_, f_ / fps, med_))
                    nodrop_.append(f_)
        if nodrop_: os.environ['V2T_NODROP'] = ','.join(str(f_) for f_ in nodrop_)
        if (single or nodrop_) and not timemap and 'V2T_DROP1' not in os.environ:
            # 1 枚だけの跳び(流れ 2 倍)は、小節線がちょうど 1 フレーム早まる段差になっているものだけ録画の落ちとして直し、検出をやり直す(候補はキャッシュ)
            ok_ = []; unk_ = []
            for f_ in single:
                r_, med_ = prep60.bar_step(bars, f_ / fps, fps)
                print('[%s] 1 枚だけの跳び フレーム %d(%.2fs): 小節線の段差 %s → %s' % (title, f_, f_ / fps, 'なし' if med_ is None else '%+.1fms' % med_, '録画の落ちとして直す' if r_ else '直さない'))
                if r_: ok_.append(f_)
                elif med_ is None: unk_.append(f_)
            # 録画の時計のずれは一定の速さで溜まるので、録画の落ちは等間隔に起きる(六兆年と一夜物語: 4483・5479・6475・7471 と 996 フレームおき)。
            # 小節線で段差を測れない跳び(近くに小節線がそろっていない)でも、確かめた落ち 3 つ以上の間隔が ±1 フレームでそろい、その周期に
            # 乗っていれば録画の落ち(周期 n 個ぶん離れていれば ±n フレーム。8467 = 7471 + 996: 以後の 143〜146s が全部 1 フレームずれていた)
            if len(ok_) >= 3 and unk_:
                gaps_ = np.diff(sorted(ok_))
                per_ = float(np.median(gaps_))
                if per_ > 0 and np.all(np.abs(gaps_ - per_) <= 1):
                    for f_ in unk_:
                        n_ = round((f_ - min(ok_)) / per_)
                        if abs(f_ - (min(ok_) + n_ * per_)) <= abs(n_):      # 間隔のぶれ ±1 フレームが周期ごとに積もる
                            print('[%s] 1 枚だけの跳び フレーム %d: 確かめた落ちの周期 %.0f フレームに乗る → 録画の落ちとして直す' % (title, f_, per_))
                            ok_.append(f_)
            if ok_ or nodrop_:
                os.environ['V2T_DROP1'] = ','.join(str(f_) for f_ in ok_)
                fps, onsets = detect2.run_parallel(video, start, max(end, end_g), dump=None, dbg=True, cal=cal, skip_frames=skip, clock=clock, auto_dups=not timemap,
                                                     gogo_range=(max(start, lo), max(min(end, hi), end_g)))
                del os.environ['V2T_DROP1']
                os.environ.pop('V2T_NODROP', None)
                bars = detect2.run_parallel.bars; bvel = detect2.run_parallel.bar_vels
                _lap('検出(やり直し)'); _live('検出(やり直し)', onsets, fps, bars, bvel)
        # 音源が動画と違う(カバー音源・別の長さの版: 廻廻奇譚・ミックスナッツ・ALiVE)と Δ が外れ、検出する範囲(音源の範囲)の端で音符の列が切れる。
        # 最初(最後)の音符が範囲の端から 1 秒以内まで来ていたら、曲が範囲の外まで続いているとみなし、動画全体で検出し直して範囲では切らない
        dk_t = sorted(o['t'] for o in onsets if o['color'] in ('don', 'ka'))
        cut_lo = bool(dk_t) and start > 0.5 and dk_t[0] - start < 1.0
        # 終わり側: 検出はゴーゴーのために音源の後 12 秒まで行うので、範囲の終わりの直前 1 秒と、すぐ外の 2 秒の両方に音符があるときだけ
        # (曲の後、間を空けて出る結果画面の誤検出を「続いている」と取り違えた: ココドコ・ウルトラマンX)
        cut_hi = _vlen > 0 and end < _vlen - 0.5 and any(end - 1.0 <= t <= end for t in dk_t) and any(end < t <= end + 2.0 for t in dk_t)
        audio_mismatch = False
        if (cut_lo or cut_hi) and not os.environ.get('V2T_NO_REWIDEN'):
            audio_mismatch = True
            print('[%s] 警告: 音符の列が検出の範囲の%sで切れている(最初 %.2fs・最後 %.2fs、範囲 %.2f〜%.2fs)→ 音源が動画と違う(カバー音源・別の版)可能性。'
                  '動画全体で検出し直し、音源の範囲では切らない。OFFSET は当てにならない' % (title, '頭' if cut_lo and not cut_hi else ('終わり' if cut_hi and not cut_lo else '頭と終わり'), dk_t[0], dk_t[-1], start, end))
            start, end = 0.0, _vlen - 0.1
            fps, onsets = detect2.run_parallel(video, start, end, dump=None, dbg=True, cal=cal, skip_frames=skip, clock=clock, auto_dups=not timemap,
                                                 gogo_range=(start, end))
            bars = detect2.run_parallel.bars; bvel = detect2.run_parallel.bar_vels
            lo, hi = start, end
            _lap('検出(動画全体)')
        # 曲の音声区間の外(リザルト画面など)の検出は捨てる。音符は音声の中にしか無い
        onsets = [o for o in onsets if lo <= o['t'] <= hi]
        bb = [(b, v) for b, v in zip(bars, bvel) if lo <= b <= hi]; bars = [b for b, _ in bb]; bvel = [v for _, v in bb]
        hidden_mode = os.environ.get('V2T_HIDDEN', 'ring')        # ring: 円を追いかける(軽い) / synth: 描いて比べる / 0: しない
        if hidden_mode == 'ring':
            import ringtrack
            add_r, gone_r = ringtrack.recover(video, onsets, fps, lo, hi, cal, dbg=True)
            if gone_r:
                print('[%s] 途中から現れた偽の音符 %d 個を外した: %s' % (title, len(gone_r), ' '.join('%s %.3f' % (o['color'], o['t']) for o in gone_r)))
                onsets = [o for o in onsets if not any(o is g_ for g_ in gone_r)]
                # 外した音符が持っていた「控え」(同じ時刻に判定枠の手前で現れた組: detect2 の emerge)を戻す。同じ色の音符が 35ms 以内に残っていれば戻さない
                back_ = [g_['_emerge_alt'] for g_ in gone_r if g_.get('_emerge_alt')
                         and not any(q['color'] == g_['_emerge_alt']['color'] and abs(q['t'] - g_['_emerge_alt']['t']) < 0.035 for q in onsets)]
                if back_:
                    print('[%s] 外した音符の控えを %d 個戻した: %s' % (title, len(back_), ' '.join('%s %.3f' % (o['color'], o['t']) for o in back_)))
                    onsets = sorted(onsets + back_, key=lambda o: o['t'])
            if add_r:
                print('[%s] 隠れて流れる音符を %d 個足した: %s' % (title, len(add_r), ' '.join('%s %.3f(速さ %.1f)' % (o['color'], o['t'], o['vel']) for o in add_r)))
                onsets = sorted(onsets + add_r, key=lambda o: o['t'])
            _lap('隠れた音符'); _live('隠れた音符', onsets, fps, bars, bvel)
        if hidden_mode == 'synth' and os.environ.get('V2T_SYNTH', '1') != '0':
            # 速さの違う音符に追い越されて隠れながら流れる音符(カオスタイム)を、描いて比べて拾い直す(synth.py)
            import synth
            add, gone = synth.recover(video, onsets, fps, lo, hi, cal, dbg=True)
            if gone:
                print('[%s] 描き直して入れ替えた音符 %d 個: %s' % (title, len(gone), ' '.join('%s %.3f' % (o['color'], o['t']) for o in gone)))
                onsets = [o for o in onsets if not any(o is g for g in gone)]
            if add:
                print('[%s] 隠れて流れる音符を %d 個足した: %s' % (title, len(add), ' '.join('%s %.3f(速さ %.1f)' % (o['color'], o['t'], o['vel']) for o in add)))
                onsets = sorted(onsets + add, key=lambda o: o['t'])
            _lap('隠れた音符')
        if hidden_mode == 'synth' and os.environ.get('V2T_SYNTH', '1') != '0' and any(o.get('emerged') or o.get('emerge') for o in onsets):
            import synth
            gone_e = synth.verify_emerged(video, onsets, fps, cal, dbg=True)
            if gone_e:
                print('[%s] 途中から現れた偽の音符 %d 個を外した: %s' % (title, len(gone_e), ' '.join('%s %.3f' % (o['color'], o['t']) for o in gone_e)))
                onsets = [o for o in onsets if not any(o is g_ for g_ in gone_e)]
            _lap('現れた音符')
        # 判定枠での目撃(detect2 V2T_SIGHT)で足した音符は、目撃の後の段(断片の直線・ringtrack など)で同じ音符が見つかっていれば、目撃の方を外す
        # (夏祭り 52.00s: 積み重なったドンの一番上を目撃した時点では、そのドン 52.003 はまだ見つかっていなかった)。同じとみなす範囲は目撃のときと同じ
        sg_ = [o for o in onsets if o.get('sighted')]
        if sg_:
            dup_ = [o for o in sg_ if any(q is not o and not q.get('sighted') and (q['color'] == o['color'] or (o['color'] == 'don' and q['color'] == 'roll_end')) and o['sight_lo'] <= o['sight_ts'] - q['t'] <= o['sight_hi'] for q in onsets)]
            if dup_:
                print('[%s] 判定枠で目撃した音符のうち、後で見つかった音符と同じもの %d 個を外した: %s' % (title, len(dup_), ' '.join('%s %.3f' % (o['color'], o['t']) for o in dup_)))
                onsets = [o for o in onsets if not any(o is d_ for d_ in dup_)]
        if any(o.get('slow') for o in onsets):
            import synth
            onsets = synth.slow_hits(video, onsets, fps, cal, dbg=True); _lap('遅い音符')
        import balloon_count; balloon_count.annotate(video, onsets, cal)      # 風船の打数(吹き出しの数字を読む)
        _lap('風船'); _live('風船', onsets, fps, bars, bvel)
        gogo_iv = None
        # 120fps 録画の時刻補正は行わない(prep60.py 冒頭: tick の偶奇に時刻の情報が無く、直すと悪化した)
    if os.environ.get('V2T_GOGO') == 'redo': gogo_iv = None     # 検出は再利用し、ゴーゴーだけ測り直す
    if gogo_iv is None:                       # ゴーゴー(レーン背景の明るさ)。古い JSON を再利用したときもここで測る(数秒)
        try:
            import gogo as gogo_mod
            nt = [o['t'] for o in onsets if o['color'] in ('don', 'ka')]
            ro = sorted((o['t'], o['color'], bool(o.get('big'))) for o in onsets if o['color'] in ('renda', 'balloon', 'roll_end'))
            rolls = []; h_ = None
            for t_, c_, b_ in ro:
                if c_ != 'roll_end': h_ = (t_, c_ == 'balloon' or b_)      # (頭, 帯がゴーゴーの判定行に掛かるか: 大連打・風船)
                elif h_ is not None: rolls.append((h_[0], t_, h_[1])); h_ = None
            gf_ = getattr(detect2.run_parallel, 'gogo_feats', None) if not os.environ.get('REUSE_ONS') else None
            # 画面の上の物(音符・連打の頭と棒): (頭の時刻, 終わりの時刻, 速さ px/フレーム, 半径)。右端の色の窓に掛かるフレームは色を測らない
            objs_ = []
            ro_ = sorted((o for o in onsets if o['color'] in ('renda', 'balloon', 'roll_end') and o.get('vel')), key=lambda o: o['t'])
            # 連打・風船の頭だけ(顔の赤い口や縁が背景の色に混ざる。普通の音符と棒は背景の画素の割合 BG_MIN で扱う: 全部の音符で除くと右端はほぼいつも覆われる)
            for o in ro_:
                if o['color'] != 'roll_end': objs_.append((o['t'], o['t'], abs(o['vel']), 84.0 if (o.get('big') or o['color'] == 'balloon') else 50.0))
            gogo_iv = gogo_mod.intervals(video, max(start, lo), max(min(end, hi), end_g), cal, bars, nt, rolls=rolls, feats=gf_, objs=objs_)
            _lap('ゴーゴー'); _live('ゴーゴー', onsets, fps, bars, bvel, gogo_iv)
        except Exception as _ex:
            print('[%s] ゴーゴーの検出に失敗: %s' % (title, _ex)); gogo_iv = []
    # 画面(web)の重ね描き用: 動画のフレーム f → 音符の時刻と同じ時計のフレーム(録画の落ち・複製の並べ直しと描画の揺れ。ringtrack._eff と同じ)。
    # 動画の再生時刻のままで描くと、落ちたフレームの後はそのフレーム数ぶん円がずれて見えた
    teff = None
    if os.environ.get('REUSE_ONS') and os.path.exists(prefix + '_ons.json'):
        teff = json.load(open(prefix + '_ons.json', encoding='utf-8')).get('teff')
    elif getattr(detect2.run_parallel, 'clk', None) is not None or getattr(detect2.run_parallel, 'gdelta', None) is not None:
        import ringtrack
        f_lo, f_hi = int(max(start, lo) * fps), int(min(end, hi) * fps) + 1
        teff = {'f0': f_lo, 'd': [round(ringtrack._eff(f, detect2.run_parallel.clk) - f, 3) for f in range(f_lo, f_hi)]}
    # 連打の終わりを譜面の終わりの点へ(追跡の点のままだと小連打は 18px 早く、大連打は 18px 遅い: detect2.ROLL_END_S の注)。前回の検出を使うときは済んでいる
    if not (os.environ.get('REUSE_ONS') and os.path.exists(prefix + '_ons.json')) and os.environ.get('V2T_ROLLEND', '1') != '0':
        detect2.roll_end_point(onsets, fps); onsets.sort(key=lambda o: o['t'])
    json.dump({'fps': fps, 'start': max(start, lo), 'end': min(end, hi), 'onsets': onsets, 'bars': bars, 'bar_vels': bvel, 'gogo': gogo_iv, 'teff': teff},
              open(prefix + '_ons.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[%s] detected %d onsets, %d bars' % (title, len(onsets), len(bars)))
    # 全コースと照合。参照 TJA を指定していなければ、同じ音源を使うフォルダの TJA を全部照合する(裏譜面だけの別ファイル「(Ura).tja」など。
    # 曲探しが選んだ音源の最初の TJA が裏だけのファイルで、鬼の動画を裏と比べた: エンジェルドリーム)。TJA ごとに自分の OFFSET で
    mt, mc, rng = load_ons(prefix + '_ons.json')
    best = None; best_tja = (tja, offset)
    ref_tjas = [tja]
    if not want or os.environ.get('V2T_REF_AUTO'):
        for t_ in tjas:
            if t_ == tja: continue
            try: w_ = re.search(r'^WAVE:\s*(.+)$', open(t_, encoding='utf-8-sig', errors='replace').read(), re.M)
            except Exception: w_ = None
            if w_ and w_.group(1).strip() == os.path.basename(ogg): ref_tjas.append(t_)
    for tj_r in ref_tjas:
        txt_r = open(tj_r, encoding='utf-8-sig', errors='replace').read()
        m_o = re.search(r'^OFFSET:\s*([-\d.]+)', txt_r, re.M); off_r = float(m_o.group(1)) if m_o else offset
        for crs in re.findall(r'^COURSE:\s*(\w+)', txt_r, re.M):
            bpm, tn = parse_tja_times(tj_r, crs)
            if len(tn) < 10: continue
            tt = np.array([n[0] for n in tn]); tc = np.array([n[1] for n in tn])
            print('== %s vs %s%s (vid_offset %.3f) ==' % (title, crs, '' if tj_r == tja else '(%s)' % os.path.basename(tj_r), delta - off_r))
            r = evaluate(mt, mc, tt, tc, delta - off_r, rng)
            if r and (best is None or r[0] + r[1] > best[1]):
                best = (crs, r[0] + r[1], r); best_tja = (tj_r, off_r)
    if best_tja[0] != tja:
        tja, offset = best_tja; txt = open(tja, encoding='utf-8-sig', errors='replace').read()
        m_t = re.search(r'^TITLE:(.*)$', txt, re.M); title = m_t.group(1).strip() if m_t else title
        print('[%s] 参照 TJA: %s のコースが最も合った' % (title, os.path.basename(tja)))
    if best:
        crs, _, r = best
        print('[%s] best course: %s  recall %.1f%% precision %.1f%%  misses %d extras %d' % (title, crs, r[0], r[1], len(r[3]), len(r[4])))
        if r[3]: print('   misses (video s):', ' '.join('%.2f' % x for x in r[3][:40]))
        if r[4]: print('   extras (video s):', ' '.join('%.2f' % x for x in r[4][:40]))
        # 相違の自動判定: 長く綺麗なトラックで、前後の音符が参照と揃っているのに同種の参照音符だけ 25〜150ms ずれている → 参照側のずれ(実機準拠)
        if r[3] or r[4]:
            import refcheck
            _, tn_ = parse_tja_times(tja, crs)
            vd = refcheck.judge(video_raw if timemap else video, onsets, [(n[0], n[1]) for n in tn_], r[3], r[4], delta - offset, prefix + '_refcheck', cal)
            print('[%s] 相違の判定(証拠画像: %s_refcheck/):' % (title, prefix)); print(refcheck.report(vd))
            n_ref = sum(1 for v in vd if v[3] == 'reference-off'); n_unc = sum(1 for v in vd if v[3] == 'uncertain'); n_miss = sum(1 for v in vd if v[3] == 'unmatched')
            print('[%s] 実機基準: 参照側のずれ %d / 判定保留 %d / 取りこぼし %d(余分 %d)' % (title, n_ref, n_unc, n_miss, len(r[4]) - n_ref - n_unc))
    # 同期(音源に対する時刻の原点)。2 通りで出して両方報告する:
    #  ・av: 叩いたカッの音で「映像→音声の遅れ」を測る(参照譜面を使わない。公式譜面 10 曲で平均 0.0ms sd 3.8ms 最大 8.7ms)
    #  ・ref: 参照譜面の音符との時刻差(参照が正しければ 1〜3ms だが、参照に頼る。参照が怪しいと引きずられる)
    # 既定は av。V2T_SYNC=ref で従来通り、V2T_SYNC=av で av を強制(測れなければ ref に落ちる)
    corr_av = None
    try:
        import av_sync
        fix, sd_, cm_, nseg_, ok_ = av_sync.audio_stability(video_raw, ogg, delta)
        _lap('音声安定')
        ka_t = [o['t'] for o in onsets if o['color'] == 'ka']
        if timemap and ka_t:
            import prep60
            ka_t = [float(x) for x in prep60.apply_map(timemap, ka_t)]     # 60fps に直した時刻 → 元動画の時刻軸
        lag = av_sync.measure(video_raw, ka_t)[0] if ok_ else None
        if not ok_: print('[%s] 音源と動画音声が合っていないので av の同期は使わない' % title)
        # Δ は区間ごとの遅れの多数決(fix)で直す: Δ_true = Δ − fix。
        # (2026-09-20 に一度 +fix で入れてツンデレが −4.7→−14.3ms と悪化し外したが、それは符号の誤りだった)
        if lag is not None: corr_av = -float(lag) - fix
    except Exception as _ex:
        print('[%s] av_sync が使えなかった: %s' % (title, _ex))
    corr = 0.0
    if best:
        crs, _, r = best
        _, tn = parse_tja_times(tja, crs); tt = np.array([n[0] for n in tn]) + (delta - offset)
        # 最近傍の差の中央値だと、24 分の流れ(48ms 間隔)で 50ms のずれが「隣の音符との差 −2ms」に化ける(彁で発生)。
        # 評価器と同じ「ソフトスコアが最大になるシフト」を 8 秒窓ごとに求め、その中央値を補正量にする(ずれが半音価を超えても迷わない)
        from eval_ons import bestshift
        mt_s = np.array([o['t'] for o in onsets if o['color'] in ('don', 'ka') and not o['big']])
        if len(mt_s) >= 50 and len(tt):
            vo = delta - offset; g0 = bestshift(mt_s, tt - vo, vo, 0.6)
            shifts = []; t0 = mt_s.min(); sh = g0
            while t0 < mt_s.max():
                win = mt_s[(mt_s >= t0) & (mt_s < t0 + 8.0)]
                if len(win) >= 3: sh = bestshift(win, tt - vo, sh, 0.25); shifts.append(sh)
                t0 += 8.0
            corr = float(np.median(shifts) - vo) if shifts else float(g0 - vo)
            print('[%s] 参照との時刻差 %+.1fms(検出 − 参照、8 秒窓の整列の中央値。窓ごとの範囲 %+.0f〜%+.0fms)→ 音源との同期をその分補正' % (
                title, corr * 1000, (min(shifts) - vo) * 1000 if shifts else corr * 1000, (max(shifts) - vo) * 1000 if shifts else corr * 1000))
    # 小節線による同期: 参照にそのコースが無い動画(参照に裏譜面しか無い曲の表など)は音符が対応しないので、
    # 検出した小節線を参照の小節頭に合わせる(小節の格子は曲の音楽に従うので全コース共通)
    crs_ref = best[0] if best else next(iter(re.findall(r'^COURSE:\s*(\w+)', txt, re.M)), None)
    if bars and crs_ref:
        bs = bar_shift(bars, parse_tja_bars(tja, crs_ref), delta - offset)
        if bs:
            print('[%s] 小節線による同期: ずれの中央値 %.1fms(%d 本が 30ms 以内)→ 補正 %+.1fms' % (
                title, bs[0] * 1000, bs[2], (bs[1] - (delta - offset)) * 1000))
            weak = (best is None) or best[2][0] < 90.0 or best[2][1] < 90.0
            if weak and bs[0] < 0.02:
                corr = bs[1] - (delta - offset)
                print('[%s] 参照にこのコースが無い(一致が弱い)ので、小節線の同期を採用' % title)
    # どちらを採るか
    mode = os.environ.get('V2T_SYNC', 'av')
    corr_ref = corr
    have_ref = bool(best) or (corr != 0.0)            # 参照のコースと合ったか、小節線で同期できたか
    if mode != 'ref' and corr_av is not None:
        corr = corr_av
    print('[%s] 同期: 採用 %+.1fms(%s)  参照なし(av) %s  参照あり(ref) %s%s' % (
        title, corr * 1000, 'av' if (mode != 'ref' and corr_av is not None) else ('ref' if have_ref else 'なし'),
        ('%+.1fms' % (corr_av * 1000)) if corr_av is not None else '測れず',
        ('%+.1fms' % (corr_ref * 1000)) if have_ref else '参照なし',
        ('  ← 差 %+.1fms' % ((corr_av - corr_ref) * 1000)) if (corr_av is not None and have_ref) else ''))
    if corr_av is None and not have_ref:
        print('[%s] 注意: 参照なしの同期が測れず、参照譜面も無いので、音声の対応(Δ)だけで時刻を決めた(動画の映像と音声のずれの分だけ OFFSET がずれる)' % title)
    if corr_av is not None and have_ref and abs(corr_av - corr_ref) > 0.015:
        print('[%s] 注意: 2 つの同期が %.0fms 食い違う。参照譜面側の OFFSET がずれている可能性(彁・魔宵月で実例)' % (
            title, abs(corr_av - corr_ref) * 1000))
    if '--no-tja' not in sys.argv:
        os.environ['VID2AUDIO'] = '%.4f' % (delta + corr)
        if os.environ.get('FORCE_COURSE'): os.environ['COURSE'] = os.environ['FORCE_COURSE']    # コースを明示(参照に無いコースの動画)
        elif best: os.environ['COURSE'] = best[0]                    # 最も合った参照コース名で書く(裏なら Edit)
        import subprocess
        subprocess.run([sys.executable, 'make_tja.py', prefix + '_ons.json', str(start), str(end), prefix + '.tja', title], check=False)
        _lap('譜面作成')
        # 清書版(<prefix>_clean.tja): 音符・小節線を採譜から 5ms 以内、速さを 3% 以内に保って、セル数を最小に・細かい BPM/SCROLL をまとめる
        # (clean_tja.py。採譜そのもの <prefix>.tja は残す。試験・照合は採譜そのものを使う)
        if os.path.exists(prefix + '.tja') and os.environ.get('V2T_CLEAN', '1') != '0':
            try:
                import clean_tja
                raw_ = open(prefix + '.tja', encoding='utf-8-sig', errors='replace').read()
                new_, info_ = clean_tja.clean_text(raw_)
                clean_tja.write_tja(prefix + '_clean.tja', new_)
                s0, s1 = clean_tja.stats(raw_), clean_tja.stats(new_)
                mv = max([v.get('worst_ms', 0) for v in info_.values() if isinstance(v, dict)] or [0])
                print('[%s] 清書: 行 %d → %d、#BPMCHANGE %d → %d、#SCROLL %d → %d(音符が動いた最大 %.1fms)→ %s' % (
                    title, s0['lines'], s1['lines'], s0['bpm'], s1['bpm'], s0['scroll'], s1['scroll'], mv, os.path.basename(prefix + '_clean.tja')), flush=True)
            except Exception as ex:
                print('[%s] 清書に失敗(採譜そのものは %s): %s' % (title, os.path.basename(prefix + '.tja'), ex), flush=True)
            _lap('清書')

if __name__ == '__main__':
    calib.low_priority()
    main()
