"""120fps などの録画(OBS)を検出器用の 60fps 動画に直す。
ゲームは 60Hz で描画しているので 120fps の録画は半分が直前フレームの複製。fps フィルタで 1 枚おきに間引くと、
ゲームと録画の位相が流れるにつれて「新しい絵」と「古い絵(8.3ms 前)」が入れ替わり、音符の時刻が ±8ms の塊でずれる(Xa: 70s 以降 sd 5ms)。
ここでは
  1) 全フレームで直前との差(譜面レーンの平均絶対差)を取り、差が小さい(<3)フレームを複製とみなす
  2) 60Hz の各スロット(tick 2k, 2k+1)から新しい絵のフレームを選んで CFR 60fps の動画を書く(時刻軸は元動画のまま)
  3) map(JSON)に各スロットで使った tick(use)を書く。新しい絵が無かったスロット(use が直前と同じ)は run_song が
     skip_frames で候補を空にする(直前の絵を入れると 1 フレームぶん古い点が混ざる)。時刻の補正はしない(偶奇に情報が無かった)
  結果(Xa): fps フィルタの 1 枚おき間引きと同じ精度(参照比 sd 3.8ms。前半 1.3ms、後半 4〜6ms はゲーム側の描画タイミングの揺れで動画に情報が無い)
run_song は map があれば検出した時刻を map で補正する(apply_map)。
使い方: prep60.py <raw.mp4> <out60.mp4>   → out60.mp4 と out60.map.json
"""
import sys, os, json, subprocess, numpy as np, cv2

DIFF_TH = 3.0      # 譜面レーン(小節線の帯を含む y290-480 × x630-1910)の平均絶対差。複製は再エンコードのノイズで 0.05〜1.2、新しい絵は 20 以上(Xa)
ROI = (290, 480, 630, 1910)   # 全画面だと音符しか動かないフレームで差が薄まり、新しい絵を複製と誤る
# レーンの平均だけだと、動く物が小さいときに新しい絵を複製と誤る(検出テスト ファイナル 94.5〜96s: 叩かれない黄色い連打の棒がレーンを埋め、動くのは
# 棒の端と後ろのドンの三日月だけ。平均の差 < 3 で 80 枚中 77 枚を複製として捨て、ドンを落とした)。16px 四方の塊ごとの差の最大も見る:
# 複製は再エンコードのノイズだけで ≤ 4(同じ区間)、動く音符の縁(白い縁と暗い背景で 200 段)が 1px 動くだけで塊の差は 12 以上、実測 50 以上
BLOCK = 16
LOCAL_TH = 20.0
RULE = 2           # 新しい絵の決め方の版(変換済みの 60fps 動画を作り直す目印。run_song.ensure_60fps が見る)
QUEUE = 32         # 読む・書くスレッドとの受け渡しに溜める枚数(1080p で 1 枚 6MB → 200MB ずつ)。読み(HEVC 120fps で 480 枚/秒)と
                   # 書き出し(libx264 で 280 枚/秒)の速さの揺れを吸うだけで、多くしても速くならない
def ENC_ARGS():
    """60fps の中間の動画の書き出し方。既定は libx264(CPU)。V2T_PREP_ENC=nvenc で GPU(NVENC、画素が少し変わるので精度を確かめる用)"""
    if os.environ.get('V2T_PREP_ENC') == 'nvenc':
        return ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq', '-rc', 'constqp', '-qp', '16']
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '16']


_W3 = np.full((1, 3), 1 / 3.0, np.float32)


def _diff(s, prev):
    """直前のフレームとのレーンの差: (チャンネル平均の絶対差の平均, 16px 四方の塊ごとの平均の最大)。
    以前は numpy の int16 で同じ値を計算していた(1 枚 4.0ms → 0.9ms。120fps・18742 枚で新しい絵の判定が 1 枚も変わらないことを確かめた:
    差の最大 1e-4、閾値の ±1e-3 に入る枚も無し)"""
    g = cv2.transform(cv2.absdiff(s, prev).astype(np.float32), _W3); h_, w_ = g.shape
    d = float(cv2.mean(g)[0])
    b = cv2.resize(g[:h_ // BLOCK * BLOCK, :w_ // BLOCK * BLOCK], (w_ // BLOCK, h_ // BLOCK), interpolation=cv2.INTER_AREA)
    return d, float(b.max())


def _fresh(s, prev):
    d, bm = _diff(s, prev)
    return d >= DIFF_TH or bm >= LOCAL_TH, d

def analyze(raw, verbose=True):
    cap = cv2.VideoCapture(raw); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fresh = np.zeros(n, bool); diffs = np.zeros(n, np.float32); prev = None; i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        s = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        if prev is None: fresh[i] = True
        else: fresh[i], diffs[i] = _fresh(s, prev)
        prev = s; i += 1
    cap.release(); fresh = fresh[:i]; diffs = diffs[:i]
    if verbose: print('prep60: %d frames @%.0ffps, fresh %.3f (diff<%.1f = 複製)' % (i, fps, fresh.mean(), DIFF_TH))
    return fps, fresh, diffs

def build_map(fps, fresh):
    """新しい絵の tick 列から、60Hz スロットごとの (使うフレーム, 真の描画時刻[秒]) を作る"""
    r = int(round(fps / 60.0)); n = len(fresh); nslot = n // r
    F = np.flatnonzero(fresh)                                  # 新しい絵の tick(m 番目)
    m_of_tick = -np.ones(n, int); m_of_tick[F] = np.arange(len(F))
    # 絵の古さ(描画からキャプチャ tick までの時間、0〜1 tick): 新しい絵の間隔は普通 r tick。
    # 間隔 r−1(ゲームの方が速く、同じスロットに 2 枚)の直後は古さ ≈ 0、間隔 r+1(ゲームの方が遅く、スロットに 1 枚も無い)の直後は古さ ≈ 1 tick。
    # その間は位相が一様に流れるので直線でつなぐ。レーンが空で新しい絵が判らない区間(間隔がもっと長い)は事象にしない
    gaps = np.diff(F); ev_m = []; ev_age = []
    for j, g in enumerate(gaps):
        if g == r - 1: ev_m.append(j + 1); ev_age.append(0.0)
        elif g == r + 1: ev_m.append(j + 1); ev_age.append(1.0)
    # 実測(Xa、OBS 120fps): ゲームの描画間隔は 15.5/17.5/21ms と揺れていて(フレームペーシング)、tick の偶奇と音符の時刻誤差に関係が無かった
    # (偶奇別の平均誤差 −0.8 / −1.0ms)。偶奇で ±4.2ms 直すと悪化(sd 3.8 → 5.6ms)。時刻は補正せず、スロットの時刻 k/60 をそのまま使う
    age = np.full(len(F), 0.0)
    true_t = F / fps
    use = np.zeros(nslot, int); tt = np.zeros(nslot)
    last = 0
    for k in range(nslot):
        cand = [t for t in range(r * k, r * k + r) if fresh[t]]
        if cand: last = cand[0]                                # 2 枚あれば先(1 枚は捨てる: ゲームの方が速い側の slip)
        use[k] = last                                           # 無ければ直前の新しい絵(古さ 1 tick 超の点が 1 つ混じるだけ)
        tt[k] = k / 60.0                                        # 時刻は補正しない(上記)
    n_slip = len(ev_m)
    return {'rule': RULE, 'fps_raw': fps, 'r': r, 'use': use.tolist(), 'true_t': tt.tolist(), 'age_ms': (age * 1000 / fps).tolist(), 'n_slip': n_slip,
            'fresh_frac': float(fresh.mean())}

def _prepare_1pass(raw, out, verbose=True):
    """1 回読みながら書く。読む(cv2)・比べる・書き出す(ffmpeg へ渡す)を別々のスレッドにして重ねる
    (以前は 1 本の流れで順に行い、120fps・156 秒の録画で 145 秒。読み 39 秒・差 61 秒・書き出し 33 秒の足し算だった)"""
    import threading, queue
    cap = cv2.VideoCapture(raw); fps = cap.get(cv2.CAP_PROP_FPS); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = int(round(fps / 60.0)); n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', '60', '-i', '-',
           '-i', raw, '-map', '0:v', '-map', '1:a?', ] + ENC_ARGS() + ['-c:a', 'copy', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rq, wq = queue.Queue(QUEUE), queue.Queue(QUEUE); err = []

    def reader():
        try:
            while True:
                ok, img = cap.read()
                rq.put(img if ok else None)
                if not ok: break
        except Exception as ex:
            err.append(ex); rq.put(None)

    def writer():
        try:
            while True:
                img = wq.get()
                if img is None: break
                p.stdin.write(img.data if img.flags.c_contiguous else img.tobytes())
        except Exception as ex:
            err.append(ex)
            while wq.get() is not None: pass                # 読み側を止めないように捨てる

    tr = threading.Thread(target=reader, daemon=True); tw = threading.Thread(target=writer, daemon=True); tr.start(); tw.start()
    fresh = []; use = []; prev = None; i = 0; last_i = 0; last_img = None; slot = []
    import detect2
    try:
        while True:
            img = rq.get()
            if img is None: break
            s_ = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            fr_ = True if prev is None else _fresh(s_, prev)[0]
            prev = s_; fresh.append(fr_); slot.append((i, img, fr_)); i += 1
            if i % 600 == 0: detect2.progress('準備', i, n_all, '%.0ffps の録画を 60fps に直す(新しい絵だけを選ぶ)' % fps)
            if len(slot) == r:                                  # スロットの tick がそろった: 使う絵を決めて書く(build_map と同じ決め方)
                c_ = [(t, im) for t, im, f in slot if f]
                if c_: last_i, last_img = c_[0]
                if last_img is None: last_img = slot[0][1]
                use.append(last_i); wq.put(last_img); slot = []
    finally:
        wq.put(None); tw.join(); tr.join(timeout=5)
        p.stdin.close(); p.wait(); cap.release()
    if err: raise RuntimeError('prep60: 読み書きの途中で止まった: %s' % err[0])
    fresh = np.array(fresh, bool)
    mp = build_map(fps, fresh)
    if mp['use'] != use: raise RuntimeError('prep60: 1 回読みの use が build_map と違う')
    if verbose: print('prep60: %d frames @%.0ffps, fresh %.3f (diff<%.1f = 複製)' % (i, fps, fresh.mean(), DIFF_TH))
    return mp, len(use)


def write60(raw, out, use, fps_raw):
    cap = cv2.VideoCapture(raw); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', '60', '-i', '-',
           '-i', raw, '-map', '0:v', '-map', '1:a?', ] + ENC_ARGS() + ['-c:a', 'copy', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    want = np.asarray(use); i = 0; k = 0; buf = None; last_img = None
    while k < len(want):
        ok, img = cap.read()
        if not ok: break
        while k < len(want) and want[k] == i:
            p.stdin.write(img.tobytes()); k += 1
        i += 1
    p.stdin.close(); p.wait(); cap.release()
    return k

def prepare(raw, out, verbose=True):
    """1 回読むだけで作る(analyze → build_map → write60 と同じ動画・map)。スロット k で使う絵は、そのスロットの tick(r 枚)の新しい絵の先頭か、
    無ければ直前の新しい絵(build_map)で、先を読まずに決まるので、読みながら書く(以前は全部を 2 回読んだ: 120fps・156 秒の録画で準備 179 秒)"""
    if os.environ.get('V2T_PREP_2PASS') == '1':
        fps, fresh, diffs = analyze(raw, verbose)
        mp = build_map(fps, fresh)
        n = write60(raw, out, mp['use'], fps)
    else:
        mp, n = _prepare_1pass(raw, out, verbose)
    json.dump(mp, open(out + '.map.json', 'w'))
    if verbose:
        age = np.array(mp['age_ms'])
        print('prep60: wrote %d frames → %s  (slip %d 回、絵の古さ 0〜%.1fms を補正)' % (n, out, mp['n_slip'], age.max() if len(age) else 0))
    return out, out + '.map.json'

def prepare_up(raw, out, verbose=True):
    """30fps などの動画を 60fps のスロットに並べ直す(ゲームは 60Hz 描画なので、30fps の録画は 1 枚おきの絵しか無い)。
    元のフレーム k(時刻 t_k)をスロット round(t_k*60) に置き、間のスロットは直前の絵で埋めて map の use で「複製」と印を付ける。
    run_song は複製スロットを skip_frames で空にするので、検出器は 60fps 基準の速さ[px/frame]のまま、点が 1 枚おきになるだけ。
    29.97fps は 30 で割り切れないので、スロットは時刻から決める(k/30 とすると 100 秒で 0.1 秒縮む)"""
    cap = cv2.VideoCapture(raw); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', '60', '-i', '-',
           '-i', raw, '-map', '0:v', '-map', '1:a?', ] + ENC_ARGS() + ['-c:a', 'copy', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    use = []; k = 0; last = None
    while True:
        ok, img = cap.read()
        if not ok: break
        slot = int(round(k / fps * 60.0))
        while len(use) < slot and last is not None:          # 間のスロットは直前の絵(複製)
            p.stdin.write(last.tobytes()); use.append(use[-1])
        if len(use) <= slot:
            p.stdin.write(img.tobytes()); use.append(k)
        last = img; k += 1
    p.stdin.close(); p.wait(); cap.release()
    mp = {'fps_raw': fps, 'r': 0, 'use': use, 'true_t': [j / 60.0 for j in range(len(use))], 'age_ms': [], 'n_slip': 0,
          'fresh_frac': float(k) / max(1, len(use)), 'upsampled': True}
    json.dump(mp, open(out + '.map.json', 'w'))
    if verbose: print('prep60: %.2ffps の %d 枚を 60fps の %d スロットに並べた(複製 %d)' % (fps, k, len(use), len(use) - k))
    return out, out + '.map.json'

def _shifts(video, t0, t1, cal=None):
    """各フレームで、直前のフレームからレーンが流れた量[px](行プロファイルの相互相関)。レーンが空なら -99"""
    import calib
    M = None
    if cal is not None and not calib.is_canonical(cal): M, _ = calib.affine(cal)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    prev = None; sh = []; idx = []; diff = []; prev_band = None
    n = 2048
    for k in range(f0, f1):
        ok, img = cap.read()
        if not ok: break
        if M is not None: img = cv2.warpAffine(img, M, (1920, 1080))
        band = cv2.cvtColor(img[350:420, 760:1880], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff.append(float(np.abs(band - prev_band).mean()) if prev_band is not None else 99.0); prev_band = band
        row = band.mean(0)
        row -= row.mean()
        if prev is not None and row.std() > 5 and prev.std() > 5:
            cc = np.fft.irfft(np.fft.rfft(prev, n) * np.conj(np.fft.rfft(row, n)), n)
            cand = np.r_[cc[0:80], cc[n - 3:n]]; lag = np.r_[np.arange(80), np.arange(-3, 0)]
            sh.append(int(lag[int(np.argmax(cand))])); idx.append(k)
        else:
            sh.append(-99); idx.append(k)
        prev = row
    cap.release()
    _shifts.diff = np.array(diff)
    return idx, np.array(sh)


def shifts_from_rows(rows):
    """行プロファイルの列 → 各フレームの流れ[px](_shifts と同じ計算を一括で)。レーンが空なら -99"""
    R = np.asarray(rows, np.float32); R = R - R.mean(1, keepdims=True); n = 2048
    F = np.fft.rfft(R, n, axis=1); sd = R.std(1)
    sh = np.full(len(R), -99.0)
    lag = np.r_[np.arange(80), np.arange(-3, 0)]
    for i in range(1, len(R)):
        if sd[i] > 5 and sd[i - 1] > 5:
            cc = np.fft.irfft(F[i - 1] * np.conj(F[i]), n)
            c_ = np.r_[cc[0:80], cc[n - 3:n]]; k = int(np.argmax(c_)); l0 = int(lag[k])
            # 山の前後で放物線を当てて 1px 未満まで求める(22px/フレームなら 1px = 0.05 フレーム。小数のゲーム時刻に使う)
            ym, y0, yp = cc[(l0 - 1) % n], cc[l0 % n], cc[(l0 + 1) % n]
            den = ym - 2 * y0 + yp
            sh[i] = l0 + (0.5 * (ym - yp) / den if den < 0 else 0.0)
    return sh


def frac_offsets(idx, sh, clock=None, win=61):
    """フレームごとのゲーム時刻の小数のずれ[フレーム](整数の時計からの差)。そのフレームで進んだゲームのフレーム数 = 流れ ÷ 近くの速さ
    (複製 0・処理落ち 2・描画間隔の揺れ 0.8〜1.2 など)。整数の時計の進みとの差を累積し、前後 win 枚の平均を引く(録画は平均すれば 60fps)。
    戻り値 {フレーム番号: ずれ}"""
    sh = np.asarray(sh, float); n = len(sh)
    m = np.full(n, np.nan)
    for i in range(n):
        for R in (10, 40):
            w = sh[max(0, i - R):i + R + 1]; w = w[(w != -99) & (np.abs(w) > 1)]
            if len(w) >= 6: break
        if len(w) < 6: continue
        lo = np.percentile(w, 10); w = w[w <= 1.4 * lo]
        if len(w): m[i] = float(np.median(w))
    base = np.ones(n)
    if clock:
        g = np.array([clock.get(int(k), np.nan) for k in idx], float)
        d = np.diff(g, prepend=g[0] - 1); base = np.where(np.isnan(d), 1.0, d)
    with np.errstate(invalid='ignore', divide='ignore'):
        st = sh / m
    ok = (sh != -99) & ~np.isnan(m) & (m >= 4) & (st >= -0.2) & (st <= 3.2)
    e = np.where(ok, st - base, 0.0)
    E = np.cumsum(e)
    k = np.ones(win) / win
    Em = np.convolve(np.pad(E, (win // 2, win // 2), mode='edge'), k, mode='valid')
    r = E - Em
    return {int(i): float(v) for i, v in zip(idx, r)}


def dups_from_shifts(idx, sh, verbose=True):
    """流れの列から複製フレーム(流れ ≤1px、前後 10 枚の中央値 ≥4px)を拾う。戻り値 (set, 割合)"""
    dups = set()
    for i, s_ in enumerate(sh):
        if s_ == -99 or abs(s_) > 1: continue
        nb = sh[max(0, i - 10):i + 11]; nb = nb[(nb != -99) & (np.abs(nb) > 1)]
        if len(nb) >= 5 and np.median(nb) >= 4: dups.add(idx[i])
    frac = len(dups) / max(1, int((sh != -99).sum()))
    if verbose: print('prep60: 複製フレーム %d 枚(流れの見えたフレームの %.0f%%)' % (len(dups), frac * 100))
    return dups, frac


def find_dups(video, t0, t1, cal=None, verbose=True):
    """60fps の動画の中の「複製フレーム」(直前と同じ絵)を、レーンが流れた量で見つける。
    YouTube には 60fps の入れ物に 30fps の中身が入った動画がある(IRIS OUT むずかしい: 流れが 37,0,37,0…px)。
    複製フレームも時刻 k/60 として使うと、1 枚おきに 1 フレーム古い点が混ざり、重複の位相が途中で入れ替わるたびに
    音符の時刻が ±8ms の 2 つの山に割れる(sd 9ms)。複製を skip_frames で外すと 30fps 動画と同じ扱いになる。
    画素の差分(prep60.analyze)は、まばらな譜面でレーンが空だと新しい絵でも差が小さく、使えなかった。
    判定: そのフレームの流れが 1px 以下で、前後 10 フレームの(流れのある)中央値が 4px 以上。止まっている音符(ドンカマ)は
    前後も小さいので複製にしない。戻り値: (複製のフレーム番号の set, 複製の割合)"""
    idx, sh = _shifts(video, t0, t1, cal)
    dups = set()
    for i, s_ in enumerate(sh):
        if s_ == -99 or abs(s_) > 1: continue
        nb = sh[max(0, i - 10):i + 11]; nb = nb[(nb != -99) & (np.abs(nb) > 1)]
        if len(nb) >= 5 and np.median(nb) >= 4: dups.add(idx[i])
    frac = len(dups) / max(1, int((sh != -99).sum()))
    if verbose: print('prep60: 複製フレーム %d 枚(流れの見えたフレームの %.0f%%)' % (len(dups), frac * 100))
    find_dups.clock = game_clock(idx, sh, verbose, diff=_shifts.diff)
    return dups, frac


CAND_X1 = 1800        # detect2.run_parallel がゲームの時計に渡す候補の x の上限(800 ≤ x ≤ 1800)


def game_clock(idx, sh, verbose=True, diff=None, cands=None):
    """ゲームの描画フレームの番号を数える(録画側が挟んだ複製・落としたフレームを直す)。
    VERTeX(1080p60): 43.43s に 1 枚だけ複製(流れ 0px、前後 14px、画素の差 1.3)があり、以降の音符が全部 +16.7ms 遅れ、
    その前後は 0.13ms/s で早まっていく(録画の時計がわずかに遅く、遅れが 1 フレームに溜まった所で複製を挟んで音声に追いつく鋸歯)。
    ビルダーはこの段差をグリッドに乗せられず、26 小節を音符ごとの BPM 指定で書いていた。
    各フレームの進み = round(流れ / 近くの速さ)(0, 1, 2)。近くの速さ = 前後 30 枚の「普通の」フレーム(近く ±10 枚の中央値の ±10%)の中央値で、
    それらが ±10% に揃っている所だけ数える(SCROLL の切り替わりでは数えない)。
    0(複製)は画素の差が ≤3 で、前後のフレームが流れているときだけ(風船・曲の終わりは何枚も止まり、演出で相関が 0 に出たフレームは
    画素が変わっている: みんなのかき氷 116.78s)。複製と落ちたフレームは隣り合って組になることが多い(食べるな 71〜73s は 0,34,0,33…)。
    戻り値: {フレーム番号: ゲームのフレーム番号} か None(事象が無い/多すぎる)"""
    game_clock.blank = set(); game_clock.unpaired = []; game_clock.flow_steps = []; game_clock.single = []
    sh = np.asarray(sh); n = len(sh)
    valid = (sh != -99) & (np.abs(sh) > 1)
    # 近くの速さ m[i] = 前後 10 枚(足りなければ 40 枚)の流れのうち「小さい側の群れ」(下位 10% 点の 1.4 倍以下)の中央値。
    # 処理落ちが密な所(食べるな 71〜73s: 17,0,34,0,33,0,33…)は 2 倍の値の方が多く、ただの中央値だと 33 を「普通」と誤る
    m = np.full(n, np.nan)
    for i in range(n):
        for R in (10, 40):
            w = sh[max(0, i - R):i + R + 1]; w = w[(w != -99) & (np.abs(w) > 1)]
            if len(w) >= 6: break
        if len(w) < 6: continue
        lo = np.percentile(w, 10); w = w[w <= 1.4 * lo]
        if len(w): m[i] = float(np.median(w))
    with np.errstate(invalid='ignore', divide='ignore'):
        normal = valid & (np.abs(sh / np.where(np.isnan(m), 1e9, m) - 1) <= 0.1)
    step = np.ones(n, int); nd = nx = 0
    for i in range(1, n - 1):
        if sh[i] == -99 or np.isnan(m[i]) or m[i] < 4: continue
        if abs(sh[i]) <= 1:
            # 複製: 画素で確かめられる(同じ絵、差 ≤2。本物は 0.0〜1.3、VERTeX 91.58s の SCROLL の切り替わりで相関が 0 に出たフレームは 2.7)。
            # 前後が流れていて、前後の普通のフレームの流れが 25% 以内で揃っている(切り替わりの境目を除く。速さが 23→27px と変わっていく
            # ドンカマ 115.53s の複製は拾う)
            # (前後の揃いは見ない。処理落ちが密な所では普通のフレームが足りず、組の相手の複製を落としていた。孤立した複製として時刻を直すときは
            #  下で前後 15 枚に乱れが無いことを別に確かめる)
            if diff is None or diff[i] > 2.0 or not (valid[i - 1] and valid[i + 1]): continue
            step[i] = 0; nd += 1
        elif abs(sh[i] / m[i] - 2) < 0.2:
            step[i] = 2; nx += 1          # 落ちたフレーム(等間隔の音符で相関が 2 倍を拾う誤りは、下で近くに複製があるものだけ採って除く)
    # 組み合わせ: 時刻の順に見て、直前に残っている反対側の事象(5 枚以内)と組ませる(近い順に組ませると、0,24,48,0,48 のように続く所で
    # 隣の組の相手を横取りして組にならないものが残った: 活声 41〜45s、段位 祭 39.4s)
    zeros = list(np.flatnonzero(step == 0)); twos = list(np.flatnonzero(step == 2))
    ev = sorted([(int(i), 0) for i in zeros] + [(int(i), 2) for i in twos])
    stack = []; pairs_ = []
    for i, typ in ev:
        if stack and stack[-1][1] != typ and i - stack[-1][0] <= 5:
            j, _ = stack.pop(); pairs_.append((j, i))
        else:
            stack.append((i, typ))
    paired = {x for pr in pairs_ for x in pr}
    zeros_set = set(int(z) for z in zeros); twos_set = set(int(t) for t in twos)
    npair = len(pairs_)
    # 組は時刻の軸を並べ替えず、組の間の「予定とずれた絵」のフレームを検出から外すだけにする(game_clock.blank)。
    #  落ち → 複製(49,24,0): 落ちたフレームから複製の手前までが 1 フレーム先の絵(活声 85.1s: 追跡が切れてドンを 1 個落とした)。
    #  複製 → 追いつき(0,26): 複製の 1 枚が古い絵。
    # 並べ替えると、処理落ちが 2〜3 枚ごとに続く所(段位 祭 39.5〜42s)で 1 フレームぶん時刻がずれた
    # 外すのは「落ち → 複製」の組の間(先走った絵)だけ。「複製 → 追いつき」の古い絵 1 枚は追跡がそのままこなせていて、外すと
    # 処理落ちが続く所の点が減って段位 祭の時刻 sd が 1.3→2.5ms に悪化した(V2T_BLANK_ALL=1 で両方外す)
    # 「落ち → 複製」(49,24,0)の組は、その間のフレームを 1 フレーム先の絵として時刻を並べ替える(落ちで +2、複製で 0。組なので全体の長さは同じ)。
    # 外す(blank)と、複製と落ちが数枚おきに続く所(活声 146.8〜147.1s)で点が足りなくなり、ドンを 1 個落とした。
    # 「複製 → 追いつき」(0,26)は並べ替えも外しもしない(追跡がそのままこなせる。外すと段位 祭の sd 1.3→2.5ms)
    blank = set(); df_pairs = [(j, i) for j, i in pairs_ if step[j] == 2]
    game_clock.blank = blank
    # 時刻を直すのは孤立した複製だけ: 組にならず、前後 15 枚に他の乱れ(複製・流れが近くの速さの ±20% 外)が一つも無いもの
    # (VERTeX 43.43s・ドンカマ 115.53s: 録画の時計の遅れを取り戻すために挟まれた複製で、以降が全部 1 フレーム遅れる)。
    # 落ちたフレームだけのものは数えない(等間隔の音符で相関が 2 倍を拾う: HYPERNOVA)
    with np.errstate(invalid='ignore', divide='ignore'):
        rough = valid & (np.abs(sh / np.where(np.isnan(m), 1e9, m) - 1) > 0.2)
    iso = set()
    for z in zeros:
        if z in paired: continue
        nb = [j for j in range(max(0, z - 15), min(n, z + 16)) if j != z and (step[j] != 1 or rough[j] or (sh[j] != -99 and abs(sh[j]) <= 1))]
        if not nb: iso.add(z)
    game_clock.unpaired = [(int(idx[i]), t) for i, t in stack]
    # 落ちたフレーム(録画がゲームのフレームを 2〜3 枚落とし、以降の音符が全部そのぶん早く見える。プレリュード 裏 41.0s: 流れ 45px = 近くの速さの 3.06 倍)。
    # 流れが近くの速さと合わず、前後 15 枚に他の乱れが無いフレームで、音符の候補から 3〜4 倍(2〜3 枚落ち)と決まったものだけ。1 枚だけの跳び(2 倍)はゲーム側の揺れでも起きる
    # (段位 AtoZ 50.6s: 直すと以降の音符が 1 フレーム遅れた)ので、従来どおり複製と組になるときだけ扱う。
    # 等間隔の音符で相関が k 倍を拾う誤り(HYPERNOVA)があるので音符の候補で確かめる: 前の絵の候補を k 倍ずらすと同じ色の候補の 5 割以上に重なり、
    # 1 倍ずらしでは 2 割以下にしか重ならないときだけ採る
    drops = {}; game_clock.single = []
    _drop1 = set(x for x in __import__('os').environ.get('V2T_DROP1', '').split(',') if x)
    _nodrop = set(x for x in __import__('os').environ.get('V2T_NODROP', '').split(',') if x)
    if cands is not None:
        for i in range(1, n - 1):
            if sh[i] == -99 or np.isnan(m[i]) or m[i] < 4 or i in paired: continue
            if not rough[i] or abs(sh[i]) <= 1: continue                 # 流れが近くの速さどおり(普通)・複製は対象外
            if any((step[j] != 1 or rough[j]) for j in range(max(0, i - 15), min(n, i + 16)) if j != i): continue
            a_ = cands.get(int(idx[i])); b_ = cands.get(int(idx[i - 1]))
            # 候補が 2〜3 個の疎な所(ほうかご☆マジシャン 117.8s・ドドドドドンだフル! 92.6s の 2 枚落ち)は、全部が k 倍ずらしで重なり、1 倍では 1 つも
            # 重ならないときだけ採る(割合の閾値は候補 4 個以上のとき)
            if a_:          # 同じ音符の二重の候補は 1 つに数える。大音符は色の塊と顔の候補が 14px ずれて 2 つ出る(23時54分 15.85s)ので、同じ色で
                # 小音符の内円の半径の半分(20px)以内は同じ物
                u_ = []
                for col, x in sorted(a_, key=lambda c: c[1]):
                    if not any(c2 == col and abs(x - x2) <= 20 for c2, x2 in u_): u_.append((col, x))
                a_ = u_
            if not a_ or not b_ or len(a_) < 2: continue
            # 絵が何枚ぶん跳んだなら、そのフレームの流れは近くの速さの何倍にもなる。流れが普通の 1 歩より小さいのは遅れて描かれた絵で、落ちではない。
            # 等間隔に並んだ音符の列では、前の絵の候補を k 倍ずらすと偶然重なる(一括試験 オペレート・ミー 161.0s: 流れ 13.5・速さ 27.1 の半歩の絵を、
            # 100px 間隔の列で 4 倍ずらしが重なって「3 枚落ち」とし、以降の音符が 50ms 遅れ、1 本の直線で動画の時刻に戻す所で曲全体が 34ms 流れた)
            if os.environ.get('V2T_DROP_FLOW', '1') != '0' and 1 < abs(sh[i]) < 0.9 * m[i]: continue

            # 前の絵での位置(x + kk × 流れ)が候補を集めた範囲(x ≤ CAND_X1)の外になる候補は、重なる相手がいようがないので数えない
            # (23時54分 15.85s: 2 枚落ちで、右端から入ったばかりのカッを数えて 3 倍の重なりが 2/3 になり採らず、大カッが前半・後半の 2 本に分かれた)
            def score(kk):
                hit = den = 0
                for col, x in a_:
                    if x + kk * m[i] > CAND_X1 - 3: continue
                    den += 1
                    if any(c2 == col and abs(x2 - kk * m[i] - x) <= 3 for c2, x2 in b_): hit += 1
                return (hit / den if den else 0.0), den
            # 流れ(相関)は 80px までしか測れない(速さ 27 の 4 倍 = 107px)ので、何倍進んだかは候補で決める(1〜4 倍のうち一番重なるもの)
            scd = [score(kk) for kk in (1, 2, 3, 4)]; sc = [v for v, _ in scd]
            k = int(np.argmax(sc)) + 1; sk = sc[k - 1]; s1 = sc[0]; den_k = scd[k - 1][1]
            second = max(v for j, v in enumerate(sc) if j != k - 1)
            sure = (sk >= 0.5 and s1 <= 0.2) if den_k >= 4 else (den_k >= 2 and sk == 1.0 and s1 == 0.0)
            if k >= 3 and sure and second < 0.8 * sk:
                # run_song が直したあとの小節線で確かめ、ちょうど直した分だけ遅れたもの(ゲーム側の処理落ち)は V2T_NODROP で外してやり直す
                if str(int(idx[i])) not in _nodrop: drops[i] = k
            elif k == 2 and sure and second < 0.8 * sk:
                # 1 枚だけの跳び: 録画の追いつき(プレリュード 67.4s・124.2s)とゲーム側の揺れ(AtoZ 50.6s)があり、ここでは決めない。
                # run_song が小節線の段差で確かめ(bar_step)、録画のものだけ V2T_DROP1 で指定してやり直す
                if str(int(idx[i])) in _drop1: drops[i] = 2
                else: game_clock.single.append(int(idx[i]))
            if verbose: print('prep60: 落ちたフレーム? %d 流れ %.1f 近くの速さ %.1f、候補の重なり 1〜4 倍 %s → %s' % (idx[i], sh[i], m[i], ' '.join('%.2f' % v for v in sc), '%d 枚落ち' % (k - 1) if i in drops else '採らない'))
    game_clock.drops = [(int(idx[i]), k) for i, k in drops.items()]
    step = np.ones(n, int)
    for z in iso: step[z] = 0
    for i, k in drops.items(): step[i] = k
    # 組にならない複製(画素で同じ絵と確かめたもの)が本当に時刻の段差か(その後どこかで追いついていないか)は、前後 20 枚ずつのレーンの流れの合計で測る:
    # ゲームが進んだフレーム数 = 流れの合計 ÷ 近くの速さ。動画のフレーム数より 1 少なければ段差(中間の 0.5 で分ける)。
    # 段位 祭 44.37s(処理落ちが続く区間の中、以降の音符が全部 +14ms): 段差。メズマライザー 46.83s: 追いついていて段差ではない
    steps_ = []
    if __import__('os').environ.get('V2T_CLOCK_FLOW', '0') == '1':
        W = 20
        for z in zeros:
            if z in paired or z in iso or np.isnan(m[z]) or m[z] < 4: continue
            w = sh[max(0, z - W):min(n, z + W + 1)]; ok_ = w != -99
            if ok_.mean() < 0.9: continue                      # レーンが空の所が多いと合計が当てにならない
            deficit = float(ok_.sum() - w[ok_].sum() / m[z])
            steps_.append((int(idx[z]), round(deficit, 2)))
            if deficit >= 0.5: step[z] = 0; nd += 1 if False else 0
    game_clock.flow_steps = steps_
    for j, i in df_pairs: step[j] = 2; step[i] = 0
    if __import__('os').environ.get('V2T_CLOCK_DC', '0') == '1':
        # 試験: 「複製 → 追いつき」(0,26)の組も時計で直す(複製は直前と同じゲームフレーム = 古い絵なので外れ、追いつきは +2)
        for j, i in pairs_:
            if step[j] == 1 and step[i] == 1 and j in zeros_set and i in twos_set: step[j] = 0; step[i] = 2
    nd = len(iso) + sum(1 for _, d_ in game_clock.flow_steps if d_ >= 0.5) + len(drops)
    nv = int(valid.sum())
    if verbose and drops: print('prep60: 孤立した落ちたフレーム %d 箇所(時刻を直す): %s' % (len(drops), ' '.join('%d(+%d)' % (idx[i], k - 1) for i, k in sorted(drops.items()))))
    if verbose: print('prep60: ゲームの時計 — 孤立した複製 %d 枚(時刻を直す)・処理落ちの組 %d(うち落ち→複製 %d 組を並べ替え)・組にならない事象 %d、流れの見えた %d 枚中' % (nd, npair, len(df_pairs), len(stack), nv))
    if npair > 0.05 * max(1, nv): game_clock.blank = set()
    dc_ = __import__('os').environ.get('V2T_CLOCK_DC', '0') == '1' and any(step == 0)
    if (nd == 0 and not df_pairs and not dc_) or nd > 0.05 * max(1, nv) or len(df_pairs) > 0.05 * max(1, nv): return None       # 多すぎるのは 30fps 相当など(find_dups の側で扱う)
    g = np.cumsum(step) - step[0]
    return {int(k): int(g[i] + idx[0]) for i, k in enumerate(idx)}


def bar_offset(bars, te, fps):
    """時刻 te の後の小節線が、前の小節線 6 本の直線の延長からどれだけずれているか(ms の中央値, それに ±3ms でそろう本数)。確かめられないときは (None, 0)"""
    b = np.asarray(sorted(bars), float)
    pre = b[b < te - 0.05][-6:]; post = b[b > te + 0.05][:6]
    if len(pre) < 4 or len(post) < 4: return None, 0
    n = np.arange(len(pre)); a = np.polyfit(n, pre, 1)
    if (pre - np.polyval(a, n)).std() > 0.002: return None, 0
    k = np.round((post - pre[-1]) / a[0]) + len(pre) - 1
    d = (post - np.polyval(a, k)) * 1000
    med = float(np.median(d)); return med, int((np.abs(d - med) <= 3).sum())


def bar_step(bars, te, fps):
    """時刻 te の前後で、小節線がちょうど 1 フレーム早まる段差になっているか(録画が 1 枚落として追いついた印)。
    前の小節線 6 本がそろった間隔(直線の残差 ≤2ms)で、後の小節線 6 本のうち 4 本以上が前の直線の延長より 1 フレーム(±4ms)早く、
    互いに ±3ms にそろうとき True。小節線が足りない・間隔がそろわない(BPM の変わり目など)ときは確かめられないので False。
    ゲーム側の揺れ(AtoZ 50.6s)は延長との差が −13〜−29ms とそろわない"""
    b = np.asarray(sorted(bars), float)
    pre = b[b < te - 0.05][-6:]; post = b[b > te + 0.05][:6]
    if len(pre) < 4 or len(post) < 4: return False, None
    n = np.arange(len(pre)); a = np.polyfit(n, pre, 1)
    if (pre - np.polyval(a, n)).std() > 0.002: return False, None
    k = np.round((post - pre[-1]) / a[0]) + len(pre) - 1
    d = (post - np.polyval(a, k)) * 1000
    med = float(np.median(d)); good = int((np.abs(d - med) <= 3).sum())
    return (abs(med + 1000.0 / fps) <= 4 and good >= 4), med


def apply_map(mapfile, t):
    """60fps 動画の時刻 t(秒、配列可)→ 真の描画時刻(元動画の時刻軸)"""
    mp = json.load(open(mapfile)); tt = np.asarray(mp['true_t']); k = np.asarray(t, float) * 60.0
    return np.interp(k, np.arange(len(tt)), tt)

if __name__ == '__main__':
    if '--map-only' in sys.argv:          # 動画は書かず map だけ作り直す
        fps, fresh, diffs = analyze(sys.argv[1]); mp = build_map(fps, fresh); json.dump(mp, open(sys.argv[2] + '.map.json', 'w')); print('map only')
    else:
        prepare(sys.argv[1], sys.argv[2])

