"""音源(ogg)も参照TJAも無い動画から TJA を作る(ライブラリに無い曲)。音声は動画から切り出す。
使い方: run_video_only.py <video> <out_prefix> <TITLE> [COURSE(既定 Oni)]
  出力: <out_prefix>.tja・<out_prefix>_ons.json・<out_prefix>_song/<TITLE>.ogg(TJA の WAVE はこの音源)
中身は run_song.py と同じ(隠れた音符・ゴーゴー・風船・120fps の録画も同じに扱う):
  1. 動画の音声を丸ごと切り出し、音符の無い仮の譜面と一緒に仮の曲フォルダを作って、run_song で動画全体を検出する(Δ = 0)
  2. 音符の主な塊(8 秒以上途切れない塊で最大のもの)をプレイ中とし、その前後の無音で音声を切る(曲選び・結果の画面の誤検出と音を落とす)
  3. 切った音源の仮の曲フォルダで、検出を使い回して(REUSE_ONS)譜面だけ作り直す(OFFSET は切った音源に対して)
"""
import sys, os, json, subprocess, shutil
import numpy as np
import audio_align as A

HOP = 0.05
V = os.path.dirname(os.path.abspath(__file__))


def silence_edge(rms, i_from, i_to, thr, need):
    """i_from から i_to へ向かって(どちら向きでも)最初に need ホップ連続で thr 未満になる位置を返す"""
    step = 1 if i_to >= i_from else -1
    run = 0
    for i in range(i_from, i_to, step):
        if 0 <= i < len(rms) and rms[i] < thr:
            run += 1
            if run >= need: return i - step * (need - 1) if step > 0 else i + (need - 1)
        else: run = 0
    return None


def song_dir(path, title, course, ogg_src, ss=None, t=None):
    """音源(動画の音声の [ss, ss+t])と、音符の無い仮の譜面だけの曲フォルダを作る"""
    os.makedirs(path, exist_ok=True)
    ogg = os.path.join(path, title + '.ogg')
    cmd = ['ffmpeg', '-v', 'error', '-y']
    if ss is not None: cmd += ['-ss', '%.3f' % ss]
    if t is not None: cmd += ['-t', '%.3f' % t]
    subprocess.run(cmd + ['-i', ogg_src, '-vn', '-ac', '2', '-c:a', 'libvorbis', '-q:a', '6', ogg], check=True)
    open(os.path.join(path, title + '.tja'), 'w', encoding='utf-8-sig').write(
        'TITLE:%s\nWAVE:%s.ogg\nBPM:120\nOFFSET:0\n\nCOURSE:%s\nLEVEL:10\n\n#START\n#END\n' % (title, title, course))
    return path


def run_song(video, sdir, prefix, extra_env):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', **extra_env)
    # 出力はログに書き、画面(web)から動かしたとき(V2T_PROGRESS=1)は段階と進み具合の行を流す(以前は終わるまで何も出なかった)
    p = subprocess.Popen([sys.executable, '-u', os.path.join(V, 'run_song.py'), video, sdir, prefix], cwd=V, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    lines = []; show = os.environ.get('V2T_PROGRESS') == '1'
    for raw in iter(p.stdout.readline, b''):
        ln = raw.decode('utf-8', 'replace').rstrip('\r\n'); lines.append(ln)
        if show and (ln.startswith('[時間]') or ln.startswith('[進み]') or 'Traceback' in ln or 'Error' in ln): print(ln, flush=True)
    code = p.wait()
    out = '\n'.join(lines) + '\n'
    open(prefix + '.log', 'w', encoding='utf-8').write(out)
    if code != 0: raise SystemExit('run_song が失敗しました(%s.log)' % prefix)
    return out


def main():
    video, prefix, title = sys.argv[1], sys.argv[2], sys.argv[3]
    course = sys.argv[4] if len(sys.argv) > 4 else 'Oni'
    dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', video]).decode().strip())
    # 1. 動画全体(音源 = 動画の音声そのまま、Δ = 0)
    full = prefix + '_full'
    run_song(video, song_dir(full + '_song', title, course, video), full, {'VID2OGG': '0'})
    d = json.load(open(full + '_ons.json', encoding='utf-8'))
    onsets = sorted(d['onsets'], key=lambda o: o['t'])
    # 2. 音符の主な塊(プレイ前後の画面の誤検出を落とす)と、前後の無音で音声を切る所
    ts = np.array([o['t'] for o in onsets])
    segs = np.split(np.arange(len(ts)), np.where(np.diff(ts) > 8.0)[0] + 1)
    seg = max(segs, key=len); t_first, t_last = float(ts[seg[0]]), float(ts[seg[-1]])
    bars = sorted(float(b) for b in d['bars']); b = np.array(bars)
    gd = np.diff(b); gd = gd[(gd > 0.3) & (gd < 4.0)]; gmed = float(np.median(gd)) if len(gd) else 2.0
    # 曲の頭の小節線: 最初の音符の直前の小節線から前へ、小節の間隔(±15%)で並ぶものだけたどる。曲が始まる前の画面の演出は小節線に見えるが
    # 33ms おきに並ぶ(Shine!! 6.9〜9.x 秒)ので、そこで途切れる
    k0 = int(np.searchsorted(b, t_first + 0.05)) - 1
    if k0 >= 0:
        kk = k0
        while kk > 0 and 0.85 * gmed <= b[kk] - b[kk - 1] <= 1.15 * gmed: kk -= 1
        t_bar0 = float(b[kk])
    else:
        t_bar0 = t_first
    a = A.load(video); hop = int(A.SR * HOP); n = len(a) // hop
    rms = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1))
    lvl = float(np.median(rms[int(t_first / HOP):int(t_last / HOP)])); thr = 0.05 * lvl
    i = silence_edge(rms, int(t_bar0 / HOP), max(0, int((t_bar0 - 8) / HOP)), thr, 6)
    t0 = (i + 1) * HOP if i is not None else max(0.0, t_bar0 - 1.0)
    # 切る頭は、曲の頭(最初の小節線と最初の音符の早い方)の 0.5〜2 秒前に収める。2 秒より前は曲が始まる前の画面の音、0.5 秒より後ろは
    # 最初の音符を切りうる(虹: 最初の音符の直前が静かで、無音の終わりを 10.20 秒 = 最初の音符 10.17 秒より後ろにして 1 音落とした)
    s0 = min(t_bar0, t_first)
    t0 = float(np.clip(t0, max(0.0, s0 - 2.0), max(0.0, s0 - 0.5)))
    j = silence_edge(rms, int(t_last / HOP) + 1, min(n, int((t_last + 20) / HOP)), thr, 10)
    t1 = j * HOP if j is not None else min(dur, t_last + 8.0)
    keep = lambda t: t0 <= t <= t1
    d['onsets'] = [o for o in onsets if keep(o['t']) and t_first - 1e-6 <= o['t'] <= t_last + 1e-6]
    bv = d.get('bar_vels') or [None] * len(d['bars'])
    kb = [k for k, x in enumerate(d['bars']) if keep(x) and t_bar0 - 1e-6 <= x <= t_last + 1.5 * gmed]
    d['bars'] = [d['bars'][k] for k in kb]; d['bar_vels'] = [bv[k] for k in kb]
    print('[%s] 音符 %d(プレイの外の %d を落とした)・小節線 %d。音声を %.2f〜%.2f 秒で切る' % (title, len(d['onsets']), len(onsets) - len(d['onsets']), len(d['bars']), t0, t1))
    # 3. 切った音源で譜面だけ作り直す(検出は使い回す)。Δ = 切り出しの頭
    json.dump(d, open(prefix + '_ons.json', 'w', encoding='utf-8'), ensure_ascii=False)
    sdir = song_dir(prefix + '_song', title, course, video, t0, t1 - t0)
    run_song(video, sdir, prefix, {'VID2OGG': '%.4f' % t0, 'REUSE_ONS': '1'})
    shutil.rmtree(full + '_song', ignore_errors=True)
    print('譜面: %s.tja(音源 %s)' % (prefix, os.path.join(sdir, title + '.ogg')))
    if os.environ.get('V2T_EXT_REF'): compare_ref(prefix, os.environ['V2T_EXT_REF'], title)


def coarse_offset(mt, tt):
    """検出した音符の時刻 mt と譜面の音符の時刻 tt の差を全部の組で数え、いちばん多いずれ(10ms 刻み)。全部の音符がそろうずれが一番高い山になる
    (音源が無く、音声で時刻を合わせられないときの目安。細かい合わせは evaluate が ±0.6 秒で行う)"""
    d = (np.asarray(mt)[:, None] - np.asarray(tt)[None, :]).ravel()
    h, e = np.histogram(d, bins=np.arange(d.min(), d.max() + 0.02, 0.01))
    k = int(np.argmax(np.convolve(h, np.ones(3), mode='same')))
    return float(e[k] + 0.005)


def compare_ref(prefix, ref, title):
    """音源の無い譜面(ライブラリに無い TJA だけ)と答え合わせする。時刻の原点は音符の並びで合わせる(coarse_offset)。
    出力は run_song と同じ形(== 曲 vs コース (vid_offset …) ==、best course: …、misses / extras)"""
    import re
    from eval_timing import parse_tja_times
    from eval_ons import evaluate, load_ons
    mt, mc, rng = load_ons(prefix + '_ons.json')
    txt = open(ref, encoding='utf-8-sig', errors='replace').read()
    best = None
    for crs in re.findall(r'^COURSE:\s*(\w+)', txt, re.M):
        try: _, tn = parse_tja_times(ref, crs)
        except Exception as ex: print('[%s] %s を読めない: %s' % (title, crs, ex)); continue
        if len(tn) < 10 or len(mt) < 10: continue
        tt = np.array([n[0] for n in tn]); tc = np.array([n[1] for n in tn])
        vo = coarse_offset(mt, tt)
        print('== %s vs %s (vid_offset %.3f) ==' % (title, crs, vo))
        r = evaluate(mt, mc, tt, tc, vo, rng)
        if r and (best is None or r[0] + r[1] > best[1]): best = (crs, r[0] + r[1], r)
    if best:
        crs, _, r = best
        print('[%s] 参照 %s(音源が無いので、音符の並びで時刻を合わせた)' % (title, os.path.basename(ref)))
        print('[%s] best course: %s  recall %.1f%% precision %.1f%%  misses %d extras %d' % (title, crs, r[0], r[1], len(r[3]), len(r[4])))
        if r[3]: print('   misses (video s):', ' '.join('%.2f' % x for x in r[3][:40]))
        if r[4]: print('   extras (video s):', ' '.join('%.2f' % x for x in r[4][:40]))
    else:
        print('[%s] 参照 %s と照合できなかった(音符 10 個以上のコースが無い)' % (title, os.path.basename(ref)))


if __name__ == '__main__':
    main()
