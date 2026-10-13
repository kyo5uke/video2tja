"""video2tja の処理を見ながら動かすローカルの Web ページ。
使い方: python web/server.py [--port 8765]  → ブラウザで http://127.0.0.1:8765 を開く
動画を放り込み(または testvideos から選び)、曲フォルダを選んで開始すると、run_song.py の各段階の進み具合と
見つかったものを順に表示し、終わったら元の動画の上に検出した音符を重ねて再生する。
出力は testvideos/work/web/<番号>/ に書く(Songs/ には書かない)。
"""
import os, sys, json, re, time, glob, threading, subprocess, urllib.parse, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS = os.environ.get('V2T_SONGS') or os.path.join(V, 'Songs')
VIDEOS = os.path.join(V, 'testvideos')
WORK = os.path.join(VIDEOS, 'work')
WEB_OUT = os.path.join(WORK, 'web')
UPLOADS = os.path.join(WORK, 'uploads')
EXT_SONGS = os.path.join(WORK, 'ext_songs')      # ライブラリ(Songs)に無い譜面(TJA と、あれば音源)を置く所。曲フォルダは 'ext:<名前>' で表す
HERE = os.path.dirname(os.path.abspath(__file__))

_songs = None
JOBS = {}
_lock = threading.Lock()
_sf_env = None                       # 音源の索引(songfind。最初の問い合わせで読む)
_sf_lock = threading.Lock()


def find_song(video):
    """動画の音声から曲フォルダの候補を探す(songfind.py)。[{score, dir, tja, title, delta}] を強い順に"""
    global _sf_env
    sys.path.insert(0, V); import songfind
    with _sf_lock:
        if _sf_env is None: _sf_env = songfind.index(verbose=False)
        env = _sf_env
    ch, groups = songfind.choose(songfind.find(video, 8, env=env))
    out = []
    for g in groups[:5]:
        w, p, d, s = g[0][:4]
        tj, ti = songfind._title(p)
        out.append({'score': round(w, 3), 'dir': os.path.relpath(os.path.dirname(p), SONGS).replace(os.sep, '/'), 'tja': tj, 'title': ti, 'delta': round(d, 3),
                    'auto': ch is g[0], 'same': [os.path.relpath(os.path.dirname(c[1]), SONGS).replace(os.sep, '/') for c in g[1:]]})
    return out


def song_list():
    """TJA と音源(ogg)のある曲フォルダ(Songs からの相対パス)"""
    global _songs
    if _songs is None:
        out = []
        for root, dirs, files in os.walk(SONGS):
            depth = os.path.relpath(root, SONGS).count(os.sep)
            if depth >= 3: dirs[:] = []
            tj = [f for f in files if f.lower().endswith('.tja')]
            if tj and any(f.lower().endswith('.ogg') for f in files):
                rel = os.path.relpath(root, SONGS).replace(os.sep, '/')
                out.append({'dir': rel, 'tjas': sorted(tj), 'title': _tja_title(rel, None) or os.path.basename(rel)})
        _songs = sorted(out, key=lambda s: s['dir'])
    return _songs


_vmeta = None


def _tja_title(song_dir, ref):
    """参照 TJA の曲名(TITLEJA があればそれ、無ければ TITLE)"""
    try:
        p = _ref_tja(song_dir, ref)
        txt = open(p, encoding='utf-8-sig', errors='replace').read(4000)
        m = re.search(r'^TITLEJA:\s*(.+)$', txt, re.M) or re.search(r'^TITLE:\s*(.+)$', txt, re.M)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def video_meta():
    """動画 ID → 曲名・曲フォルダ・参照 TJA(全曲試験 regress/suite.json と一括試験 testvideos/wide/all.json から)。ファイル名は動画の ID なので曲名で出す"""
    global _vmeta
    if _vmeta is None:
        m = {}
        for jp in (os.path.join(V, 'regress', 'suite.json'), os.path.join(VIDEOS, 'wide', 'all.json')):
            try:
                for e in json.load(open(jp, encoding='utf-8')):
                    vid = e.get('video')
                    if vid and vid not in m:
                        title = e.get('name') if 'wide' in jp else (_tja_title(e.get('dir'), e.get('ref')) or os.path.basename(e.get('dir', '')).split(' ', 1)[-1] or e.get('name'))
                        if 'wide' not in jp and e.get('course') in ('Edit', 'Ura'): title += '(裏)'
                        m[vid] = {'title': title or vid, 'dir': _real_dir(e.get('dir')), 'ref': e.get('ref'), 'course': e.get('course')}
            except Exception:
                pass
        _vmeta = m
    return _vmeta


def _guess_song(fname):
    """ファイル名に含まれる曲名(TJA の TITLE)から曲フォルダを探す(アップロードした「【太鼓の達人】曲名(裏譜面) 全良 [ID].mp4」など)。
    いちばん長く一致した曲名を採る。裏譜面・裏 と書いてあればコースは Edit"""
    f = fname.replace('　', ' ')
    best = None
    for sg in song_list():
        t = (sg.get('title') or '').strip()
        if len(t) >= 2 and t in f and (best is None or len(t) > len(best['title'])): best = sg
    if not best: return None
    ura = '裏' in f.replace(best['title'], '')
    return {'title': best['title'] + ('(裏)' if ura else ''), 'dir': best['dir'], 'ref': None, 'course': 'Edit' if ura else None}


def video_list():
    vs = []; meta = video_meta()
    paths = glob.glob(os.path.join(VIDEOS, '*.mp4')) + glob.glob(os.path.join(VIDEOS, 'wide', 'yt', '*.mp4')) + glob.glob(os.path.join(UPLOADS, '*'))
    for p in paths:
        if os.path.isfile(p) and not p.endswith('_z0.mp4'):
            vid = os.path.splitext(os.path.basename(p))[0]
            mb = re.search(r'\[([\w-]{11})\]', vid)             # 「曲名 [ID].webm」の形の名前
            mm = meta.get(vid) or (meta.get(mb.group(1)) if mb else None) or _guess_song(vid) or {}
            vs.append({'path': p, 'name': os.path.basename(p), 'id': vid, 'title': mm.get('title') or os.path.basename(p), 'dir': mm.get('dir'),
                       'ref': mm.get('ref'), 'course': mm.get('course'), 'size': os.path.getsize(p), 'mtime': os.path.getmtime(p),
                       'where': 'アップロード' if p.startswith(UPLOADS) else ('一括試験' if os.sep + 'wide' + os.sep in p else '試験')})
    vs.sort(key=lambda v: (v['where'] != 'アップロード', v['title'].lower()))
    return vs


WIDE = os.path.join(VIDEOS, 'wide')


def _real_dir(spec):
    """'08 Namco Original/Donkama 2000' → 番号付きの実フォルダ(Songs からの相対。regress/suite.py の song_dir と同じ探し方)"""
    if not spec or os.path.isdir(os.path.join(SONGS, spec)): return spec
    cat, name = spec.rsplit('/', 1) if '/' in spec else ('', spec)
    for h in glob.glob(os.path.join(SONGS, cat, '[0-9]* ' + name)):
        if os.path.basename(h).split(' ', 1)[1] == name: return os.path.relpath(h, SONGS).replace(os.sep, '/')
    return spec


def _song_path(song_dir):
    """曲フォルダの実際の場所: Songs からの相対、または 'ext:<名前>'(ライブラリに無い譜面を置いた作業フォルダ)"""
    if song_dir and song_dir.startswith('ext:'): return os.path.join(EXT_SONGS, re.sub(r'[\\/:*?"<>|]+', '_', song_dir[4:]))
    return os.path.join(SONGS, _real_dir(song_dir))


def ext_song(token):
    """ライブラリに無い譜面の作業フォルダの中身: TJA(曲名・コース・音源があるか)"""
    d = _song_path('ext:' + token)
    if not os.path.isdir(d): return None
    tjas = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith('.tja'): continue
        txt = open(os.path.join(d, f), encoding='utf-8-sig', errors='replace').read()
        m = re.search(r'^TITLEJA:\s*(.+)$', txt, re.M) or re.search(r'^TITLE:\s*(.+)$', txt, re.M)
        w = re.search(r'^WAVE:\s*(.+?)\s*$', txt, re.M)
        tjas.append({'tja': f, 'title': m.group(1).strip() if m else os.path.splitext(f)[0], 'courses': re.findall(r'^COURSE:\s*(\w+)', txt, re.M),
                     'wave': w.group(1) if w else None, 'audio': bool(w and os.path.isfile(os.path.join(d, w.group(1))))})
    return {'dir': 'ext:' + token, 'title': tjas[0]['title'] if tjas else token, 'tjas': [t['tja'] for t in tjas], 'detail': tjas,
            'audio': any(t['audio'] for t in tjas)}


def _ref_tja(song_dir, ref=None):
    d = _song_path(song_dir)
    song_dir = _real_dir(song_dir) if not song_dir.startswith('ext:') else song_dir
    if ref and os.path.exists(os.path.join(d, ref)): return os.path.join(d, ref)
    tj = sorted(f for f in os.listdir(d) if f.lower().endswith('.tja')) if os.path.isdir(d) else []
    base = os.path.basename(song_dir.rstrip('/'))
    for f in tj:
        if os.path.splitext(f)[0].lower() in base.lower(): return os.path.join(d, f)
    return os.path.join(d, tj[0]) if tj else None


def result_list():
    """開ける結果: 一括試験(testvideos/wide)と、このページで変換したもの(testvideos/work/web)"""
    out = []
    for rp in glob.glob(os.path.join(WIDE, 'res', '*.json')):
        try: r = json.load(open(rp, encoding='utf-8'))
        except Exception: continue
        v = r.get('video')
        if 'recall' not in r or not os.path.exists(os.path.join(WIDE, 'yt', v + '.mp4')) or not os.path.exists(os.path.join(WIDE, 'out', v + '_ons.json')): continue
        out.append({'kind': 'wide', 'id': v, 'name': r.get('name') or v, 'course': r.get('course'), 'perfect': bool(r.get('perfect')),
                    'diff': (r.get('best_misses') or 0) + (r.get('best_extras') or 0), 'mtime': os.path.getmtime(rp)})
    # 全曲試験(regress/suite.json と regress/suite_out)の結果
    SUITE_OUT = os.path.join(V, 'regress', 'suite_out'); meta = video_meta()
    try: suite = json.load(open(os.path.join(V, 'regress', 'suite.json'), encoding='utf-8'))
    except Exception: suite = []
    for e in suite:
        n = e['name']; op = os.path.join(SUITE_OUT, n + '_ons.json'); rp = os.path.join(SUITE_OUT, n + '.json')
        vp = e['video'] if os.path.isabs(e['video']) else os.path.join(VIDEOS, e['video'] + '.mp4')
        if not (os.path.exists(op) and os.path.exists(vp)): continue
        try: r = json.load(open(rp, encoding='utf-8'))
        except Exception: r = {}
        title = (meta.get(e['video']) or {}).get('title') or _tja_title(e.get('dir'), e.get('ref')) or n
        out.append({'kind': 'suite', 'id': n, 'name': title, 'course': e.get('course'), 'perfect': None,
                    'diff': (r.get('misses') or 0) + (r.get('extras') or 0) if r else None, 'mtime': os.path.getmtime(op)})
    for d in glob.glob(os.path.join(WEB_OUT, '*')):
        mp = os.path.join(d, 'meta.json')
        if not (os.path.exists(mp) and os.path.exists(os.path.join(d, 'out_ons.json'))): continue
        m = json.load(open(mp, encoding='utf-8'))
        vt = next((v['title'] for v in video_list() if v['path'] == m['video']), None)
        out.append({'kind': 'web', 'id': os.path.basename(d), 'name': vt or _tja_title(m['song_dir'], m.get('ref_tja')) or os.path.basename(m['song_dir']),
                    'course': None, 'perfect': None, 'diff': None, 'mtime': os.path.getmtime(mp)})
    out.sort(key=lambda x: -x['mtime'])
    return out


def open_result(kind, rid):
    """結果を開く: 動画・検出(ons)・ログから拾った要点・参照の音符(動画の秒)・書き出した TJA"""
    if kind == 'wide':
        r = json.load(open(os.path.join(WIDE, 'res', rid + '.json'), encoding='utf-8'))
        video = os.path.join(WIDE, 'yt', rid + '.mp4'); prefix = os.path.join(WIDE, 'out', rid)
        ref = _ref_tja(r['dir'], r.get('ref')); course = r.get('course'); title = r.get('name')
    elif kind == 'suite':
        e = next(x for x in json.load(open(os.path.join(V, 'regress', 'suite.json'), encoding='utf-8')) if x['name'] == rid)
        video = e['video'] if os.path.isabs(e['video']) else os.path.join(VIDEOS, e['video'] + '.mp4')
        prefix = os.path.join(V, 'regress', 'suite_out', rid)
        ref = _ref_tja(e['dir'], e.get('ref')); course = e.get('course')
        title = (video_meta().get(e['video']) or {}).get('title') or _tja_title(e['dir'], e.get('ref')) or rid
    else:
        d = os.path.join(WEB_OUT, rid); m = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
        video = m['video']; prefix = os.path.join(d, 'out'); course = None
        ref = _ref_tja(m['song_dir'], m.get('ref_tja')) if m.get('song_dir') else m.get('ext_ref')   # 曲フォルダの無い変換(nosong)は、ライブラリに無い譜面を渡していればそれ
        title = (next((v['title'] for v in video_list() if v['path'] == video), None) or (m.get('title') if m.get('nosong') else None)
                 or _tja_title(m['song_dir'], m.get('ref_tja')) or os.path.basename(m['song_dir']))
    log = open(prefix + '.log', encoding='utf-8', errors='replace').read() if os.path.exists(prefix + '.log') else ''
    res = {'info': {'video': video, 'title': title, 'kind': kind, 'id': rid}, 'log': log}
    res['ons'] = json.load(open(prefix + '_ons.json', encoding='utf-8'))
    try: res['tja'] = open(prefix + '.tja', encoding='utf-8-sig', errors='replace').read()
    except Exception: res['tja'] = None
    res['tja_path'] = prefix + '.tja'
    _add_clean(res, prefix)
    try:
        sys.path.insert(0, V); import calib
        res['calib_ref'] = calib.REF
    except Exception: res['calib_ref'] = None
    # 参照: 一致の一番良いコース(ログの best course)を、そのコースの vid_offset で動画の秒に
    mb = re.search(r'best course: (\w+)', log); course = (mb.group(1) if mb else None) or course
    res['ref'] = []; res['ref_course'] = course
    if ref and course:
        mo = re.search(r'vs %s \(vid_offset ([-\d.]+)\)' % re.escape(course), log)
        if mo:
            try:
                from eval_timing import parse_tja_times
                _, tn = parse_tja_times(ref, course, marks='12345678')        # 連打・風船(5〜8)も
                off = float(mo.group(1))
                # 照合と同じく、曲全体で一番よく合うずれ(eval_ons.bestshift: vid_offset の ±0.6 秒)で揃える。vid_offset のままだと
                # 数十 ms ずれて参照の音符が後ろに見えた(ドンカマ2000)
                try:
                    import numpy as np
                    from eval_ons import bestshift
                    mt = np.array(sorted(o['t'] for o in res['ons']['onsets'] if o['color'] in ('don', 'ka')))
                    tt = np.array(sorted(t for t, c, *_ in tn if c in '1234'))
                    if len(mt) >= 10 and len(tt) >= 10:
                        g0 = float(bestshift(mt, tt, off, 0.6)); res['ref_shift_ms'] = round((g0 - off) * 1000, 1); off = g0
                except Exception as ex:
                    res['ref_shift_error'] = str(ex)
                res['ref'] = [[round(t + off, 4), c] for t, c, *_ in tn]
                from eval_timing import parse_tja_bars
                res['ref_bars'] = [round(t + off, 4) for t in parse_tja_bars(ref, course)]      # 譜面の画面の小節の区切り
            except Exception as ex:
                res['ref_error'] = str(ex)
    return res


def _live_ref(j, note_t):
    """変換中の参照(本家の譜面)。音声合わせの Δ(ログの「Δ(VID2OGG) = …」)が出たら、参照 TJA の各コースを動画の秒(譜面の秒 + Δ − OFFSET、
    run_song の vid_offset と同じ)に直しておき、今までに見つかった音符(note_t)と一番よく合うコースを返す。戻り値 (コース, [[秒, 記号]]) か None。
    ずれは結果を開くときと同じく、見つかった音符と曲全体で一番よく合う値に詰める(eval_ons.bestshift: ±0.6 秒。Δ − OFFSET のままだと
    音声の同期の補正ぶん数十 ms 早く見えた)。詰めは音符の数が変わったときだけ計算する"""
    if j.nosong: return None
    if getattr(j, '_lref', None) is None:
        delta = None
        for ln in list(j.lines):
            mm = re.search(r'Δ\(VID2OGG\) = ([-\d.]+)', ln)
            if mm: delta = float(mm.group(1))
        if delta is None: return None
        j._lref = {}
        try:
            ref = _ref_tja(j.song_dir, j.ref_tja)
            txt = open(ref, encoding='utf-8-sig', errors='replace').read()
            mo = re.search(r'^OFFSET:\s*([-\d.]+)', txt, re.M); off = float(mo.group(1)) if mo else 0.0
            sys.path.insert(0, V)
            from eval_timing import parse_tja_times
            for crs in dict.fromkeys(re.findall(r'^COURSE:\s*(\w+)', txt, re.M)):
                try: _, tn = parse_tja_times(ref, crs, marks='12345678')
                except Exception: continue
                if tn:
                    from eval_timing import parse_tja_bars
                    j._lref[crs] = (delta - off, tn, parse_tja_bars(ref, crs))
        except Exception:
            pass
    if not j._lref: return None
    import numpy as np
    mt = np.array(sorted(note_t), float); best = None
    for crs, (base, tn0, _b) in j._lref.items():
        tn = [[t + base, c] for t, c in tn0]
        tt = np.array(sorted(t for t, c in tn if c in '1234'), float)
        if len(mt) and len(tt):
            # 両向きの一致の調和平均(F 値): 見つかった音符のうち ±60ms に参照の音符がある割合と、見つかった範囲の参照の音符のうち ±60ms に
            # 見つかった音符がある割合(仮の音符は数十 ms ずれることがある)。片向きだけ数えると、音符の少ないコース(Oni)で 1 つの参照に
            # 何個も対応して点が 1 を超え、Edit の動画に Oni を選んだ
            def near(a, b):
                k = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
                return np.minimum(np.abs(b[k] - a), np.abs(b[k - 1] - a)) <= 0.06
            tr = tt[(tt >= mt.min() - 0.5) & (tt <= mt.max() + 0.5)]
            if len(tr) and len(mt) >= 2:
                p_ = float(near(mt, tt).mean()) if len(tt) >= 2 else 0.0; r_ = float(near(tr, mt).mean())
                score = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
            else:
                score = 0.0
        else:
            score = len(tt) * 1e-9                 # まだ音符が無い: 音符の多いコース
        if best is None or score > best[0]: best = (score, crs, tn)
    crs = best[1]; base, tn0, bars0 = j._lref[crs]
    key = (crs, len(mt))
    if getattr(j, '_lshift', (None, None))[0] != key:
        g0 = base
        if len(mt) >= 10:
            try:
                from eval_ons import bestshift
                tt0 = np.array(sorted(t for t, c in tn0 if c in '1234'), float)
                if len(tt0) >= 10: g0 = float(bestshift(mt, tt0, base, 0.6))
            except Exception:
                pass
        j._lshift = (key, g0)
    g0 = j._lshift[1]
    return crs, [[round(t + g0, 4), c] for t, c in tn0], [round(t + g0, 4) for t in bars0]


class Job:
    def __init__(self, video, song_dir, ref_tja, cpu='8', title=None, ref_auto=False, ext_ref=None, prep_enc=None):
        self.id = time.strftime('%Y%m%d_%H%M%S')
        self.video, self.song_dir, self.ref_tja = video, song_dir, ref_tja
        self.dir = os.path.join(WEB_OUT, self.id); os.makedirs(self.dir, exist_ok=True)
        self.prefix = os.path.join(self.dir, 'out')
        # 曲フォルダの無い変換(ライブラリに無い曲): 動画の音声を音源にする(run_video_only.py)
        self.nosong = not song_dir; self.title = title or os.path.splitext(os.path.basename(video))[0]
        self.prep_enc = 'nvenc' if prep_enc == 'nvenc' else 'cpu'
        json.dump({'video': video, 'song_dir': song_dir, 'ref_tja': ref_tja, 'cpu': cpu, 'nosong': self.nosong, 'title': self.title, 'ext_ref': ext_ref,
                   'prep_enc': self.prep_enc},
                  open(os.path.join(self.dir, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        self.lines = []; self.done = False; self.code = None; self.t0 = time.time()
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', V2T_WORK=WORK, V2T_PROGRESS='1')   # 段階の中の進み具合も出す
        env['V2T_LIVE'] = self.prefix + '_live.json'      # 途中経過(仮の音符・段階ごとの音符)。変換中の画面が読む
        if ref_tja: env['REF_TJA'] = ref_tja
        if ref_auto: env['V2T_REF_AUTO'] = '1'            # 同じ音源を使う TJA を全部照合する(run_song)
        if ext_ref: env['V2T_EXT_REF'] = ext_ref          # 音源の無い譜面: 動画の音声で作り、この TJA と答え合わせする(run_video_only)
        env['V2T_CPU'] = str(cpu or '8'); self.cpu = env['V2T_CPU']
        # 120fps 録画を 60fps に直すときの書き出し(prep60.ENC_ARGS)。GPU(NVENC)は準備が速いが画素が少し変わり、重なった音符の判定がずれる
        # (ファイナル: 取りこぼし 72 → 79・余分 22 → 40)ので既定は CPU。中間の動画は別の名前(_60nv.mp4)なので CPU 版のキャッシュと混ざらない
        if self.prep_enc == 'nvenc': env['V2T_PREP_ENC'] = 'nvenc'
        else: env.pop('V2T_PREP_ENC', None)
        cmd = ([sys.executable, '-u', 'run_video_only.py', video, self.prefix, self.title] if self.nosong
               else [sys.executable, '-u', 'run_song.py', video, _song_path(song_dir), self.prefix])
        self.proc = subprocess.Popen(cmd, cwd=V, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in iter(self.proc.stdout.readline, b''):
            self.lines.append(raw.decode('utf-8', 'replace').rstrip('\r\n'))
        self.code = self.proc.wait(); self.done = True
        self.lines.append('[web] 終了 (コード %s、%.0f 秒)' % (self.code, time.time() - self.t0))
        try: open(self.prefix + '.log', 'w', encoding='utf-8').write('\n'.join(self.lines) + '\n')     # 後で「結果を開く」から見られるように
        except OSError: pass

    def cancel(self):
        if not self.done:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(self.proc.pid)], capture_output=True)

    def info(self):
        return {'id': self.id, 'video': self.video, 'song_dir': self.song_dir, 'done': self.done, 'code': self.code, 'cpu': self.cpu, 'prep_enc': self.prep_enc,
                'cpu_total': os.cpu_count(), 'elapsed': round(time.time() - self.t0, 1)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def _file(self, path, ctype):
        size = os.path.getsize(path); rng = self.headers.get('Range')
        a, b = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)', rng or '')
        if m:
            if m.group(1): a = int(m.group(1))
            if m.group(2): b = min(int(m.group(2)), size - 1)
            elif not m.group(1): a = max(0, size - int(m.group(2) or 0))
            self.send_response(206); self.send_header('Content-Range', 'bytes %d-%d/%d' % (a, b, size))
        else:
            self.send_response(200)
        self.send_header('Content-Type', ctype); self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(b - a + 1)); self.end_headers()
        with open(path, 'rb') as fh:
            fh.seek(a); left = b - a + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        p = u.path
        if p == '/':
            return self._file(os.path.join(HERE, 'index.html'), 'text/html; charset=utf-8')
        if p == '/api/songs':
            return self._json(song_list())
        if p == '/api/videos':
            return self._json(video_list())
        if p == '/api/results':
            return self._json(result_list())
        if p == '/api/open':
            try: return self._json(open_result(q.get('kind', [''])[0], q.get('id', [''])[0]))
            except Exception as ex: return self._json({'error': '開けませんでした: %s' % ex}, 400)
        if p == '/api/extsong':
            es = ext_song(q.get('token', [''])[0])
            return self._json(es) if es else self._json({'error': '見つかりません'}, 404)
        if p == '/api/findsong':
            video = os.path.abspath(q.get('video', [''])[0])
            if not os.path.isfile(video): return self._json({'error': '動画が見つかりません'}, 400)
            try: return self._json(find_song(video))
            except Exception as ex: return self._json({'error': '曲を探せませんでした: %s' % ex}, 500)
        if p == '/api/jobs':
            return self._json([j.info() for j in sorted(JOBS.values(), key=lambda j: j.id, reverse=True)])
        if p == '/media':
            path = os.path.abspath(q.get('path', [''])[0])
            if not (path.startswith(os.path.abspath(VIDEOS)) and os.path.isfile(path)): return self._json({'error': 'not found'}, 404)
            return self._file(path, 'video/mp4')
        m = re.match(r'/api/job/([\w]+)/live$', p)
        if m and m.group(1) in JOBS:
            # 変換中の画面: 途中経過のファイル(run_song・detect2 が段階ごとに書く)。まだ無ければ段階だけ空で返す
            j = JOBS[m.group(1)]; d = {}
            try: d = json.load(open(j.prefix + '_live.json', encoding='utf-8'))
            except Exception: pass
            try:
                sys.path.insert(0, V); import calib
                d['calib_ref'] = calib.REF
            except Exception: pass
            if getattr(j, 'duration', None) is None:       # 動画の長さ(画面の全体図の幅。動画を読み込む前でも曲全体を出す)
                try:
                    j.duration = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', j.video],
                                                      capture_output=True, text=True, timeout=20).stdout.strip())
                except Exception: j.duration = 0.0
            d['video'] = j.video; d['done'] = j.done; d['duration'] = j.duration
            try:
                nt = [o['t'] for o in d['ons']['onsets'] if o.get('color') in ('don', 'ka')] if 'ons' in d else [p_[0] for p_ in d.get('prov') or []]
                lr = _live_ref(j, nt)
                if lr: d['ref_course'], d['ref'], d['ref_bars'] = lr
            except Exception as ex:
                d['ref_error'] = str(ex)
            return self._json(d)
        m = re.match(r'/api/job/([\w]+)/(events|result)$', p)
        if m:
            j = JOBS.get(m.group(1))
            if j is None: return self._json({'error': 'no job'}, 404)
            if m.group(2) == 'events':
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache'); self.end_headers()
                k = int(q.get('from', ['0'])[0])
                try:
                    while True:
                        n = len(j.lines)
                        if k < n:
                            for ln in j.lines[k:n]:
                                self.wfile.write(('data: ' + json.dumps(ln, ensure_ascii=False) + '\n\n').encode('utf-8'))
                            k = n; self.wfile.flush()
                        elif j.done:
                            self.wfile.write(b'event: done\ndata: {}\n\n'); self.wfile.flush(); return
                        else:
                            time.sleep(0.15)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    return
            res = {'info': j.info(), 'log': '\n'.join(j.lines)}
            try: res['ons'] = json.load(open(j.prefix + '_ons.json', encoding='utf-8'))
            except Exception: res['ons'] = None
            try: res['tja'] = open(j.prefix + '.tja', encoding='utf-8-sig', errors='replace').read()
            except Exception: res['tja'] = None
            res['tja_path'] = j.prefix + '.tja'
            _add_clean(res, j.prefix)
            try:
                sys.path.insert(0, V); import calib
                res['calib_ref'] = calib.REF
            except Exception: res['calib_ref'] = None
            return self._json(res)
        return self._json({'error': 'not found'}, 404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); p = u.path
        if p == '/api/upload':
            name = urllib.parse.unquote(self.headers.get('X-Filename') or 'upload.mp4')
            name = re.sub(r'[\\/:*?"<>|]+', '_', os.path.basename(name)) or 'upload.mp4'
            os.makedirs(UPLOADS, exist_ok=True)
            dst = os.path.join(UPLOADS, name); left = int(self.headers.get('Content-Length') or 0)
            with open(dst, 'wb') as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk: break
                    fh.write(chunk); left -= len(chunk)
            return self._json({'path': dst, 'name': name})
        if p == '/api/upload_song':
            # ライブラリに無い譜面: TJA と音源などを 1 つずつ受け取り、X-Folder の名前の作業フォルダに置く
            name = urllib.parse.unquote(self.headers.get('X-Filename') or 'song.tja')
            name = re.sub(r'[\\/:*?"<>|]+', '_', os.path.basename(name)) or 'song.tja'
            tok = re.sub(r'[^\w-]+', '_', self.headers.get('X-Folder') or time.strftime('%Y%m%d_%H%M%S'))
            if not name.lower().endswith(('.tja', '.ogg', '.mp3', '.wav', '.png', '.jpg')): return self._json({'error': '譜面(.tja)か音源(.ogg / .mp3 / .wav)を選んでください'}, 400)
            d = _song_path('ext:' + tok); os.makedirs(d, exist_ok=True)
            left = int(self.headers.get('Content-Length') or 0)
            with open(os.path.join(d, name), 'wb') as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk: break
                    fh.write(chunk); left -= len(chunk)
            return self._json({'token': tok, 'name': name})
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if p == '/api/run':
            with _lock:
                if any(not j.done for j in JOBS.values()):
                    return self._json({'error': '実行中の処理があります(CPU を全部使うので 1 本ずつ)'}, 409)
                video = os.path.abspath(body.get('video', ''))
                if not os.path.isfile(video): return self._json({'error': '動画が見つかりません'}, 400)
                if body.get('nosong'):
                    j = Job(video, '', None, body.get('cpu') or '8', title=(body.get('title') or '').strip() or None, prep_enc=body.get('prep_enc')); JOBS[j.id] = j
                    return self._json(j.info())
                sd = body.get('song_dir', '')
                if sd.startswith('ext:'):
                    # ライブラリに無い譜面。音源があれば普通の曲フォルダと同じ。音源が無ければ動画の音声で作り、その譜面と答え合わせする
                    es = ext_song(sd[4:])
                    if not es or not es['tjas']: return self._json({'error': '譜面(.tja)が見つかりません'}, 400)
                    ref = body.get('ref_tja') or es['tjas'][0]
                    det = next((t for t in es['detail'] if t['tja'] == ref), es['detail'][0])
                    if not det['audio']:
                        j = Job(video, '', None, body.get('cpu') or '8', title=det['title'], ext_ref=os.path.join(_song_path(sd), det['tja']), prep_enc=body.get('prep_enc')); JOBS[j.id] = j
                        return self._json(j.info())
                elif not os.path.isdir(os.path.join(SONGS, sd)): return self._json({'error': '曲フォルダが見つかりません'}, 400)
                j = Job(video, body['song_dir'], body.get('ref_tja') or None, body.get('cpu') or '8', ref_auto=bool(body.get('ref_auto')), prep_enc=body.get('prep_enc')); JOBS[j.id] = j
            return self._json(j.info())
        m = re.match(r'/api/toai/([\w]+)$', p)
        if m:
            # 変換した譜面をゲームの「13 AI譜面」に入れる(to_ai_folder.py: 本家の音源とヘッダー + 自動採譜のコース)。このページで変換した結果だけ
            d = os.path.join(WEB_OUT, m.group(1)); tja = os.path.join(d, 'out.tja')
            try: meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
            except Exception: return self._json({'error': '変換の記録が見つかりません'}, 404)
            if not os.path.isfile(tja): return self._json({'error': '譜面(out.tja)がありません'}, 404)
            # 既定は清書版(音符・小節線は採譜から 5ms 以内、速さ 3% 以内。clean_tja.py)。{"raw": true} なら採譜そのまま
            if not body.get('raw'):
                _add_clean({}, os.path.join(d, 'out'))
                if os.path.isfile(os.path.join(d, 'out_clean.tja')): tja = os.path.join(d, 'out_clean.tja')
            if meta.get('nosong'):
                # 本家の曲フォルダが無い: 動画から切り出した音源と譜面をそのまま置く(convert.install_plain)
                sys.path.insert(0, V); import convert
                try: return self._json({'ok': True, 'dir': convert.install_plain(os.path.join(d, 'out'), meta.get('title') or 'out', tja).replace('作成: ', '')})
                except Exception as ex: return self._json({'error': str(ex)}, 500)
            crs = re.search(r'^COURSE:\s*(\w+)', open(tja, encoding='utf-8-sig', errors='replace').read(), re.M)
            cmd = [sys.executable, 'to_ai_folder.py', _song_path(meta['song_dir']), tja, crs.group(1) if crs else 'Oni']
            if meta.get('ref_tja'): cmd += ['--ref', meta['ref_tja']]
            r = subprocess.run(cmd, cwd=V, capture_output=True, text=True, encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONUTF8='1'))
            if r.returncode != 0: return self._json({'error': (r.stderr or r.stdout)[-400:]}, 500)
            mm = re.search(r'作成:\s*(.+)', r.stdout)
            return self._json({'ok': True, 'dir': mm.group(1).strip() if mm else '', 'log': r.stdout})
        m = re.match(r'/api/job/([\w]+)/cancel$', p)
        if m and m.group(1) in JOBS:
            JOBS[m.group(1)].cancel(); return self._json({'ok': True})
        return self._json({'error': 'not found'}, 404)


def _add_clean(res, prefix):
    """清書版の譜面(<prefix>_clean.tja、clean_tja.py)を結果に足す。無ければ(古い変換)その場で作る"""
    path = prefix + '_clean.tja'
    try:
        if not os.path.exists(path) and os.path.exists(prefix + '.tja'):
            sys.path.insert(0, V); import clean_tja
            txt = open(prefix + '.tja', encoding='utf-8-sig', errors='replace').read()
            new, _ = clean_tja.clean_text(txt)
            clean_tja.write_tja(path, new)
        res['tja_clean'] = open(path, encoding='utf-8-sig', errors='replace').read()
        res['tja_clean_path'] = path
    except Exception as ex:
        res['tja_clean'] = None; res['tja_clean_err'] = str(ex)


def main():
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8765
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = 'http://127.0.0.1:%d/' % port
    print('video2tja の可視化ページ: %s (止めるには Ctrl+C)' % url, flush=True)
    if '--no-browser' not in sys.argv: webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    for j in JOBS.values(): j.cancel()


if __name__ == '__main__':
    main()
