"""安定版の試験スイート: たくさんの曲(段位道場 2026 の課題曲など)を採譜して本家と照合し、基準と比べる。

使い方:
  python regress/suite.py                 # 全曲(検出からやり直す。1 曲 1〜2 分)
  python regress/suite.py --reuse         # 検出は前回の JSON を使い、ビルダー以降だけ(数秒/曲)
  python regress/suite.py --only 名前 ...  # 一部の曲だけ
  python regress/suite.py --update        # 今回の結果を基準(suite_baseline.json)にする
  python regress/suite.py --jobs 6        # 同時に回す曲数(既定 4)
結果: regress/suite_out/<名前>.{log,json,tja}、表を標準出力に。基準より悪くなった曲があれば終了コード 1。

曲の一覧は regress/suite.json(名前, 動画 ID, 曲フォルダ, 参照 TJA, コース)。
動画は testvideos/<ID>.mp4(YouTube から取ったキャプチャ。直撮りは入れない)。
曲フォルダは ゲーム側で番号が振り直されることがあるので「カテゴリ/番号を除いた曲名」で探す。
"""
import sys, os, re, json, glob, subprocess, time

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS = os.environ.get('V2T_SONGS') or os.path.join(V, 'Songs')
VIDEOS = V + '/testvideos'
OUT = V + '/regress/suite_out'
MANIFEST = V + '/regress/suite.json'
BASE = V + '/regress/suite_baseline.json'
ENV = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
ENV.setdefault('V2T_CPU', 'all')      # 試験は全部の CPU を使う(1 曲ずつの変換の既定は 8 本: detect2.cpu_budget)


def song_dir(spec):
    """'08 Namco Original/Taberuna 2000' → 番号付きの実フォルダ。'12 Dan Dojo 2026/Nijiiro 2026/15 Judan' はそのまま"""
    p = SONGS + '/' + spec
    if os.path.isdir(p): return p
    cat, name = spec.rsplit('/', 1)
    hits = [h for h in glob.glob(SONGS + '/' + cat + '/[0-9]* ' + name)
            if os.path.basename(h).split(' ', 1)[1] == name]      # 番号の後ろが曲名と一致(「Zoku Shimedore 2000」を「Shimedore 2000」で拾わない)
    if not hits: raise SystemExit('曲フォルダが見つからない: ' + spec)
    return hits[0]


def sh(cmd, env=None):
    return subprocess.run(cmd, capture_output=True, cwd=V, env=env or ENV).stdout.decode('utf-8', 'replace')


def run(e, reuse):
    os.makedirs(OUT, exist_ok=True)
    name = e['name']; prefix = OUT + '/' + name
    video = e['video'] if os.path.isabs(e['video']) else VIDEOS + '/' + e['video'] + '.mp4'
    sdir = song_dir(e['dir']); ref = sdir + '/' + e['ref']
    env = dict(ENV, REF_TJA=e['ref'], V2T_WORK=VIDEOS + '/work')
    if reuse: env['REUSE_ONS'] = '1'
    if os.environ.get('V2T_WORKERS'): env['V2T_WORKERS'] = os.environ['V2T_WORKERS']
    for k, v in (e.get('env') or {}).items(): env[k] = v
    t0 = time.time()
    log = sh([sys.executable, 'run_song.py', video, sdir, prefix], env)
    open(prefix + '.log', 'w', encoding='utf-8').write(log)
    r = {'name': name, 'sec': round(time.time() - t0)}
    m = re.search(r'best course: (\w+)\s+recall ([\d.]+)% precision ([\d.]+)%\s+misses (\d+) extras (\d+)', log)
    if m: r.update(course=m.group(1), recall=float(m.group(2)), precision=float(m.group(3)), misses=int(m.group(4)), extras=int(m.group(5)))
    # 評価の行の間にビルダーの出力が混ざることがある(標準出力のバッファ)ので、見出しの後ろ 1500 文字の中を探す
    m = re.search(r'vs ' + re.escape(r.get('course') or e['course']) + r' \(vid_offset[\s\S]{0,1500}?recall [^\n]*?type\(color\) ([\d.]+)% [^\n]*?big/small ([\d.]+)% [^\n]*?timing sd ([\d.]+)ms', log)
    if m: r.update(color=float(m.group(1)), big=float(m.group(2)), sd=float(m.group(3)))
    m = re.search(r'同期: 採用 ([-+\d.]+)ms\((\w+)\)  参照なし\(av\) (\S+)  参照あり\(ref\) ([-+\d.]+)ms', log)
    if m:
        r.update(sync_mode=m.group(2), sync_ref=float(m.group(4)))
        try: r['sync_av'] = float(m.group(3).replace('ms', ''))
        except ValueError: r['sync_av'] = None
    if 'Traceback' in log: r['error'] = log[log.index('Traceback'):][:400]
    tja = prefix + '.tja'
    if os.path.exists(tja):
        c = e['course']
        g = sh([sys.executable, 'regress/gogo_cmp.py', tja, ref, c], env)      # 曲ごとの分岐の道(BRANCH_PATH)を使う
        gm = re.search(r'生成 (\d+) 区間 / 参照 (\d+) 区間', g)
        if gm: r['gogo'] = [int(gm.group(1)), int(gm.group(2))]
        d = [abs(int(x)) for x in re.findall(r'差 +([-+]?\d+) / +[-+]?\d+ ms', g)]          # 開始の差(終わりは最終音符より後ろのことがあるので見ない)
        r['gogo_start_max'] = max(d) if d else None
        # 書き出した TJA を読み直して検出と照らし合わせる(格子への当てはめ・BPM・小節・OFFSET の誤りはここにしか出ない)
        rt = sh([sys.executable, 'regress/roundtrip.py', prefix + '_ons.json', tja], env)
        rm = re.search(r'matched (\d+) unmatched (\d+) type-mismatch (\d+)', rt)
        rs = re.search(r'sd ([\d.]+)ms max\|\.\| ([\d.]+)ms[\s\S]*?>20ms: (\d+)', rt)
        if rm: r.update(rt_unmatched=int(rm.group(2)), rt_type=int(rm.group(3)))
        if rs: r.update(rt_sd=float(rs.group(1)), rt_gt20=int(rs.group(3)))
        # 連打・風船(種類・頭・終わり)を参照と比べる
        rc = sh([sys.executable, 'regress/roll_cmp.py', name], env)
        cm = re.search(r'合計: 連打・風船 (\d+)  無し (\d+)  余分 (\d+)  種類違い (\d+)  頭 >30ms (\d+)  終わり >50ms (\d+)', rc)
        if cm: r['roll'] = [int(cm.group(k)) for k in range(1, 7)]
        s = sh([sys.executable, 'regress/sym_cmp.py', tja, ref, c], env)
        sm = re.search(r'音符\(1234\) 生成 (\d+) / 参照 (\d+)', s)
        if sm: r['notes'] = [int(sm.group(1)), int(sm.group(2))]
    json.dump(r, open(prefix + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return r


def worse(r, b):
    """基準 b より悪くなったところ"""
    out = []
    if 'error' in r: out.append('エラー')
    for k in ('recall', 'precision', 'color', 'big'):      # big: 大小の一致率(チャーリーダッシュ! の大音符 4 個が小になったのを見逃していた)
        if b.get(k) is not None and (r.get(k) is None or r[k] < b[k] - 0.05): out.append('%s %s→%s' % (k, b[k], r.get(k)))
    if b.get('sd') is not None and (r.get('sd') is None or r['sd'] > b['sd'] + 1.0): out.append('sd %s→%s' % (b['sd'], r.get('sd')))
    if b.get('gogo') and (not r.get('gogo') or abs(r['gogo'][0] - r['gogo'][1]) > abs(b['gogo'][0] - b['gogo'][1])):
        out.append('ゴーゴー %s→%s' % (b['gogo'], r.get('gogo')))           # 本家との区間数の差が広がったときだけ
    for k in ('rt_unmatched', 'rt_type', 'rt_gt20'):         # 書き出した TJA と検出の食い違いの数
        if b.get(k) is not None and (r.get(k) is None or r[k] > b[k]): out.append('%s %s→%s' % (k, b[k], r.get(k)))
    if b.get('rt_sd') is not None and (r.get('rt_sd') is None or r['rt_sd'] > b['rt_sd'] + 1.0): out.append('rt_sd %s→%s' % (b['rt_sd'], r.get('rt_sd')))
    if b.get('roll') and (not r.get('roll') or sum(r['roll'][1:]) > sum(b['roll'][1:])):      # 連打の 無し+余分+種類違い+頭+終わり
        out.append('連打 %s→%s' % (b['roll'], r.get('roll')))
    if b.get('gogo_start_max') is not None and (999 if r.get('gogo_start_max') is None else r['gogo_start_max']) > b['gogo_start_max'] + 10:
        out.append('ゴーゴー開始 %s→%s' % (b['gogo_start_max'], r.get('gogo_start_max')))
    return out


def main():
    args = sys.argv[1:]
    reuse = '--reuse' in args; update = '--update' in args
    only = args[args.index('--only') + 1:] if '--only' in args else None
    songs = json.load(open(MANIFEST, encoding='utf-8'))
    base = json.load(open(BASE, encoding='utf-8')) if os.path.exists(BASE) else {}
    res = {}; bad = 0
    print('%-30s %-6s %6s %6s %5s %4s | %-7s %5s | %-9s | %s' % ('曲', 'コース', '再現率', '適合率', '色', 'sd', 'ゴーゴー', '開始差', '同期 av/ref', '基準との比較'))
    todo = [e for e in songs if not only or e['name'] in only]
    # 何曲か同時に回す(16 コア: 既定 4 曲。各曲の検出ワーカーは 28 ÷ 曲数。GPU デコードは曲をまたいで共有)
    jobs = int(args[args.index('--jobs') + 1]) if '--jobs' in args else min(4, len(todo))
    if jobs > 1: os.environ.setdefault('V2T_WORKERS', str(max(4, 28 // jobs)))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(run, e, reuse): e for e in todo}
        for fu in as_completed(futs):
            e = futs[fu]; r = fu.result(); res[e['name']] = r
            w = worse(r, base.get(e['name'], {})); bad += bool(w)
            print('%-30s %-6s %6s %6s %5s %4s | %-7s %5s | %4s/%-4s | %s' % (
            e['name'][:30], r.get('course'), r.get('recall'), r.get('precision'), r.get('color'), r.get('sd'),
            '%d/%d' % tuple(r['gogo']) if r.get('gogo') else '-', r.get('gogo_start_max'),
            ('%.0f' % r['sync_av']) if r.get('sync_av') is not None else '-', ('%.0f' % r['sync_ref']) if r.get('sync_ref') is not None else '-',
            ('悪化: ' + ', '.join(w)) if w else ('OK' if e['name'] in base else '(基準なし)')), flush=True)
    print('所要 %.0f 秒(%d 曲、同時 %d 曲)' % (time.time() - t_all, len(res), jobs))
    if update:
        base.update(res); json.dump(base, open(BASE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('基準を更新: %d 曲' % len(res))
    print('悪化 %d 曲 / %d 曲' % (bad, len(res)))
    sys.exit(1 if bad and not update else 0)


if __name__ == '__main__':
    main()
