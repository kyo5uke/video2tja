"""たくさんの曲の一括試験: 画面録画を 1 本ずつ取得 → 採譜 → 参照と照合。途中から再開できる。
使い方: python regress/wide.py <targets.json> <作業フォルダ> [--jobs N] [--limit N] [--dl N(先回りのダウンロード、既定 2)]
  targets.json: [{"video": id, "name": 曲名, "dir": Songs からの曲フォルダ, "ref": TJA ファイル名, "course": Oni/Edit}]
  結果: <作業フォルダ>/res/<id>.json、まとめ <作業フォルダ>/summary.txt。動画は <作業フォルダ>/yt/ に残す(V2T_WIDE_DELETE=1 で 100% の動画を消す)
普段使いと同じ軽い動かし方(V2T_CPU 既定 8)。候補のキャッシュは使わない(ディスクを食うので)
"""
import json, os, re, subprocess, sys, time, random
from concurrent.futures import ThreadPoolExecutor
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS = os.environ.get('V2T_SONGS') or os.path.join(V, 'Songs')
WARP = 'C:/Program Files/Cloudflare/Cloudflare WARP/warp-cli.exe'
ENV = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', V2T_CACHE='0')
ENV.setdefault('V2T_CPU', '8')


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, env=kw.pop('env', ENV), **kw)
    return r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')


def download(vid, ytdir, retry=True):
    p = os.path.join(ytdir, vid + '.mp4')
    if os.path.exists(p): return p
    out = sh(['yt-dlp', '-q', '--no-warnings', '-f', 'bv*[height=1080][fps>50]+ba/bv*[height<=1080]+ba/b', '--merge-output-format', 'mp4',
              '-o', os.path.join(ytdir, '%(id)s.%(ext)s'), 'https://www.youtube.com/watch?v=' + vid])
    # WARP のつなぎ直しは PC 全体の通信経路を切り替え、ほかの通信を途中で切る(「API Error: Connection lost」の原因)。
    # V2T_WARP_TOGGLE=1 のときだけつなぎ直す。既定はつながり方を変えない(つなぎたければ先に手でつないでおく)
    if not os.path.exists(p) and retry and 'bot' in out and os.path.exists(WARP) and os.environ.get('V2T_WARP_TOGGLE') == '1':
        # 「ボット確認」で止められたら Cloudflare WARP につないで取り直す
        subprocess.run([WARP, '--accept-tos', 'connect'], capture_output=True); time.sleep(8)
        try: return download(vid, ytdir, retry=False)
        finally: subprocess.run([WARP, '--accept-tos', 'disconnect'], capture_output=True)
    return p if os.path.exists(p) else None


def course_stats(log, course):
    m = re.search(r'== .*? vs %s \(vid_offset[^\n]*\n[^\n]*\n(recall [^\n]*)' % re.escape(course), log)
    if not m: return None
    s = m.group(1)
    g = lambda k: float(re.search(k + r' ([\d.]+)', s).group(1)) if re.search(k + r' ([\d.]+)', s) else None
    return {'recall': g('recall'), 'precision': g('precision'), 'color': g(r'type\(color\)'), 'big': g('big/small'), 'sd': g('timing sd')}


_DL = {}          # 先回りのダウンロード {動画: Future[(パス, 秒)]}(main で起動。採譜の枠はダウンロードを待たずに済む)


def _dl_timed(vid, ytdir):
    t0 = time.time(); p = download(vid, ytdir); return p, time.time() - t0


def one(t, work):
    vid = t['video']; rp = os.path.join(work, 'res', vid + '.json')
    if os.path.exists(rp): return json.load(open(rp, encoding='utf-8'))
    ytdir = os.path.join(work, 'yt')
    p, t_dl = _DL[vid].result() if vid in _DL else _dl_timed(vid, ytdir)
    r = dict(t, dl=round(t_dl, 1))
    if not p:
        r['error'] = 'download'; json.dump(r, open(rp, 'w', encoding='utf-8'), ensure_ascii=False); return r
    sdir = os.path.join(SONGS, t['dir']); prefix = os.path.join(work, 'out', vid)
    t0 = time.time()
    log = sh([sys.executable, 'run_song.py', p, sdir, prefix], cwd=V, env=dict(ENV, REF_TJA=t['ref']))
    r['sec'] = round(time.time() - t0, 1)
    open(prefix + '.log', 'w', encoding='utf-8').write(log)
    st = course_stats(log, t['course'])
    if st: r.update(st)
    m = re.search(r'best course: (\w+)\s+recall ([\d.]+)% precision ([\d.]+)%\s+misses (\d+) extras (\d+)', log)
    if m: r['best'] = m.group(1); r['best_misses'] = int(m.group(4)); r['best_extras'] = int(m.group(5))
    if 'Traceback' in log: r['error'] = log[log.index('Traceback'):][-600:]
    perfect = st and st['recall'] == 100.0 and st['precision'] == 100.0 and st['color'] == 100.0 and st['big'] == 100.0
    r['perfect'] = bool(perfect)
    # 動画は残す(回し直しで全曲を試せるように。1 本 約 100MB)。消すのは V2T_WIDE_DELETE=1 のときだけ
    if perfect and os.environ.get('V2T_WIDE_DELETE') == '1':
        try: os.remove(p)
        except OSError: pass
    json.dump(r, open(rp, 'w', encoding='utf-8'), ensure_ascii=False)
    return r


def summary(work, targets):
    rs = []
    for t in targets:
        rp = os.path.join(work, 'res', t['video'] + '.json')
        if os.path.exists(rp): rs.append(json.load(open(rp, encoding='utf-8')))
    done = [r for r in rs if 'recall' in r]
    perfect = [r for r in done if r.get('perfect')]
    lines = ['済み %d / %d 本(取得失敗 %d、照合できず %d)。100%%: %d 本' % (len(rs), len(targets), sum(1 for r in rs if r.get('error') == 'download'),
             len(rs) - len(done) - sum(1 for r in rs if r.get('error') == 'download'), len(perfect))]
    if done:
        secs = sorted(r['sec'] for r in done)
        lines.append('1 曲の時間: 中央値 %.0f 秒、最大 %.0f 秒' % (secs[len(secs) // 2], secs[-1]))
    for r in sorted(done, key=lambda r: (r['recall'] or 0) + (r['precision'] or 0)):
        if r.get('perfect'): continue
        lines.append('%-12s %-5s 再現 %6.2f 適合 %6.2f 色 %6.2f 大小 %6.2f  %s / %s' % (r['video'], r['course'], r['recall'] or 0, r['precision'] or 0,
                     r['color'] or 0, r['big'] or 0, r['name'], r['dir']))
    for r in rs:
        if 'recall' not in r and r.get('error') != 'download': lines.append('失敗 %s %s %s' % (r['video'], r['name'], (r.get('error') or '')[-200:].replace('\n', ' ')))
    open(os.path.join(work, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    return lines


def main():
    targets = json.load(open(sys.argv[1], encoding='utf-8')); work = sys.argv[2]
    jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 2
    if '--limit' in sys.argv: targets = targets[:int(sys.argv[sys.argv.index('--limit') + 1])]
    for d in ('res', 'yt', 'out'): os.makedirs(os.path.join(work, d), exist_ok=True)
    n = 0
    # ダウンロードは採譜と別の係(--dl 本、既定 2)が曲の順に先回りして進める。採譜の枠がダウンロード(1 本 約 16 秒)で空かない
    ndl = int(sys.argv[sys.argv.index('--dl') + 1]) if '--dl' in sys.argv else 2
    dl_ex = ThreadPoolExecutor(ndl)
    for t in targets:
        if not os.path.exists(os.path.join(work, 'res', t['video'] + '.json')):
            _DL[t['video']] = dl_ex.submit(_dl_timed, t['video'], os.path.join(work, 'yt'))
    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(lambda t: one(t, work), targets):
            n += 1
            st = '100%' if r.get('perfect') else ('%.1f/%.1f' % (r.get('recall') or 0, r.get('precision') or 0) if 'recall' in r else r.get('error', '?')[:40])
            print('[%d/%d] %s %s %s %ss' % (n, len(targets), r['video'], r['name'], st, r.get('sec')), flush=True)
            if n % 10 == 0: summary(work, targets)
    print('\n'.join(summary(work, targets)[:3]))


if __name__ == '__main__':
    sys.path.insert(0, V); import calib; calib.low_priority()
    main()
