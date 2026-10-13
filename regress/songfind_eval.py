"""一括試験の動画(testvideos/wide/all.json)で、songfind が参照の曲フォルダを選ぶかを数える。
使い方: python regress/songfind_eval.py [--jobs 4] [--skip N] [--limit N] [--only ID,...] [--out 結果.json]  → 結果は testvideos/wide/songfind_eval.json
数えるもの: 自動で選んだ曲(songfind.choose)が参照の曲フォルダか(auto_ok)・選ばなかったか(auto なし)・1 位の組が参照か(top_ok)"""
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V)
MAIN = os.path.join(V, 'testvideos', 'wide')
_env = None


def _init():
    global _env
    import songfind
    _env = songfind.index(verbose=False)


def _rel(p):
    import songfind
    return os.path.relpath(os.path.dirname(p), songfind.SONGS).replace(os.sep, '/')


def _one(t):
    import songfind
    p = os.path.join(MAIN, 'yt', t['video'] + '.mp4')
    if not os.path.exists(p): return dict(t, err='no video')
    try:
        r = songfind.find(p, 8, env=_env)
        ch, groups = songfind.choose(r)
    except Exception as e: return dict(t, err=str(e)[:200])
    top = [(round(w, 3), _rel(q), round(d, 3), round(s, 3), round(L, 1)) for w, q, d, s, L in r]
    gtop = [[_rel(c[1]) for c in g] for g in groups[:3]]
    auto = _rel(ch[1]) if ch else None
    return dict(t, top=top, groups=gtop, auto=auto, auto_ok=auto == t['dir'], top_ok=bool(gtop) and t['dir'] in gtop[0])


if __name__ == '__main__':
    tg = json.load(open(os.path.join(MAIN, 'all.json'), encoding='utf-8'))
    if '--only' in sys.argv: tg = [t for t in tg if t['video'] in sys.argv[sys.argv.index('--only') + 1].split(',')]
    if '--skip' in sys.argv: tg = tg[int(sys.argv[sys.argv.index('--skip') + 1]):]
    if '--limit' in sys.argv: tg = tg[:int(sys.argv[sys.argv.index('--limit') + 1])]
    jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 4
    out = []
    with ProcessPoolExecutor(jobs, initializer=_init) as ex:
        for k, r in enumerate(ex.map(_one, tg)):
            out.append(r)
            if not r.get('auto_ok'):
                print('[%d] %s %s 参照 %s → 自動 %s / 1 位の組 %s / %s' % (k + 1, r['video'], r['name'], r['dir'], r.get('auto'), (r.get('groups') or [None])[0],
                                                               (r.get('top') or r.get('err'))[:3]), flush=True)
            if (k + 1) % 100 == 0: print('  %d / %d 本' % (k + 1, len(tg)), flush=True)
    outp = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(MAIN, 'songfind_eval.json')
    json.dump(out, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n = len(out)
    print('自動で参照の曲フォルダを選んだ: %d / %d、選ばなかった: %d、別の曲を選んだ: %d、1 位の組に参照: %d' % (
        sum(1 for r in out if r.get('auto_ok')), n, sum(1 for r in out if not r.get('auto') and not r.get('err')),
        sum(1 for r in out if r.get('auto') and not r.get('auto_ok')), sum(1 for r in out if r.get('top_ok'))))
