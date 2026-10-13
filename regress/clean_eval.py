"""清書(clean_tja)の前後を、1100 曲(testvideos/wide)の採譜 TJA と本家で比べる。
使い方: clean_eval.py [--jobs 16] [--only 動画ID ...] [--ms=8.33]
結果: regress/clean_out/<動画ID>_clean.tja と clean_eval.json、集計を標準出力に"""
import sys, os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V); sys.path.insert(0, os.path.join(V, 'regress'))
MAIN = V
WIDE = os.path.join(MAIN, 'testvideos', 'wide')
SONGS = os.environ.get('V2T_SONGS') or os.path.join(V, 'Songs')
OUT = os.path.join(V, 'regress', 'clean_out')


def one(e, T):
    import io, contextlib
    import clean_tja as C, tja_cmp as TC
    gen = os.path.join(WIDE, 'out', e['video'] + '.tja')
    ref = os.path.join(SONGS, e['dir'], e['ref'])
    if not os.path.exists(gen) or not os.path.exists(ref): return e['video'], {'skip': 'ファイルが無い'}
    t0 = time.time()
    txt = open(gen, encoding='utf-8-sig', errors='replace').read()
    try:
        new, info = C.clean_text(txt, T)
    except Exception as ex:
        import traceback; return e['video'], {'error': traceback.format_exc()[-600:]}
    dst = os.path.join(OUT, e['video'] + '_clean.tja')
    open(dst, 'w', encoding='utf-8-sig', newline='\n').write(new + '\n')
    r = {'sec': round(time.time() - t0, 2), 'info': info, 'raw': C.stats(txt, e['course']), 'clean': C.stats(new, e['course']),
         'ref': C.stats(open(ref, encoding='utf-8-sig', errors='replace').read(), e['course'])}
    al = None
    for k, g in (('cmp_raw', gen), ('cmp_clean', dst)):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # 揃えるのは直線(一定のずれ + 本家と動画の版のテンポの違い)だけ。直線は清書前で当てはめ、清書後も同じ直線を使う
                # (局所の窓で揃えると譜面自身の揺れまで揃えて揺れを残した清書前が有利になり、別々に当てはめると当てはめの違いで数が跳ぶ)。
                # 清書後は曲全体の一定のずれ(OFFSET の直し)だけ足し直す(tja_cmp の align)
                c = TC.compare(g, ref, e['course'], verbose=False, mode='line', align=al)
            if al is None: al = c['align']
            c.pop('err', None); c.pop('align', None); c.pop('align', None); c['miss'] = len(c['miss']); c['extra'] = len(c['extra']); r[k] = c
        except SystemExit as ex: r[k] = {'error': str(ex)}
        except Exception as ex: r[k] = {'error': repr(ex)}
    return e['video'], r


def main():
    a = sys.argv[1:]
    jobs = int(a[a.index('--jobs') + 1]) if '--jobs' in a else 16
    only = a[a.index('--only') + 1:] if '--only' in a else None
    T = next((float(x[5:]) / 1000 for x in a if x.startswith('--ms=')), None)
    import clean_tja as C
    T = T or C.T_MOVE
    os.makedirs(OUT, exist_ok=True)
    songs = json.load(open(os.path.join(WIDE, 'all.json'), encoding='utf-8'))
    if only: songs = [e for e in songs if e['video'] in only]
    res = {}; t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(one, e, T) for e in songs]
        for k, fu in enumerate(as_completed(futs)):
            v, r = fu.result(); res[v] = r
            if (k + 1) % 100 == 0: print('  %d / %d(%.0f 秒)' % (k + 1, len(songs), time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(V, 'regress', 'clean_eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    ok = [r for r in res.values() if 'cmp_raw' in r and 'cmp_clean' in r and 'error' not in r['cmp_raw'] and 'error' not in r['cmp_clean']]
    err = [(v, r) for v, r in res.items() if 'error' in r]
    print('曲 %d(照合できた %d、清書の失敗 %d、所要 %.0f 秒)' % (len(res), len(ok), len(err), time.time() - t0))
    for v, r in err[:5]: print('  失敗', v, r['error'][-300:])
    S = lambda key, sub: sum(r[key][sub] for r in ok)
    print('%-22s %10s %10s %10s' % ('(コースの本文)', '清書前', '清書後', '本家'))
    for sub, lab in (('lines', '行'), ('bpm', '#BPMCHANGE'), ('scroll', '#SCROLL'), ('measure', '#MEASURE')):
        print('%-22s %10d %10d %10d' % (lab, S('raw', sub), S('clean', sub), S('ref', sub)))
    for sub, lab in (('match', '音符の一致(連打込み)'), ('match_dk', 'ドン・カッの一致'), ('miss', '取りこぼし'), ('extra', '余分'), ('type', '種類違い'), ('gt10', '時刻 >10ms'), ('gt25', '時刻 >25ms'),
                     ('speed', '速さ >3%'), ('bars', '小節線の一致')):
        print('%-22s %10d %10d' % (lab, S('cmp_raw', sub), S('cmp_clean', sub)))
    import numpy as np
    for sub, lab in (('med', '時刻の差 中央(曲の中央値)'), ('p95', '時刻の差 95%(曲の中央値)'), ('max', '時刻の差 最大(曲の中央値)')):
        print('%-22s %10.2f %10.2f' % (lab, np.median([r['cmp_raw'][sub] for r in ok]), np.median([r['cmp_clean'][sub] for r in ok])))
    perfect = lambda c: c['match_dk'] == c['ref_dk'] == c['gen_dk'] and c['type'] == 0
    print('ドン・カッが全部一致した曲(種類違い 0)  清書前 %d  清書後 %d' % (sum(perfect(r['cmp_raw']) for r in ok), sum(perfect(r['cmp_clean']) for r in ok)))
    worse = [(v, r) for v, r in res.items() if r in ok and (r['cmp_clean']['match_dk'] < r['cmp_raw']['match_dk'] or r['cmp_clean']['type'] > r['cmp_raw']['type']
             or r['cmp_clean']['gt25'] > r['cmp_raw']['gt25'])]
    print('一致・種類・>25ms が悪くなった曲 %d' % len(worse))
    for v, r in worse[:15]:
        a_, b_ = r['cmp_raw'], r['cmp_clean']
        print('  %s  一致 %d→%d  種類 %d→%d  >25ms %d→%d  >10ms %d→%d' % (v, a_['match_dk'], b_['match_dk'], a_['type'], b_['type'], a_['gt25'], b_['gt25'], a_['gt10'], b_['gt10']))
    fr = sum(sum(x.get('frozen', 0) for x in r['info'].values() if isinstance(x, dict)) for r in ok)
    ms_ = sum(sum(x.get('measures', 0) for x in r['info'].values() if isinstance(x, dict)) for r in ok)
    print('そのまま残した小節 %d / %d' % (fr, ms_))


if __name__ == '__main__':
    main()
