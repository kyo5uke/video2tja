"""一括試験の結果を、保存した検出(out/<id>_ons.json)で照合だけやり直す(照合を直したとき用)。使い方: python regress/wide_reeval.py <作業フォルダ>"""
import glob, json, os, re, sys
import numpy as np
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V)
from eval_ons import evaluate, load_ons
from eval_timing import parse_tja_times
work = sys.argv[1]; n = ch = 0
for rp in sorted(glob.glob(os.path.join(work, 'res', '*.json'))):
    r = json.load(open(rp, encoding='utf-8')); v = r['video']
    op = os.path.join(work, 'out', v + '_ons.json'); lp = os.path.join(work, 'out', v + '.log')
    if 'recall' not in r or not os.path.exists(op) or not os.path.exists(lp): continue
    log = open(lp, encoding='utf-8', errors='replace').read()
    m = re.search(r'vs \w+ \(vid_offset ([-\d.]+)\)', log)
    if not m: continue
    vo = float(m.group(1)); tja = os.path.join(os.environ.get('V2T_SONGS') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Songs'), r['dir'], r['ref'])
    txt = open(tja, encoding='utf-8-sig', errors='replace').read()
    mt, mc, rng = load_ons(op); best = None; own = None
    for crs in re.findall(r'^COURSE:\s*(\w+)', txt, re.M):
        try: _, tn = parse_tja_times(tja, crs)
        except Exception: continue
        if len(tn) < 10: continue
        tt = np.array([x[0] for x in tn]); tc = np.array([x[1] for x in tn])
        res = evaluate(mt, mc, tt, tc, vo, rng, verbose=False)
        if res and (best is None or res[0] + res[1] > best[1]): best = (crs, res[0] + res[1], res)
        if crs == r['course']: own = res
    if not best: continue
    old = (r.get('best_misses'), r.get('best_extras'))
    r['best'] = best[0]; r['best_misses'] = len(best[2][3]); r['best_extras'] = len(best[2][4])
    if own: r['recall'], r['precision'], r['color'] = round(own[0], 1), round(own[1], 1), round(own[2], 1)
    r['perfect'] = bool(own and own[0] == 100.0 and own[1] == 100.0 and own[2] == 100.0 and r.get('big') == 100.0)
    json.dump(r, open(rp, 'w', encoding='utf-8'), ensure_ascii=False); n += 1
    if old != (r['best_misses'], r['best_extras']): ch += 1
print('照合し直した %d 本、食い違いの数が変わった %d 本' % (n, ch))
