"""全曲の連打・風船の頭と終わりを 参照 と比べる: roll_cmp.py [曲名 …]
各曲で 参照 と採譜の連打を時刻の近い順に対応させ(頭が 0.15 秒以内)、頭・終わりの差と種類の違いを数える"""
import sys, os, json
import numpy as np
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V); sys.path.insert(0, V + '/regress')
import render_compare_ref as R
from suite import song_dir
only = sys.argv[1:]
tot = {'n': 0, 'miss': 0, 'extra': 0, 'kind': 0, 'head30': 0, 'end50': 0}
for e in json.load(open(V + '/regress/suite.json', encoding='utf-8')):
    if only and e['name'] not in only: continue
    tja = V + '/regress/suite_out/%s.tja' % e['name']
    os.environ['BRANCH_PATH'] = (e.get('env') or {}).get('BRANCH_PATH', 'M')      # 譜面分岐のたどる道(曲ごと)
    if not os.path.exists(tja): continue
    try:
        en, _, er = R.parse_tja(song_dir(e['dir']) + '/' + e['ref'], e['course'])
        mn, _, mr = R.parse_tja(tja, e['course'] if ('COURSE:' + e['course']) in open(tja, encoding='utf-8-sig').read() else 'Oni')
    except Exception as ex:
        print(e['name'], '読めない', ex); continue
    if not er and not mr: continue
    # 曲全体のずれ(音符)を合わせる
    ta = np.array([x[0] for x in en]); tb = np.array(sorted(x[0] for x in mn))
    sh = float(np.median([tb[np.argmin(np.abs(tb - t))] - t for t in ta])) if len(ta) and len(tb) else 0.0
    used = set(); rows = []
    for h, en_, c, sp in er:
        j = min((k for k in range(len(mr)) if k not in used), key=lambda k: abs(mr[k][0] - (h + sh)), default=None)
        if j is None or abs(mr[j][0] - (h + sh)) > 0.15: rows.append(('無', h, c, None)); continue
        used.add(j); rows.append(('対', h, c, (mr[j][0] - h - sh, mr[j][1] - en_ - sh, mr[j][2])))
    extra = len(mr) - len(used)
    bad = []
    for kind, h, c, d in rows:
        tot['n'] += 1
        if kind == '無': tot['miss'] += 1; bad.append('%.2f %s 無し' % (h, c)); continue
        dh, de, c2 = d
        k2 = {'5': 'S', '6': 'B', '7': 'b', '9': 'b'}
        if k2[c] != k2[c2]: tot['kind'] += 1; bad.append('%.2f 種類 %s→%s' % (h, c, c2))
        if abs(dh) > 0.03: tot['head30'] += 1; bad.append('%.2f 頭 %+.0fms' % (h, dh * 1000))
        if abs(de) > 0.05: tot['end50'] += 1; bad.append('%.2f 終わり %+.0fms' % (h, de * 1000))
    tot['extra'] += extra
    if bad or extra: print('%-28s 参照 %2d 採譜 %2d  %s%s' % (e['name'][:28], len(er), len(mr), ' / '.join(bad[:6]), ('  余分 %d' % extra) if extra else ''))
print('合計: 連打・風船 %(n)d  無し %(miss)d  余分 %(extra)d  種類違い %(kind)d  頭 >30ms %(head30)d  終わり >50ms %(end50)d' % tot)
