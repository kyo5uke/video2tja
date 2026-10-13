"""生成 TJA の音符時刻(OFFSET 込み)を検出 JSON の時刻と突き合わせる: roundtrip.py <ons.json> <tja> [course]
VID2AUDIO=0 で作った TJA なら動画秒どうしで比べられる"""
import sys, os, re, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_timing import parse_tja_times
js, tja = sys.argv[1], sys.argv[2]; crs = sys.argv[3] if len(sys.argv) > 3 else None
txt = open(tja, encoding='utf-8-sig').read()
if crs is None: crs = re.search(r'^COURSE:\s*(\w+)', txt, re.M).group(1)
offset = float(re.search(r'^OFFSET:\s*([-\d.]+)', txt, re.M).group(1))
_, tn = parse_tja_times(tja, crs)
tt = np.array([n[0] for n in tn]) - offset; tc = [n[1] for n in tn]
d = json.load(open(js, encoding='utf-8'))
det = [o for o in d['onsets'] if o['color'] in ('don', 'ka')]
dt = np.array([o['t'] for o in det])
# 書き出した TJA は音源の時刻(OFFSET は音源基準)。検出は動画の時刻なので、全体のずれ(動画↔音源の差と同期の補正)を 1 つだけ推定して引く
if len(tt) and len(dt):
    ts_, ds_ = np.sort(tt), np.sort(dt); m_ = min(len(ts_), len(ds_))
    c0 = float(np.median(ts_[:m_] - ds_[:m_]))                 # おおよそのずれ(音源と動画の差 ~十数秒)。並べた順の差の中央値
    d0 = np.array([tt[int(np.argmin(np.abs(tt - c0 - x)))] - x for x in dt])
    for _ in range(3):
        sh0 = float(np.median(d0)) if len(d0) else 0.0
        d0 = np.array([tt[int(np.argmin(np.abs(tt - sh0 - x)))] - sh0 - x for x in dt]) + sh0
        d0 = d0[np.abs(d0 - sh0) < 0.05] if np.any(np.abs(d0 - sh0) < 0.05) else d0
    tt = tt - float(np.median(d0))
sym = lambda o: ('3' if o['big'] else '1') if o['color'] == 'don' else ('4' if o['big'] else '2')
errs = []; unmatched = 0; typeerr = 0
for o in det:
    j = int(np.argmin(np.abs(tt - o['t'])))
    if abs(tt[j] - o['t']) > 0.05: unmatched += 1; continue
    errs.append((tt[j] - o['t']) * 1000)
    if tc[j] != sym(o): typeerr += 1
errs = np.array(errs)
body = txt.split('#START')[1].split('#END')[0]
print('TJA notes %d (det %d)  matched %d unmatched %d type-mismatch %d' % (len(tt), len(det), len(errs), unmatched, typeerr))
print('tja - det: mean %+.1fms sd %.1fms max|.| %.1fms  p90|.| %.1fms  (>10ms: %d, >20ms: %d)' % (errs.mean(), errs.std(), np.abs(errs).max(), np.percentile(np.abs(errs), 90), (np.abs(errs) > 10).sum(), (np.abs(errs) > 20).sum()))
print('lines: measures %d  BPMCHANGE %d  SCROLL %d  MEASURE %d' % (body.count(','), body.count('#BPMCHANGE'), body.count('#SCROLL'), body.count('#MEASURE')))
if '-v' in sys.argv:
    bad = [(o['t'], e) for o, e in zip([o for o in det if abs(tt[int(np.argmin(np.abs(tt - o['t'])))] - o['t']) <= 0.05], errs) if abs(e) > 10]
    print('  >10ms:', ' '.join('%.2f(%+.0f)' % b for b in bad[:60]))
