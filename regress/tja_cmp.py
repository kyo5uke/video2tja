"""TJA どうしの照合(採譜・清書した TJA と本家): 音符の時刻・種類・流れる速さ、小節線を 1 個ずつ突き合わせる。
使い方: tja_cmp.py <生成tja> <参照tja> [コース(既定: 生成側の最初のコース)] [--list]
時刻は「音源の秒」(#START からの秒 − OFFSET)で比べる。全体のずれ(OFFSET の差)は 1 つだけ推定して引き、その値も出す。
出力(1 行目が集計):
  音符 一致 N/参照 R(再現率 %)・生成 G(適合率 %)・種類違い・時刻の差 |中央値|/95%/最大 ms・>10ms・>25ms の数・速さ >3% 違い
  小節線 一致(見える線どうし)
"""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clean_tja as C

TOL = 0.035      # 音符の対応を取る幅(eval_ons.evaluate と同じ)


def load(path, course=None):
    txt = open(path, encoding='utf-8-sig', errors='replace').read()
    head, courses = C.split_tja(txt)
    bpm0 = float(C.header_value(head, 'BPM')); off = float(C.header_value(head, 'OFFSET', '0'))
    pick = None
    for c in courses:
        name = C.header_value(c['head'], 'COURSE', '')
        if course is None or name.lower() == course.lower() or name == {'oni': '3', 'edit': '4', 'hard': '2'}.get((course or '').lower()):
            if C.header_value(c['head'], 'STYLE', 'Single').lower() != 'double': pick = c; break
    if pick is None: raise SystemExit('コース %s が無い: %s' % (course, path))
    name = C.header_value(pick['head'], 'COURSE', '')
    if any(t.startswith('#BRANCH') for m in pick['body'] for _, t in m.cmds):
        # 譜面分岐: 動画に映る道(BRANCH_PATH、既定 M)だけを eval_timing の読み方でたどる(速さは比べない)
        from eval_timing import parse_tja_times, parse_tja_bars
        _, tn = parse_tja_times(path, name, '12345678')
        return [(t - off, ch, None) for t, ch in tn], [(t - off, None) for t in parse_tja_bars(path, name)], name
    notes, bars, _ = C.timeline(pick['body'], bpm0)
    nt = [(t - off, ch, v) for t, ch, v, *_ in notes if ch in '12345678']
    bt = [(t - off, v) for t, v, on, _ in bars if on]
    return nt, bt, C.header_value(pick['head'], 'COURSE', '')


def pair(a, b, tol):
    """時刻の列 a, b(昇順)を差の小さい組から貪欲に対応させる → [(i, j)]"""
    cand = []
    for i, x in enumerate(a):
        j = int(np.searchsorted(b, x))
        for jj in range(max(0, j - 3), min(len(b), j + 3)):
            if abs(b[jj] - x) <= tol: cand.append((abs(b[jj] - x), i, jj))
    cand.sort(); ua = set(); ub = set(); out = []
    for _, i, j in cand:
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j); out.append((i, j))
    return sorted(out)


W = 8.0         # 局所のずれを求める窓(秒)。eval_ons.evaluate と同じ(録画の揺れ・版の違いで曲の中のずれが数十 ms 動く)
# 本家と動画のずれを揃える直線の探し方。1100 曲で見た最大は 版のテンポの違い 0.08%(Re:End of a Dream: 2 分で 100ms)と一定のずれ 90ms なので、
# 傾きは ±LINE_SLOPE_MAX を LINE_SLOPE_STEP 刻み(2 分で 6ms)、ずれは ±LINE_SPAN の組の差の山(LINE_BIN 刻みの 3 つ分 = ±LINE_INLIER)で探す
LINE_SLOPE_MAX = 0.003
LINE_SLOPE_STEP = 0.00005
LINE_SPAN = 0.25
LINE_BIN = 0.005
LINE_INLIER = 0.0075
# 局所の残りをたどる幅(今のずれから ±LOCAL_GATE の組だけ使う)と回数。録画の継ぎ目で途中から 20ms 飛ぶ動画があり、その 2 倍
LOCAL_GATE = 0.04
LOCAL_ITERS = 4
REFIT_GATE = 0.02   # 別の譜面のずれを借りるとき、一定のずれ(清書の OFFSET の直し ≤ 5ms)を足し直すのに使う組の幅
SPEED_TOL = 0.03    # 速さの違いを数える幅(clean_tja.V_TOL・make_tja が音符ごとに #SCROLL を挟む基準と同じ)


def compare(gen, ref, course=None, verbose=True, align=None, W=W, mode='local'):
    """時刻の差は「8 秒窓ごとのずれ」を引いてから測る(eval_ons.evaluate と同じ考え方。曲全体のずれ 1 つだと、本家と動画で
    曲の中のずれが動く曲(版の違い)は採譜が正しくても差が大きく出る)。種類・時刻の統計はドン・カッ(1234)だけ、連打は一致の数にだけ入れる。
    align = 別の譜面の照合で求めた (時刻, ずれ) の列。
    mode = 'line' なら揃えるのは直線(一定のずれ + 版のテンポの違いの傾き)だけ。清書の前後を公平に比べるとき用
    (局所の窓で揃えると、その譜面自身の揺れまで揃えてしまい、揺れを残した方が有利になる)"""
    gn, gb, crs = load(gen, course)
    rn, rb, _ = load(ref, course or crs)
    ga = np.array([n[0] for n in gn]); ra = np.array([n[0] for n in rn])
    # 全体のずれ: 近い組の差の中央値を 3 回詰める
    sh = 0.0
    for tol in (0.1, 0.05, 0.02):
        pr = pair(ga - sh, ra, tol)
        if pr: sh = float(np.median([ga[i] - ra[j] for i, j in pr]))
    # 曲の中のずれ = 直線(録画・版のテンポの違い: 0.08% で 2 分に 100ms)+ 局所の残り。
    # 直線は「差 ±250ms の全部の組」のうち ±7.5ms に入る組がいちばん多い傾き・切片(傾き ±0.3% を 0.005% 刻み)
    rs = np.sort(ra)
    gi, rj = [], []
    for i, x in enumerate(ga):
        lo_, hi_ = np.searchsorted(rs, x - LINE_SPAN), np.searchsorted(rs, x + LINE_SPAN)
        gi += [i] * (hi_ - lo_); rj += list(range(lo_, hi_))
    gi = np.array(gi, int); rj = np.array(rj, int)
    line = (sh, 0.0)
    if len(gi):
        d = ga[gi] - rs[rj]; tt = ga[gi]; t0 = float(np.median(ga)); best_ = None
        for b in np.arange(-LINE_SLOPE_MAX, LINE_SLOPE_MAX + 1e-6, LINE_SLOPE_STEP):
            rr = d - b * (tt - t0)
            h, e = np.histogram(rr, bins=np.arange(-LINE_SPAN, LINE_SPAN + 1e-4, LINE_BIN))
            k = int(np.argmax(np.convolve(h, np.ones(3), mode='same')))
            if best_ is None or h[max(0, k - 1):k + 2].sum() > best_[0]: best_ = (h[max(0, k - 1):k + 2].sum(), b, (e[k] + e[k + 1]) / 2)
        _, b, a = best_
        rr = d - b * (tt - t0); inl = np.abs(rr - a) <= LINE_INLIER
        if inl.sum() >= 3: a = float(np.median(rr[inl]))
        line = (a, b, t0)
    fit = (lambda t: line[0] + line[1] * (t - line[2])) if len(line) == 3 else (lambda t: sh)
    # 局所の残り: 今のずれから ±40ms の組(音符ごとにいちばん近いもの)の、前後 W/2 秒の中央値。4 回繰り返してたどる
    # (録画の継ぎ目で途中から 20ms 飛ぶ動画がある)
    loc = fit(ga)
    for _ in range(LOCAL_ITERS if mode == 'local' else 0):
        pr0 = pair(ga - loc, ra, LOCAL_GATE)
        kk = np.array(sorted(i for i, _ in pr0), int); dd = {i: ga[i] - loc[i] - ra[j] for i, j in pr0}
        kt = ga[kk] if len(kk) else np.zeros(0); kd = np.array([dd[i] for i in kk]) if len(kk) else np.zeros(0)
        new = loc.copy()
        for i in range(len(ga)):
            a_, b_ = np.searchsorted(kt, ga[i] - W / 2), np.searchsorted(kt, ga[i] + W / 2)
            if b_ - a_ >= 3: new[i] = loc[i] + float(np.median(kd[a_:b_]))
        if np.allclose(new, loc, atol=1e-5): break
        loc = new
    if align is not None:
        # 別の譜面のずれを使う。曲全体の一定のずれ(清書で OFFSET を数 ms 直した分)だけは近い組の中央値で足す
        loc = np.interp(ga, align[0], align[1])
        p_ = pair(ga - loc, ra, REFIT_GATE)
        if p_: loc = loc + float(np.median([ga[i] - loc[i] - ra[j] for i, j in p_]))
    # 局所のずれで揃えて対応させる(全体のずれだけで揃えると、曲の中のずれ(録画・版の違い)が時刻の差に入り、採譜の正しさを測れない)
    al = ga - loc; pr = pair(al, ra, TOL)
    dk = [(i, j) for i, j in pr if gn[i][1] in '1234' and rn[j][1] in '1234']
    err = np.array([(al[i] - ra[j]) * 1000 for i, j in dk]) if dk else np.zeros(0)
    kind = lambda ch: ch if ch in '1234' else ('5' if ch in '567' else '8')
    tbad = [(i, j) for i, j in pr if kind(gn[i][1]) != kind(rn[j][1])]
    vbad = [(i, j) for i, j in pr if gn[i][2] and rn[j][2] and gn[i][1] != '8' and abs(gn[i][2] / rn[j][2] - 1) > SPEED_TOL] if pr else []
    gbt = np.array([b[0] for b in gb]); rbt = np.array([b[0] for b in rb])
    bp = pair(gbt - sh, rbt, TOL)
    berr = np.array([(gbt[i] - sh - rbt[j]) * 1000 for i, j in bp]) if bp else np.zeros(0)
    ae = np.abs(err)
    r = {'match': len(pr), 'ref': len(rn), 'gen': len(gn), 'recall': 100.0 * len(pr) / max(1, len(rn)), 'precision': 100.0 * len(pr) / max(1, len(gn)),
         'match_dk': len(dk), 'ref_dk': sum(1 for n in rn if n[1] in '1234'), 'gen_dk': sum(1 for n in gn if n[1] in '1234'),
         'type': len(tbad), 'speed': len(vbad), 'shift_ms': sh * 1000,
         'med': float(np.median(ae)) if len(ae) else 0, 'p95': float(np.percentile(ae, 95)) if len(ae) else 0, 'max': float(ae.max()) if len(ae) else 0,
         'gt10': int((ae > 10).sum()), 'gt25': int((ae > 25).sum()),
         'bars': len(bp), 'bars_ref': len(rb), 'bars_gen': len(gb), 'bar_max': float(np.abs(berr).max()) if len(berr) else 0,
         'miss': [ra[j] for j in sorted(set(range(len(rn))) - {j for _, j in pr})], 'extra': [al[i] for i in sorted(set(range(len(gn))) - {i for i, _ in pr})],
         'err': [(ra[j], (al[i] - ra[j]) * 1000) for i, j in dk], 'align': (ga, loc)}
    if verbose:
        print('音符 一致 %d / 参照 %d(%.1f%%)・生成 %d(%.1f%%)  ドン・カッ %d / %d・%d  種類違い %d  時刻の差 中央 %.1f / 95%% %.1f / 最大 %.1f ms  >10ms %d  >25ms %d  速さ>3%%違い %d  全体のずれ %+.1f ms' % (
            r['match'], r['ref'], r['recall'], r['gen'], r['precision'], r['match_dk'], r['ref_dk'], r['gen_dk'], r['type'], r['med'], r['p95'], r['max'], r['gt10'], r['gt25'], r['speed'], r['shift_ms']))
        print('小節線 一致 %d / 参照 %d・生成 %d  最大の差 %.1f ms' % (r['bars'], r['bars_ref'], r['bars_gen'], r['bar_max']))
        if r['miss']: print('  取りこぼし(音源 s):', ' '.join('%.2f' % x for x in r['miss'][:30]))
        if r['extra']: print('  余分(音源 s):', ' '.join('%.2f' % x for x in r['extra'][:30]))
    return r


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    r = compare(a[0], a[1], a[2] if len(a) > 2 else None)
    if '--list' in sys.argv:
        for t, e in r['err']:
            if abs(e) > 10: print('  %.3f  %+.1f ms' % (t, e))
