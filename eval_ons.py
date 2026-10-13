"""検出オンセットJSON(動画秒, 判定枠到達時刻)を正解TJAと照合して精度を出す。
使い方: eval_ons.py <ons.json> [course=Edit] [vid_offset=309.254] [-m]
  正解の動画時刻 = chart_t + vid_offset。8秒窓ごとに ±0.25s の局所シフトで整列(fpsドリフト対策)。
  整列スコアは最近傍誤差のソフトスコア(16分の連続でも±半音価のプラトーで迷わない)。
  正解は検出区間 [start+1.2, end] 内のものだけ数える(区間外は見えないので)。
  密度別(隣接IOI)の recall も出す。
"""
import sys, json, numpy as np
from eval_timing import parse_tja_times
import os
TJA = os.environ.get("TJA", "")   # 環境変数 TJA で曲を切替

def load_ons(path):
    d = json.load(open(path, encoding='utf-8'))
    rng = None
    if isinstance(d, dict):
        if 'start' in d: rng = (d['start'], d['end'])
        d = d['onsets']
    ons = [o for o in d if o['color'] in ('don', 'ka')]
    t = np.array([o['t'] for o in ons])
    c = np.array([('3' if o['big'] else '1') if o['color'] == 'don' else ('4' if o['big'] else '2') for o in ons])
    return t, c, rng

def nearest_err(mine, b):
    j = np.searchsorted(b, mine)
    j0 = np.clip(j - 1, 0, len(b) - 1); j1 = np.clip(j, 0, len(b) - 1)
    e0 = np.abs(b[j0] - mine); e1 = np.abs(b[j1] - mine)
    return np.minimum(e0, e1)

def bestshift(mine, tt, base, span, tol=0.04, step=0.002, near=0.0):
    # 点が最良から near 以内のずれのうち base に最も近いものを採る(near=0 なら最良。同点なら base に近い方)
    shs = np.arange(base - span, base + span, step)
    sc_ = np.array([(np.clip(1 - nearest_err(mine, tt + sh) / tol, 0, None) ** 2).sum() for sh in shs])
    ok_ = np.flatnonzero(sc_ >= sc_.max() - near - 1e-9)
    return float(shs[ok_[np.argmin(np.abs(shs[ok_] - base))]])


def evaluate(mt, mc, tt, tc, vid_offset, rng=None, W=8.0, tol=0.035, verbose=True):
    """検出 (mt, mc) と正解 (tt, tc) の照合。8秒窓ごとに局所シフトを求めてから、全体を一度に貪欲対応付けする
    (窓ごとに対応付けると窓の境界をまたぐ音符が取りこぼしに数えられる)。"""
    if len(mt) == 0:
        print('no onsets'); return
    if rng is not None:
        tlo, thi = rng[0] + 1.2, rng[1]
    else:
        tlo, thi = mt.min() - 0.1, mt.max() + 0.1
    g0 = bestshift(mt, tt, vid_offset, 0.6)
    # 1) 窓ごとのシフト → 各検出に「その窓のシフト」を割り当て、正解側の時刻に揃える
    shifts = []; t0 = tlo; sh = g0
    while t0 < thi:
        t1 = min(t0 + W, thi)
        win = mt[(mt >= t0) & (mt < t1)]
        # 窓ごとのずれは曲全体のずれ g0 の ±80ms だけ探す(±250ms だと、繰り返しの多いリズムで 1 拍隣に合わせてしまい、
        # 夏祭り(新音源)は全音符 12ms 以内で一致しているのに 66% と出た。動画の揺れで窓ごとに動くのは数十 ms まで)
        if len(win) >= 3: sh = bestshift(win, tt, g0, 0.08)
        shifts.append((t0, t1, sh)); t0 = t1
    def shift_at(t):
        for a, b_, s_ in shifts:
            if a <= t < b_: return s_
        return shifts[-1][2] if t >= shifts[-1][1] else shifts[0][2]
    # 窓ごとのずれで揃えた照合と、曲全体のずれ g0 だけで揃えた照合の両方を行い、対応の多い方を採る。等間隔の列では窓の端の音符 1 個の数え方で
    # 1 音符ぶん(+78ms)ずれた窓のずれが勝ち、全部一致しているのに取りこぼし 3 + 余分 3 と数えていた(朱の旋律 110〜118s。基準のずれが 0.3ms
    # 違うだけで結果が変わった)。窓ごとのずれは、録画の揺れで曲の中のずれが数十 ms 動く動画に要る
    best_ = None
    for mt_al in (np.array([x - shift_at(x) for x in mt]), mt - g0):
        used_d, used_t = _pair(mt_al, tt, tol)
        if best_ is None or len(used_d) > len(best_[1]): best_ = (mt_al, used_d, used_t)
    mt_al, used_d, used_t = best_
    return _summary(mt, mc, tt, tc, mt_al, used_d, used_t, g0, tlo, thi, verbose)


def _pair(mt_al, tt, tol):
    """全体で貪欲対応付け(誤差の小さいペアから)。戻り値: ({検出: 参照}, {参照: 検出})"""
    pairs = []
    for i, x in enumerate(mt_al):
        j = np.searchsorted(tt, x)
        for jj in (j - 2, j - 1, j, j + 1):
            if 0 <= jj < len(tt) and abs(tt[jj] - x) <= tol: pairs.append((abs(tt[jj] - x), i, jj))
    pairs.sort()
    used_d = {}; used_t = {}
    import bisect
    order_i = []; order_j = []            # 採用した対(検出の番号順)。対応が交差する割り当ては採らない
    for e, i, jj in pairs:
        if i in used_d or jj in used_t: continue
        # 順番の入れ替わり(検出 i < i' なのに参照 j > j')を作る対は捨てる。40ms 間隔の組に窓のずれが 25ms 乗ると、
        # 誤差の小さい順の貪欲では交差して割り当て、取りこぼし+余分と数えていた(活声ライクリフド 144.9s)
        k = bisect.bisect_left(order_i, i)
        if (k > 0 and order_j[k - 1] > jj) or (k < len(order_i) and order_j[k] < jj): continue
        order_i.insert(k, i); order_j.insert(k, jj)
        used_d[i] = jj; used_t[jj] = i
    return used_d, used_t


def _summary(mt, mc, tt, tc, mt_al, used_d, used_t, g0, tlo, thi, verbose):
    # 3) 集計(正解は検出区間内のものだけ)
    tt_vid = tt + g0
    in_range = (tt_vid >= tlo) & (tt_vid < thi)
    tot = dict(true=int(in_range.sum()), matched_true=0, det=len(mt), matched_det=len(used_d), type_ok=0, big_ok=0)
    err = []; misses = []; extras = [mt[i] for i in range(len(mt)) if i not in used_d]
    dens = {}
    ioi_prev = np.diff(tt, prepend=tt[0] - 9); ioi_next = np.diff(tt, append=tt[-1] + 9)
    dmin = np.minimum(ioi_prev, ioi_next)
    for i, jj in used_d.items():
        err.append(mt_al[i] - tt[jj])
        if mc[i].translate(str.maketrans('34', '12')) == tc[jj].translate(str.maketrans('34', '12')): tot['type_ok'] += 1
        if (mc[i] in '34') == (tc[jj] in '34'): tot['big_ok'] += 1
    for j in np.where(in_range)[0]:
        k = '16th+' if dmin[j] >= 0.06 else ('24th' if dmin[j] >= 0.04 else ('32nd' if dmin[j] >= 0.03 else '48th+'))
        dens.setdefault(k, [0, 0]); dens[k][0] += 1
        if j in used_t: dens[k][1] += 1; tot['matched_true'] += 1
        else: misses.append(tt_vid[j])
    err = np.array(err) if err else np.zeros(1)
    rec = tot['matched_true'] / max(1, tot['true']) * 100; prec = tot['matched_det'] / max(1, tot['det']) * 100
    typ = tot['type_ok'] / max(1, tot['matched_det']) * 100; bigp = tot['big_ok'] / max(1, tot['matched_det']) * 100
    if verbose:
        print('det %d  true(in range %.1f-%.1f) %d' % (tot['det'], tlo, thi, tot['true']))
        print('recall %.1f%%  precision %.1f%%  type(color) %.1f%%  big/small %.1f%%  timing sd %.1fms mean %.1fms' % (rec, prec, typ, bigp, err.std() * 1000, err.mean() * 1000))
        for k in ('16th+', '24th', '32nd', '48th+'):
            if k in dens: print('  density %-6s true %4d  recalled %4d  (%.1f%%)' % (k, dens[k][0], dens[k][1], dens[k][1] / dens[k][0] * 100))
    return rec, prec, typ, misses, extras

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    path = args[0]; crs = args[1] if len(args) > 1 else 'Edit'
    vo = float(args[2]) if len(args) > 2 else 309.254
    bpm, tn = parse_tja_times(TJA, crs)
    tt = np.array([n[0] for n in tn]); tc = np.array([n[1] for n in tn])
    mt, mc, rng = load_ons(path)
    r = evaluate(mt, mc, tt, tc, vo, rng)
    if r and '-m' in sys.argv:
        print('misses (video s):', ' '.join('%.2f' % m for m in r[3][:80]))
        print('extras (video s):', ' '.join('%.2f' % m for m in r[4][:80]))
