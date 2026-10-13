"""オンセット列 → BPM/オフセット推定 → グリッド量子化 → TJA。
使い方: make_tja.py <video> <start> <end> <out.tja> [TITLE] [bpm_lo bpm_hi]
音符種: don->1 ka->2 大don->3 大ka->4 renda(黄)->5..8(連打), balloon(橙)->7..8。
自動生成なので連打の長さ・分岐・BPM変化は要手直し。
"""
import sys, os, numpy as np
# 既定は方式2(列プロファイル)。DETECTOR=1 で旧ブロブ追跡方式。
if os.environ.get('DETECTOR') == '1':
    import detect
else:
    import detect2 as detect

def _phase_and_score(times, bpm):
    # 16分(拍/4)と3連8分(拍/3)の両グリッドで測り、良い方を採用(3連主体の曲にも対応)
    best = None
    for div in (4, 3):
        step = 60.0 / bpm / div
        ang = 2 * np.pi * times / step
        phase = np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()) / (2 * np.pi) * step
        res = np.abs(((times - phase + step / 2) % step) - step / 2)
        tol = 0.13 * step
        score = (res < tol).mean() * (1.0 if div == 4 else 0.97)   # 同点は16分(binary)優先
        key = (round(score, 4), -res.mean() / step)
        if best is None or key > best[0]:
            best = (key, phase, score, res.mean() / step)
    return best[1], best[2], best[3]

def bpm_from_bars(bars):
    """小節線の通過間隔から小節長 → BPM(4/4想定)。間隔は小節長の整数倍になるので最頻の基本周期を取る。"""
    bars = np.sort(np.asarray(bars, float)); d = np.diff(bars); d = d[(d > 0.3) & (d < 4.0)]
    if len(d) < 4: return None
    P = float(np.median(d))
    k = np.round(d / P); keep = (k >= 1) & (np.abs(d - k * P) < 0.06 * P)
    if keep.sum() < 4: return None
    P = float(np.sum(d[keep] * k[keep]) / np.sum(k[keep] ** 2))
    return 240.0 / P

def estimate_bpm(times, lo=115.0, hi=300.0, hint=None):
    """IOIヒストグラムの最頻ピーク(音価)を起点にBPM候補を作り、16分整合で選ぶ。"""
    times = np.sort(np.asarray(times))
    ioi = np.diff(times); ioi = ioi[(ioi > 0.04) & (ioi < 0.6)]
    if len(ioi) < 8:
        # 疎すぎるとき: 総当たり
        cand = np.arange(lo, hi, 0.1)
    else:
        # ±6% 窓で最も密なIOI値=最頻音価
        peak, pc = None, -1
        for p in np.arange(0.05, 0.45, 0.001):
            c = ((ioi > p * 0.94) & (ioi < p * 1.06)).sum()
            if c > pc: pc, peak = c, p
        # ピークが 16分/12分/8分/付点/4分 のどれか → 4分に直してBPM
        cand = []
        for m in (1, 1.5, 2, 3, 4, 6):
            bpm = 60.0 / (peak * m)
            if lo <= bpm <= hi:
                cand.append(bpm)
        cand = np.array(sorted(set(round(b, 2) for b in cand)) or [60.0 / (peak * 4)])
    # 各候補をまず局所精密化(±3.5%)してからスコア比較 → 正しいオクターブが勝つ
    results = []
    for bpm_c in cand:
        loc = None
        for bpm in np.arange(bpm_c * 0.94, bpm_c * 1.06, 0.02):
            ph, sc, ne = _phase_and_score(times, bpm)
            key = (round(sc, 4), -ne)
            if loc is None or key > loc[0]:
                loc = (key, bpm, ph)
        results.append((loc[0], loc[1], loc[2], bpm_c))
    results.sort(reverse=True)
    # 小節線から出したBPM(hint)があれば、その±3%にある候補を優先(3連主体の曲で 3/4 倍のBPMに化けるのを防ぐ)
    if hint is not None:
        near = [r for r in results if abs(r[1] - hint) / hint < 0.03]
        if near:
            print('小節線周期による BPM %.2f に合う候補 %.2f を採用(音符だけの最良は %.2f)' % (hint, near[0][1], results[0][1]))
            results = near + [r for r in results if r not in near]
        else:
            # 音符の最頻音価からは出てこない BPM でも、小節線の周期の BPM で音符の格子がほぼ同じ良さで合うならそちらを採る
            # (ドンカマ2000: 32 分主体で 135 が出たが小節は 2.0s = 4/4 の 120。彁は 62.5 だと格子が合わないので採らない)
            loc = None
            for bpm in np.arange(hint * 0.97, hint * 1.03, 0.02):
                ph, sc, ne = _phase_and_score(times, bpm); key = (round(sc, 4), -ne)
                if loc is None or key > loc[0]: loc = (key, bpm, ph)
            if loc is not None and results and loc[0][0] >= 0.9 * results[0][0][0]:
                print('小節線周期による BPM %.2f(格子スコア %.3f)を採用(音符だけの最良 %.2f はスコア %.3f)' % (loc[1], loc[0][0], results[0][1], results[0][0][0]))
                results = [(loc[0], loc[1], loc[2], hint)] + results
    import os
    if os.environ.get('BPMDBG'):
        print('  peak %.4f  精密化後: %s' % (peak if len(ioi) >= 8 else -1,
              [(round(r[1], 2), r[0][0]) for r in results]))
    return results[0][1], results[0][2]

def choose_octave(times, bpm, phase):
    return bpm

def bar_phase(bars, bpm):
    """小節線の判定枠通過時刻から、小節頭の位相(mod 小節長)を円平均で決める。信頼度(0-1)も返す。"""
    bars = np.asarray(bars, float)
    measure = 240.0 / bpm
    if len(bars) < 3: return None, 0.0
    ang = 2 * np.pi * bars / measure
    ph = np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()) / (2 * np.pi) * measure
    r = float(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()))
    return ph % measure, r

def _body_tokens(body):
    """小節の本文を「命令行」と「セル 1 文字」に分ける"""
    toks = []; i = 0
    while i < len(body):
        if body[i] == '#':
            j = body.find(chr(10), i)
            if j < 0: j = len(body)
            toks.append(body[i:j]); i = j + 1
        elif body[i] == chr(10):
            i += 1
        else:
            toks.append(body[i]); i += 1
    return toks


def _cell_count(body):
    return sum(1 for t in _body_tokens(body) if not t.startswith('#'))


def insert_at_cell(body, cell, text):
    """body の cell 番目のセルの直前に命令行 text を入れる(cell がセル数以上なら末尾)"""
    toks = _body_tokens(body)
    n = 0; pos = len(toks)
    for i, t in enumerate(toks):
        if t.startswith('#'): continue
        if n == cell: pos = i; break
        n += 1
    toks.insert(pos, text)
    out = ''
    for t in toks:
        if t.startswith('#'):
            out += ('' if (not out or out.endswith(chr(10))) else chr(10)) + t + chr(10)
        else:
            out += t
    return out


def _measure_lines(m, beats, L, scrolls=None, cur_scroll=None, cmds=None, t0=0.0):
    """1小節ぶんの音符 (frac, kind, big) を TJA のセル文字列にする。beats=拍数、L=小節長(秒)。
    セル数の候補は 拍×{4,3,6,12,8,16}(16分,3連,24分,48分,32分,64分)のうち整数になるもの。
    scrolls(音符ごとの SCROLL)が渡されたら、値が変わる音符の前に #SCROLL 行を挟む(連打だけ速い仕掛けなど)。"""
    if not m:
        if scrolls is None: return ''
        if not cmds: return '', cur_scroll
        # 音符の無い小節にゴーゴーなどの命令が来たら、拍×12 のセルに分けて時刻どおりの位置に置く
        # (小節の先頭に置くと最大 1 小節ずれる: 倍倍FIGHT! のゴーゴーが 1.5 秒早まった)
        cells = max(1, int(round(beats * 12)))
        body = '0' * cells
        for t, txt in cmds: body = insert_at_cell(body, max(0, min(cells, int(round((t - t0) / L * cells)))), txt)
        return body, cur_scroll
    cands = []
    for k in (4, 3, 6, 12, 8, 16, 24, 48):       # 16分,3連,24分,48分,32分,64分, 24分+32分(拍/24), 拍/48
        c = beats * k
        if abs(c - round(c)) < 1e-6 and round(c) >= 1: cands.append(int(round(c)))
    if not cands: cands = [max(1, int(round(beats * 4)))]
    TOL_BEAT = min(0.03, 0.006 * beats / L)      # 6ms 以内(拍の 3% だと 128 BPM で 14ms: 雲間 204.15s の 24 分+32 分の小節が 64 セルに乗って +13ms)
    prec = [f for f, k, _ in m if k in ('don', 'ka')] or [f for f, _, _ in m]
    best = None
    for mult in (1, 2, 3, 6):
        for cells in cands:
            worst = max(abs(f * cells - round(f * cells)) / cells * beats for f in prec)
            if worst <= TOL_BEAT * mult: best = cells; break
        if best is not None: break
    if best is None:
        best = min(cands, key=lambda cells: sum(abs(f * cells - round(f * cells)) / cells for f, _, _ in m))
    # 連打の頭・終端(5/6/7/8)は上の判定に入っていない(don/ka だけ)。粗い格子に丸めると終端が数十 ms ずれるので
    # (メヌエットの 8 は本家 35/48 小節、16 分割だと 33/48 に丸まって −71ms)、細かい候補に上げる。
    # 閾値は 40ms: 検出した連打の端は 1 個ずつ ±30ms ほどぶれるので、20ms だとぶれに忠実になって
    # ツンデレの終端 +31→+70ms、TDM の風船の頭 +16→+41ms と悪化した。40ms なら 8 曲で悪化ゼロ
    # (以前(2026-09-20)は検出の終端自体が 28〜55ms 遅れていたので逆効果で戻した。終端の位置を直したあとに入れ直した)
    roll_f = [f for f, k, _ in m if k not in ('don', 'ka')]
    RG = float(os.environ.get('ROLL_GRID_MS', '40')) / 1000.0
    # ずれは実際に置けるセルで測る。小節の終わりの直前の端は次の小節線(セル cells)に丸まっても最後のセル(cells − 1)に置かれるので、
    # 次の小節線までの差で測ると細かい格子に上げ損ねた(メカデス。裏 50.53s: 大連打の終わりが小節線の 26ms 前で、16 分割の最後のセルに置かれて −62ms。本家は 48 分)
    def _cell(f, cells): return min(cells - 1, max(0, int(round(f * cells))))
    if roll_f and RG > 0:
        def _roll_err(cells): return max(abs(f * cells - _cell(f, cells)) for f in roll_f) / cells * L
        if _roll_err(best) > RG:
            for cells in sorted(c for c in cands if c > best and c % best == 0):     # don/ka の位置はそのまま保てる倍数だけ
                if _roll_err(cells) <= RG: best = cells; break
        # 連打の頭(5/6)は位置が正確(±10ms)なので 20ms を超えたら、12ms 以内に収まる倍数の格子に上げる
        # (Dimpy Dimper 107.97s: 24 分の小節の 1.5 セル目の頭が 2 セル目に丸まって +31ms。本家は 48 分)
        head_f = [f for f, k, _ in m if k == 'renda']
        RH = float(os.environ.get('ROLL_HEAD_MS', '20')) / 1000.0
        if head_f and RH > 0:
            def _head_err(cells): return max(abs(f * cells - _cell(f, cells)) for f in head_f) / cells * L
            if _head_err(best) > RH:
                for cells in sorted(c for c in cands if c > best and c % best == 0):
                    if _head_err(cells) <= 0.012 and _roll_err(cells) <= max(RG, _roll_err(best)): best = cells; break
    # 違う音符・連打の端が同じセルに入るなら細かい候補に上げる(resume:story 裏 20.6s: 45ms の連打の頭と終わりが 16 分の同じセルに入り、
    # 終わりで上書きされて連打が消えた)
    def _collide(cells): return len({min(cells - 1, max(0, int(round(f * cells)))) for f, _, _ in m}) < len(m)
    pos = None
    if _collide(best):
        coarse = best
        for cells in sorted(c for c in cands if c > best and c % best == 0):
            if not _collide(cells): best = cells; break
    slots = ['0'] * best; sc_at = {}
    for j, (f, k, big) in enumerate(m):
        idx = pos[j] if pos is not None else min(best - 1, max(0, int(round(f * best))))
        slots[idx] = symbol(k, big)
        if scrolls is not None and scrolls[j] is not None and k != 'roll_end': sc_at[idx] = scrolls[j]
    if scrolls is None: return ''.join(slots)
    out = ''; cur = cur_scroll
    for idx, ch in enumerate(slots):
        if idx in sc_at and abs(sc_at[idx] - cur) >= 0.03:
            out += (chr(10) if out else '') + '#SCROLL %g' % sc_at[idx] + chr(10); cur = sc_at[idx]
        out += ch
    for t, txt in (cmds or []):       # ゴーゴーなどの命令。この小節は等間隔なので時刻の比でセルが決まる
        out = insert_at_cell(out, max(0, min(best, int(round((t - t0) / L * best)))), txt)
    return out, cur

EXACT_MS = 5.0   # 選んだ候補でも音符の最大残差がこれを超える小節は、音符ごとに #BPMCHANGE を入れて時刻を正確に再現する

def _measure_lines_exact(m, beats, L, vels, t0, t_head, k_speed, cur_bpm, cur_scroll, bar_vel=None, cmds=None):
    """テンポが小節の中で変わる(ラストの減速など)か、どのグリッドにも乗らない小節の書き方。
    セル位置は「音符自身の速さから求めた局所 BPM」で拍に換算して決め、隣り合う音符の間の BPM は
    実測時刻がぴったり合う値にする(公式譜面の「BPM を刻んで間隔を作る」表現と同じ)。
    見た目の速さは #SCROLL = V/(K·区間BPM) で保つ。小節頭の区間は TJA の累積時刻 t_head から始めて、
    次の小節線ちょうどで終わるようにする。戻り値: (本文, 最後の区間BPM, cur_scroll, BPM変更回数)"""
    cells = int(round(beats * 24))
    ev = sorted([(t0 + f * L, k, big, (v if v else None)) for (f, k, big), v in zip(m, vels)], key=lambda e: e[0])
    if not ev:
        B = beats * 60.0 / max(t0 + L - t_head, 1e-3)
        out = ('#BPMCHANGE %.2f' % B + chr(10)) if abs(B - cur_bpm) > 0.05 else ''
        out = out + '0' * cells
        for t, txt in (cmds or []): out = insert_at_cell(out, max(0, min(cells, int(round((t - t0) / L * cells)))), txt)
        return out, (B if out else cur_bpm), cur_scroll, (1 if out else 0)
    Bloc = []; last = cur_bpm
    for e in ev:
        if e[3] and e[1] != 'roll_end': last = e[3] / k_speed
        Bloc.append(last)
    pos = []; acc = 0.0; tprev = t0; Bprev = Bloc[0]
    for j, e in enumerate(ev):
        acc += (e[0] - tprev) * Bprev / 60.0; pos.append(acc); tprev = e[0]; Bprev = Bloc[j]
    total = acc + (t0 + L - tprev) * Bprev / 60.0
    scale = beats / total if total > 0 else 1.0
    idx = []
    for q in pos:
        c = int(round(q * scale * 24)); c = max(c, (idx[-1] + 1) if idx else 0); idx.append(min(c, cells - 1))
    # 小節頭(cell 0)に置けるのは小節線ちょうどの音符だけ。それより後の音符を cell 0 に置くと、小節線→音符の区間が無くなって
    # 音符の時刻ぶんだけ小節全体が後ろへずれ、以後の小節に蓄積する(彁 裏: 小節線の直後の低速大ドンで 3〜20ms/小節)
    if idx and ev[0][0] > t_head + 0.004 and idx[0] == 0:   # 4ms 以内は小節頭に置く(境界は t_head のままで蓄積しない。1ms だと 1〜2ms 後ろの音符に最小区間 5ms が入り、VERTeX 冒頭の速い大ドンが小節線の 17px 右に見えた)
        for j in range(len(idx)):
            if j == 0 or idx[j] <= idx[j - 1]: idx[j] = (idx[j - 1] + 1) if j else 1
            else: break
    for j in range(len(idx) - 2, -1, -1):
        if idx[j] >= idx[j + 1]: idx[j] = idx[j + 1] - 1
    bounds = [(0, t_head)]
    for c, e in zip(idx, ev):
        if c == bounds[-1][0]: bounds.append((c, bounds[-1][1]))     # 小節頭(cell 0)の音符: 境界は小節頭 t_head のまま(音符の時刻にすると小節線→音符の差だけ小節が短くなり、Ultraman X で 80 小節 × 0.8ms = 7.5ms 蓄積)
        else: bounds.append((c, max(e[0], bounds[-1][1] + 0.005)))
    bounds.append((cells, t0 + L))
    seg = []
    for (c0, ta), (c1, tb) in zip(bounds[:-1], bounds[1:]):
        seg.append(((c1 - c0) / 24.0 * 60.0 / max(tb - ta, 0.001)) if c1 > c0 else None)   # 下限 1ms(20ms だと小節線直後の音符で小節全体が十数 ms 遅れる)
    if os.environ.get('BUILDDBG'): print('      exact: t_head %.3f t0 %.3f L %.3f cells %d idx %s seg %s' % (t_head, t0, L, cells, idx, [round(s, 1) if s else None for s in seg]))
    slots = ['0'] * cells; at = {}
    for j, (c, e) in enumerate(zip(idx, ev)):
        slots[c] = symbol(e[1], e[2]); at[c] = j
    st = {'out': '', 'bpm': cur_bpm, 'sc': cur_scroll, 'n': 0}
    def emit_bpm(B):
        if B is not None and abs(B - st['bpm']) > 0.005:
            st['out'] += ('' if (not st['out'] or st['out'].endswith(chr(10))) else chr(10)) + '#BPMCHANGE %.3f' % B + chr(10); st['bpm'] = B; st['n'] += 1
    emit_bpm(seg[0])
    if bar_vel and st['bpm']:
        # 小節線は小節頭の SCROLL で流れる。検出した小節線の速度に合わせる(先頭の音符が cell 0 に無いときだけ独立に指定できる)
        s = round(bar_vel / (k_speed * st['bpm']), 3)
        if abs(s - st['sc']) >= 0.005 and 0 not in at:
            st['out'] += '#SCROLL %g' % s + chr(10); st['sc'] = s
    for c in range(cells):
        if c in at:
            j = at[c]; emit_bpm(seg[j + 1]); v = ev[j][3]
            if v and ev[j][1] != 'roll_end':
                s = round(v / (k_speed * st['bpm']), 3)     # 寄せない: 区間 BPM が音符ごとに違うので 0.25 刻みに寄せると速さが最大 3% ずれる
                if abs(s - st['sc']) >= 0.005:
                    st['out'] += ('' if (not st['out'] or st['out'].endswith(chr(10))) else chr(10)) + '#SCROLL %g' % s + chr(10); st['sc'] = s
        st['out'] += slots[c]
    if cmds:
        # セル位置は時刻に比例しない(区間ごとに BPM が違う)。bounds の (セル, 時刻) の間を線形に見る
        def cell_of(t):
            if t <= bounds[0][1]: return 0
            for (c0, ta), (c1, tb) in zip(bounds[:-1], bounds[1:]):
                if ta <= t <= tb: return c0 if tb <= ta else c0 + (c1 - c0) * (t - ta) / (tb - ta)
            return cells
        body = st['out']
        for t, txt in cmds: body = insert_at_cell(body, max(0, min(cells, int(round(cell_of(t))))), txt)
        st['out'] = body
    return st['out'], st['bpm'], st['sc'], st['n']

K_SPEED = 0.0988   # 音符の速さ [px/frame] = K_SPEED × BPM × SCROLL(1080p アーケード配置・60fps で3曲とも同値)

def drift_ms(fracs, beats, L):
    """小節内の「ずれの傾き」(先頭から末尾までのずれの変化, ms)。小節の長さの取り違え(隠れた小節線で 2 つの小節が
    1 つに見えている)やテンポ変化では、ずれが位置に比例して増える。録画のゆらぎは位置と無関係なので傾きは小さい"""
    cells = max(1, int(round(beats * 24)))
    if len(fracs) < 4: return 0.0
    f = np.asarray(sorted(fracs), float)
    res = (f * cells - np.round(f * cells)) / cells * L * 1000
    return abs(float(np.polyfit(f, res, 1)[0])) * (f[-1] - f[0])

def _nice_beats(x, tol):
    """x 拍が 1/4 拍刻み(または 1/8 拍刻み)の「きれいな」拍数に近ければその値、そうでなければ None"""
    for step in (0.25, 0.125):
        r = round(x / step) * step
        if abs(x - r) <= tol and r >= 0.5: return r
    return None

def _snap_scroll(s):
    """SCROLL 値の丸め: 0.25 の倍数に 0.035 以内なら寄せる(1.03→1, 1.99→2, 3.03→3, 4.52→4.5)。それ以外は2桁"""
    q = round(s * 4) / 4
    return q if abs(s - q) <= 0.035 else round(s, 2)

def _meter_of(beats):
    n64 = int(round(beats * 16))
    if n64 % 16 == 0: return (n64 // 16, 4)
    if n64 % 8 == 0: return (n64 // 8, 8)
    if n64 % 4 == 0: return (n64 // 4, 16)
    if n64 % 2 == 0: return (n64 // 2, 32)
    return (n64, 64)

def build_var(times, kinds, bigs, vels, bpm, bars, k_speed=None, bar_vels=None, gogo=None):
    """小節線をそのまま小節境界にし、小節ごとに BPM・拍子・SCROLL を決める。
    使う情報: 小節長 L(小節線の間隔)と、その小節の音符の流れる速さ V(= K × BPM × SCROLL)。
    候補: c1 直前の BPM を維持(拍数が 1/4 拍刻みで割り切れれば) → SCROLL = V/(K·BPM)
          c2 主BPM を維持
          c3 速さから BPM = V/K (SCROLL=1 と仮定) → 拍数が割り切れれば BPM を小節長から精密化
          c4 4/4 で BPM = 240/L(最後の手段)
    音符が候補のグリッド(拍/12)に乗る度合い(最大残差 ms)で決め、差が 2ms 以内なら c1>c2>c3>c4 の順(連続性優先)。
    戻り値: offset, 行のリスト(#MEASURE/#BPMCHANGE/#SCROLL を含む)"""
    bars_a = np.asarray(bars, float); order = np.argsort(bars_a); bars = bars_a[order]
    bv = np.asarray(bar_vels, float)[order] if (bar_vels is not None and len(bar_vels) == len(bars)) else None
    vels = np.asarray(vels, float)
    # 0) 小節線の時刻を音符に合わせる: 小節線の中心(明るい列の中央)は音符の中心(右端−半径)より 2.5〜2.8px 右に出るので、
    #    小節頭の音符が小節線より 2〜3ms 早く検出される(6 曲+3 曲すべてで同じ向き)。小節頭 ±15ms の音符との差の中央値で小節線側を補正する
    #    (格子の小節が音符ごとの小節と 2.5ms 段違いになるのと、その残差で不要に exact になるのを防ぐ)
    t_dk = np.array([t for t, k in zip(times, kinds) if k in ('don', 'ka')])
    if len(t_dk) and len(bars):
        dd = []
        for b in bars:
            k = int(np.argmin(np.abs(t_dk - b)))
            if abs(t_dk[k] - b) < 0.015: dd.append(t_dk[k] - b)
        if len(dd) >= 10:
            sh = float(np.median(dd)); bars = bars + sh
            print('小節線の時刻を音符に合わせて %+.1fms 補正(小節頭の音符 %d 個の中央値)' % (sh * 1000, len(dd)))
    if k_speed is None:
        # 速さ→BPM×SCROLL の定数は基準配置(キャリブレーション後)なら K_SPEED で固定。「音符の速さの中央値 / 推定 BPM」は
        # 主な層が SCROLL 1 のときしか合わない(彁: 中央値 24.6px/frame ÷ 208 BPM = 0.118 になり、全 SCROLL が 16% 遅く出た)
        k_speed = float(os.environ.get('K_SPEED', K_SPEED))
    # 1) 小節線の掃除と補間
    iv = np.diff(bars)
    gmed = float(np.median(iv[iv > 0.3])) if (iv > 0.3).any() else float(np.median(iv))
    mask = bars <= times[-1] + 1.5 * gmed                 # 曲が終わった後(リザルト演出など)の小節線は捨てる
    bars = bars[mask]; bv = bv[mask] if bv is not None else None
    iv = np.diff(bars)
    # 飾り小節線(resume:story: 1/64 拍子の極小小節を並べて小節線を違う速度で流す演出)は、短い間隔(0.045 秒以上・普通の小節の 45% 未満)が
    # 4 つ以上連続する所だけ「極小小節」として残す。孤立した短い間隔(偽線)は従来どおり捨てる
    locs = []
    for i in range(1, len(bars)):
        win = np.concatenate([iv[max(0, i - 6):i - 1], iv[i:i + 6]]); win = win[win > 0.3]   # 自分の間隔は除く(欠けで長くなった間隔が自分を「普通」にしてしまう)
        locs.append(float(np.median(win)) if len(win) else gmed)
    short = [(iv[i - 1] < 0.45 * locs[i - 1]) and iv[i - 1] >= 0.045 for i in range(1, len(bars))]
    micro = [False] * len(bars); nominal = [None] * len(bars)
    j = 0
    while j < len(short):
        if short[j]:
            k = j
            while k + 1 < len(short) and short[k + 1]: k += 1
            if k - j + 1 >= 4:
                gaps = np.array([iv[q] for q in range(j, k + 1)])
                nom = float(np.min(gaps[gaps >= 0.045])) if (gaps >= 0.045).any() else float(np.median(gaps))
                # 極小小節の基本間隔 = ランの中で最も短い間隔(欠けた線があると 2 倍・4 倍の間隔が混じる)
                near = gaps[np.abs(gaps / nom - 1) < 0.25]
                nom = float(np.median(near)) if len(near) else nom
                for q in range(j, k + 2): micro[q] = True; nominal[q] = nom
            j = k + 1
        else: j += 1
    keep = [(bars[0], bv[0] if bv is not None else None)]
    for i in range(1, len(bars)):
        L = bars[i] - keep[-1][0]
        loc = locs[i - 1]   # 「普通の長さ」の小節から局所基準を取る(演出の連続偽線に引きずられない)
        v_i = bv[i] if bv is not None else None
        if L < 0.45 * loc:
            if micro[i] and L >= 0.045:
                # 欠けた線の補間: 間隔が基本間隔の k 倍(k≥2)なら k−1 本を等間隔で足す(風船の絵に隠れて追跡が切れた所など)。
                # 速度は前後の線から幾何補間(本家の掃引は等比で減る)
                kk = int(round(L / nominal[i])) if nominal[i] else 1
                # 補間するのは「両隣の間隔が基本間隔」の内側の欠けだけ(ランの端にある本物の長い小節(1/4 拍子など)は埋めない)
                nb_ok = (i >= 2 and abs(iv[i - 2] / nominal[i] - 1) < 0.25) and (i < len(iv) and abs(iv[i] / nominal[i] - 1) < 0.25)
                if os.environ.get('BUILDDBG') and kk >= 2: print('  小節線の欠け? t=%.2f L=%.3f nominal=%.4f k=%d 両隣=%s' % (bars[i], L, nominal[i], kk, nb_ok))
                if kk >= 2 and abs(L / nominal[i] - kk) < 0.25 and nb_ok:
                    t0_, v0_ = keep[-1]
                    for q in range(1, kk):
                        f = q / float(kk)
                        vq = (v0_ ** (1 - f) * v_i ** f) if (v0_ and v_i and v0_ > 0 and v_i > 0) else (v_i if v_i else v0_)
                        keep.append((t0_ + f * L, vq))
                keep.append((bars[i], v_i))
            continue
        k = int(round(L / loc))
        if k >= 2 and abs(L / loc - k) < 0.12 and not micro[i]:
            base = keep[-1][0]
            for j2 in range(1, k): keep.append((base + j2 * L / k, None))
        keep.append((bars[i], v_i))
    starts = [t for t, _ in keep]; svel = [v for _, v in keep]
    # 最初の小節線より前・最後の小節線より後ろを延ばす。端の区間が極小小節(飾り小節線の掃引)なら普通の小節の長さ(中央値)で延ばす
    # (夢と希望: 冒頭は #BARLINEOFF で最初に見える小節線が掃引の 0.088 秒の小節。その長さで 13.2s まで延ばして極小小節を 110 個作り、
    #  掃引の SCROLL の平滑化も偽の小節ごと当てはめて、段々速くなる小節線が等間隔になった)
    n_pre = 0
    while starts[0] > times[0] - 0.02:
        L = starts[1] - starts[0]
        if L < 0.45 * gmed: L = gmed
        starts.insert(0, starts[0] - L); svel.insert(0, None); n_pre += 1
    t_last = max([times[-1]] + [float(b) for a, b in (gogo or [])])     # ゴーゴーの終わりが最終音符より後ろなら、そこまで空の小節を足す(第六天魔王 裏)
    while starts[-1] < t_last + 0.02:
        L = starts[-1] - starts[-2]
        if L < 0.45 * gmed: L = gmed
        starts.append(starts[-1] + L); svel.append(None)
    starts = np.array(starts)
    i0 = max(0, int(np.searchsorted(starts, times[0] + 0.008) - 1))     # 最初の音符が小節線の 8ms 以内前なら「その小節線の頭」(±3ms のジッタで空の先頭小節が付いたり消えたりしない)
    starts = starts[i0:]; svel = svel[i0:]
    n_hidden = max(0, n_pre - i0)       # 延ばした先頭の小節(動画に小節線が見えていない: 音符はあるのに小節線が検出されなかった)は #BARLINEOFF で隠す
    nmeas = len(starts) - 1
    per = [[] for _ in range(nmeas)]; pv = [[] for _ in range(nmeas)]
    TOL = 1.0 / 64
    pvel = [[] for _ in range(nmeas)]; vfirst = [None] * nmeas
    t_prev = None
    for t, k, big, v in zip(times, kinds, bigs, vels):
        # 小節線の直前の音符は小節線に寄せて次の小節の頭に置く(検出のジッタ)。ただし直前の音符との間隔の 4 割まで
        # (彁 裏: 3.84 秒の小節で 1/64 = 60ms の寄せが 24 分(48ms)の音符を次の小節に飛ばしていた)
        tol_t = min(TOL * (starts[1] - starts[0]), 0.008)     # 上限 8ms(検出のジッタは sd 2〜3ms。resume 147.39s: 本家は小節線の 10ms 前に置いた音符で、17ms の寄せが小節頭に吸って +14ms)
        if t_prev is not None: tol_t = min(tol_t, 0.4 * (t - t_prev))
        mi = int(np.searchsorted(starts, t + tol_t) - 1)
        if k in ('don', 'ka'): t_prev = t
        if not (0 <= mi < nmeas): continue
        per[mi].append(((t - starts[mi]) / (starts[mi + 1] - starts[mi]), k, big)); pvel[mi].append(v)
        if k in ('don', 'ka'): pv[mi].append(v)
        if k != 'roll_end' and v and vfirst[mi] is None: vfirst[mi] = v
    # 小節ごとの速さ(音符が無い小節は前後から)
    V = np.array([np.median(x) if x else np.nan for x in pv])
    if np.all(np.isnan(V)): V[:] = k_speed * bpm
    first = next(i for i in range(nmeas) if not np.isnan(V[i]))
    V[:first] = V[first]
    for i in range(first + 1, nmeas):
        if np.isnan(V[i]): V[i] = V[i - 1]            # 音符の無い小節は直前の速さを保持
    # 2) 小節ごとに解釈を決める
    # 拍数が n/16 拍子で表せる(拍×4 が整数: 5/8, 7/8, 9/16, 15/16, 13/8 …)なら普通の拍子とみなす。
    # 1/8 拍単位でしか表せない拍数(27/32 など)は珍しいので 8ms 分不利にする
    def is_common(beats): return abs(beats * 4 - round(beats * 4)) < 1e-6
    def worst_ms(fracs, beats, L):
        cells = max(1, int(round(beats * 24)))      # 拍/24 のグリッド(16分・3連・24分・32分・48分を含む)
        if not fracs: return 0.0
        return max(abs(f * cells - round(f * cells)) for f in fracs) / cells * L * 1000
    def med_ms(fracs, beats, L):
        """グリッドからのずれの中央値。候補選びは最悪値(worst_ms)で良いが、「グリッドに乗らない小節」の判定に最悪値を使うと
        音符の多い小節が不利になる(32 分の 60 音符なら、ゆらぎ 1.2ms でも最大は 3〜4ms になる)。中央値なら音符数に依らない"""
        cells = max(1, int(round(beats * 24)))
        if not fracs: return 0.0
        return float(np.median([abs(f * cells - round(f * cells)) for f in fracs])) / cells * L * 1000
    # 多重スクロールの仕掛け譜面(1 小節に速さの違う音符が混ざる小節が 4 割以上: 彁は 86%、普通の曲は HS の切り替わりで 8% 以下)は、
    # グリッドに乗らない小節(残差 > 3ms)を自動で音符ごとの BPM 指定(exact)にする。環境変数 EXACT_RES=<ms> で上書きできる
    multi = [max(x) / min(x) - 1 > 0.25 for x in pv if len(x) >= 2]
    frac_multi = (sum(multi) / len(multi)) if multi else 0.0
    exact_res = float(os.environ.get('EXACT_RES', '3') or 0)      # 既定 3ms: グリッドに乗らない小節は音符ごとに BPM を刻む(2026-09-16、6 曲の往復 sd ≤2.4ms)
    # exact の閾値。flat(既定) = 固定 EXACT_RES。noise = 曲の残差の中央値に合わせて上げる(録画のゆらぎが大きい動画で
    # ゆらぎを譜面に書き込まないため。ただし小節線が隠れている譜面(魔宵月: 94 小節のうち 26 本が #BARLINEOFF)では
    # 残差が「解釈の誤り」なので上げると悪化する: 往復 sd 1.1→3.0ms、20ms 超が 0→5 個。既定は flat)
    EXACT_POLICY = os.environ.get('EXACT_POLICY', 'flat')
    EXACT_K = float(os.environ.get('EXACT_K', '2.0'))
    # その曲の「普通の小節の残差」(検出のゆらぎ + グリッドの取り方)の中央値。1/4 拍刻みの拍数で小節長に合う BPM の格子で測る
    _rm = []
    for i in range(nmeas):
        if not per[i]: continue
        L_ = starts[i + 1] - starts[i]; pr_ = [f for f, k, _ in per[i] if k in ('don', 'ka')]
        if not pr_: continue
        b_ = _nice_beats(L_ * bpm / 60.0, 0.06)
        _rm.append(worst_ms(pr_, b_ if b_ else max(0.5, round(L_ * bpm / 60.0 * 4) / 4), L_))
    res_med = float(np.median(_rm)) if _rm else 0.0
    if os.environ.get('BUILDDBG'): print('  小節の残差の中央値 %.1fms(exact の閾値: 仕掛けの証拠が無い小節は %.1fms)' % (res_med, max(exact_res, EXACT_K * res_med)))
    if frac_multi > 0.4: print('多重スクロールの譜面(速さの混ざる小節 %.0f%%)' % (frac_multi * 100))
    plan = []; cur_bpm = bpm; carry = 0.0
    for i in range(nmeas):
        L = starts[i + 1] - starts[i]; v = V[i]
        prec = [f for f, k, _ in per[i] if k in ('don', 'ka')]
        # 飾り小節線の掃引: 短い小節(0.45 拍未満)か、「0.6 拍未満の短い小節が 4 つ以上並び、小節線の速さが隣と 1% 以上違う」もの。
        # resume:story の 2 つ目の掃引(0.5 拍間隔、v43.9→22.4)は 0.45 拍の条件だけだと普通の 1/8 小節に見えて速度が一定で出ていた
        beats_raw = L * cur_bpm / 60.0
        def _sweep_like(i_):
            if not (0 <= i_ < nmeas) or (starts[i_ + 1] - starts[i_]) * cur_bpm / 60.0 >= 0.6: return False
            v0 = svel[i_] if i_ < len(svel) else None; v1 = svel[i_ + 1] if i_ + 1 < len(svel) else None
            return bool(v0 and v1 and abs(v1 / v0 - 1) > 0.01)      # 掃引は 1 本ごとに 2% 前後減る(検出のぶれで 1.9% のこともある)
        sweep = beats_raw < 0.6 and sum(_sweep_like(q) for q in range(i - 3, i + 4)) >= 4
        if beats_raw < 0.45 or sweep:
            # 極小小節(飾り小節線): BPM は保ち、長さは 1/64 小節(1/16 拍)単位に丸める(丸め誤差は次へ持ち越して流れないように)。
            # 小節線の速さ = その小節頭の SCROLL なので、検出した小節線の速度から SCROLL を決める(音符は音符ごとの SCROLL で別に出る)
            n = max(1, int(round((L + carry) * cur_bpm / 60.0 * 16)))
            beats = n / 16.0
            carry = (L + carry) - beats * 60.0 / cur_bpm
            sc = (svel[i] / (k_speed * cur_bpm)) if (i < len(svel) and svel[i]) else (plan[-1][2] if plan else 1.0)
            if os.environ.get('BUILDDBG'):
                print('  m%3d start %.2f L=%.3f  極小小節 %d/64 SCROLL %.2f(小節線の速度 %s)' % (i, starts[i], L, n, sc, ('%.1f' % svel[i]) if (i < len(svel) and svel[i]) else '-'))
            plan.append([cur_bpm, beats, sc, L, 'cm', False, None, False])
            continue
        carry = 0.0
        cands = []
        nb = [starts[q + 1] - starts[q] for q in range(max(0, i - 2), min(nmeas, i + 3)) if q != i]
        L_ref = float(np.median(nb)) if len(nb) >= 2 else L
        for tag, B in (('c1', cur_bpm), ('c2', bpm)):
            if tag == 'c2' and abs(bpm - cur_bpm) < 0.05: continue
            beats = _nice_beats(L * B / 60.0, 0.06)
            if beats: cands.append((tag, B, beats, worst_ms(prec, beats, L)))
            # 1 小節だけの小さなテンポ変化(resume:story 54.4s: 小節が 18ms 長く、音符もその中で直線的に遅れる)。自分の長さだと珍しい拍数(8.125)に
            # なるが隣の小節の長さ(1.2% 以内)だと普通の拍数(8)のとき、その拍数で BPM を小節長にぴったり合わせた候補を足す(BPM は平滑化しない)
            if tag == 'c1' and abs(L / L_ref - 1) < 0.012 and (not beats or not is_common(beats)):
                b2 = _nice_beats(L_ref * B / 60.0, 0.06)
                if b2 and is_common(b2): cands.append(('c1n', b2 * 60.0 / L, b2, worst_ms(prec, b2, L)))
        Bv = v / k_speed
        beats = _nice_beats(L * Bv / 60.0, 0.12)
        if beats:
            B = beats * 60.0 / L
            cands.append(('c3', B, beats, worst_ms(prec, beats, L)))
        cands.append(('c4', 240.0 / L, 4.0, worst_ms(prec, 4.0, L)))
        # 珍しい拍数(15/16, 21/16 など)は 8ms 分不利にして、同程度なら普通の拍子を選ぶ
        scored = [(c[3] + (0.0 if is_common(c[2]) else 8.0), c) for c in cands]
        best = min(s for s, _ in scored)
        half_slot = 60.0 / max(c[1] for c in cands) / 48 * 1000       # 拍/24 グリッドの半スロット(ms) の目安
        thr = max(best + 3.5, 0.6 * half_slot)
        v_sc = vfirst[i] if vfirst[i] else (plan[-1][2] * k_speed * plan[-1][0] if plan else v)
        # テンポ変化(減速/加速)の証拠: 音符の速さが 4 小節(3 回の変化)にわたって同じ向きに 2% 以上変わり続けている
        # 区間に入っている。HS 変更は 1〜2 段の階段で戻ることが多く、拍子変更は速さが変わらないので、これで区別する
        spread = (max(pv[i]) - min(pv[i])) / float(np.median(pv[i])) if len(pv[i]) >= 2 else 0.0
        def dv(a, b): return V[b] / V[a] - 1 if 0 <= a < nmeas and 0 <= b < nmeas else 0.0
        tempo_evi = False
        def step_ok(a):   # 小節 a→a+1 が「BPM だけ変わった」形か: 速さが 2% 以上・小節長が 1% 以上変わり、拍数×SCROLL(∝ L×V)はほぼ保たれる
            if not (0 <= a and a + 1 < nmeas): return False       # (HS の階段は小節長が変わらず、拍子変更は速さが変わらない)
            La, Lb = starts[a + 1] - starts[a], starts[a + 2] - starts[a + 1]
            return abs(dv(a, a + 1)) > 0.02 and abs(Lb / La - 1) > 0.01 and abs((Lb * V[a + 1]) / (La * V[a]) - 1) < 0.15
        for a in range(i - 3, i + 1):
            d3 = [dv(a, a + 1), dv(a + 1, a + 2), dv(a + 2, a + 3)]
            if all(step_ok(a + k) for k in range(3)) and (all(x > 0 for x in d3) or all(x < 0 for x in d3)): tempo_evi = True; break
        if tempo_evi:
            # 減速中は公式譜面も「拍子と SCROLL は保ち、BPM を刻む」書き方なので、自然さ(4/4 か直前と同じ拍子、
            # SCROLL≈1、珍しくない拍数)で選ぶ。グリッド残差は後で音符ごとの BPM 指定で吸収するので閾値は緩める
            prev_beats = plan[-1][1] if plan else 4.0
            def nat(c):
                return (0 if c[2] in (4.0, prev_beats) else 1) + (0 if abs(v_sc / (k_speed * c[1]) - 1) <= 0.12 else 1) + (0 if is_common(c[2]) else 2)
            thr_t = max(best + 8.0, 12.0)
            chosen = min((nat(c), n, c) for n, (s, c) in enumerate(scored) if s <= thr_t)[2]
        else:
            chosen = next(c for s, c in scored if s <= thr)                # 閾値内で最初(連続性優先)の候補                # 閾値内で最初(連続性優先)の候補
        tag, B, beats, res = chosen
        v_sc = vfirst[i] if vfirst[i] else (plan[-1][2] * k_speed * plan[-1][0] if plan else v)   # 表示用: 最初の音符の速さ(無ければ前小節を保持)
        # 小節頭の SCROLL は小節線の速さそのもの。小節線の速度が測れていて、小節頭(1/64 拍以内)に音符が無ければ小節線の速度を使う
        # (resume:story の掃引の最後の線: 直前の極小小節の値を引き継ぐと 2% 速く、到着 1.3 秒前で 20px 先を流れていた)
        v_bar = svel[i] if i < len(svel) else None
        head_note = any(f * L <= 60.0 / B / 16 + 0.002 for f, k, _ in per[i] if k != 'roll_end')
        if v_bar and v_bar > 0 and not head_note and plan and plan[-1][4] == 'cm': v_sc = v_bar     # 掃引(極小小節のラン)の直後の小節だけ(他は小節線の速度の測り損ねを拾う: TDM 260.9s)
        if os.environ.get('BUILDDBG') and v_bar and abs(v_bar / (vfirst[i] or v_bar) - 1) > 0.03: print('      head scroll: bar v=%.1f note v=%s head_note=%s -> v_sc %.1f' % (v_bar, vfirst[i], head_note, v_sc))
        scroll = v_sc / (k_speed * B)
        if os.environ.get('BUILDDBG'):
            print('  m%3d start %.2f L=%.3f V=%.1f n=%d | %s | -> %s B=%.1f beats=%g scroll=%.2f%s' % (
                i, starts[i], L, v, len(prec), ' '.join('%s:%.1f/%g/%.0fms' % (c[0], c[1], c[2], c[3]) for c in cands), tag, B, beats, scroll, (' TEMPO' if tempo_evi else '')))
        exact = tempo_evi and len(per[i]) <= 12 and (abs(dv(i - 1, i)) > 0.02 or spread > 0.025)   # 減速中で速さが実際に変わった小節は全部
        # 残差でも exact にする。ただし「仕掛けの証拠」(小節内で速さが混ざる/テンポ変化の区間)が無い小節では
        # 閾値をその曲の残差の中央値に合わせて上げる(検出のゆらぎをそのまま書き込むと、グリッドに乗せて平均するより
        # 参照から遠くなる: HYPERNOVA 中央値 3.35→1.96ms、雲間 2.67→1.00ms。EXACT_POLICY=flat で従来の固定閾値)
        thr_x = exact_res
        res_d = drift_ms(prec, beats, L)
        # 閾値を上げるのは「ずれが録画のゆらぎに見える小節」だけ。ずれの半分以上が傾き(位置に比例)なら小節の解釈の問題
        # (魔宵月は 94 小節のうち 26 本の小節線が #BARLINEOFF で描かれず、2 小節が 1 つに見えて長さを取り違える)なので上げない
        if exact_res > 0 and EXACT_POLICY != 'flat' and not (tempo_evi or spread > 0.025 or frac_multi > 0.4) and res_d < 0.5 * res:
            thr_x = max(exact_res, EXACT_K * res_med)
            # EXACT_POLICY=trick: 仕掛けの証拠が無い曲では残差だけで exact にしない(グリッドに乗せる)。
            # 集計は良くなる(HYPERNOVA 3.35→1.96ms、雲間 2.67→1.00ms、行数も 1/4)が、resume:story の 147.39s(本家が小節線の
            # 10ms 前に置いた音符。ユーザーが気づいた所)が −10ms に戻り、54.4s の 1 小節テンポ変化も ±3ms 悪化するので既定にしない
            if EXACT_POLICY == 'trick': thr_x = 1e9
        # exact にする条件(どれか):
        #  ・ずれの中央値が閾値超え … 小節ぜんぶがグリッドから外れている(仕掛け)
        #  ・ずれの傾きが閾値の 2 倍超え … 小節の中でテンポが変わっている(ドンカマ2000。最悪値だと音符の多い小節が不利)
        #  ・最悪値が閾値の 4 倍超え … 1 音符だけ大きく外れている(遅い層の音符)
        if exact_res > 0 and res > thr_x and per[i]: exact = True    # どのグリッドにも乗らない小節は音符ごとの BPM 指定で書く
        plan.append([B, beats, scroll, L, tag, exact, None, tempo_evi])
        if os.environ.get('BUILDDBG') and exact: print('      -> exact (res %.1fms 傾き %.1f > 閾値 %.1f, spread %.3f, n=%d)' % (res, res_d, thr_x, spread, len(per[i])))
        cur_bpm = B
    # 2b) 極小小節(飾り小節線)の SCROLL の平滑化。検出速度の揺れ(±2%)や丸めが、判定枠から遠い線では「時間×速度差」で
    #     数十 px の間隔の乱れになる。本家の掃引は等比(1 本ごとに ×0.976)なので、ランごとに log(速度) を一次(合わなければ二次)で
    #     当てはめて置き換える。速い/遅いが交互に来る区間(SCROLL 16/2.09/14.68/2.18…)は偶数番目と奇数番目を別々に当てはめる
    i = 0
    while i < nmeas:
        if plan[i][4] != 'cm': i += 1; continue
        j = i
        while j + 1 < nmeas and plan[j + 1][4] == 'cm': j += 1
        seq = np.array([plan[q][2] for q in range(i, j + 1)], float)
        if len(seq) >= 4 and (seq > 0).all():
            d = np.diff(np.log(seq))
            alt = len(d) >= 3 and np.mean(np.sign(d[1:]) != np.sign(d[:-1])) >= 0.8
            groups = [np.arange(0, len(seq), 2), np.arange(1, len(seq), 2)] if alt else [np.arange(len(seq))]
            fitted = seq.copy()
            for g in groups:
                if len(g) < 3: continue
                y = np.log(seq[g]); x = g.astype(float)
                for deg in (1, 2):
                    c = np.polyfit(x, y, deg); yh = np.polyval(c, x)
                    if np.max(np.abs(yh - y)) < 0.04 or deg == 2: break   # 4% 以内で乗れば一次(等比)
                fitted[g] = np.exp(yh)
            for q, v_ in zip(range(i, j + 1), fitted): plan[q][2] = float(v_)
            if os.environ.get('BUILDDBG'): print('  極小小節ラン m%d-%d: SCROLL を%s平滑化 %s → %s' % (i, j, '偶奇別に' if alt else '', ' '.join('%.2f' % v for v in seq[:6]), ' '.join('%.3f' % v for v in fitted[:6])))
        i = j + 1
    # 3) BPM の平滑化: 連続する小節で BPM が 1.5% 以内なら同じ値に揃える。値は「区間の総拍数×60 / 区間の実測総時間」
    #    (平均の比ではなく比の平均だと区間の合計時間がずれて、後ろの小節ほど時刻が流れる)
    i = 0
    while i < nmeas:
        if plan[i][4] == 'c1n': i += 1; continue          # 小節長にぴったり合わせた BPM(1 小節のテンポ変化)は平滑化しない
        j = i
        while j + 1 < nmeas and plan[j + 1][4] != 'c1n' and abs(plan[j + 1][0] - plan[i][0]) / plan[i][0] < 0.015: j += 1
        if j > i:
            Bm = float(sum(p[1] for p in plan[i:j + 1]) * 60.0 / sum(p[3] for p in plan[i:j + 1]))
            for p in plan[i:j + 1]: p[0] = Bm
        i = j + 1
    # 3b) 閉ループ補正: TJA の累積時刻(拍数×60/BPM の積算)が実測の小節線から 6ms 以上ずれたら、
    #    その小節の BPM だけを「次の小節線ちょうどで終わる値」にして戻す(拍子の解釈違いや丸めの蓄積を断つ)
    t_tja = starts[0]; n_corr = 0
    for i in range(nmeas):
        e = t_tja - starts[i]; plan[i][6] = t_tja
        if plan[i][5]:
            t_tja = starts[i + 1]; continue
        # 小節の終わり(次の小節頭)で許容を超えるなら、この小節で直す。小節頭で見てから直すと 1 小節遅れ、ずれの出た小節の頭がずれたまま残る
        # (段位 祭 30.36s: 本家は序盤で小節ごとに BPM を変え(1.677〜1.728 秒)、BPM 141 のまま書いた小節の次の小節頭が 25ms ずれた)。V2T_LOOP_AHEAD=0 で従来
        tol_ = 0.004 if plan[i][7] else float(os.environ.get('LOOP_MS', '5')) / 1000.0
        if os.environ.get('V2T_LOOP_AHEAD', '1') != '0' and i + 1 < len(starts):
            e = t_tja + plan[i][1] * 60.0 / plan[i][0] - starts[i + 1]
        if abs(e) > tol_ and i + 1 < len(starts) and starts[i + 1] - t_tja > 0.2:   # 既定 5ms(20ms だと小節頭が最大 20ms 流れ、ココドコで 70 音符が 20ms 超)   # 動画のフレーム時刻の鋸歯状ジッタ(±8ms が蓄積して十数ms)は追わず、拍子解釈の誤り(≥数十ms)だけ直す
            plan[i][0] = plan[i][1] * 60.0 / (starts[i + 1] - t_tja); n_corr += 1
        t_tja += plan[i][1] * 60.0 / plan[i][0]
    # 4) 行の生成
    # ゴーゴー: 区間の端を「その時刻が入る小節の、いちばん近いセル」に置く(本家も小節の途中に置くことがある)
    gogo_ev = []
    for a, b in (gogo or []):
        gogo_ev.append((float(a), '#GOGOSTART')); gogo_ev.append((float(b), '#GOGOEND'))
    gogo_ev.sort()
    n_gogo = [0]

    def _cmds_for(i):
        lo = -1e9 if i == 0 else starts[i]            # 最初の小節より前・最後の小節より後ろの端はその小節に寄せる
        hi = 1e9 if i == nmeas - 1 else starts[i + 1]
        out = [(t, c) for t, c in gogo_ev if lo <= t < hi]
        n_gogo[0] += len(out)
        return out

    lines = []; cur_meter = (4, 4); cur_bpm = None; cur_scroll = 1.0; n_bpm = n_meter = n_scroll = n_exact = 0
    after_exact = False   # 直前の小節が音符ごとの BPM 指定(_measure_lines_exact)だったか
    for i in range(nmeas):
        B, beats, scroll, L, tag, exact, t_head, _tempo = plan[i]
        if n_hidden and i == 0: lines.append('#BARLINEOFF')
        if n_hidden and i == n_hidden: lines.append('#BARLINEON')
        meter = _meter_of(beats)
        if meter != cur_meter:
            lines.append('#MEASURE %d/%d' % meter); cur_meter = meter; n_meter += 1
        if exact:
            if cur_bpm is None: cur_bpm = B
            body, cur_bpm, cur_scroll, nb = _measure_lines_exact(per[i], beats, L, pvel[i], starts[i], t_head, k_speed, cur_bpm, cur_scroll,
                                                                 svel[i] if i < len(svel) else None, cmds=_cmds_for(i))
            lines.append(body + ','); n_bpm += nb; n_exact += 1; after_exact = True
            continue
        if cur_bpm is None or abs(B - cur_bpm) > 0.05:
            if cur_bpm is not None: lines.append('#BPMCHANGE %.2f' % B); n_bpm += 1
            cur_bpm = B
        if tag == 'cm':
            sc = round(scroll, 3)                      # 飾り小節線: 寄せずに 3 桁(等比の刻みをそのまま出す)
            if abs(sc - cur_scroll) >= 0.005:
                lines.append('#SCROLL %g' % sc); cur_scroll = sc; n_scroll += 1
        elif i > 0 and plan[i - 1][4] == 'cm':
            sc = round(scroll, 3)                      # 掃引の直後の小節頭 = 掃引の最後の線。3 桁のまま、0.005 以上違えば出す(0.03 だと 1 本前の値を引き継いで 2% 速い)
            if abs(sc - cur_scroll) >= 0.005:
                lines.append('#SCROLL %g' % sc); cur_scroll = sc; n_scroll += 1
        else:
            sc = _snap_scroll(scroll)
            # 音符ごとの BPM 指定の小節は、区間 BPM に合わせた細かい SCROLL(0.972 など)で終わる。次の小節で 3% 未満の差を見逃すと
            # その値が曲の終わりまで残る(星河一天: 0.972 のまま 70 小節、本家より 2.8% 遅く流れた)ので、直後は 0.005 の差でも戻す
            if abs(sc - cur_scroll) >= (0.005 if after_exact else 0.03):
                lines.append('#SCROLL %g' % sc); cur_scroll = sc; n_scroll += 1
        # 音符ごとの SCROLL(小節の代表値と 3% 以上違う音符・連打の頭の前に #SCROLL を挟む)
        note_sc = [_snap_scroll(vv / (k_speed * B)) if vv else None for vv in pvel[i]]
        body, cur_scroll = _measure_lines(per[i], beats, L, note_sc, cur_scroll, cmds=_cmds_for(i), t0=starts[i])
        lines.append(body + ','); after_exact = False
    head_bpm = plan[0][0]
    print('小節 %d(小節線から)。拍子変更 %d 回、BPM変更 %d 回(うち時刻補正 %d、音符ごと指定の小節 %d)、SCROLL変更 %d 回、ゴーゴー %d 個。先頭 BPM %.2f' % (
        nmeas, n_meter, n_bpm, n_corr, n_exact, n_scroll, len(gogo or []), head_bpm))
    if gogo_ev and n_gogo[0] < len(gogo_ev): print('注意: ゴーゴーの命令 %d 個のうち %d 個しか置けなかった' % (len(gogo_ev), n_gogo[0]))
    return -starts[0], lines, head_bpm

def build(times, kinds, bigs, bpm, phase, bars=()):
    beat = 60.0 / bpm
    measure = beat * 4
    # 最初の音符が入る小節から開始。OFFSET は「小節頭の時刻」をTJAの負値で表す
    # 小節線が取れていればその位相を使う(音符だけでは16分グリッドの位相しか決まらず、小節頭が8分ずれたりする)
    bph, conf = bar_phase(bars, bpm)
    if bph is not None and conf > 0.8:
        # 小節線位相に最も近いグリッド点を小節頭にする。音符グリッドの位相は16分(拍/4)か3連(拍/3)の
        # どちらかで出ているので、両方を含む 48分(拍/12)刻みで寄せる(拍/4刻みだと3連位相のとき拍に乗らず 1/48小節ずれる)
        step = beat / 12
        k = np.round((bph - phase) / step)
        phase_bar = phase + k * step
        print('小節線 %d 本から小節頭位相を決定 (信頼度 %.2f, グリッド補正 %+.0fms)' % (len(bars), conf, (phase_bar - bph) * 1000))
    else:
        phase_bar = phase
        print('小節線が使えない(%d本, 信頼度 %.2f) → 小節頭は音符グリッド任せ(8分単位でずれ得る)' % (len(bars), conf))
    m0 = int(np.floor((times[0] - phase_bar) / measure))
    bar0_time = phase_bar + m0 * measure
    offset = -bar0_time
    last = times[-1]
    nmeas = int(np.ceil((last - bar0_time) / measure)) + 1
    # 小節頭の時刻列。動画のフレーム時刻には鋸歯状のドリフト(約30小節ごとに8ms跳ぶ)があるので、
    # 固定グリッドではなく「実際に検出した小節線」を各小節の基準にする(無い小節は前後から補間)
    starts = bar0_time + np.arange(nmeas + 1) * measure
    if bph is not None and conf > 0.8 and len(bars) > 0:
        bars_a = np.sort(np.asarray(bars, float)); anchored = np.full(nmeas + 1, np.nan)
        for i in range(nmeas + 1):
            j = np.argmin(np.abs(bars_a - starts[i]))
            if abs(bars_a[j] - starts[i]) < 0.04 * measure: anchored[i] = bars_a[j]
        ok = ~np.isnan(anchored)
        if ok.sum() >= 3:
            idx = np.arange(nmeas + 1)
            starts = np.interp(idx, idx[ok], anchored[ok])       # 区間外は端の値で一定 → 下で名目値に戻す
            starts[:idx[ok][0]] = anchored[idx[ok][0]] - (idx[ok][0] - idx[:idx[ok][0]]) * measure
            starts[idx[ok][-1] + 1:] = anchored[idx[ok][-1]] + (idx[idx[ok][-1] + 1:] - idx[ok][-1]) * measure
            print('小節線で %d/%d 小節の頭を実測に合わせた' % (ok.sum(), nmeas + 1))
    offset = -starts[0]
    # 各音符を (小節, 小節内位相0..1) に
    per = [[] for _ in range(nmeas)]
    TOL = 1.0 / 64          # 小節頭ぴったりの音符が数ms早くても前の小節の末尾に落ちないための許容幅
    for t, k, big in zip(times, kinds, bigs):
        mi = int(np.searchsorted(starts, t + TOL * measure) - 1)
        if not (0 <= mi < nmeas): continue
        frac = (t - starts[mi]) / (starts[mi + 1] - starts[mi])      # [-TOL, 1-TOL)
        per[mi].append((frac, k, big))
    SUBS = [16, 12, 24, 48, 32, 64]
    lines = []
    for m in per:
        if not m:
            lines.append(',')
            continue
        # 分割数の選択: 「全音符の残差が許容(0.045拍≈13ms@212)以内」を満たす最も粗い分割を採る。
        # 合計誤差の最小化だと、検出ノイズ(数ms)と細かい分割のペナルティの兼ね合いで
        # 24分混じりの小節が16分に丸められたり、3連が64分になったりして安定しなかった。
        TOL_BEAT = 0.03                        # 8.5ms@212。32分を48分グリッドに(半スロット=11.8ms)乗せない
        prec = [f for f, k, _ in m if k in ('don', 'ka')] or [f for f, _, _ in m]   # 連打/風船の頭・終端は時刻が甘いので除外
        best_sub = None
        for mult in (1, 2, 3, 6):              # 外れ値(数十ms)があるときは許容を段階的に緩めて、粗い分割を優先し続ける
            for sub in SUBS:                   # 16,12,24,48,32,64 の順(48=16分+24分混在が32分より普通)
                worst = max(abs(f * sub - round(f * sub)) / sub * 4 for f in prec)      # 拍単位
                if worst <= TOL_BEAT * mult:
                    best_sub = sub; break
            if best_sub is not None: break
        if best_sub is None:                   # どれも合わない → 合計誤差最小
            best_sub = min(SUBS, key=lambda sub: sum(abs(f * sub - round(f * sub)) / sub for f, _, _ in m))
        slots = ['0'] * best_sub
        for f, k, big in m:
            idx = min(best_sub - 1, max(0, int(round(f * best_sub))))
            slots[idx] = symbol(k, big)
        lines.append(''.join(slots) + ',')
    return offset, lines

def symbol(kind, big):
    if kind == 'don':    return '3' if big else '1'
    if kind == 'ka':     return '4' if big else '2'
    if kind == 'renda':  return '6' if big else '5'     # 連打開始(大連打は6)
    if kind == 'balloon':return '7'
    if kind == 'roll_end': return '8'                   # 連打/風船の終端(棒の右端が判定枠に来た時刻)
    return '1'

def fix_rolls(onsets, bpm):
    """5/6/7 の後には必ず 8 が来るように整える(TJAとして壊れないための保険)。
    ・頭(5/6/7)の後、次のドン/カより前に終端(8)が無ければ、次の音符の1拍前(無ければ頭の2拍後)に 8 を補う
    ・頭の無い 8 は捨てる
    ・8 が次の音符と同時刻以降なら次の音符の16分前に寄せる"""
    beat = 60.0 / bpm
    ons = sorted(onsets, key=lambda o: o['t'])
    out = []; open_head = None
    last_head_t = -9
    for o in ons:
        if o['color'] in ('renda', 'balloon'):
            if o['t'] - last_head_t < 0.12: continue        # 頭が2つ続くことは無い(棒の左端の誤検出) → 後の方を捨てる
            last_head_t = o['t']
            if open_head is not None:                       # 連打中に別の頭 → 直前に閉じる
                # 連打のはしご(頭どうしが 0.3 秒未満)は終端が次の頭に隠れて見えない。本家は次の頭の 23〜30ms 前に 8 を置く
                # (resume:story)。16 分前(beat/4)だと 46ms 早かった
                gap = 0.025 if o['t'] - open_head['t'] < 0.3 else beat / 4
                out.append({'t': o['t'] - gap, 'color': 'roll_end', 'big': False, 'synth': True})
            out.append(o); open_head = o
        elif o['color'] == 'roll_end':
            if open_head is None: continue                  # 頭の無い終端は捨てる
            if o['t'] <= open_head['t'] + beat / 16: continue        # 1/16 拍より近い終わりは誤検出(1/8 拍だと、もぺもぺの 0.05 秒の連打の終わりを捨てていた)
            out.append(o); open_head = None
        else:
            if open_head is not None:                       # 終端が来ないまま次の音符 → 手前で閉じる
                t8 = max(open_head['t'] + beat / 4, o['t'] - beat)
                out.append({'t': t8, 'color': 'roll_end', 'big': False, 'synth': True}); open_head = None
            out.append(o)
    if open_head is not None:
        out.append({'t': open_head['t'] + 2 * beat, 'color': 'roll_end', 'big': False, 'synth': True})
    out.sort(key=lambda o: o['t'])
    # 風船の長さ: 袋(尻尾)は描かれないことが多く(resume:story の 39 打は小節線演出の中で袋が無い)、終端の検出は当てにならない。
    # 打数 count が読めていれば 長さ = count/15 秒(参照 8 曲で 3 打 0.14s, 10 打 0.68s, 40 打 2.67s と ±25% で合う)。
    # 次の音符の 1/4 拍手前までに収める
    for i, o in enumerate(out):
        if o['color'] != 'balloon' or not o.get('count'): continue
        j = next((k for k in range(i + 1, len(out)) if out[k]['color'] == 'roll_end'), None)
        if j is None: continue
        nxt = next((out[k]['t'] for k in range(i + 1, len(out)) if out[k]['color'] not in ('roll_end',)), None)
        # 検出した終端は使わない(袋の終端は短く出る: 銀の黎明か…の 6 打は 0.08s と出たが参照譜面は 0.393s、3 打は 0.197s = count/15)
        # 長さ = 打数 / 11 秒、次の音符の 1/2 拍前まで(試験スイートの参照 140 個: 打/秒の中央値 12、終わり→次の音符は 1/2 拍か 1 拍が大半。
        # 打数/15・1/4 拍前だと 69 個が 0.1 秒以上短く(割れない窓になる)、打数/11・1/2 拍前で短すぎ 33・誤差の中央値 0)
        want = o['t'] + max(o['count'] / 11.0, beat / 4)
        if nxt is not None: want = min(want, nxt - beat / 2)
        out[j]['t'] = max(want, o['t'] + beat / 8)
    out.sort(key=lambda o: o['t'])
    return out

def main():
    video, start, end, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    title = sys.argv[5] if len(sys.argv) > 5 else 'AutoChart'
    lo = float(sys.argv[6]) if len(sys.argv) > 6 else 90.0
    hi = float(sys.argv[7]) if len(sys.argv) > 7 else 310.0
    bars = []; bar_vels = None; gogo_iv = []
    if video.lower().endswith('.json'):      # 検出済みJSONを再利用(検出は時間がかかるので)
        import json
        d = json.load(open(video, encoding='utf-8'))
        onsets = [o for o in (d['onsets'] if isinstance(d, dict) else d) if start <= o['t'] < end]
        if isinstance(d, dict): bars = d.get('bars', []); bar_vels = d.get('bar_vels'); gogo_iv = d.get('gogo') or []
    else:
        cal = None
        if os.environ.get('NOCAL') != '1' and hasattr(detect, 'run') and detect.__name__ == 'detect2':
            import calib                                     # 動画の配置を自動推定(基準配置ならそのまま)
            mid = (start + end) / 2
            cal = calib.calibrate(video, max(start, mid - 20), min(end, mid + 20))
        runner = getattr(detect, 'run_parallel', detect.run)
        fps, onsets = runner(video, start, end, cal=cal) if cal is not None else runner(video, start, end)
        bars = getattr(runner, 'bars', []); bar_vels = getattr(runner, 'bar_vels', None)
        import balloon_count; balloon_count.annotate(video, onsets, cal)     # 風船の打数(吹き出しの数字)
    # 連打(黄)を含む点は位置がぶれやすいので BPM推定には don/ka のみ使う
    tk = [o for o in onsets if o['color'] in ('don', 'ka')]
    times_all = np.array([o['t'] for o in onsets])
    times = np.array([o['t'] for o in tk])
    hint = bpm_from_bars(bars) if len(bars) >= 8 else None
    if hint: print('小節線 %d 本の周期から BPM ≈ %.2f' % (len(bars), hint))
    bpm, phase = estimate_bpm(times, lo, hi, hint)
    bpm = choose_octave(times, bpm, phase)
    # (estimate_bpm 内で精密化済み)
    print('推定 BPM %.2f  音符 %d (don/ka %d)' % (bpm, len(onsets), len(tk)))
    onsets = fix_rolls(onsets, bpm)
    # 判定枠に止まっていた音符(detect2 の stationary): 時刻は「叩かれて消えたフレーム」なので ±25ms ぶれる。
    # 直前の小節線から始まる 1/4 拍の格子(直前の小節の長さを 4 拍とみなす)に 40ms 以内なら寄せる(ドンカマ2000 の最後の大ドン)
    if len(bars) >= 2:
        bs = np.sort(np.asarray(bars, float))
        for o in onsets:
            if not o.get('stationary'): continue
            k = int(np.searchsorted(bs, o['t'])) - 1
            if k < 1: continue
            b0 = bs[k]; step = (bs[k] - bs[k - 1]) / 16.0
            g = b0 + round((o['t'] - b0) / step) * step
            if abs(g - o['t']) <= 0.04:
                print('止まっている音符 %.3f → 拍の格子 %.3f(%+.0fms)' % (o['t'], g, (g - o['t']) * 1000)); o['t'] = float(g)
    times_all = np.array([o['t'] for o in onsets])
    kinds = [o['color'] for o in onsets]; bigs = [o['big'] for o in onsets]
    if len(bars) >= 8 and os.environ.get('FIXED_METER') != '1':
        vels = [o.get('vel', 0.0) for o in onsets]
        offset, lines, bpm = build_var(times_all, kinds, bigs, vels, bpm, bars, bar_vels=bar_vels, gogo=gogo_iv)   # 小節線ベース(小節ごとに BPM/拍子/SCROLL)
    else:
        offset, lines = build(times_all, kinds, bigs, bpm, phase, bars)
    # OFFSET は動画時刻基準。曲の音源(ogg)に合わせるなら、動画音声↔ogg の相互相関で出したΔ(ogg_t = video_t − Δ)を
    # 環境変数 VID2AUDIO で渡す(HYPERNOVA: 308.081)。渡さなければ動画の音声をそのまま WAVE にする前提。
    delta = float(os.environ.get('VID2AUDIO', '0'))
    offset += delta
    body = '\n'.join(lines)
    nballoon = sum(ln.count('7') for ln in body.split(chr(10)) if not ln.startswith('#'))   # 命令行の数字は数えない
    bcounts = [int(o.get('count', 5)) for o in sorted(onsets, key=lambda o: o['t']) if o['color'] == 'balloon']   # balloon_count が読んだ打数(無ければ 5)
    kusu = [bool(o.get('kusu')) for o in sorted(onsets, key=lambda o: o['t']) if o['color'] == 'balloon']
    if any(kusu) and len(kusu) == nballoon:
        # くす玉(detect2 の kusu)は 7 でなく 9。BALLOON: の打数は 7 と 9 を合わせた順
        k_ = iter(kusu); out_l = []
        for ln in body.split(chr(10)):
            if not ln.startswith('#'): ln = ''.join(('9' if next(k_) else '7') if ch == '7' else ch for ch in ln)
            out_l.append(ln)
        body = chr(10).join(out_l)
    if len(bcounts) != nballoon:
        print('注意: 風船の数が検出 %d 個と本文 %d 個で違うので、足りない分は 5 打にする' % (len(bcounts), nballoon))
        bcounts = (bcounts + [5] * nballoon)[:nballoon]
    balloon = ('BALLOON:' + ','.join(str(c) for c in bcounts) + chr(10)) if nballoon else ''
    tja = ("TITLE:%s\nSUBTITLE:--%s\nBPM:%.2f\nWAVE:%s.ogg\nOFFSET:%.3f\n"
           "DEMOSTART:0\n\nCOURSE:%s\nLEVEL:10\n%s\n#START\n%s\n#END\n"
           % (title, os.environ.get('SUBTITLE', 'video2tja 自動採譜'), bpm, title, offset, os.environ.get('COURSE', 'Oni'), balloon, body))
    open(out, 'w', encoding='utf-8-sig', newline='\r\n').write(tja)
    nn = sum(c in '1234' for c in body)
    print('書き出し', out, ' 小節数', body.count(','), ' 音符記号', nn, ' OFFSET %.3f' % offset)

if __name__ == '__main__':
    main()
