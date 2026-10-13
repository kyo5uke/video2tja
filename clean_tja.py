"""TJA の清書: 採譜した TJA を、音符の時刻・種類・流れる速さと小節線を変えずに、職人が書くような短い形に直す。
使い方: clean_tja.py <in.tja> [out.tja(既定: 上書きせず <in>_clean.tja)]

採譜の TJA(make_tja)は「測った時刻と速さを正確に再現する」ことを優先して書いてあるので、
  - 小節のセル数が細かいまま(本家 `22,` → `2000000020000000,`)
  - 時刻の合わせ込みの #BPMCHANGE と、それで変わる見た目の速さを戻す #SCROLL が並ぶ(307.142 / 0.993 …)
  - 値が細かい(BPM:305.01)
になる。清書は次の順に直し、1 つ直すたびに「全部の音符・小節線がどれだけ動いたか」を計り直して、許す幅を超える直しは入れない。
  1. セル数を最小に(音符と命令の位置を保つ最大の約数で割る。時刻は 1 つも変わらない)
  2. 小節の中の #BPMCHANGE を 1 つの BPM にまとめ、音符を最小の格子に乗せる(小節の長さ = 小節線の時刻は保つ)
  3. BPM・SCROLL を丸める・同じ値の続きをまとめる
許す幅は採譜の分解能に合わせる: 時刻は検出の揺れ(make_tja が「同じ」とみなす幅と同じ)、速さは SCROLL を挟む基準と同じ 3%。
"""
import sys, os, re, math

NL = chr(10)


class Measure:
    """1 小節: セル文字列と、セル位置つきの命令 [(pos, 行)]。pos = その命令の前にあるセルの数(0 = 小節の頭)"""
    __slots__ = ('cells', 'cmds')

    def __init__(self, cells='', cmds=None):
        self.cells = cells; self.cmds = list(cmds or [])

    def copy(self): return Measure(self.cells, list(self.cmds))


def parse_body(lines):
    """#START と #END の間の行 → (小節の列, 最後のカンマの後ろに残った命令)"""
    ms = []; cur = Measure()
    for raw in lines:
        s = raw.split('//')[0].strip()
        if not s: continue
        if s.startswith('#'):
            cur.cmds.append((len(cur.cells), s)); continue
        for ch in s:
            if ch == ',':
                ms.append(cur); cur = Measure()
            elif not ch.isspace():
                cur.cells += ch
    return ms, cur


def split_tja(txt):
    """TJA の全文 → (ヘッダの行, [コース: {'head': 行, 'body': 小節, 'tail': 残り命令, 'after': #END の後ろの行}])"""
    lines = txt.replace('\r', '').split(NL)
    head = []; courses = []; i = 0
    while i < len(lines) and not lines[i].strip().startswith('COURSE:'):
        head.append(lines[i]); i += 1
    while i < len(lines):
        c = {'head': [], 'body': None, 'tail': None, 'after': []}
        while i < len(lines) and not lines[i].strip().startswith('#START'):
            c['head'].append(lines[i]); i += 1
        i += 1; body = []
        while i < len(lines) and not lines[i].strip().startswith('#END'):
            body.append(lines[i]); i += 1
        i += 1
        c['body'], c['tail'] = parse_body(body)
        while i < len(lines) and not lines[i].strip().startswith('COURSE:'):
            c['after'].append(lines[i]); i += 1
        courses.append(c)
    return head, courses


def header_value(lines, key, default=None):
    for ln in lines:
        m = re.match(r'\s*' + key + r':\s*(.*)$', ln)
        if m and m.group(1).strip(): return m.group(1).strip()
    return default


def timeline(ms, bpm0):
    """小節の列 → (音符 [(時刻, 文字, 速さ BPM×SCROLL, 小節, セル)], 小節線 [(時刻, 速さ, 見える, 小節)], 終わりの時刻)。
    時刻は #START からの秒(OFFSET は含まない)。eval_timing.parse_tja_times と同じ数え方(#MEASURE・#BPMCHANGE・#DELAY)"""
    t = 0.0; bpm = bpm0; sc = 1.0; mult = 1.0; bar_on = True
    notes = []; bars = []

    def apply(txt):
        nonlocal bpm, sc, mult, bar_on, t
        m = re.match(r'#BPMCHANGE\s+([-\d.]+)', txt)
        if m: bpm = float(m.group(1)); return
        m = re.match(r'#SCROLL\s+([-\d.]+)', txt)
        if m: sc = float(m.group(1)); return
        m = re.match(r'#MEASURE\s+(\d+)/(\d+)', txt)
        if m: mult = int(m.group(1)) / int(m.group(2)); return
        m = re.match(r'#DELAY\s+([-\d.]+)', txt)
        if m: t += float(m.group(1)); return
        if txt.startswith('#BARLINEOFF'): bar_on = False
        elif txt.startswith('#BARLINEON'): bar_on = True

    for mi, m in enumerate(ms):
        n = len(m.cells)
        at = {}
        for p, txt in m.cmds: at.setdefault(p, []).append(txt)
        for txt in at.get(0, []): apply(txt)
        bars.append((t, bpm * sc, bar_on, mi))
        if n == 0:
            t += 60.0 / bpm * 4 * mult
            continue
        for p in range(n):
            if p > 0:
                for txt in at.get(p, []): apply(txt)
            ch = m.cells[p]
            if ch != '0': notes.append((t, ch, bpm * sc, mi, p))
            t += 60.0 / bpm * 4 * mult / n
        for txt in at.get(n, []): apply(txt)
    return notes, bars, t


def write_body(ms, tail):
    """小節の列 → 本文の行。命令はセル位置の前で改行して入れる(本家と同じ書き方)"""
    out = []
    for m in ms:
        at = {}
        for p, txt in m.cmds: at.setdefault(p, []).append(txt)
        line = ''
        for p in range(len(m.cells) + 1):
            if p in at:
                if line: out.append(line); line = ''
                out.extend(at[p])
            if p < len(m.cells): line += m.cells[p]
        out.append(line + ',')
    if tail is not None:                  # 最後のカンマの後ろ(#END の前の命令)
        for _, txt in tail.cmds: out.append(txt)
        if tail.cells: out.append(tail.cells)
    return out


# ---------------------------------------------------------------- 清書
T_MOVE = 0.005       # 音符・小節線を動かしてよい幅: make_tja が「同じ時刻」とみなす幅(EXACT_MS・LOOP_MS)と同じ。1100 曲で本家と比べて
#   8.3ms(画面の半コマ)は BPM 変更が 1/6 まで減るが、本当にテンポが揺れる曲(ともに)を均して 10ms 超が 1.5 倍、3ms は減りが半分
V_TOL = 0.03         # 音符の流れる速さを変えてよい割合(make_tja が音符ごとの #SCROLL を挟む基準・_snap_scroll の寄せ幅と同じ)
V_TOL_BAR = 0.005    # 音符の無い小節の小節線(飾り小節線)の速さ。線の位置そのものが中身なので make_tja の 3 桁の刻みと同じ細かさで保つ
GRID_PER_BEAT = (4, 3, 6, 12, 8, 16, 24, 48)   # 格子の候補(1 拍のセル数)。make_tja._measure_lines と同じ順(16 分・3 連を先に)
# 格子に乗ったとみなすずれ: make_tja._measure_lines の TOL_BEAT = min(拍の 3%, 6ms) と、乗らなければ 1・2・3・6 倍まで広げる段と同じ
# (採譜が「この格子」と決めたのと同じ基準で選び直すため。make_tja を変えたらここも合わせる)。
# 拍の 3%: 32 分を 48 分の格子に乗せない幅(212 BPM で 8.5ms、半セル 11.8ms)。6ms: 128 BPM の 24 分+32 分の小節が 64 セルに乗って +13ms ずれたので上限を付けた
GRID_TOL_BEAT = 0.03
GRID_TOL_MAX = 0.006
GRID_TOL_STEPS = (1, 2, 3, 6)
GRID_CELL_FRAC = 0.25   # ずれはセルの 1/4 まで(_grid のコメント)。それより大きいと隣のセル・細かい格子と見分けられない
# 拍子を読み違えた小節用の格子(拍数に掛ける数)。1 拍 3 つ ↔ 4 つの読み違い(3/4 ↔ 4/4)と、2 拍 ↔ 3 拍(6/8 系)。_grid のコメント
METER_SCALES = (1.0, 0.75, 4.0 / 3, 1.5, 2.0 / 3)
# BPM の区間を選ぶときの「書く手間」: #BPMCHANGE 1 行 = 1、値の小数 1 桁 = DIGIT_COST(2 桁の値 1 つと BPM 変更 1 行が同じ重み)、
# そのまま残す小節 1 つ = FROZEN_COST(残す小節は音符ごとに #BPMCHANGE・#SCROLL を書くので BPM 変更 1 行よりずっと重い: 銀の黎明の 1 小節で #BPMCHANGE 10 行・#SCROLL 9 行)。
# どちらも試験の数字に合わせた値ではない(桁より行の数を減らすこと、残す小節を増やさないことを優先する向きだけ決めている)
DIGIT_COST = 0.5
FROZEN_COST = 3
# BPM の値の候補を集める量(計算量の上限。結果は変えない): 1 つの桁で 400 刻みより広いときは真ん中 ±200 刻みだけ見る(整数なら ±200 BPM)、
# 桁の少ない方から 8 個集まるまで桁を増やし、先読み(次の区間の長さ)で比べるのは桁の少ない順・真ん中に近い順の 12 個まで
BPM_CAND_SPAN = 400
BPM_CAND_MIN = 8
BPM_CAND_LOOKAHEAD = 12
TIMING = ('#BPMCHANGE', '#SCROLL', '#MEASURE', '#DELAY')


def _is_timing(txt): return bool(txt.strip()) and txt.split()[0] in TIMING


def walk(ms, bpm0):
    """小節ごとの実際の時刻と状態: start/dur/beats、小節頭の BPM・SCROLL、小節線の速さ・見えるか、
    音符 [(時刻, 文字, 速さ, セル)]、時刻以外の命令 [(セル, 行, 時刻)]"""
    t = 0.0; st = {'bpm': bpm0, 'sc': 1.0, 'mult': 1.0, 'bar': True, 'odd': False, 'meter': None}
    recs = []

    def apply(txt):
        m = re.match(r'#BPMCHANGE\s+([-\d.]+)', txt)
        if m: st['bpm'] = float(m.group(1)); return
        m = re.match(r'#SCROLL\s+([-\d.]+)', txt)
        if m: st['sc'] = float(m.group(1)); return
        m = re.match(r'#MEASURE\s+(\d+)/(\d+)', txt)
        if m: st['mult'] = int(m.group(1)) / int(m.group(2)); st['meter'] = txt; return
        if txt.startswith('#BARLINEOFF'): st['bar'] = False
        elif txt.startswith('#BARLINEON'): st['bar'] = True

    for m in ms:
        n = len(m.cells); at = {}
        for p, txt in m.cmds: at.setdefault(p, []).append(txt)
        r = {'n': n, 'raw': m}
        st['odd'] = False; cmds = []
        for txt in at.get(0, []):
            if txt.startswith('#DELAY'):          # #DELAY は清書しない(その小節はそのまま残す)
                st['odd'] = True; m_ = re.match(r'#DELAY\s+([-\d.]+)', txt); t += float(m_.group(1)) if m_ else 0.0; continue
            apply(txt)
            if not _is_timing(txt): cmds.append((0, txt, t))
        r.update(start=t, bpm0=st['bpm'], sc0=st['sc'], mult=st['mult'], beats=4.0 * st['mult'], meter=st['meter'],
                 bar_on=st['bar'], bar_v=st['bpm'] * st['sc'])
        notes = []
        if n == 0:
            t += 60.0 / st['bpm'] * 4 * st['mult']
        else:
            for p in range(n + 1):
                if p > 0:
                    for txt in at.get(p, []):
                        if txt.startswith('#DELAY'):
                            st['odd'] = True; m_ = re.match(r'#DELAY\s+([-\d.]+)', txt); t += float(m_.group(1)) if m_ else 0.0; continue
                        if txt.startswith('#MEASURE'): st['odd'] = True      # 小節の途中の拍子変更は扱わない
                        apply(txt)
                        if not _is_timing(txt): cmds.append((p, txt, t))
                if p == n: break
                ch = m.cells[p]
                if ch != '0': notes.append((t, ch, st['bpm'] * st['sc'], p))
                t += 60.0 / st['bpm'] * 4 * st['mult'] / n
        r.update(end=t, dur=t - r['start'], notes=notes, cmds=cmds, odd=st['odd'], bpm_out=st['bpm'], sc_out=st['sc'])
        recs.append(r)
    return recs


def _fmt(x, d=3):
    s = ('%.*f' % (d, x)).rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


def _nicest(lo, hi, ctr, prefer=None):
    """[lo, hi] の中で桁の少ない値(整数 → 小数 1 桁 → 2 桁 …)。同じ桁なら ctr に近いもの。prefer が入っていればそれ"""
    if prefer is not None and lo - 1e-9 <= prefer <= hi + 1e-9: return prefer
    for d in range(0, 7):
        s = 10.0 ** -d
        k0 = math.ceil(lo / s - 1e-9); k1 = math.floor(hi / s + 1e-9)
        if k0 <= k1: return round(min(max(round(ctr / s), k0), k1) * s, d)
    return ctr


def _grid(r, c0, dur, T):
    """小節 r の音符と命令を、小節頭 c0・長さ dur の格子に乗せる。make_tja と同じ順・同じ許容(GRID_TOL_BEAT・GRID_TOL_MAX を
    GRID_TOL_STEPS 倍まで広げる)で最初に乗る格子を選び、どの音符も元の時刻から T 以内のものだけ。戻り値 (セル数, 音符のセル, 命令のセル) か None"""
    beats = r['beats']; notes = r['notes']; cmds = r['cmds']
    if not notes and not cmds: return (1 if r['n'] else 0), [], []
    cands = []
    # 拍子を読み違えた小節の格子も後ろに足す(make_tja が 4/4・220 BPM と読んだ小節が本家では 3/4・165 BPM で、1 拍 3 つの 36 セル
    # = 4/4 では 1 拍 9 つになり候補に無かった: Scars of FAUNA・SORA-Ⅶ)。拍数 × 3/4・4/3・3/2・2/3 の拍子として数えた格子
    for f in METER_SCALES:
        for k in GRID_PER_BEAT:
            c = beats * f * k
            if abs(c - round(c)) < 1e-6 and round(c) >= 1 and int(round(c)) not in cands: cands.append(int(round(c)))
    if not cands: return None
    tol0 = min(GRID_TOL_BEAT * dur / beats, GRID_TOL_MAX)
    for mult in GRID_TOL_STEPS:
        tol = min(tol0 * mult, T)
        for n in cands:
            cell = [min(n - 1, max(0, int(round((t - c0) / dur * n)))) for t, *_ in notes]
            if any(b <= a for a, b in zip(cell, cell[1:])): continue
            # 格子の位置が決まると言えるのは、ずれがセルの 1/4 以内のときだけ(それより大きいと隣のセルや細かい格子と区別できない:
            # 305 BPM の 48 分(16ms)に 8ms の幅で乗せると、16 分の音符が 1 セル前の 48 分に乗った)
            if any(abs(c0 + c * dur / n - t) > min(tol, dur / n * GRID_CELL_FRAC) for c, (t, *_) in zip(cell, notes)): continue
            # 命令(ゴーゴーなど): 元と同じ音符の間(前の音符より後ろ、次の音符以前)で、時刻に近いセル
            cc = []; ok = True
            for p, txt, t in cmds:
                nb = sum(1 for nt in notes if nt[3] < p)
                lo_ = (cell[nb - 1] + 1) if nb else 0
                hi_ = cell[nb] if nb < len(cell) else n
                c = min(max(int(round((t - c0) / dur * n)), lo_), hi_)
                if lo_ > hi_ or abs(c0 + c * dur / n - t) > T: ok = False; break
                cc.append(c)
            if ok: return n, cell, cc
        if tol >= T: break
    return None


def _grid_fit(r, T):
    """元の小節頭・長さで格子に乗らない小節(小節線だけ数 ms ずれた小節: 録画の引っかかりで検出した小節線が 10ms 早い等)を、
    小節頭と終わりをそれぞれ ±T の中で 1ms 刻みに動かして格子に乗せる。音符の多い格子の当てはまりを優先(make_tja の順で最初に乗るもの)"""
    best = None; k = int(T * 1000)
    for da in range(-k, k + 1):
        for db in range(-k, k + 1):
            s = r['start'] + da / 1000.0; e = r['end'] + db / 1000.0
            g = _grid(r, s, e - s, T)
            if g is None: continue
            # 当てはまりのよさ: 音符の残差の最大(小さいほどよい)。同じなら動かす量の小さいもの
            res = max([abs(s + c * (e - s) / g[0] - nt[0]) for c, nt in zip(g[1], r['notes'])] or [0.0])
            key = (_GRID_RANK.get(g[0] / r['beats'], 99), res, abs(da) + abs(db))
            if best is None or key < best[0]: best = (key, g)
    return best[1] if best else None


_GRID_RANK = {k: i for i, k in enumerate(GRID_PER_BEAT)}


def _extend(recs, grids, frozen, T, i, ci):
    """小節 i(小節頭の時刻 ci)から同じ BPM で続けられる所まで伸ばす。戻り値 (j, 1 拍の秒の下限, 上限, 使った (拍, 元の時刻))。
    j == i なら 1 小節も収まらない"""
    N = len(recs); lo, hi = 0.0, float('inf'); Q = 0.0; j = i; used = []
    while j < N and j not in frozen:
        r = recs[j]; n, cell, _ = grids[j]; lo2, hi2 = lo, hi
        anchors = [(Q + r['beats'] * cc / n, nt[0]) for cc, nt in zip(cell, r['notes'])] if n else []
        anchors.append((Q + r['beats'], r['end']))
        bad = False
        for q, t in anchors:
            if q <= 1e-12:
                if abs(ci - t) > T + 1e-9: bad = True
                continue
            lo2 = max(lo2, (t - ci - T) / q); hi2 = min(hi2, (t - ci + T) / q)
        if bad or lo2 > hi2: break
        lo, hi = lo2, hi2; Q += r['beats']; j += 1; used += anchors
    return j, lo, hi, used, Q


def _digits(x):
    for d in range(7):
        if abs(x * 10 ** d - round(x * 10 ** d)) < 1e-6: return d
    return 7


def _tempo_runs_from(recs, grids, frozen, T, c0):
    """BPM を変えずに続けられるだけ小節をつなぐ(どの音符・小節線も元の時刻から T 以内)。c0 = 最初の小節頭の時刻(OFFSET の直し)。
    区間の BPM は、その区間に収まる値のうち桁の少ないもの。同じ桁なら「次の区間がいちばん長く続く」もの
    (区間の終わりのずれが許す幅の端に来ると、次の区間が録画の揺れを収めきれずに短く切れる)。
    戻り値 (小節ごとの BPM(そのまま残す小節は None), 小節頭の時刻, 1 小節でも収まらず残すことにした小節, 書く手間(区間 1 + 桁 DIGIT_COST), 最大のずれ)"""
    N = len(recs); c = [0.0] * (N + 1); c[0] = c0; B = [None] * N; newly = set(); last_B = None; runs = 0; worst = abs(c0)
    i = 0
    while i < N:
        if i in frozen or i in newly:
            c[i + 1] = c[i] + recs[i]['dur']; i += 1; continue
        j, lo, hi, used, Q = _extend(recs, grids, frozen | newly, T, i, c[i])
        if j == i:
            newly.add(i); continue
        Blo = 60.0 / hi if hi > 0 else 0.0; Bhi = 60.0 / lo if lo > 0 else 1e9
        x_mid = (lo + hi) / 2 if hi < float('inf') else lo
        cands = []
        if last_B is not None and Blo - 1e-9 <= last_B <= Bhi + 1e-9: cands = [last_B]
        else:
            for d in range(0, 4):
                s = 10.0 ** -d; k0 = math.ceil(Blo / s - 1e-9); k1 = math.floor(Bhi / s + 1e-9)
                if k1 - k0 > BPM_CAND_SPAN:
                    k0, k1 = int(round(60.0 / x_mid / s)) - BPM_CAND_SPAN // 2, int(round(60.0 / x_mid / s)) + BPM_CAND_SPAN // 2
                cands += [round(k * s, d) for k in range(k0, k1 + 1)]
                if len(cands) >= BPM_CAND_MIN: break
            cands.append(_nicest(Blo, Bhi, 60.0 / x_mid))
            cands = sorted(set(cands), key=lambda b: (_digits(b), abs(b - 60.0 / x_mid)))[:BPM_CAND_LOOKAHEAD]
        if j < N and len(cands) > 1:
            def reach(b):
                cj = c[i] + 60.0 * Q / b
                jj = j
                while jj < N and (jj in frozen or jj in newly): cj += recs[jj]['dur']; jj += 1
                return _extend(recs, grids, frozen | newly, T, jj, cj)[0] if jj < N else N
            Bsel = max(cands, key=lambda b: (-_digits(b), reach(b), -abs(b - 60.0 / x_mid)))
        else:
            Bsel = cands[0]
        for q, t in used: worst = max(worst, abs(c[i] + q * 60.0 / Bsel - t))
        for k in range(i, j):
            B[k] = Bsel; c[k + 1] = c[k] + 60.0 * recs[k]['beats'] / Bsel
        if last_B is None or abs(Bsel - last_B) > 1e-9: runs += 1 + DIGIT_COST * _digits(Bsel)     # 書く手間: 1 行 + 桁
        last_B = Bsel; i = j
    return B, c, newly, runs, worst


def _tempo_runs(recs, grids, frozen, T, shift=True):
    """_tempo_runs_from を、最初の小節頭を ±T の中で 1ms 刻み(OFFSET の桁)に動かして試し、書く手間(BPM の区間の数と桁、
    そのまま残す小節)がいちばん少なく、次にずれがいちばん小さいものを採る(録画の揺れが片側に寄っていると、元の OFFSET のままでは揺れの幅が
    許す幅の端に来て、途中の小節が収まらなくなる)"""
    best = None
    for k in (range(-int(T * 1000), int(T * 1000) + 1) if shift else (0,)):
        B, c, newly, runs, worst = _tempo_runs_from(recs, grids, frozen, T, k / 1000.0)
        key = (runs + FROZEN_COST * len(newly), worst)
        if best is None or key < best[0]: best = (key, B, c, newly)
    _, B, c, newly = best
    frozen |= newly
    return B, c, newly


def _scroll_runs(recs, grids, B, frozen, vtol=V_TOL, vtol_bar=V_TOL_BAR):
    """流れる速さ(BPM×SCROLL)を保つ SCROLL を、変えずに続けられるだけ続ける。戻り値 {(小節, セル): SCROLL}(値が変わる所だけ)。
    続けてよいのは、区間の要る値(速さ ÷ BPM)の最大と最小の比が許す幅(3%)以内のときだけ。値は区間の中央値の ±半幅の中で桁の少ないもの
    (両側 3% の幅が重なるだけで続けると、1.00 と 1.06 が 1.03 にまとまり、前半の音符がずっと 3% 速く流れた: HIMAWARI HAPPY)"""
    out = {}; st = {'cur': None, 'run': [], 'vals': []}

    def close():
        if not st['run']: return
        vals = st['vals']; mt = min(t for _, t in vals)
        lo = max(v * (1 - t) for v, t in vals); hi = min(v * (1 + t) for v, t in vals)
        med = float(sorted(v for v, _ in vals)[len(vals) // 2])
        lo2, hi2 = max(lo, med * (1 - mt / 2)), min(hi, med * (1 + mt / 2))
        if lo2 <= hi2: s = _nicest(lo2, hi2, med, prefer=st['cur'] if st['cur'] is not None else 1.0)
        else: s = _nicest(lo, hi, min(max(med, lo), hi))
        if st['cur'] is None or abs(s - st['cur']) > 1e-9: out[st['run'][0]] = s
        st['cur'] = s

    for i, r in enumerate(recs):
        if i in frozen:
            close(); st.update(run=[], vals=[], cur=r['sc_out']); continue
        n, cell, _ = grids[i]; pts = {}
        if r['bar_on']:
            pts.setdefault(0, []).append((r['bar_v'] / B[i], vtol if r['notes'] else vtol_bar))
        for cc, (t, ch, v, p) in zip(cell, r['notes']):
            if ch == '8': continue          # 連打の終わりは頭の速さで流れる(make_tja も SCROLL を付けない)
            pts.setdefault(cc, []).append((v / B[i], vtol))
        for cc in sorted(pts):
            cand = st['vals'] + pts[cc]
            ss = [abs(v) for v, _ in cand]; mt = min(t for _, t in cand)
            same_sign = all(v > 0 for v, _ in cand) or all(v < 0 for v, _ in cand) or all(v == 0 for v, _ in cand)
            ok = same_sign and (min(ss) == max(ss) or (min(ss) > 0 and max(ss) / min(ss) <= 1 + mt + 1e-12))
            if not ok:
                close(); st.update(run=[], vals=list(pts[cc]))
            else:
                st['vals'] = cand
            st['run'].append((i, cc))
    close()
    return out


def _emit(recs, grids, B, frozen, scr, bpm_head):
    """清書した小節の列を作る(そのまま残す小節は元の命令ごと。入る時の BPM・SCROLL・拍子が違えば頭で合わせる)"""
    out = []; cur_b = bpm_head; cur_s = 1.0; cur_m = 1.0
    for i, r in enumerate(recs):
        m = Measure()
        if i in frozen:
            raw = r['raw']; head0 = [t for p, t in raw.cmds if p == 0]
            if abs(r['mult'] - cur_m) > 1e-12 and not any(t.startswith('#MEASURE') for t in head0):
                m.cmds.append((0, r['meter'] or '#MEASURE 4/4'))
            if abs(r['bpm0'] - cur_b) > 1e-9 and not any(t.startswith('#BPMCHANGE') for t in head0):
                m.cmds.append((0, '#BPMCHANGE ' + _fmt(r['bpm0'], 6)))
            if abs(r['sc0'] - cur_s) > 1e-9 and not any(t.startswith('#SCROLL') for t in head0):
                m.cmds.append((0, '#SCROLL ' + _fmt(r['sc0'], 6)))
            m.cells = raw.cells; m.cmds += list(raw.cmds)
            cur_b, cur_s, cur_m = r['bpm_out'], r['sc_out'], r['mult']
            out.append(m); continue
        n, cell, cc = grids[i]
        if abs(r['mult'] - cur_m) > 1e-12:
            m.cmds.append((0, r['meter'] or '#MEASURE 4/4')); cur_m = r['mult']
        if abs(B[i] - cur_b) > 1e-9:
            m.cmds.append((0, '#BPMCHANGE ' + _fmt(B[i], 6))); cur_b = B[i]
        slots = ['0'] * n
        for c, nt in zip(cell, r['notes']): slots[c] = nt[1]
        m.cells = ''.join(slots)
        pos_cmd = [(c, txt) for (p, txt, t), c in zip(r['cmds'], cc)]
        for c in sorted(set([0] + list(cell))):
            if (i, c) in scr and abs(scr[(i, c)] - cur_s) > 1e-9:
                pos_cmd.append((c, '#SCROLL ' + _fmt(scr[(i, c)], 6))); cur_s = scr[(i, c)]
        # 同じセルでは #SCROLL を先に、ゴーゴーなどは元の順で後ろに(どちらでも時刻は同じ)
        pos_cmd.sort(key=lambda x: (x[0], 0 if x[1].startswith('#SCROLL') else 1))
        m.cmds += pos_cmd
        out.append(m)
    return out


def reduce_grid(ms):
    """セル数を最小に(音符と命令の位置を保つ最大の約数で割る)。時刻は 1 つも変わらない"""
    out = []
    for m in ms:
        n = len(m.cells)
        if n <= 1: out.append(m.copy()); continue
        g = n
        for p, ch in enumerate(m.cells):
            if ch != '0': g = math.gcd(g, p)
        for p, _ in m.cmds: g = math.gcd(g, p)
        out.append(Measure(m.cells[::g], [(p // g, t) for p, t in m.cmds]) if g > 1 else m.copy())
    return out


def diff(raw_ms, new_ms, bpm_raw, bpm_new, T=T_MOVE, vtol=V_TOL, vtol_bar=V_TOL_BAR, c0=0.0):
    """清書の前後の食い違い: 音符(数・種類・時刻 T 以内・速さ)と小節線(数・見えるか・時刻・速さ)、命令(ゴーゴーなど)の並び。
    c0 = 清書で動かした OFFSET(清書の時刻 + c0 が元の時刻に対応)。戻り値 (食い違った小節の集合, 音符・小節線の最大の時刻差[s])"""
    a = walk(raw_ms, bpm_raw); b = walk(new_ms, bpm_new)
    for r in b:
        r['start'] += c0; r['end'] += c0; r['notes'] = [(t + c0,) + tuple(x) for t, *x in r['notes']]
    if len(a) != len(b): return set(range(len(a))), float('inf')
    bad = set(); worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        worst = max(worst, abs(x['start'] - y['start']))
        if abs(x['start'] - y['start']) > T + 1e-9 or x['bar_on'] != y['bar_on']: bad.add(i)
        if x['bar_on'] and (abs(y['bar_v'] - x['bar_v']) > 1e-9 if x['bar_v'] == 0 else abs(y['bar_v'] / x['bar_v'] - 1) > (vtol if x['notes'] else vtol_bar) + 1e-9): bad.add(i)
        if len(x['notes']) != len(y['notes']): bad.add(i); continue
        for p, q in zip(x['notes'], y['notes']):
            worst = max(worst, abs(p[0] - q[0]))
            if p[1] != q[1] or abs(p[0] - q[0]) > T + 1e-9: bad.add(i)
            if p[1] != '8' and (abs(q[2] - p[2]) > 1e-9 if p[2] == 0 else abs(q[2] / p[2] - 1) > vtol + 1e-9): bad.add(i)
        seq = lambda r: [(txt, sum(1 for nt in r['notes'] if nt[3] < p)) for p, txt, _ in r['cmds']]
        if seq(x) != seq(y): bad.add(i)
    if a and abs(a[-1]['end'] - b[-1]['end']) > T + 1e-9: bad.add(len(a) - 1)
    return bad, worst


def clean_course(ms, bpm0, T=T_MOVE, shift=True):
    """1 コースの小節の列を清書する。shift: OFFSET を動かしてよいか(コースが 1 つのとき)。
    戻り値 (新しい小節の列, 先頭 BPM, 動かした OFFSET の量 c0(新しい OFFSET = 元 − c0), 情報)"""
    recs = walk(ms, bpm0)
    frozen = {i for i, r in enumerate(recs) if r['odd']}
    for _ in range(len(recs) + 1):
        grids = {}
        for i, r in enumerate(recs):
            if i in frozen: continue
            g = _grid(r, r['start'], r['dur'], T) or _grid_fit(r, T)
            if g is None: frozen.add(i)
            else: grids[i] = g
        # 格子は元の小節頭と長さ(検出した小節線)で選ぶ。清書した小節頭(T 以内で動く)で選び直すと、小節頭が片側に寄ったとき
        # 全部の音符が 1/96 小節ずれた細かい格子が「よく当てはまる」ことになり、16 分の小節が 96 分割の 1 セルずらしになった
        B, c, _ = _tempo_runs(recs, grids, frozen, T, shift)
        head = recs[0]['bpm0'] if (recs and 0 in frozen) else next((b for b in B if b is not None), bpm0)
        scr = _scroll_runs(recs, grids, B, frozen)
        new = reduce_grid(_emit(recs, grids, B, frozen, scr, head))
        bad, worst = diff(ms, new, bpm0, head, T, c0=c[0])
        if not bad: return new, head, c[0], {'frozen': len(frozen), 'worst_ms': worst * 1000, 'measures': len(recs), 'offset_ms': c[0] * 1000}
        frozen |= bad
    return reduce_grid(ms), bpm0, 0.0, {'frozen': len(recs), 'worst_ms': 0.0, 'measures': len(recs), 'offset_ms': 0.0}


def clean_text(txt, T=T_MOVE):
    """TJA の全文を清書した全文にする(BPM: の値も書き換える)。戻り値 (全文, 情報)"""
    head, courses = split_tja(txt)
    bpm0 = float(header_value(head, 'BPM'))
    if any(t.startswith(('#BRANCH', '#N', '#E', '#M')) and not t.startswith('#MEASURE')
           for c in courses for m in c['body'] for _, t in m.cmds):
        return txt, {'skip': '譜面分岐は清書しない'}
    info = {}; outs = []; head_bpm = None; c0 = 0.0
    for c in courses:
        new, hb, c0, inf = clean_course(c['body'], bpm0, T, shift=len(courses) == 1)
        if head_bpm is None: head_bpm = hb
        elif abs(hb - head_bpm) > 1e-9 and new:     # コースごとに先頭 BPM が違えば、2 つ目以降は頭に #BPMCHANGE を足す
            new[0] = Measure(new[0].cells, [(0, '#BPMCHANGE ' + _fmt(hb, 6))] + [x for x in new[0].cmds if not (x[0] == 0 and x[1].startswith('#BPMCHANGE'))])
        outs.append((c, new)); info[header_value(c['head'], 'COURSE', '?')] = inf
    lines = []
    for ln in head:
        if re.match(r'\s*BPM:', ln): ln = 'BPM:' + _fmt(head_bpm if head_bpm is not None else bpm0, 6)
        elif re.match(r'\s*OFFSET:', ln) and c0: ln = 'OFFSET:' + _fmt(float(header_value([ln], 'OFFSET')) - c0, 6)
        lines.append(ln)
    for c, new in outs:
        lines += c['head'] + ['#START'] + write_body(new, c['tail']) + ['#END'] + c['after']
    return NL.join(lines), info


def write_tja(path, txt):
    """TJA を UTF-8(BOM 付き)・改行 LF で書く(最後に改行が無ければ足す)"""
    open(path, 'w', encoding='utf-8-sig', newline=NL).write(txt if txt.endswith(NL) else txt + NL)


def stats(txt, course=None):
    """行数(空行を除く)と命令の数。course を渡すとそのコースの #START〜#END だけ(本家は 1 つのファイルに全コースがある)"""
    body = txt.replace('\r', '').split(NL)
    if course is not None:
        out = []; on = False; cur = None
        for l in body:
            m = re.match(r'\s*COURSE:\s*(\S+)', l)
            if m: cur = m.group(1)
            if l.strip().startswith('#START'):
                on = cur is not None and (cur.lower() == course.lower() or cur == {'oni': '3', 'edit': '4', 'hard': '2'}.get(course.lower())); continue
            if l.strip().startswith('#END'):
                if on and out: break
                on = False; continue
            if on: out.append(l)
        body = out
    return {'lines': len([l for l in body if l.strip()]), 'bpm': sum(l.startswith('#BPMCHANGE') for l in body),
            'scroll': sum(l.startswith('#SCROLL') for l in body), 'measure': sum(l.startswith('#MEASURE') for l in body)}


def main():
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    T = T_MOVE
    for x in sys.argv[1:]:
        if x.startswith('--ms='): T = float(x[5:]) / 1000.0
    src = a[0]; dst = a[1] if len(a) > 1 else os.path.splitext(src)[0] + '_clean.tja'
    txt = open(src, encoding='utf-8-sig', errors='replace').read()
    new, info = clean_text(txt, T)
    open(dst, 'w', encoding='utf-8-sig', newline='\n').write(new if new.endswith(NL) else new + NL)
    s0, s1 = stats(txt), stats(new)
    print('清書: %s → %s' % (os.path.basename(src), os.path.basename(dst)))
    print('  行 %d → %d、#BPMCHANGE %d → %d、#SCROLL %d → %d、#MEASURE %d → %d' % (
        s0['lines'], s1['lines'], s0['bpm'], s1['bpm'], s0['scroll'], s1['scroll'], s0['measure'], s1['measure']))
    for k, v in info.items():
        if isinstance(v, dict) and 'worst_ms' in v:
            print('  %s: 動いた最大 %.1f ms(許す幅 %.1f ms)、そのまま残した小節 %d / %d' % (k, v['worst_ms'], T * 1000, v['frozen'], v['measures']))
        else: print('  %s: %s' % (k, v))


if __name__ == '__main__':
    main()
