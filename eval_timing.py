import os, sys, re, numpy as np, detect

def parse_tja_times(path, course='Oni', marks='1234'):
    """TJA の指定コースを解析して [(chart秒, 音符文字), ...] を返す(marks の文字だけ。連打・風船も欲しければ '12345678')。
    小節はカンマ区切り。#BPMCHANGE は小節の途中にも入る(その時点以降のセルの長さが変わる)ので、
    小節をセルの区間ごとに BPM を持たせて積算する。#MEASURE と #DELAY にも対応。行末コメント(//...)は剥がす。"""
    txt = open(path, encoding='utf-8-sig', errors='replace').read()
    m = re.search(r'COURSE:\s*%s(.*?)(#END|\Z)' % course, txt, re.S | re.I)
    seg = m.group(1)
    hb = re.search(r'^BPM:\s*([\d.]+)', txt, re.M)
    bpm = float(hb.group(1))
    st = seg.find('#START'); seg = seg[st + 6:]
    seg = re.sub(r'//[^\n]*', '', seg)      # コメント(//16 など)を除去。残すと数字が音符として数えられ小節もずれる
    t = 0.0; bpmv = bpm; measure_mult = 1.0; notes = []
    tokens = re.split(r'(,|#[A-Z]+[^\n]*)', seg)
    segs = []            # 今の小節の [セル文字列, そのセルたちのBPM] の列
    want = (os.environ.get('BRANCH_PATH') or 'M').upper()   # 譜面分岐でたどる道(動画は 1 本しか映らない。全良なら M)
    branch = None        # 分岐の中で今読んでいる道(None = 分岐の外)
    for tok in tokens:
        tok = tok.strip('\r')
        if tok in ('#N', '#E', '#M'):
            branch = tok[1]; continue
        if tok.startswith('#BRANCHSTART'):
            branch = None; continue
        if tok.startswith('#BRANCHEND'):
            branch = None; continue
        if branch is not None and branch != want:
            continue     # 選ばなかった道は読み飛ばす(時間も進めない)
        if tok == ',':
            cells = ''.join(c for c, _ in segs)
            n = len(cells) if cells else 1
            if not cells:
                t += (60.0 / bpmv) * 4 * measure_mult
            else:
                for c, b in segs:
                    dur_cell = (60.0 / b) * 4 * measure_mult / n
                    for ch in c:
                        if ch in marks: notes.append((t, ch))
                        t += dur_cell
            segs = []
        elif tok.startswith('#'):
            mm = re.match(r'#BPMCHANGE\s+([\d.]+)', tok)
            if mm: bpmv = float(mm.group(1))
            mm = re.match(r'#MEASURE\s+(\d+)/(\d+)', tok)
            if mm: measure_mult = int(mm.group(1)) / int(mm.group(2))
            mm = re.match(r'#DELAY\s+([-\d.]+)', tok)
            if mm: t += float(mm.group(1))
        else:
            cells = re.sub(r'\s', '', tok)
            if cells: segs.append([cells, bpmv])
    return bpm, notes

def parse_tja_bars(path, course='Oni'):
    """TJA の指定コースの小節頭の時刻(chart 秒)を返す。parse_tja_times と同じ走査。
    コースが違う動画の同期に使う(音符は対応しないが、小節の格子は曲の音楽に従うので全コース共通)"""
    txt = open(path, encoding='utf-8-sig', errors='replace').read()
    m = re.search(r'COURSE:\s*%s(.*?)(#END|\Z)' % course, txt, re.S | re.I)
    if not m: return []
    seg = m.group(1)
    bpm = float(re.search(r'^BPM:\s*([\d.]+)', txt, re.M).group(1))
    st = seg.find('#START'); seg = seg[st + 6:]
    seg = re.sub(r'//[^\n]*', '', seg)
    t = 0.0; bpmv = bpm; measure_mult = 1.0; heads = []; segs = []
    want = (os.environ.get('BRANCH_PATH') or 'M').upper()
    branch = None
    for tok in re.split(r'(,|#[A-Z]+[^\n]*)', seg):
        tok = tok.strip('\r')
        if tok in ('#N', '#E', '#M'):
            branch = tok[1]; continue
        if tok.startswith('#BRANCHSTART') or tok.startswith('#BRANCHEND'):
            branch = None; continue
        if branch is not None and branch != want:
            continue
        if tok == ',':
            heads.append(t)
            cells = ''.join(c for c, _ in segs); n = len(cells) if cells else 1
            if not cells:
                t += (60.0 / bpmv) * 4 * measure_mult
            else:
                for c, b in segs: t += (60.0 / b) * 4 * measure_mult / n * len(c)
            segs = []
        elif tok.startswith('#'):
            mm = re.match(r'#BPMCHANGE\s+([\d.]+)', tok)
            if mm: bpmv = float(mm.group(1))
            mm = re.match(r'#MEASURE\s+(\d+)/(\d+)', tok)
            if mm: measure_mult = int(mm.group(1)) / int(mm.group(2))
            mm = re.match(r'#DELAY\s+([-\d.]+)', tok)
            if mm: t += float(mm.group(1))
        else:
            cells = re.sub(r'\s', '', tok)
            if cells: segs.append([cells, bpmv])
    return heads

def bar_shift(bars, heads, base, span=0.8, step=0.0005):
    """検出した小節線を参照の小節頭に合わせるシフトを探す。戻り値 (ずれの中央値[秒], シフト, 対応した本数)"""
    import numpy as np
    b = np.asarray(bars, float); h = np.asarray(sorted(heads), float)
    if len(b) < 5 or len(h) < 5: return None
    best = None
    for sh in np.arange(base - span, base + span, step):
        idx = np.searchsorted(h, b - sh)
        lo = np.clip(idx - 1, 0, len(h) - 1); hi = np.clip(idx, 0, len(h) - 1)
        d = np.minimum(np.abs(b - sh - h[lo]), np.abs(b - sh - h[hi]))
        m = float(np.median(d))
        if best is None or m < best[0]: best = (m, float(sh), int((d < 0.03).sum()))
    return best

def main():
    video, s, e, tja = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    bpm, tnotes = parse_tja_times(tja)
    tt = np.array([n[0] for n in tnotes]); tc = [n[1] for n in tnotes]
    fps, ons = detect.run(video, s, e)
    ons = [o for o in ons if o['color'] in ('don', 'ka')]
    mt = np.array([o['t'] for o in ons])
    mc = [('3' if o['big'] else '1') if o['color'] == 'don' else ('4' if o['big'] else '2') for o in ons]
    tt0 = tt - tt[0]; mt0 = mt - mt[0]
    best = None
    for sh in np.arange(-0.5, 0.5, 0.005):
        c = 0
        b = tt0 + sh
        bi = 0
        for x in mt0:
            while bi < len(b) and b[bi] < x - 0.025: bi += 1
            if bi < len(b) and abs(b[bi] - x) <= 0.025: c += 1
        if best is None or c > best[0]: best = (c, sh)
    sh = best[1]
    b = tt0 + sh
    matched_true = set(); matched_mine = 0; type_ok = 0
    order = np.argsort(mt0)
    used = np.zeros(len(b), bool)
    for k in order:
        x = mt0[k]
        lo = np.searchsorted(b, x - 0.03)
        cand = [(abs(b[j]-x), j) for j in range(lo, min(len(b), lo+4)) if not used[j] and abs(b[j]-x)<=0.03]
        if cand:
            _, j = min(cand); used[j]=True; matched_mine += 1; matched_true.add(j)
            if mc[k].replace('3','1').replace('4','2') == tc[j].replace('3','1').replace('4','2'): type_ok += 1
    rec = len(matched_true)/len(tt)*100; prec = matched_mine/len(mt)*100
    print('BPM(true) %.0f  正解音符 %d  検出 %d  シフト %.3fs' % (bpm, len(tt), len(mt), sh))
    print('recall %.1f%%  precision %.1f%%  マッチ内の種別(色)正解 %.1f%%' % (rec, prec, type_ok/max(1,matched_mine)*100))

if __name__ == "__main__":
    main()
