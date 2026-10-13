"""元動画のレーン(上)・参照譜面 参照(中)・自動採譜(下)を縦に並べた比較動画。譜面はゲームと同じ速さ(BPM×SCROLL)で流す。
相違のある音符に赤丸、画面下に説明。等速(音あり)のあと、同じ区間を 1/4 スロー(無音)で繰り返す。

使い方: python render_compare_ref.py <out.mp4> <動画> <参照.tja> <採譜.tja> <コース> <動画 T0> <動画 T1> [題名] [採譜の検出 JSON]
  ・時刻はすべて動画の秒。参照 と採譜の OFFSET は同じ音源(参照 の WAVE)基準
  ・動画と音源の対応(Δ: 音源 t = 動画 t − Δ)は run_song.find_delta で求める。映像のずれは採譜の検出 JSON と採譜 TJA の音符の差で合わせる
環境変数: SLOW(既定 0.25。0 でスロー無し)、MATCH_TOL(相違とみなす時刻差、既定 0.035 秒)
"""
import sys, os, re, json, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W = 1280
FPS = 30
JUDGE_X = 240
SCALE = 2.0 / 3.0                           # ゲーム画面(1920 幅)を 1280 幅に縮小
K_PX_PER_S = 0.0988 * 60 * SCALE            # 音符の速さ [px/s] = K × BPM × SCROLL(ゲーム画面と同じ、縮小後)
VID_Y0, VID_H = 60, 127                     # 動画の帯(レーン y290-480 を縮小)
LANES = (((255, 190, 90), '参照譜面', 262), ((120, 200, 255), '採譜', 402))
H = 560
R, RBIG = 16, 24
FONT = next((p for p in (r'C:/Windows/Fonts/meiryob.ttc', r'C:/Windows/Fonts/meiryo.ttc', r'C:/Windows/Fonts/YuGothB.ttc', r'C:/Windows/Fonts/msgothic.ttc') if os.path.exists(p)), None)
NAME = {'1': 'ドン', '2': 'カッ', '3': '大ドン', '4': '大カッ', '5': '連打', '6': '大連打', '7': '風船', '9': '風船', '8': '終端'}


GOGO = {}


def parse_tja(path, course):
    """→ 音符 [(音源秒, 記号, 速さ BPM×SCROLL)], 小節線 [(音源秒, 速さ, 見えるか)], 連打 [(頭, 終わり, 記号, 速さ)]。
    #MEASURE #BPMCHANGE #SCROLL #DELAY #BARLINEOFF/ON、譜面分岐は達人(#M)をたどる(採点の eval_timing と同じ)"""
    txt = open(path, encoding='utf-8-sig').read()
    head = txt.split('COURSE:')[0]
    bpm = float(re.search(r'^BPM:\s*([\d.]+)', head, re.M).group(1)); off = float(re.search(r'^OFFSET:\s*([-\d.]+)', head, re.M).group(1))
    m = re.search(r'COURSE:\s*' + course + r'\s.*?#START(.*?)#END', txt, re.S | re.I)
    t = -off; scroll = 1.0; num, den = 4.0, 4.0; barline = True; branch = None
    notes = []; bars = []; buf = []; gg = []
    def flush():
        nonlocal t
        n = sum(1 for x in buf if x[0] not in 'GE')
        for ch, b, sc, bl in buf:
            if ch == 'G': gg.append([t, None]); continue
            if ch == 'E':
                if gg and gg[-1][1] is None: gg[-1][1] = t
                continue
            if ch in '123456789': notes.append((t, ch, b * sc))
            t += num / den * 4 * 60.0 / b / n
        buf.clear()
    for raw in m.group(1).split('\n'):
        line = raw.split('//')[0].strip()
        if not line: continue
        if line.startswith('#'):
            p = line.split()
            if p[0] in ('#N', '#E', '#M'): branch = p[0]; continue
            if p[0].startswith('#BRANCHSTART'): branch = 'X'; continue
            if p[0] == '#BRANCHEND': branch = None; continue
            if branch not in (None, '#' + os.environ.get('BRANCH_PATH', 'M').upper()): continue
            if p[0] == '#BPMCHANGE': bpm = float(p[1])
            elif p[0] == '#SCROLL': scroll = float(p[1])
            elif p[0] == '#MEASURE': num, den = [float(x) for x in p[1].split('/')]
            elif p[0] == '#DELAY': t += float(p[1])
            elif p[0] == '#BARLINEOFF': barline = False
            elif p[0] == '#BARLINEON': barline = True
            elif p[0] == '#GOGOSTART': buf.append(('G', bpm, scroll, barline))
            elif p[0] == '#GOGOEND': buf.append(('E', bpm, scroll, barline))
            continue
        if branch not in (None, '#' + os.environ.get('BRANCH_PATH', 'M').upper()): continue
        if not any(x[0] not in 'GE' for x in buf): bars.append((t, bpm * scroll, barline))
        for ch in line.rstrip(','):
            if ch.isdigit(): buf.append((ch, bpm, scroll, barline))
        if line.endswith(','):
            if not any(x[0] not in 'GE' for x in buf): buf.append(('0', bpm, scroll, barline))
            flush()
    rolls = []; hd = None
    for tt, c, sp in notes:
        if c in '5679':
            if hd is None: hd = (tt, c, sp)        # 連打中の 2 つ目の頭は続き(Bai Bai FIGHT! の「9009,8」のくす玉は最初の 9 が頭)
        elif c == '8' and hd: rolls.append((hd[0], tt, hd[1], hd[2])); hd = None
    GOGO[path] = [(a, b if b is not None else t) for a, b in gg]      # ゴーゴー区間(音源秒)。戻り値は変えずに別に残す
    return [x for x in notes if x[1] in '1234'], bars, rolls


def match(a, b, tol=0.035, slip=0.12):
    """音符の対応付け(区間シフト無し: 時刻の絶対値で)。戻り値: 相違 [(音源秒, 説明, aのindex, bのindex)]"""
    ta = np.array([x[0] for x in a]); tb = np.array([x[0] for x in b])
    pairs = sorted((abs(ta[i] - tb[j]), i, j) for i in range(len(a)) for j in np.where(np.abs(tb - ta[i]) <= slip)[0])
    ua = set(); ub = set(); out = []
    for d, i, j in pairs:
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j)
        dd = tb[j] - ta[i]
        if abs(dd) > tol: out.append((ta[i], '%s: 採譜が %+.0fms' % (NAME[a[i][1]], dd * 1000), i, j))
        elif a[i][1] != b[j][1]: out.append((ta[i], '種類: 参照=%s / 採譜=%s' % (NAME[a[i][1]], NAME[b[j][1]]), i, j))
    for i in range(len(a)):
        if i not in ua: out.append((ta[i], '参照 だけにある %s' % NAME[a[i][1]], i, None))
    for j in range(len(b)):
        if j not in ub: out.append((tb[j], '採譜だけにある %s' % NAME[b[j][1]], None, j))
    return sorted(out)


def text(img, s, xy, size=22, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    dr.text(xy, s, font=ImageFont.truetype(FONT, size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def color(c): return (60, 60, 235) if c in '13' else (235, 190, 60)


def draw_lane(img, y0, now, notes, bars, rolls, marks):
    cv2.rectangle(img, (0, y0 - 40), (W, y0 + 40), (40, 40, 40), -1)
    for tb, sp, vis in bars:
        x = int(JUDGE_X + (tb - now) * sp * K_PX_PER_S)
        if vis and 0 <= x < W: cv2.line(img, (x, y0 - 40), (x, y0 + 40), (150, 150, 150), 1)
    for h, e, c, sp in rolls:
        x0 = int(JUDGE_X + (h - now) * sp * K_PX_PER_S); x1 = int(JUDGE_X + (e - now) * sp * K_PX_PER_S)
        if x1 >= -40 and x0 <= W + 40:
            rr = RBIG if c in '69' else R
            cv2.rectangle(img, (max(-40, x0), y0 - rr + 4), (min(W + 40, x1), y0 + rr - 4), (40, 200, 240), -1)
            cv2.circle(img, (x0, y0), rr, (40, 200, 240) if c in '56' else (60, 120, 240), -1); cv2.circle(img, (x0, y0), rr, (255, 255, 255), 2)
    for k in range(len(notes) - 1, -1, -1):                    # 早い音符を上に描く(ゲームと同じ)
        tn, c, sp = notes[k]
        x = int(JUDGE_X + (tn - now) * sp * K_PX_PER_S)
        if -40 < x < W + 40:
            r = RBIG if c in '34' else R
            cv2.circle(img, (x, y0), r, color(c), -1); cv2.circle(img, (x, y0), r, (255, 255, 255), 2)
            if k in marks: cv2.circle(img, (x, y0), r + 8, (0, 0, 255), 3)


def main():
    out, video, ref_tja, mine, course, v0, v1 = sys.argv[1:8]; v0, v1 = float(v0), float(v1)
    title = sys.argv[8] if len(sys.argv) > 8 else os.path.basename(out)
    ons_json = sys.argv[9] if len(sys.argv) > 9 else None
    slow = float(os.environ.get('SLOW', '0.25'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    wave = re.search(r'^WAVE:\s*(.+)$', open(ref_tja, encoding='utf-8-sig').read(), re.M).group(1).strip()
    ogg = os.path.join(os.path.dirname(ref_tja), wave)
    import run_song
    delta = run_song.find_delta_cached(video, ogg)[0]            # 音源 t = 動画 t − Δ
    en, eb, er = parse_tja(ref_tja, course); mn, mb, mr = parse_tja(mine, course if course in open(mine, encoding='utf-8-sig').read() else 'Oni')
    # 映像のずれ: 採譜の検出(動画秒)と採譜 TJA の音符(音源秒)の差の中央値。映像は音声より遅れて映ることがある
    pic = delta
    if ons_json and os.path.exists(ons_json):
        det = np.array(sorted(o['t'] for o in json.load(open(ons_json, encoding='utf-8'))['onsets'] if o['color'] in ('don', 'ka')))
        # 曲全体で ±200ms を 1ms 刻みで探し、5ms 以内に検出がある採譜の音符が最も多いずれを採る(一番近い音符との差だと、
        # 16 分の間隔(VERTeX 63ms)と映像のずれ(64ms)が重なって隣の音符と対応させ、ずれを 0 と誤認した)
        mt = np.array([x[0] for x in mn]) + delta
        sc = []
        # ±1 秒・3ms 以内(Nivalis 裏は動画と音源の対応 Δ の推定が 214ms ずれていて ±200ms の外だった。5ms だと 32 分の周期の偽の山と差が出ない)
        for sft in np.arange(-1.0, 1.0005, 0.001):
            j = np.clip(np.searchsorted(det, mt + sft), 1, len(det) - 1)
            sc.append((int((np.minimum(np.abs(det[j] - mt - sft), np.abs(det[j - 1] - mt - sft)) < 0.003).sum()), float(sft)))
        # 32 分が並ぶ曲は 1 周期ずれた所もほぼ同じ数だけ一致する(Nivalis 裏: +20ms 780 個 / +136ms 786 個で +136ms を選び、譜面が映像とずれた)。
        # 最多の 97% 以上のうち、ずれの小さいものを採る
        top = max(n for n, _ in sc)
        pic = delta + min((x for x in sc if x[0] >= 0.97 * top), key=lambda x: abs(x[1]))[1]
    # 参照 の OFFSET と採譜の OFFSET(映像に合わせた)は数十 ms 違うことがある。曲全体のずれ(区間の前後 20 秒で、±100ms を 1ms 刻み、
    # 10ms 以内に相手がいる音符が最多のずれ)で 参照 を採譜の時間軸に寄せてから描き・比べる(採点の eval_ons と同じ考え方)
    ta = np.array([x[0] for x in en]); tb = np.array(sorted(x[0] for x in mn))
    sel = ta[(ta + delta > v0 - 20) & (ta + delta < v1 + 20)]; ref_sh = 0.0
    if len(sel) and len(tb):
        best = None
        for sft in np.arange(-0.1, 0.1005, 0.001):
            j = np.clip(np.searchsorted(tb, sel + sft), 1, len(tb) - 1)
            n_ = int((np.minimum(np.abs(tb[j] - sel - sft), np.abs(tb[j - 1] - sel - sft)) < 0.010).sum())
            if best is None or n_ > best[0]: best = (n_, sft)
        ref_sh = best[1]
    gg_e = [(a + ref_sh, b + ref_sh) for a, b in GOGO.get(ref_tja, [])]; gg_m = GOGO.get(mine, [])
    en = [(t + ref_sh, c, sp) for t, c, sp in en]; eb = [(t + ref_sh, sp, v) for t, sp, v in eb]; er = [(h + ref_sh, e + ref_sh, c, sp) for h, e, c, sp in er]
    diffs = match(en, mn, tol=float(os.environ.get('MATCH_TOL', '0.035')))
    me = {d[2] for d in diffs if d[2] is not None}; mm = {d[3] for d in diffs if d[3] is not None}
    near = [d for d in diffs if v0 - delta - 0.2 <= d[0] <= v1 - delta + 0.2]
    print('%s: Δ %.3f 映像 %+.0fms  参照 を %+.0fms ずらす  区間の相違 %d 件' % (title, delta, (pic - delta) * 1000, ref_sh * 1000, len(near)))
    for d in near: print('   動画 %.2fs  %s' % (d[0] + pic, d[1]))
    # 区間の帯(レーン y290-480、判定枠 x617.8 が縮小後に JUDGE_X に来るよう x258 から)を ffmpeg で正確な時刻から切り出す
    # (OpenCV のシークは報告する時刻が数フレームずれることがあり、譜面と 60ms ずれて見えた)
    import tempfile
    fps_v = 60.0
    t_start = np.floor(max(0.0, v0 - 0.2) * fps_v) / fps_v; bw, bh = int(round(1662 * SCALE)), VID_H
    tmp = os.path.join(tempfile.gettempdir(), 'v2t_cmp_%d.raw' % os.getpid())
    r = subprocess.run(['ffmpeg', '-y', '-v', 'info', '-nostats', '-ss', '%.4f' % t_start, '-t', '%.4f' % (v1 - v0 + 0.5), '-copyts', '-i', video,
                        '-vf', 'crop=1662:190:258:290,scale=%d:%d,showinfo' % (bw, bh), '-f', 'rawvideo', '-pix_fmt', 'bgr24', tmp], capture_output=True)
    pts = [float(x) for x in re.findall(rb'pts_time:\s*([-\d.]+)', r.stderr)]
    raw = np.fromfile(tmp, np.uint8); os.remove(tmp)
    raw = raw[:len(raw) // (bw * bh * 3) * bw * bh * 3].reshape(-1, bh, bw, 3)
    frames = [(pts[k], raw[k]) for k in range(min(len(pts), len(raw)))]
    fts = np.array([f[0] for f in frames])
    seq = [(v0 + i / FPS, False) for i in range(int((v1 - v0) * FPS))]
    if slow > 0: seq += [(v0 + i * slow / FPS, True) for i in range(int((v1 - v0) * FPS / slow))]
    T = len(seq) / FPS
    ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
                           '-ss', '%.3f' % max(0.0, v0 - delta), '-i', ogg, '-filter_complex', '[1:a]atrim=0:%.3f,apad=whole_dur=%.3f[a]' % (v1 - v0, T),
                           '-map', '0:v', '-map', '[a]', '-t', '%.3f' % T, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', out], stdin=subprocess.PIPE)
    for vt, is_slow in seq:
        k = int(np.searchsorted(fts, vt + 1e-4)) - 1; ok = 0 <= k < len(frames)
        now = vt - pic                                        # 譜面の現在時刻(音源秒)。映像と同じ位置に来るよう映像のずれで合わせる
        img = np.full((H, W, 3), 24, np.uint8)
        if ok:
            band = frames[k][1][:VID_H, :W]; img[VID_Y0:VID_Y0 + band.shape[0], :band.shape[1]] = band
        img = text(img, title, (8, 4), 24, (255, 255, 255))
        img = text(img, '元動画', (8, VID_Y0 - 26), 18, (200, 200, 200))
        img = text(img, '動画 %.2f 秒%s' % (vt, '   1/4 スロー' if is_slow else '   等速'), (W - 330, 8), 22, (80, 220, 255) if is_slow else (200, 200, 200))
        for (lc, label, y0), (ns, bs, rs), marks, ggs in zip(LANES, ((en, eb, er), (mn, mb, mr)), (me, mm), (gg_e, gg_m)):
            draw_lane(img, y0, now, ns, bs, rs, marks)
            if any(a <= now < b for a, b in ggs):         # ゴーゴー中(判定枠の時刻で)は枠を橙にして表示
                cv2.rectangle(img, (2, y0 - 40), (W - 3, y0 + 40), (0, 140, 255), 4)
                img = text(img, 'ゴーゴー中', (W - 150, y0 - 70), 20, (0, 160, 255))
            img = text(img, label + ('   (曲全体のずれ %+.0fms を合わせて表示)' % (ref_sh * 1000) if label.startswith('参照譜面') else ''), (8, y0 - 70), 20, lc)
        cv2.line(img, (JUDGE_X, VID_Y0), (JUDGE_X, H - 70), (255, 255, 255), 2)
        cur = [d for d in near if -0.5 <= d[0] - now <= 0.8]
        if cur: img = text(img, '相違  動画 %.2f 秒ごろ  %s' % (cur[0][0] + pic, cur[0][1]), (8, H - 50), 26, (90, 90, 255))
        ff.stdin.write(img.tobytes())
    ff.stdin.close(); ff.wait()
    print('書き出し', out, '%.1f 秒' % T)


if __name__ == '__main__':
    main()
