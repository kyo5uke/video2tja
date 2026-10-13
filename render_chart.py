"""TJA の譜面を、ゲームのように流れる動画にする(譜面の確認用。自動採譜で作った譜面・試験用の譜面を目で見る)。
使い方: python render_chart.py <譜面.tja> <出力.mp4> [コース(既定 Oni)] [--from 秒] [--to 秒]
  音源(WAVE)があれば音も付ける。音符・小節線は「その音符の時点の BPM × SCROLL」の速さで流れる(ゲームと同じ決まり)。
  縮尺はゲーム画面(1080p)の 2/3: 判定枠 x=413、小音符の半径 33px、大音符 56px、速さは 1 BPM あたり 0.0988px/フレーム(60fps、1080p)。
  先に着く音符を上に描く(ゲームと同じ重なり方)。判定枠に着いた音符は消える(オートで叩いたのと同じ)。
  上の帯に区間の名前(TJA の「//----」の行)・BPM・SCROLL・小節の番号・時刻を出す。"""
import sys, os, re, subprocess
import numpy as np, cv2

W, H = 1280, 300
JX = 413                               # 判定枠(1080p の 619.4 の 2/3)
LANE_Y0, LANE_Y1 = 110, 260
CY = (LANE_Y0 + LANE_Y1) // 2
FPS = 60
K = 0.0988 * 60 * (2.0 / 3.0)          # px/秒 per (BPM × SCROLL)
R_S, R_B = 33, 56                      # 外周(黒い輪郭まで)
DON, KA = (40, 70, 235), (200, 190, 60)            # BGR
ROLL, BALLOON = (40, 200, 250), (40, 140, 250)


def parse(path, course):
    txt = open(path, encoding='utf-8-sig', errors='replace').read()
    head = txt.split('COURSE:')[0]
    bpm = float(re.search(r'^BPM:\s*([-\d.]+)', head, re.M).group(1))
    off = float(re.search(r'^OFFSET:\s*([-\d.]+)', head, re.M).group(1))
    wave = re.search(r'^WAVE:\s*(.+?)\s*$', head, re.M)
    m = re.search(r'COURSE:\s*' + re.escape(course) + r'\b.*?#START(.*?)#END', txt, re.S | re.I)
    if not m: sys.exit('コース %s が見つかりません' % course)
    body = m.group(1)
    t = -off; scroll = 1.0; num, den = 4, 4; gogo = False; barline = True; label = ''
    notes, bars, gogo_iv, labels, info = [], [], [], [], []      # info: (時刻, BPM, SCROLL, 小節の番号)
    cells = []; pending_label = None; nmeas = 0; gogo_t0 = None

    def flush():
        nonlocal t, nmeas, gogo_t0, pending_label
        if not cells: cells.append(('0', bpm, scroll, gogo))
        n = len(cells)
        nmeas += 1
        if barline_at[0]: bars.append((t, cells[0][1] * cells[0][2]))
        if pending_label: labels.append((t, pending_label)); pending_label = None
        info.append((t, cells[0][1], cells[0][2], nmeas))
        for ch, b, sc, gg in cells:
            if gg and gogo_t0 is None: gogo_t0 = t
            if not gg and gogo_t0 is not None: gogo_iv.append((gogo_t0, t)); gogo_t0 = None
            if ch in '12345678': notes.append([t, ch, b * sc])
            t += (num / float(den)) * 4 * 60.0 / b / n
        cells.clear()
    barline_at = [True]
    for raw in body.split('\n'):
        s_ = raw.strip()
        mlab = re.match(r'//-+\s*(.*?)\s*-*\s*$', s_)
        if mlab and mlab.group(1): pending_label = mlab.group(1); continue
        line = raw.split('//')[0].strip()
        if not line: continue
        if line.startswith('#'):
            p = line.split()
            if p[0] == '#BPMCHANGE': bpm = float(p[1])
            elif p[0] == '#SCROLL': scroll = float(p[1])
            elif p[0] == '#MEASURE': num, den = [int(x) for x in p[1].split('/')]
            elif p[0] == '#GOGOSTART': gogo = True
            elif p[0] == '#GOGOEND': gogo = False
            elif p[0] == '#BARLINEOFF': barline = False
            elif p[0] == '#BARLINEON': barline = True
            elif p[0] == '#DELAY': t += float(p[1])
            continue
        if not cells: barline_at[0] = barline
        for ch in line.rstrip(','):
            if ch.isdigit(): cells.append((ch, bpm, scroll, gogo))
        if line.endswith(','): flush()
    if gogo_t0 is not None: gogo_iv.append((gogo_t0, t))
    # 連打・風船: 頭(5/6/7)から次の 8 まで
    rolls = []
    for i, n_ in enumerate(notes):
        if n_[1] in '567':
            end = next((q for q in notes[i + 1:] if q[1] == '8'), None)
            if end: rolls.append((n_[0], end[0], n_[1], n_[2]))
    notes = [n_ for n_ in notes if n_[1] in '1234']
    return notes, bars, gogo_iv, rolls, labels, info, (wave.group(1) if wave else None), -off


def draw_note(img, x, big, color):
    r = R_B if big else R_S
    cv2.circle(img, (x, CY), r, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (x, CY), r - 3, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(img, (x, CY), int(r * 0.78), color, -1, cv2.LINE_AA)
    e = max(2, r // 9)                                              # 目
    cv2.circle(img, (x - r // 3, CY - r // 6), e, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (x + r // 3, CY - r // 6), e, (20, 20, 20), -1, cv2.LINE_AA)


def main():
    a = sys.argv[1:]
    if len(a) < 2: print(__doc__); return
    tja, out = a[0], a[1]
    course = a[2] if len(a) > 2 and not a[2].startswith('--') else 'Oni'
    notes, bars, gogo_iv, rolls, labels, info, wave, t_start = parse(tja, course)
    ogg = os.path.join(os.path.dirname(tja), wave) if wave else None
    if ogg and not os.path.exists(ogg): ogg = None
    t_end = max([n[0] for n in notes] + [r[1] for r in rolls] + [0]) + 2.0
    T0 = float(a[a.index('--from') + 1]) if '--from' in a else 0.0
    T1 = float(a[a.index('--to') + 1]) if '--to' in a else t_end
    print('音符 %d・小節 %d・ゴーゴー %d・連打/風船 %d、%.1f〜%.1f 秒を書き出す' % (len(notes), len(bars), len(gogo_iv), len(rolls), T0, T1), flush=True)
    _tx = open(tja, encoding='utf-8-sig', errors='replace').read()
    title = re.search(r'^TITLEJA:\s*(.+?)\s*$', _tx, re.M) or re.search(r'^TITLE:\s*(.+?)\s*$', _tx, re.M)
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-']
    if ogg: cmd += ['-ss', '%.3f' % max(0.0, T0), '-i', ogg, '-c:a', 'aac', '-b:a', '128k']
    cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '24', '-pix_fmt', 'yuv420p', '-shortest' if ogg else '-y', out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # 日本語の文字は OpenCV で描けないので、区間の名前は画像にして貼る(PIL があれば)
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for fp in ('C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/msgothic.ttc'):
            if os.path.exists(fp): font = ImageFont.truetype(fp, 22); font_s = ImageFont.truetype(fp, 16); break
    except Exception:
        font = None
    cache = {}

    def text_img(s, small=False):
        key = (s, small)
        if key not in cache:
            f_ = font_s if small else font
            im = Image.new('RGB', (W, 30), (24, 24, 24)); d = ImageDraw.Draw(im); d.text((0, 2), s, font=f_, fill=(230, 230, 230))
            cache[key] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        return cache[key]
    notes_rev = sorted(notes, key=lambda n: -n[0])           # 後に着く音符から描く(先に着く音符が上)
    li = [x[0] for x in info]
    for i in range(int((T1 - T0) * FPS)):
        now = T0 + i / FPS
        img = np.full((H, W, 3), 24, np.uint8)
        gg = any(g0 <= now < g1 for g0, g1 in gogo_iv)
        cv2.rectangle(img, (0, LANE_Y0), (W, LANE_Y1), (30, 60, 110) if gg else (45, 45, 45), -1)
        for tb, spd in bars:
            x = int(round(JX + (tb - now) * spd * K))
            if JX <= x < W: cv2.line(img, (x, LANE_Y0), (x, LANE_Y1), (210, 210, 210), 2)
        cv2.circle(img, (JX, CY), R_S, (160, 160, 160), 2, cv2.LINE_AA); cv2.circle(img, (JX, CY), R_B, (110, 110, 110), 2, cv2.LINE_AA)
        for s0, s1, kind, spd in rolls:
            if s1 < now: continue
            x0 = int(round(JX + (max(s0, now) - now) * spd * K)); x1 = int(round(JX + (s1 - now) * spd * K))
            if x0 > W + R_B: continue
            r = R_B if kind == '6' else R_S; c = BALLOON if kind == '7' else ROLL
            cv2.rectangle(img, (x0, CY - r + 4), (min(x1, W + r), CY + r - 4), c, -1)
            cv2.circle(img, (min(x1, W + r), CY), r - 4, c, -1, cv2.LINE_AA)
            draw_note(img, x0, kind == '6', c)
        for tn, c, spd in notes_rev:
            if tn < now: continue
            x = int(round(JX + (tn - now) * spd * K))
            if x > W + R_B: continue
            draw_note(img, x, c in '34', DON if c in '13' else KA)
        # 上の帯: 区間・BPM・SCROLL・小節・時刻
        k = max(0, int(np.searchsorted(li, now, side='right')) - 1)
        ti, b_, sc_, mn = info[k] if info else (0, 0, 0, 0)
        lab = [s for tl, s in labels if tl <= now + 1e-6]
        if font is not None:
            img[8:38] = text_img((title.group(1) if title else os.path.basename(tja)) + ('  ―  ' + lab[-1] if lab else ''))
            img[40:70] = text_img('BPM %g   SCROLL %g   小節 %d   %.2f 秒%s' % (b_, sc_, mn, now, '   ゴーゴー' if gg else ''), small=True)
        else:
            cv2.putText(img, 'BPM %g  SCROLL %g  bar %d  %.2fs' % (b_, sc_, mn, now), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        ff.stdin.write(img.tobytes())
        if i % (FPS * 10) == 0: print('  %.0f 秒' % (now - T0), flush=True)
    ff.stdin.close(); ff.wait()
    print('書き出し:', out)


if __name__ == '__main__':
    main()
