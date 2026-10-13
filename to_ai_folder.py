"""本家の曲フォルダ(音源と TJA のヘッダー)を元に、譜面だけ自動採譜のコースに置き換えた曲フォルダを「AI譜面」ジャンルに作る。
使い方: to_ai_folder.py <本家の曲フォルダ> <自動採譜の TJA> <コース名(Oni/Edit…)> [AI譜面フォルダ(既定 Songs/13 AI譜面)] [--ref <本家の TJA の名前>]
  - ヘッダー(TITLE/SUBTITLE/WAVE/DEMOSTART…)は本家のまま。BPM と OFFSET は自動採譜側(音源に合わせて出した値)に置き換える。
  - コースは自動採譜のものだけ入れる(OFFSET は 1 ファイルに 1 つなので、本家の他コースは同居できない)。LEVEL は本家の同コースの値。
  - 音源(WAVE)は本家からコピー。フォルダ名は「<連番> <本家フォルダの曲名>」。box.def が無ければ作る。
"""
import sys, os, re, glob, shutil

def main():
    ref = None
    if '--ref' in sys.argv:                     # 曲フォルダに TJA が複数あるとき、ヘッダーと音源を使う TJA(採譜で参照したもの)
        k = sys.argv.index('--ref'); ref = sys.argv[k + 1]; del sys.argv[k:k + 2]
    src_dir, my_tja, course = sys.argv[1], sys.argv[2], sys.argv[3]
    ai_root = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.environ.get('V2T_SONGS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Songs'), '13 AI譜面')
    src_tja = os.path.join(src_dir, ref) if ref else sorted(glob.glob(os.path.join(src_dir, '*.tja')))[0]
    orig = open(src_tja, encoding='utf-8-sig').read().replace(chr(13), '')
    mine = open(my_tja, encoding='utf-8-sig').read().replace(chr(13), '')
    head = orig.split('COURSE:')[0]
    my_head = mine.split('COURSE:')[0]
    my_bpm = re.search(r'^BPM:\s*([\d.]+)', my_head, re.M).group(1)
    my_off = re.search(r'^OFFSET:\s*([-\d.]+)', my_head, re.M).group(1)
    head = re.sub(r'^BPM:.*$', 'BPM:' + my_bpm, head, flags=re.M)
    head = re.sub(r'^OFFSET:.*$', 'OFFSET:' + my_off, head, flags=re.M)
    # 自動採譜のコース本体
    m = re.search(r'COURSE:\s*' + course + r'\s*\n(.*?)#END', mine, re.S)
    if not m: raise SystemExit('自動採譜の TJA に COURSE:%s が無い' % course)
    blk = m.group(1)
    lvl = re.search(r'COURSE:\s*' + course + r'\s*\n(?:(?!COURSE:).)*?^LEVEL:\s*(\d+)', orig, re.M | re.S)
    if lvl: blk = re.sub(r'^LEVEL:.*$', 'LEVEL:' + lvl.group(1), blk, count=1, flags=re.M)
    wave = re.search(r'^WAVE:\s*(.+)$', head, re.M).group(1).strip()
    name = re.sub(r'^\d+\s+', '', os.path.basename(os.path.normpath(src_dir)))
    os.makedirs(ai_root, exist_ok=True)
    if not os.path.exists(os.path.join(ai_root, 'box.def')):
        open(os.path.join(ai_root, 'box.def'), 'w', encoding='utf-8-sig').write('#TITLE:AI譜面\n#GENRE:AI譜面\n#EXPLANATION:動画から自動採譜した譜面(video2tja)。音源とヘッダーは本家のもの\n')
    n = len([d for d in os.listdir(ai_root) if os.path.isdir(os.path.join(ai_root, d)) and not d.endswith(name)]) + 1
    existing = [d for d in os.listdir(ai_root) if d.endswith(' ' + name)]
    dst = os.path.join(ai_root, existing[0] if existing else '%04d %s' % (n, name))
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src_dir, wave), os.path.join(dst, wave))
    out = head.rstrip('\n') + '\n\n//video2tja 自動採譜(%s)。BPM/OFFSET は自動採譜側、ヘッダーと音源は本家\n' % course + 'COURSE:' + course + '\n' + blk + '#END\n'
    open(os.path.join(dst, name + '.tja'), 'w', encoding='utf-8-sig').write(out)
    print('作成:', dst); print('  ', sorted(os.listdir(dst)))
    print('  BPM %s OFFSET %s LEVEL %s' % (my_bpm, my_off, lvl.group(1) if lvl else '(自動採譜のまま)'))

if __name__ == '__main__':
    main()
