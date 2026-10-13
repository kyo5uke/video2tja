"""生成 TJA と参照 TJA の音符記号の個数・OFFSET・BPM を並べる。
使い方: sym_cmp.py <生成tja> <参照tja> <コース>
"""
import sys, re
from collections import Counter

NL = chr(10)


def head(t, key):
    m = re.search(r'^' + key + r':\s*([-\d.]+)', t, re.M)
    return m.group(1) if m else '-'


def syms(path, course):
    t = open(path, encoding='utf-8-sig', errors='replace').read().replace(chr(13), '')
    m = re.search(r'COURSE:\s*' + course + r'(.*?)(#END|\Z)', t, re.S)
    if not m:                       # コース名が数字のこともある
        num = {'Oni': '4', 'Edit': '5', 'Hard': '3'}.get(course, course)
        m = re.search(r'COURSE:\s*' + num + r'(.*?)(#END|\Z)', t, re.S)
    seg = m.group(1); seg = seg[seg.find('#START') + 6:]
    seg = re.sub(r'//[^' + NL + r']*', '', seg)
    seg = re.sub(r'#[A-Z]+[^' + NL + r']*', '', seg)          # 命令行を落とす
    c = Counter(ch for ch in seg if ch in '123456789ABCDF')
    return c, t


g, gt = syms(sys.argv[1], sys.argv[3])
r, rt = syms(sys.argv[2], sys.argv[3])
keys = sorted(set(g) | set(r))
print('  記号  ' + '  '.join('%s:%d/%d%s' % (k, g.get(k, 0), r.get(k, 0), '' if g.get(k, 0) == r.get(k, 0) else ' ←差') for k in keys))
print('  音符(1234) 生成 %d / 参照 %d   連打風船(5-9) 生成 %d / 参照 %d' % (
    sum(g.get(k, 0) for k in '1234'), sum(r.get(k, 0) for k in '1234'),
    sum(g.get(k, 0) for k in '56789'), sum(r.get(k, 0) for k in '56789')))
print('  OFFSET 生成 %s / 参照 %s     BPM 生成 %s / 参照 %s' % (
    head(gt, 'OFFSET'), head(rt, 'OFFSET'), head(gt, 'BPM'), head(rt, 'BPM')))
