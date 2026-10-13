"""一括試験の動画だけを先に取る(採譜はしない)。使い方: python regress/wide_dl.py <targets.json> <作業フォルダ> [--jobs N]"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wide import download
targets = json.load(open(sys.argv[1], encoding='utf-8')); work = sys.argv[2]
jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 2
ytdir = os.path.join(work, 'yt'); os.makedirs(ytdir, exist_ok=True)
todo = [t['video'] for t in targets if not os.path.exists(os.path.join(ytdir, t['video'] + '.mp4'))]
print('取る動画 %d 本' % len(todo), flush=True)
def f(v):
    p = download(v, ytdir)
    print(('ok ' if p else 'NG ') + v, flush=True)
with ThreadPoolExecutor(jobs) as ex: list(ex.map(f, todo))
print('済み', flush=True)
