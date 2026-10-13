"""動画 1 本から譜面を作る。曲は動画の音声から自動で探す(songfind.py)。
使い方: python convert.py <動画> [--song <曲フォルダ(Songs からの相対でも可)> | --no-song [--title 曲名]] [--out <出力フォルダ>] [--install]
  --song    曲フォルダを指定する(省くと音声から探す。はっきり決まらなければ候補を出して止まる)
  --no-song ライブラリに無い曲: 動画の音声を切り出して音源にする(run_video_only.py。曲名は --title、既定は動画の名前)
  --out     出力フォルダ(既定: testvideos/work/cli/<動画の名前>)。<出力フォルダ>/out.tja・out_ons.json・out.log ができる
  --install 出来た譜面をゲームの Songs/13 AI譜面 に入れる(本家の音源とヘッダー + 自動採譜のコース: to_ai_folder.py)
中身は run_song.py(検出・隠れた音符・ゴーゴー・風船・譜面作り・参照があれば答え合わせ)と同じ。"""
import os, re, sys, subprocess

V = os.path.dirname(os.path.abspath(__file__))
SONGS = os.environ.get('V2T_SONGS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Songs')


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'): print(__doc__); return 0

    def opt(name, has_val=True):
        if name not in args: return None
        k = args.index(name); v = args[k + 1] if has_val else True
        del args[k:k + (2 if has_val else 1)]
        return v
    song = opt('--song'); out = opt('--out'); install = opt('--install', False); nosong = opt('--no-song', False); title = opt('--title')
    video = os.path.abspath(args[0])
    if not os.path.isfile(video): sys.exit('動画が見つかりません: %s' % video)
    if song and not os.path.isabs(song): song = os.path.join(SONGS, song)
    out = out or os.path.join(V, 'testvideos', 'work', 'cli', os.path.splitext(os.path.basename(video))[0])
    os.makedirs(out, exist_ok=True)
    prefix = os.path.join(out, 'out')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    print('変換: %s → %s' % (video, out), flush=True)
    if nosong:
        return no_song(video, prefix, title or os.path.splitext(os.path.basename(video))[0], install, env)
    lines = []
    p = subprocess.Popen([sys.executable, '-u', os.path.join(V, 'run_song.py'), video, song or 'auto', prefix], cwd=V, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in iter(p.stdout.readline, b''):
        ln = raw.decode('utf-8', 'replace').rstrip('\r\n'); lines.append(ln)
        # 進み具合と結果だけ見せる(全部は out.log)
        if re.search(r'\[時間\]|曲の候補|曲フォルダ:|best course|はっきり 1 つに|Traceback|Error', ln): print('  ' + ln.strip(), flush=True)
    code = p.wait()
    open(prefix + '.log', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    if code != 0 or not os.path.exists(prefix + '.tja'):
        if not song and any('はっきり 1 つに' in ln for ln in lines):
            sys.exit('曲をはっきり 1 つに決められませんでした。--song で曲フォルダを指定するか、ライブラリに無い曲なら --no-song で動画の音声から作ってください')
        sys.exit('変換できませんでした(ログ: %s.log)' % prefix)
    print('譜面: %s.tja' % prefix)
    if install:
        m = re.search(r'曲フォルダ:\s*(.+?)(参照 TJA (.+?))', '\n'.join(lines))
        song_dir = song or (m.group(1).strip() if m else None)
        ref = os.environ.get('REF_TJA') or (m.group(2).strip() if m and m.group(2) != 'None' else None)
        if not song_dir: sys.exit('曲フォルダが分からないので入れられませんでした')
        crs = re.search(r'^COURSE:\s*(\w+)', open(prefix + '.tja', encoding='utf-8-sig', errors='replace').read(), re.M)
        cmd = [sys.executable, os.path.join(V, 'to_ai_folder.py'), song_dir, prefix + '.tja', crs.group(1) if crs else 'Oni']
        if ref: cmd += ['--ref', ref]
        r = subprocess.run(cmd, cwd=V, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0: return r.returncode
    return 0


def no_song(video, prefix, title, install, env):
    """ライブラリに無い曲: 動画の音声を音源にして作る(run_video_only.py)。--install は切り出した音源と譜面をそのまま 13 AI譜面 に置く"""
    r = subprocess.run([sys.executable, os.path.join(V, 'run_video_only.py'), video, prefix, title], cwd=V, env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    for ln in (r.stdout + r.stderr).splitlines():
        if re.search(r'音符|譜面:|Traceback|Error|失敗', ln): print('  ' + ln.strip())
    if r.returncode != 0 or not os.path.exists(prefix + '.tja'): sys.exit('変換できませんでした(ログ: %s.log)' % prefix)
    print('譜面: %s.tja' % prefix)
    if install: print(install_plain(prefix, title))
    return 0


def install_plain(prefix, title, tja=None):
    """本家の曲フォルダの無い譜面を Songs/13 AI譜面/<連番> <曲名>/ に置く(音源は動画から切り出したもの)。tja = 置く譜面(既定 <prefix>.tja)"""
    import shutil
    ai_root = os.path.join(SONGS, '13 AI譜面'); os.makedirs(ai_root, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]+', '_', title)
    existing = [d for d in os.listdir(ai_root) if d.endswith(' ' + safe)]
    n = len([d for d in os.listdir(ai_root) if os.path.isdir(os.path.join(ai_root, d))]) + 1
    dst = os.path.join(ai_root, existing[0] if existing else '%04d %s' % (n, safe)); os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(prefix + '_song', title + '.ogg'), os.path.join(dst, title + '.ogg'))
    shutil.copy(tja or (prefix + '.tja'), os.path.join(dst, safe + '.tja'))
    return '作成: %s' % dst


if __name__ == '__main__':
    sys.exit(main())
