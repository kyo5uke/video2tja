"""動画音声と曲の音源(ogg)の相互相関で Δ を出す: ogg_t = video_t - Δ。
使い方: audio_align.py <video> <ogg> <video_start秒> <video_end秒>
  動画の [start,end] 区間の音声を取り出し、ogg 全体に対してエンベロープの相互相関でずれを求める。"""
import sys, subprocess, numpy as np, os, tempfile

SR = 11025

def load(path, ss=None, t=None):
    tmp = tempfile.mktemp(suffix='.wav')
    cmd = ['ffmpeg', '-y', '-v', 'error']
    if ss is not None: cmd += ['-ss', '%.3f' % ss]
    cmd += ['-i', path]
    if t is not None: cmd += ['-t', '%.3f' % t]
    cmd += ['-ac', '1', '-ar', str(SR), '-f', 'wav', tmp]
    subprocess.run(cmd, check=True)
    raw = open(tmp, 'rb').read(); os.remove(tmp)
    # 簡易 WAV 読み(PCM16)
    i = raw.find(b'data'); data = np.frombuffer(raw[i + 8:], np.int16).astype(np.float32) / 32768
    return data

def envelope(x, hop=64):
    n = len(x) // hop
    e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
    e = np.log1p(e * 50)
    return e - e.mean()

def align(video, ogg, vs, ve):
    a = load(video, vs, ve - vs); b = load(ogg)
    hop = 64
    ea, eb = envelope(a, hop), envelope(b, hop)
    n = 1
    while n < len(ea) + len(eb): n *= 2
    fa, fb = np.fft.rfft(ea, n), np.fft.rfft(eb, n)
    cc = np.fft.irfft(fa * np.conj(fb), n)          # cc[k] ~ sum ea[i] * eb[i-k] → k = 動画区間内位置 - ogg位置
    lag = int(np.argmax(cc)); 
    if lag > n // 2: lag -= n
    peak = cc[lag] / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9)
    # 動画時刻 vt に対応する ogg 時刻 = (vt - vs) - lag*hop/SR → Δ = vs + lag*hop/SR
    delta = vs + lag * hop / SR
    # サンプル精度で精密化(±2 hop)
    best = (delta, peak)
    for d in np.arange(delta - 2 * hop / SR, delta + 2 * hop / SR, 1.0 / SR):
        # ogg を動画区間に合わせて切り出し比較
        o0 = int(round((vs - d) * SR))
        if o0 < 0 or o0 + len(a) > len(b): continue
        seg = b[o0:o0 + len(a)]
        c = float(np.dot(a, seg) / (np.linalg.norm(a) * np.linalg.norm(seg) + 1e-9))
        if c > best[1]: best = (d, c)
    return best

if __name__ == '__main__':
    video, ogg, vs, ve = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    d, c = align(video, ogg, vs, ve)
    print('VID2OGG = %.4f  (ogg_t = video_t - VID2OGG), corr %.3f' % (d, c))
