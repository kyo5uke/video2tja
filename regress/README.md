# 試験の道具

動画と参照譜面は含まれていません。自分で用意した動画と、それに対応する参照譜面で回します。

## 全曲試験

```
python regress/suite.py                 # 全曲を読み直す
python regress/suite.py --reuse         # 前回の読み取り結果を使い、譜面作り以降だけ
python regress/suite.py --only 名前 ... # 一部の曲だけ
python regress/suite.py --update        # 今回の結果を基準にする
python regress/suite.py --jobs 8        # 同時に回す曲数（既定4）
```

曲の一覧は`regress/suite.json`です。曲ごとに、名前・動画ID・曲フォルダ・参照TJA・コースを書きます。動画の置き場は`testvideos/<ID>.mp4`です。結果は`regress/suite_out/`に出ます。基準（`regress/suite_baseline.json`）より悪くなった曲があれば、終了コードが1になります。

## 一括試験

```
python regress/wide.py <targets.json> <作業フォルダ> [--jobs N]
```

たくさんの動画を1本ずつ取得して採譜し、参照譜面と照合します。途中から再開できます。`targets.json`の形は`suite.json`と同じです。

## 譜面どうしの照合

```
python regress/tja_cmp.py <生成tja> <参照tja> [コース]   # 音符の時刻・種類・速さ・小節線を1個ずつ突き合わせる
python regress/roundtrip.py <ons.json> <tja> [コース]    # 書き出したTJAの時刻を読み取り結果と突き合わせる
python regress/sym_cmp.py <生成tja> <参照tja> <コース>   # 音符記号の個数・OFFSET・BPMを並べる
python regress/roll_cmp.py [曲名 ...]                   # 連打・風船の頭と終わりを参照と比べる
```
