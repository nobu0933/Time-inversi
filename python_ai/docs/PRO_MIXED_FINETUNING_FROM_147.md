# PRO 147回目からの混合相手・多様な序盤による追加学習

122回目と147回目の比較では、標準初期盤面だけの対戦と異なる序盤からの対戦で勝敗傾向が変わりました。この手順は、147回目の重みを起点に、MID・PRO100・122回目を固定相手とする追加学習です。元の保存点とブラウザ用モデルは変更しません。

## 準備

PowerShellで `Time-inversi/python_ai` に移動し、次の4ファイルを確認してください。

```powershell
Test-Path runs/pro-long-from-102/iteration-147.pt
Test-Path runs/pro-long-from-102/mid.pt
Test-Path runs/pro-long-from-102/pro100.pt
Test-Path runs/pro-long-from-102/iteration-122.pt
```

最初の実行では147回目の**モデル重み**を引き継ぎます。前段階でMIDとPRO100の対局例が多く入った再生バッファと最適化状態は引き継がず、新しいバッファ・AdamW最適化器を作ります。開始学習率は `0.0001` です。固定相手と起点のチェックポイントは新しい出力フォルダーにハッシュ照合してコピーします。

## 実行

最初に学習を始めずに保存点と固定相手を準備・検証するには、末尾に `--prepare-only` を付けます。

```powershell
.\.venv\Scripts\python.exe -m time_inversi.mixed_finetune --base runs/pro-long-from-102/iteration-147.pt --mid runs/pro-long-from-102/mid.pt --pro100 runs/pro-long-from-102/pro100.pt --pro122 runs/pro-long-from-102/iteration-122.pt --output runs/pro-mixed-from-147 --target-iteration 167 --device cpu --prepare-only
```

学習を開始するコマンドは次のとおりです。`--target-iteration 167` は147回目から**20回追加**する指定です。停止後も同じコマンドを再実行すれば、保存済みの回から目標まで続けます。既に目標へ達している場合は追加学習しません。目標だけを変える場合は、同じ出力先で `--target-iteration` の数字を変更できます。

```powershell
.\.venv\Scripts\python.exe -m time_inversi.mixed_finetune --base runs/pro-long-from-102/iteration-147.pt --mid runs/pro-long-from-102/mid.pt --pro100 runs/pro-long-from-102/pro100.pt --pro122 runs/pro-long-from-102/iteration-122.pt --output runs/pro-mixed-from-147 --target-iteration 167 --device cpu
```

## 学習と評価の内訳

- 各回に自己対戦4局、MID・PRO100・PRO122との対戦を各2局ずつ行います。固定相手との2局は候補が黒番・白番を1局ずつ担当します。固定相手の着手は学習例にせず、候補の着手だけを学習します。
- 各対局は50%の確率で、合法手をランダムに4～8手進めた局面から始めます。残りは標準初期盤面です。ランダム序盤の着手は学習例にしません。使用した序盤の手順は `metrics.jsonl` に記録します。
- 1手につき64探索、各回の重み更新は100回です。5回ごとに、3相手それぞれへ標準初期盤面4局と異なる序盤6局面×黒白の12局を評価します。評価では探索ノイズや重み更新を使わず、標準開始と多様な序盤の成績を別々に記録します。評価用の序盤と乱数種は毎回固定しており、経過を比較できます。
- 毎回 `runs/pro-mixed-from-147/latest.pt` を保存します。評価回には `iteration-152.pt` などの保存点、詳細な `evaluation-152.json`、集計を含む `metrics.jsonl` を残します。評価中に停止しても、再実行時に保存済みの回の評価を完了してから先へ進みます。

CPUでの所要時間は未計測です。評価は3相手×16局で、従来より多くなります。まず `--target-iteration 148` で1回分の時間を計り、目標を段階的に延ばしてください。出力フォルダーには4つの固定チェックポイントと最新の保存点が入り、評価保存点も増えるため、十分な空き容量を確保してください。

## 主な調整項目

| オプション | 既定値 | 内容 |
| --- | ---: | --- |
| `--self-play-games` | 4 | 各回の自己対戦局数 |
| `--mid-games` / `--pro100-games` / `--pro122-games` | 各2 | 固定相手との局数。各2以上の偶数 |
| `--opening-probability` | 0.5 | ランダム序盤から始める確率 |
| `--opening-min-moves` / `--opening-max-moves` | 4 / 8 | 序盤をランダムに進める手数 |
| `--learning-rate` | 0.0001 | 新しい最適化器の学習率 |
| `--eval-every` | 5 | 評価間隔。0で評価を省略 |
| `--eval-standard-games` | 4 | 固定相手1種あたりの標準開始評価局数。偶数 |
| `--eval-openings` | 6 | 固定相手1種あたりの異なる序盤の数。黒白2局ずつ |

初回の学習設定は保存点に記録されます。再開時は同じコマンドを使い、学習設定を途中で変えないでください。新しい設定を試す場合は別の出力フォルダーで始めます。候補の採用前には、保存点を異なる乱数種・序盤局面でも比較してください。
