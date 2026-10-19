# Time Inversi — Python / AlphaZero

Time InversiのルールをPythonへ移植し、自己対戦・MCTS・方策／価値ネットワークによる学習を実装しています。`Time Inversi`フォルダーだけで環境構築・学習・評価できます。参考にした`AlphaZero`、`Penrosanne`、`P3NROSE`フォルダーは不要です。

ブラウザのランダムCPUとの接続や、学習モデルのブラウザ向け変換はまだ行っていません。このフォルダーは学習・評価・Pythonからの着手選択用です。

## 1. ダウンロードしたフォルダーを開く（Windows）

以下はPythonがインストールされたWindows PCでのPowerShellの手順です。GPUは不要で、まずCPUで確認します。ライブラリの初回ダウンロードにはインターネット接続と空き容量が必要です。学習中のインターネット接続は不要です。

ダウンロードしたZIPを展開し、`Time Inversi`フォルダー内でターミナル（PowerShell）を開いてください。以降は、最初に次の2行を実行します。

```powershell
cd python_ai
python --version
```

`python_ai`フォルダー内に`requirements.txt`と`time_inversi`が見えることを確認してください。以降のコマンドはすべてこの場所から実行します。`python`が見つからない場合は、Pythonのインストール設定でPATHへの追加を有効にしてターミナルを開き直してください。

## 2. 専用環境を作る

他のPCの`.venv`はコピーして使いません。ダウンロード物に`.venv`が含まれている場合は、そのフォルダーを別名に退避してから新しく作ります。学習成果の`runs`はそのまま残してください。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import sys, torch, numpy; print(sys.version); print('PyTorch:', torch.__version__, 'NumPy:', numpy.__version__); print('CUDA:', torch.cuda.is_available())"
```

`Activate.ps1`は使わず、専用環境のPythonを直接指定するため、PowerShellの実行ポリシー変更は不要です。`CUDA: False`でもCPU学習は可能です。確認した環境は64bit Windows、Python 3.14、NumPy 2.5.4、PyTorch 2.14.1（CPU）です。すべてのPythonバージョン・OSでの動作を保証するものではありません。

同じWindows / Python 3.14環境で検証時の依存バージョンに揃えたい場合は、上の`requirements.txt`の代わりに`requirements-lock.windows-py314.txt`を指定できます。他のPythonやOSでは通常の`requirements.txt`を使用してください。GPU学習には使用するGPUに対応したPyTorch環境が別途必要です。`--device auto`はCUDAが利用可能ならCUDA、それ以外はCPUを選びます。`--device cuda`で利用できない場合はエラーになります。

macOS / Linuxでは、環境作成を`python3 -m venv .venv`とし、以降の`.\.venv\Scripts\python.exe`を`./.venv/bin/python`に置き換えます。これらのOSでの実行は未検証です。

## 3. 動作確認

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

ニューラルネットの重み更新、保存／再開の一致、黒白両方の評価AIなどを検証します。Node.jsは学習に不要です。Node.jsがないPCではJavaScript版との照合1件だけを`skipped`として省略します。Node.jsがある場合は実際の`../src/game/rules.js`と通常対局・パス・角反転・直接反転APIの遷移を照合します。検証記録は[VALIDATION.md](VALIDATION.md)を参照してください。

小規模な学習確認（初回専用の出力先を選んでください）：

```powershell
.\.venv\Scripts\python.exe -m time_inversi.train --iterations 1 --self-play-games 2 --simulations 4 --batch-size 8 --updates 2 --channels 16 --blocks 1 --history-hidden 16 --eval-every 1 --eval-games 2 --device cpu --output runs/check
```

`runs/smoke`は開発時の動作確認用です。そこにある重みは短時間の検証用であり、強さを保証する学習済みAIではありません。

成功すると`runs/check/latest.pt`と`runs/check/metrics.jsonl`が生成されます。表示された`updates`が2で、`loss`が数値になり、評価結果も出ていれば、自己対戦から学習・保存・評価まで完了しています。再実行時は別の出力名を選ぶか、次で続きから実行してください。

```powershell
.\.venv\Scripts\python.exe -m time_inversi.train --resume runs/check/latest.pt --iterations 1 --device cpu
```

## 4. 本学習を始める

```powershell
.\.venv\Scripts\python.exe -m time_inversi.train --iterations 100 --self-play-games 10 --simulations 64 --device cpu --output runs/experiment01
```

1イテレーションは「自己対戦 → リプレイバッファに蓄積 → ミニバッチ学習 → 保存」です。5イテレーションごとにランダムAI・簡易貪欲AIと評価対局します。最初の12手はMCTSの訪問割合で着手を抽選し、それ以降は最多訪問手から選びます。自己対戦の探索だけにDirichletノイズを加えます。

主な設定：

| オプション | 初期値 | 内容 |
| --- | ---: | --- |
| `--self-play-games` | 10 | イテレーションあたりの対局数 |
| `--simulations` | 64 | 1手あたりのMCTS追加探索数 |
| `--batch-size` | 64 | 学習バッチサイズ |
| `--updates` | 100 | イテレーションあたりの更新回数 |
| `--buffer-size` | 50000 | 保持する局面数 |
| `--channels` | 64 | 畳み込みのチャンネル数 |
| `--blocks` | 3 | 残差ブロック数 |
| `--history-hidden` | 64 | 履歴GRUの隠れ状態サイズ |
| `--eval-every` | 5 | 評価間隔。0なら省略 |
| `--eval-games` | 20 | 相手1種類あたりの評価局数。偶数 |
| `--threads` | 2 | PyTorchのCPUスレッド数 |
| `--max-moves` | 512 | 異常な長期対局を検出する上限 |

自己対戦は現在1プロセスで逐次実行します。参考元のRay並列化は必須にしていません。学習サンプルがバッチサイズに満たない間は更新を待ちます。対局や探索が上限に達した場合はエラーとし、未終了の局面を引き分けとして学習しません。

所要時間はPCと対局長で変わります。最初の数イテレーションの`seconds`を目安にしてください。短い動作確認で棋力向上までは判定できません。現在は最新モデルで自己対戦を続ける構成で、過去の最良モデルとの昇格判定や対戦相手プールは未実装です。

### このPCで学習時間を測る

```powershell
.\.venv\Scripts\python.exe -m time_inversi.benchmark
```

標準サイズの新規モデルで自己対戦3局（1手64探索）、バッチ64で100回の実更新、ランダムAI・簡易AIとの評価を各2局実行します。既存の学習済みモデルには触れません。CPU・2スレッドが既定です。

処理別の実測秒数と、通常の1イテレーション／評価付き1イテレーション／100イテレーションの推定時間を表示し、`runs/benchmarks/`にJSONで保存します。推定は自己対戦10局・更新100回、5イテレーションごとに各相手20局評価という設定です。`--simulations`で探索数、`--games`で測定局数、`--device`で実行デバイスを変更できます。

短い測定からの概算で、保存・起動時間は含みません。学習に伴う対局長・履歴長の変化、バッファの蓄積、他のアプリの負荷で時間は変わります。「一定の強さになるまでの時間」を示すものではありません。

2026-10-20の測定例（Intel Core i7-1165G7、CPU・2スレッド、Python 3.14.2、PyTorch 2.14.1+cpu、標準モデル、64探索）：

| 処理 | 所要時間 | 根拠 |
| --- | ---: | --- |
| 自己対戦1局 | 約14〜22秒（平均17.4秒） | 3局の実測、69〜93手 |
| バッチ64で100回更新 | 約7.4秒 | 実測 |
| 通常の1イテレーション | 約3分 | 自己対戦10局＋100回更新の推定 |
| 評価付き1イテレーション | 約10.4分 | 上記＋評価40局の推定。各相手2局の実測から換算 |
| 100イテレーション | 約7.5時間 | 自己対戦1,000局、更新10,000回、評価800局の推定 |
| 100イテレーション・評価なし | 約5時間 | `--eval-every 0`の場合の推定 |

測定ログ：`runs/benchmarks/20261020-003350-063725.json`。この値は当該PCでの短期測定の外挿で、完了時間の保証ではありません。

## 5. 停止・保存・再開

保存先は常に指定出力フォルダーの`latest.pt`です。重み、ネットワーク構成、最適化状態、バッファ、乱数状態、完了イテレーションを保存します。既存の重みがあるフォルダーへの新規学習は拒否し、誤上書きを防ぎます。`metrics.jsonl`には損失・対局長・評価成績を追記します。

```powershell
.\.venv\Scripts\python.exe -m time_inversi.train --resume runs/experiment01/latest.pt --iterations 100
```

`--iterations`は再開後に追加する回数（省略時100）です。対局数・探索数・バッチサイズ・評価間隔・乱数シードなどは保存済み設定を引き継ぎ、明示したオプションだけ変更します。モデル構成、学習率、重み減衰、バッファ容量は保存内容を復元し、異なる値の指定をエラーにします。`--device`は引き継がず、省略時`auto`です。CPUで続けるなら`--device cpu`を追加してください。同一環境のCPUでは連続実行との一致を検証済みですが、異なるデバイス・ライブラリ間の完全一致は保証しません。

停止はターミナルでCtrl+Cを押します。保存はイテレーションごとのため、処理中のイテレーションは失われ、直前の保存から再開します。最初の保存前に停止した場合は新規学習から開始してください。評価は保存の後に実施するため、評価中の停止ではモデルが保存されていても、その回の`metrics.jsonl`が未記録の場合があります。

別PCへ移すときは`Time Inversi`と必要な`runs`をコピーし、専用環境を作り直してから再開します。チェックポイントはモデルだけでなく学習データも含むため容量が増えます。同じ出力先で学習を同時に複数実行しないでください。履歴を残したいチェックポイントは学習停止中に別名で保管できます。

## 導入時のトラブル

| 表示・症状 | 対処 |
| --- | --- |
| `No module named torch` / `numpy` | `.venv`のPythonでインストールと実行の両方を行っているか確認します。 |
| `No module named time_inversi` | `Time Inversi/python_ai`へ移動してから実行します。 |
| `No matching distribution found` | Pythonのバージョン・64bit/32bit・OSと配布ライブラリの対応を確認します。検証済み環境は上記の通りです。 |
| ダウンロード失敗 | 接続・プロキシ設定を確認し、同じインストールコマンドを再実行します。 |
| `latest.pt exists` | 続けるなら`--resume`、新規なら別の`--output`を使用します。 |
| CUDA関連のエラー | まず`--device cpu`で動作確認します。 |
| メモリ不足・学習が遅い | 新規学習でバッファ容量・モデル幅・探索数・バッチサイズを小さくします。再開時はモデル幅やバッファ容量を変更できません。 |
| `exceeded max_moves` | 対局を引き分けとして扱わず停止します。必要に応じて`--max-moves`を増やして再開します。 |

## 評価と着手選択

```powershell
.\.venv\Scripts\python.exe -m time_inversi.evaluate --checkpoint runs/experiment01/latest.pt --opponent greedy --games 20 --simulations 64
.\.venv\Scripts\python.exe -m time_inversi.evaluate --checkpoint runs/experiment01/latest.pt --opponent random --games 20 --simulations 64
.\.venv\Scripts\python.exe -m time_inversi.predict --checkpoint runs/experiment01/latest.pt --state state.json
```

評価は黒番・白番を半数ずつ行い、勝ち・負け・引き分けを色別にも集計します。着手選択はブラウザの状態オブジェクトと同形式のJSONを読み込み、`action`（0～35、パスは36）、`index`（パスはnull）、訪問割合を返します。空の初期局面JSONの作成例：

```powershell
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; from time_inversi.rules import create_game; Path('state.json').write_text(json.dumps(create_game().to_dict()), encoding='utf-8')"
```

## Time Inversiへの対応

- 6×6盤、黒先手、初期配置はブラウザと同じ。
- 通常の挟み込み、配置後の逆方向履歴の消化、パス、初回の角配置による時間反転を再現。
- パス履歴は消去しない空の1手として保持。駒の消滅でも過去の裏返しは戻しません。
- 消滅と角反転の処理後に終局を判定。最初の時刻へ戻っても自動反転しません。
- 選択可能な行動は36マス＋パスの37通り。手動時間反転は現在のUIにないため行動に含めません。`invert()`はJavaScriptとの照合用の内部APIです。
- MCTSの局面識別には駒の色、配置方向、手番、時間方向、両方向の全履歴、使用済みの角を使用。時計は演出なので探索・学習入力から除外します。

ネットワークは8枚の盤面入力（自分／相手、順行／逆行駒、使用済み角、合法マス、時間方向、手番色）を残差CNNで処理し、順行・逆行の全履歴をGRUで処理します。パス・空履歴・パディングは別のトークンです。履歴の順序や長さを固定長で切り捨てません。方策37出力と、手番側から見た勝敗価値1出力を学習します。

## 参考コードからの修正点

- 簡易AIは白枚数への固定参照をやめ、常に着手する側の「自分の枚数 − 相手の枚数」を最大化。候補が全て不利でも合法手を返します。
- 評価対象は配置・消滅・角反転が完了した状態。確定した勝敗を優先します。
- 評価は黒白を同数とし、探索ノイズを無効化します。
- 保存と読み込みのパスを統一し、最適化状態・学習バッファも再開可能にしました。
- AdamWで重み減衰を適用し、損失の非有限値・不正な方策ターゲットを検出します。

## 構成

```text
python_ai/
  time_inversi/
    rules.py       # Pythonルール・JSON状態・遷移
    encoding.py    # 盤面と全履歴の入力
    network.py     # 残差CNN＋GRU、方策／価値ネット
    mcts.py        # PUCT探索
    selfplay.py    # 自己対戦・勝敗ラベル
    buffer.py      # 学習用リプレイバッファ
    baselines.py   # ランダムAI・色を正しく評価する簡易AI
    checkpoint.py  # 保存・復元・デバイス選択
    train.py       # 学習コマンド
    evaluate.py    # 両色での評価
    predict.py     # 局面JSONからの着手選択
  tests/           # JS照合、探索・学習・復元テスト
  runs/            # 重みとログ（生成物）
  requirements.txt
```
