# 100回学習済みPROの最終調整

3イテレーションの試行結果は[2026-10-23の記録](PRO_FINETUNE_PILOT_2026-10-23.md)にまとめています。

`assets/ai/pro-model.js`はブラウザ推論専用です。追加学習には、100回目の**フルチェックポイント**`runs/pro01/latest.pt`が必要です。元のPCから`runs/pro01`をコピーし、まず次を確認してください。

```powershell
cd python_ai
Test-Path runs/pro01/latest.pt
Test-Path runs/experiment01/latest.pt
```

両方とも`True`にしてください。前者が100回時点のPRO、後者がMIDです。学習環境をまだ作っていないPCでは[PRO学習手順](../PRO_TRAINING_GUIDE.md)の「このPC専用のPython環境を作る」を先に実行します。

## 追加学習

```powershell
.\.venv-pro\Scripts\python.exe -m time_inversi.finetune --base runs/pro01/latest.pt --mid runs/experiment01/latest.pt --output runs/pro-final --iterations 20 --device cpu
```

各イテレーションは、最新モデル同士の自己対戦6局、MIDとの対局2局、100回時点のPROとの対局2局、バッチ64で100回の更新です。固定相手との対局は黒白を同数にし、**学習には新PROが指した手と実際の勝敗だけ**を使います。相手の手を正解として模倣しません。初期学習より低い学習率`0.0002`で調整します。

開始時に`runs/pro-final/pro100.pt`と`mid.pt`へ固定相手をコピーします。元の`runs/pro01/latest.pt`とMIDは変更しません。学習中は`runs/pro-final/latest.pt`を毎回保存し、5回ごとに固定相手2種との各20局を評価して`iteration-105.pt`などの候補を保存します。評価結果と対局内訳は`metrics.jsonl`に記録します。最初のイテレーション開始前にも再開用の`latest.pt`を作ります。

中断した場合は、**追加する回数**を指定して再開します。固定相手は保存先のコピーから読み込むため、元の入力ファイルを再指定しません。

```powershell
.\.venv-pro\Scripts\python.exe -m time_inversi.finetune --resume runs/pro-final/latest.pt --iterations 10 --device cpu
```

時間はPCと対局長で変わります。MID・PRO100も各手で探索するため、最初の5イテレーションの実測から残りを見積もってください。`runs/pro-final`にはフルチェックポイントとリプレイバッファが複数保存され、空き容量も必要です。

## 採用モデルの選択

`metrics.jsonl`の`evaluation.mid`と`evaluation.pro100`を、黒番と白番の両方で確認します。20局の差だけで採用を決めず、有望な候補を別の乱数種で比較してください。例として105回目の候補なら：

```powershell
.\.venv-pro\Scripts\python.exe -m time_inversi.compare_models --candidate runs/pro-final/iteration-105.pt --incumbent runs/pro-final/pro100.pt --games 40 --simulations 64 --seed 31001 --device cpu
.\.venv-pro\Scripts\python.exe -m time_inversi.compare_models --candidate runs/pro-final/iteration-105.pt --incumbent runs/pro-final/mid.pt --games 40 --simulations 64 --seed 31002 --device cpu
```

最終イテレーションを使う場合は`iteration-105.pt`を`latest.pt`に置き換えます。両相手への成績と黒白の偏りを比較し、100回時点のPROより安定して良い候補だけをブラウザへ反映してください。学習中の評価と同じ乱数種だけを使って選ぶと過大評価しやすいため、採用判断には上記のような別の乱数種を使います。

採用後にブラウザ用モデルを書き出し、ZIPを再生成します。既存のPROモデルを残したい場合は、先にバックアップしてください。

```powershell
.\.venv-pro\Scripts\python.exe -m time_inversi.export_browser --checkpoint runs/pro-final/iteration-105.pt --output ../assets/ai/pro-model.js
cd ..
python scripts/package_web.py
node --test tests/*.test.cjs
```
