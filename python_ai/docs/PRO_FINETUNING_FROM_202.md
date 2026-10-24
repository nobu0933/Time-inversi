# PRO 202回目を起点とする4相手・増局数の追加学習

`time_inversi.finetune_202` は202回目の重みから新しい学習を始めます。MID、PRO100、122回目、207回目を固定相手とし、202回目の最適化器と再生バッファは引き継ぎません。新しいAdamW最適化器の学習率は `0.00005` です。元の保存点は変更しません。

## 実行

PowerShellで `C:\Users\nobus\Desktop\WebGame\Time-inversi\python_ai` に移動して実行します。初めにチェックポイントの検証とコピーのみを行う場合は、末尾に `--prepare-only` を追加してください。

```powershell
cd C:\Users\nobus\Desktop\WebGame\Time-inversi\python_ai
.\.venv\Scripts\python.exe -m time_inversi.finetune_202 --base runs/pro-mixed-from-147/iteration-202.pt --mid runs/pro-long-from-102/mid.pt --pro100 runs/pro-long-from-102/pro100.pt --pro122 runs/pro-long-from-102/iteration-122.pt --pro207 runs/pro-mixed-from-147/iteration-207.pt --output runs/pro-mixed-from-202 --target-iteration 307 --max-hours 12 --device cpu
```

既定では1回あたり自己対戦8局、4相手との対戦を各4局（黒2局、白2局）で合計24局行います。旧方式の10局から2.4倍です。各局は75%の確率で合法な4～8手のランダム序盤から始めます。序盤の着手と固定相手の着手は学習例に含めません。1手64探索、各回150回の重み更新を行います。

5回ごとに4相手それぞれへ標準初期盤面4局と、異なる8序盤を先後入れ替えた16局で評価します。1回の評価は計80局です。評価用の序盤は新学習では使わず、評価間で固定します。評価結果は標準初期盤面と異なる序盤を分けて `evaluation-N.json` と `metrics.jsonl` に保存します。評価時には `iteration-N.pt` も保存します。毎回の最新状態は `latest.pt` にあります。

旧方式の直近40回の実測は、10局の学習が平均108秒、48局の定期評価が平均438秒でした。24局と80局に換算すると、5回ごとの評価を含めて105回で約12時間と見込まれます。この見込みはCPU速度や対局の長さで変わります。`--target-iteration 307` は上限で、`--max-hours 12` により、完了した回の学習・評価時間の累計と次回の見込みを比べ、約12時間の予算内で次回を完了できない見込みなら停止します。準備・読み込み・ファイル保存にかかる時間はこの累計に含みません。多少の超過や早めの停止はあり得ます。

中断後は同じコマンドで再開できます。保存済みの最適化器、再生バッファ、乱数状態を引き継ぎます。評価中の中断なら、保存済みの回の評価を先に完了します。`--max-hours` は実行のたびにリセットされず、`metrics.jsonl` に記録した累積時間を基準とします。予算を延ばす場合は同じ出力先でこの数字と `--target-iteration` を増やしてください。学習設定は開始後に変更しないでください。

Gitから取得した保存点を307回目以降も続けるときは、`latest.pt`・同じフォルダーの4つの固定相手・`metrics.jsonl` が必要です。元の入力パスを省略して再開できます。次は327回目を上限に、累積計算時間を14時間まで延ばす例です。

```powershell
cd C:\Users\nobus\Documents\GitHub\Time-inversi\python_ai
.\.venv\Scripts\python.exe -m time_inversi.finetune_202 --output runs/pro-mixed-from-202 --target-iteration 327 --max-hours 14 --device cpu
```

新しい環境では先に `requirements.txt` の依存関係をインストールしてください。保存点の `latest.pt` だけでは固定相手や学習設定の検証を完了できません。同じ出力フォルダーの `base202.pt`、`mid.pt`、`pro100.pt`、`pro122.pt`、`pro207.pt` と `metrics.jsonl` を保持してください。

最終候補は、定期評価だけで決めず、未使用の序盤で先後を入れ替えて比較してください。307回目はブラウザ用PROへ手動で書き出し済みです。以後の追加学習ではブラウザ用モデルは自動的に更新されません。
