# 静的サイトでMID・PROと対局する

ゲーム画面、MID・PROの100回学習済みモデル、探索処理がすべてブラウザで動きます。プレイヤーのPCにPythonを入れたり、着手用のサーバーを起動したりする必要はありません。CPUボタンは`OFF → EZY → MID → PRO → OFF`。EZYはランダム、MID・PROはそれぞれのモデルで64回のMCTS探索を行います。

## 公開用ファイル

`python scripts/package_web.py`で`dist/time-inversi-web.zip`を生成できます。ZIPの直下に`index.html`と`src`、`styles`、`assets`が入ります。MIDは`assets/ai/model.js`、PROは`assets/ai/pro-model.js`（各約3MB）です。Python環境・学習データ・元のチェックポイントは含みません。PROの重みはOracle由来なので、公開前に[出典と再配布上の注意](../python_ai/pretrained/ORACLE_SOURCE.md)を確認してください。

コードやモデルを変更した後は、`Time Inversi`フォルダーで再作成します。

```powershell
python scripts/package_web.py
```

## itch.io

ゲームの種類にHTML Gameを選び、`dist/time-inversi-web.zip`をアップロードします。埋め込み表示または全画面起動を選択してプレビューしてください。埋め込みサイズは正方形（例：720×720）で、実際の表示領域に合わせてゲーム全体が拡大・縮小します。

ZIP内の`index.html`と相対パスで参照する各ファイルをまとめてアップロードする形式です。[itch.ioの公式手順](https://itch.io/docs/creators/html5)

## GitHub Pages

ZIPを展開し、中身の`index.html`、`src`、`styles`、`assets`をPagesの公開対象フォルダーに配置します。ZIPファイル自体を置くだけではゲーム画面は公開されません。リポジトリのSettings → Pagesで公開元を設定してください。

入口の`index.html`は公開対象フォルダーの直下に配置します。URLが`https://ユーザー名.github.io/リポジトリ名/`になっても、モデル・Workerを相対パスで読むため対応できます。[GitHub Pagesの公式手順](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)

## ローカル確認

```powershell
node scripts/preview.cjs
```

`http://127.0.0.1:4173/`を開きます。このサーバーはファイルを配信するだけで、AIの着手はブラウザが計算します。HTMLのファイル直接起動はブラウザのWorker制限を受けることがあるため、HTTP/HTTPSでの配信を使用してください。

開発用の`http://127.0.0.1:4173/hosting-check.html`ではサブパスのページをiframeに埋め込んだ状態を確認できます。これは公開ZIPに含めません。

## 追加学習後のモデル更新

`python_ai`フォルダーから次を実行し、その後ZIPを再作成して再アップロードします。

```powershell
.\.venv\Scripts\python.exe -m time_inversi.export_browser --checkpoint runs/experiment01/latest.pt
.\.venv\Scripts\python.exe -m time_inversi.export_browser --checkpoint runs/pro01/latest.pt --output ../assets/ai/pro-model.js
```

書き出しは学習済みの推論用重みをFloat32のまま保存します。再学習や量子化は行いません。元の`latest.pt`は変更しません。ブラウザ用の書き出し後、Pythonとの照合テストを実行できます（Node.jsが必要です）。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_browser.py -v
```

## 実装と確認範囲

`src/ai/network.js`がCNN・GroupNorm・GRUを計算し、`search.js`がブラウザ版のルールを使って探索します。順行・逆行の履歴はパスも含めて全件使用します。`worker.js`で画面とは別に計算し、`client.js`が新規対局やCPU切り替え時に不要な探索を中止します。外部CDN、Python API、特別なクロスオリジン分離ヘッダーは不要です。

各モデルはMID・PROで最初に着手するときだけ読み込みます。考慮時間は端末性能と局面で変わります。同点候補の抽選や小さな浮動小数点誤差により、Python版と常に同じ手になるとは限りません。

順行・逆行・パス・長い履歴を含む103局面で、PROのブラウザ版とPython版の方策・価値の最大差は約0.000001でした。MIDも同じ局面で照合済みです。JavaScriptの44テストが成功しています。MIDの実ブラウザでの着手は以前確認済みで、PROの着手はモデル切り替えと画面処理のテストで確認しています。
