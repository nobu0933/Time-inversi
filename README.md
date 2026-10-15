# Time Inversi

`index.html` をブラウザで開くと遊べます。外部ライブラリやビルドは不要です。

画像1に合わせた6×6盤で、黒から始まる同一端末の2人対戦です。マウスカーソルの代わりに手番のコマが追従します。合法マスの中心からマス幅の約80%以内に近づくと、最寄りの候補に着地点の影が現れ、クリックすると配置します。約100msで降下したあと一度小さく跳ね、計240msで接地してから挟んだコマが裏返ります。タッチでは明るいマスをタップして配置します。キーボードではTab・矢印キーで移動し、Enter・Spaceで配置できます。動きを減らす設定にも対応します。

TIME INVERSIONボタンで順行／逆行を切り替えます。配置またはパスの後、逆方向の履歴を1手分戻して、その手で置いた駒だけを消します。順行駒のマスは橙、逆行駒は青紫です。パス・強制順行復帰・消滅後の終了判定の詳細は [時間反転ルール](docs/rules.md) を参照してください。結果と再対戦ボタンは終局時だけ表示します。対局中のリセットはページを再読み込みしてください。CPU・BGMはありません。

描画はPenrosanneの `docs/ui-design.md` と `src/ui/BoardView.js`、P3NROSEの `docs/GAME_SPECIFICATION.md` と `src/ui/board.js`・`src/ui/editor.js` を参考に、単色の上面・下向きの側面・硬い影・100msの接地を円形コマに適用しています。参考プロジェクトには依存しません。

ルールの検証: `node --test tests/*.test.cjs`

ローカルプレビュー: `node scripts/preview.cjs` を実行して `http://127.0.0.1:4173` を開きます。

## フォルダ構成

```text
Time Inversi/
├─ index.html                # ゲームの入口
├─ README.md                 # 起動方法・構成
├─ src/
│  ├─ game/rules.js          # リバーシのルール判定
│  └─ ui/game.js             # 画面描画・入力・アニメーション
├─ styles/style.css          # 配色・レイアウト・立体表現
├─ tests/                   # 基本ルール・時間反転のテスト
├─ scripts/preview.cjs       # 開発用プレビューサーバー
└─ docs/
   ├─ rules.md              # 時間反転のルール仕様
   ├─ references/           # 元の参考画像
   └─ screenshots/          # 実装画面の記録
```
