# 天体写真自動編集 Web アプリ
このアプリケーションではスマートフォンやカメラなどで撮影した未編集の天体写真に対してユーザが自分好みの編集を行うことができる. 詳しくは灘本のこれまでの論文, 資料, および修士論文を参考にしてください. 

## アプリの構造図
アプリの構造は以下のようになっている. 
![](/readme_img/app_fig.png) 
赤文字がユーザの行動, 実線が常に実行される操作, 点線がユーザの行動によっては実行されうる操作である. 

## アプリの実行
フォルダを手元の環境にダウンロードし, 
```
python run.py
```
を実行する. 

## フォルダの説明
- dataset_ssd: SSD における訓練データおよびテストデータを格納
- /app/static/input: 入力画像, 出力画像および背景画像の候補が格納
- /app/static/model: SSD および GAN の学習済みモデルを格納
- /app/static/starcatalog: 二種類のヒッパルコス星表のデータを格納 

## アプリで実装済みの内容, 未実装の内容
### 実装済みの内容
- 天体写真に写っている星座の検出
- 天体写真に建物や影などが写っていることがわかっている場合にその部分に星の補完が行われない処理
- ユーザの好みにあった背景画像の探索

### 未実装の内容
- 天体写真に建物や影などが写っていることの自動検出
- 星のサイズの決定など背景画像の選択以外にユーザが行える編集処理
- グローバル環境で実行できる環境

## 参考文献
[SSD: Single Shot MultiBox Object Detector, in PyTorch](https://github.com/amdegroot/ssd.pytorch)\
[つくりながら学ぶ! PyTorchによる発展ディープラーニング](https://github.com/YutaroOgawa/pytorch_advanced)\
[Flaskで簡単につくる、画像処理した結果を見るだけのWebサービス](https://qiita.com/redshoga/items/60db7285a573a5e87eb6)\
[TEMPLATE PARTY](https://template-party.com/)