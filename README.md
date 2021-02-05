# Flask で機械学習

## はじめに
Python ライブラリである Flask の練習として機械学習を利用した web アプリを作ってみた. [この記事](https://aiacademy.jp/media/?p=382)を参考に scikit-learn の Toy データセット load_boston を分析させる. 完成したweb アプリは下のようになる.

![webapp](https://user-images.githubusercontent.com/47030492/106380852-5429d900-63f8-11eb-9b31-e854ca588006.jpeg)

## 機械学習のアルゴリズム
機械学習のアルゴリズムには自分で書いた[コウモリアルゴリズム](https://github.com/ChicChi120/Evolutionary_computation_algorith)を用いた. 今回は重回帰分析により５つの係数を求める. 

## web アプリについて
[Bootstrap](https://getbootstrap.jp/) で web ページを作成した. 現時点では機械学習した結果を保存する機能はないので今後, データベースを利用してブログ形式で実装することも考えている.

## 実行方法
virtualenv の仮想環境で実行することをおすすめする. 仮想環境を用意するには

```$
virtualenv env  
source env/bin/activate  
pip install -r requirements.txt
```

とし必要なパッケージをインストールする.  
実行するには, はじめに

```(env)$
python bat_alg.py
```

とコマンドラインに入力すると bats.pkl ファイルが作成
されるのを確認した後

```(env)$
FLASK_APP=app.py FLASK_ENV=development flask run
```
でコマンドラインに表示される URL にアクセスすれば良い.