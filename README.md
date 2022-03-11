# Histology_Image_Prediction
Transfer learning with Mobilenet v3 for histological images from virtual histology slides

This Read me file is Japanese only! I would like to write down in English near future.

このスクリプトは、東京医科歯科大学・医学部の組織学の講義においてAIを身近に感じてもらうため、アクティブラーニング用に作成した、簡易的な画像予測AIであり、正確性を保証するものではありません。ただし、組織学における画像だけではなく、汎用的に少ない画像からの画像予測AIを作成することができるよう拡張してあります。

元ネタは、Google Coalboratoryにある「画像分類器を再トレーニングする」<br>
https://www.tensorflow.org/hub/tutorials/tf2_image_retraining <br>
です。これを大幅に改定し、フォルダに画像を詰め込み、そのフォルダを指定するだけで簡単に誰でもできるようにしたつもりです。<br>
また、グループ学習ができるように、班の数に応じて繰り返し転移学習ができるように組みなおしたものもあります。

ここでは、<br>
(1) Google Colaboratoryで実行する<br>
Histology_Image_prediction.ipynb<br>
(2) Google Colaboratoryで班の数分実行する<br>
Histology_Image_prediction_group.ipynb<br>
(3)　(1), (2)で作製したモデルファイルから推論のみを行うための<br>
Model_test.ipynb<br>
および、上記をローカルのPCで実行するための、<br>
Histology_Image_prediction.py, Histology_Image_prediction_group.ipynb, Model_test.py

および、実行テスト用の画像ファイルフォルダである<br>
train, test, group_train, group_test<br>
を公開しています。trainとtest中には腎臓の尿細管と血管の分類を、group_train, group_testでは白血球の分類を行うための画像ファイルが含まれます。<br>

詳しい使い方は、manual.pdfを参照してください。<br><br>

使い方の動画もそのうちにアップロードする予定です。


#######使用にあたっての注意事項#######
含まれるソースコードについては、改変、再頒布可能です。
一方で、サンプル画像は東京医科歯科大学に著作権が存在しており、無断での再頒布は禁止します。
作成者および著作権者は、本ソースコード並びに画像について義務や責任を何ら負いません。

#########連絡先#################
稲葉弘哲
E mail:hinaba{at}med.mie-u.ac.jp ({at}を@に変更してください）
