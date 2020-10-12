# google_images_download: Google画像ダウンロードパッケージ の使い方メモ

## Install
https://qiita.com/taedookim/items/63759e79426514c8a729 より、2020/10時点ではpip版はエラーになる
githubのをダウンロードする必要がある（一応まるごとコピーしたのも置いておく）
```bash
$ cd C:\Users\81908\Git\gid-joeclinton
$ git clone https://github.com/Joeclinton1/google-images-download.git gid-joeclinton
```

## Usage
```bash
$ cd C:\Users\81908\Git\gid-joeclinton
$ python google_images_download\google_images_download.py -l 10 -k cat
# カレントディレクトリdownloads/cat ディレクトリができて、猫の画像10枚ダウンロードされる。一回で取れる最大枚数は100枚みたい
```

# 参考サイト
- bing版もあるみたい
	- https://medium.com/towards-artificial-intelligence/building-a-custom-image-dataset-for-deep-learning-projects-7f759d069877

