# localize-text-tesseract
LSTMベースのOCRツールで画像から数字/アルファベットを検出する
- https://www.pyimagesearch.com/2020/05/25/tesseract-ocr-text-localization-and-detection/ のほぼパクリ
- Windowsの場合 https://github.com/UB-Mannheim/tesseract/wiki からtesseractのインストーラをダウンロードおよび実行して、tesseractをインストールする必要がある

## Usage
```bash
localize_text_tesseract.py の24行目当たりの pytesseract.pytesseract.tesseract_cmd をインストールしたtesseractのexeファイルのパスに変更する
$ activate tfgpu20
$ python localize_text_tesseract.py -i apple_support.png -o output  # 1画像について、OCR検出。検出画像はoutputディレクトリに保存
$ python localize_text_tesseract.py -i input -o tmp -c 50           # 指定ディレクトリの画像全件について、OCR検出。予測スコアが50より大きい要素だけを検出
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)