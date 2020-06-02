# -*- coding: utf-8 -*-
"""
Tesseract OCRを使った数字/アルファベットのローカライズと検出
https://www.pyimagesearch.com/2020/05/25/tesseract-ocr-text-localization-and-detection/
Usage:
    $ activate tfgpu20
    $ python localize_text_tesseract.py
    $ python localize_text_tesseract.py -i D:\iPhone_pictures\2019-06\IMG_0017.PNG -o tmp
    $ python localize_text_tesseract.py -i D:\iPhone_pictures\2018-09\IMG_7231.PNG -o tmp -c 50
"""
import argparse
import glob
import os
import pathlib

import cv2
from tqdm import tqdm

import pytesseract
from pytesseract import Output
# pytesseract 実行するには tesseract のexeファイルの指定が必要
# https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
# https://github.com/UB-Mannheim/tesseract/wiki からダウンロードしたexe実行して tesseract インストールした
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_ocr(image_path: str):
    """
    画像1枚ロードしてテキスト検出。バウンディングボックス描画した画像返す
    - pytesseractはLSTMでテキスト検出してるらしい
    """

    def put_bbox(image, results):
        """
        検出結果をバウンディングボックスで描画
        """
        # 個々のテキストのローカライズをループ
        for i in range(0, len(results["text"])):
            # 現在の結果からテキスト領域のバウンディングボックス座標を抽出
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            # OCRテキスト自体を、テキストの地域化の信頼性と一緒に抽出
            text = results["text"][i]
            conf = int(results["conf"][i])

            # 弱い信頼度のテキストのローカライズをフィルタリング
            if conf > args["min_conf"]:
                # 信頼度とテキストを端末に表示
                print("Confidence: {}".format(conf))
                print("Text: {}".format(text))
                print("")

                # 非 ASCII 文字を削除して，OpenCV を用いて画像上にテキストを描画し，テキストの周囲にテキストと一緒に外接枠を描画
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 入力画像を読み込み，BGRからRGBチャンネル順に変換し，Tesseractを用いて入力画像中のテキストの各領域をローカライズ
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

    # 画像にバウンディングボックス描画
    put_bbox(image, results)

    return image, results


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', type=str, default=['apple_support.png'], nargs='*', help='paths to input image to be OCR')
    ap.add_argument('-o', '--output_dir', default='output', help='output dir path')
    ap.add_argument('-c', '--min-conf', type=int, default=0, help='mininum confidence value to filter weak text detection')
    ap.add_argument('-is_s', '--is_show', action='store_const', const=True, default=False, help='image show flag')
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    image_paths = glob.glob(os.path.join(args['images'][0], '*')) if os.path.isdir(args['images'][0]) else args['images']
    for image_path in tqdm(image_paths):
        # pytesseractで検出
        image, results = detect_ocr(image_path)

        # 出力画像保存
        if args['output_dir'] is not None:
            os.makedirs(args['output_dir'], exist_ok=True)
            cv2.imwrite(os.path.join(args['output_dir'], pathlib.Path(image_path).stem + '_ocr.png'), image)

        if args['is_show']:
            # 出力画像を表示
            cv2.imshow('Image', image)
            cv2.waitKey(0)
