# -*- coding: utf-8 -*-
"""
画像を、ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

Usage:
    $ activate tfgpu20
    $ python blur_face.py --images examples/input/adrian.jpg --face face_detector --method simple
    $ python blur_face.py --images examples/input/adrian.jpg examples/input/scarlett_johansson.png --method pixelated
    $ python blur_face.py --images examples/input/ --method simple
    $ python blur_face.py --images examples/input/ --method pixelated
    $ python blur_face.py -i D:\iPhone_pictures\2011-09 -o tmp
    $ python blur_face.py -i D:\iPhone_pictures\2019-05\IMG_9842.JPG -o tmp
    $ python blur_face.py -i D:\iPhone_pictures\2019-05\IMG_9830.JPG -o tmp
"""
import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
from tqdm import tqdm

from face_blurring import anonymize_face_pixelate
from face_blurring import anonymize_face_simple


def blur_face_img(args, image_path, net):
    """
    1枚の画像を、ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
    """
    # ディスクから入力画像をロードし，クローンを作成し，画像の空間寸法を取得します
    image = cv2.imread(image_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # 画像からブロブを作成する。OpenCVでテンソル作成?
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # ネットワークを介してブロブを通過させ、顔検出を取得します
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # 検出されたものをループ
    for i in range(0, detections.shape[2]):
        # 検出に関連する信頼度（すなわち，確率）を抽出する
        confidence = detections[0, 0, i, 2]

        # 信頼度が最小信頼度よりも大きいことを確認することで，弱い検出をフィルタリング
        if confidence > args["confidence"]:
            # オブジェクトのバウンディングボックスの (x, y)座標を計算
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 顔のROIを抽出
            face = image[startY:endY, startX:endX]

            # シンプルな顔のぼかし方を適用しているかどうかを確認
            if args["method"] == "simple":
                face = anonymize_face_simple(face, factor=3.0)
            # そうでなければ、"ピクセル化された "顔の匿名化メソッドを適用
            else:
                face = anonymize_face_pixelate(face, blocks=args["blocks"])

            # 出力画像にぼかした顔を保存
            image[startY:endY, startX:endX] = face

    return orig, image


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, nargs='*', help="paths to input image")
    ap.add_argument("-o", "--output_dir", default=r"examples\output", help="output dir path")
    ap.add_argument("-f", "--face", default="face_detector", help="path to face detector model directory")
    ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"], help="face blurring/anonymizing method")
    ap.add_argument("-b", "--blocks", type=int, default=20, help="# of blocks for the pixelated blurring method")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-is_s", "--is_show", action='store_const', const=True, default=False, help="image show flag")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    # ディスクからシリアル化された顔検出器モデル(ssd)をロードする
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    image_paths = glob.glob(os.path.join(args['images'][0], '*')) if os.path.isdir(args['images'][0]) else args['images']
    for image_path in tqdm(image_paths):
        # ぼかし実行
        orig, image = blur_face_img(args, image_path, net)

        # ぼかし画像保存
        if args['output_dir'] is not None:
            cv2.imwrite(os.path.join(args['output_dir'], pathlib.Path(image_path).stem + '_blur.png'), image)

        # 元画像と出力画像にぼかしをかけた顔を並べて表示
        if args['is_show']:
            output = np.hstack([orig, image])
            cv2.imshow("Output", output)
            cv2.waitKey(0)
