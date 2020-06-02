# -*- coding: utf-8 -*-
"""
画像を、ssdで顔検出して、年齢クラスを予測するcnnで年齢予測する
https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/

年齢クラスを予測するcnnは2015年のレガシーネットワーク(AlexNetっぽいの)
簡単にするため年齢回帰ではなく、5歳ごとに分けた年齢クラスをsoftmaxで予測する
https://talhassner.github.io/home/publication/2015_CVPR

Usage:
    $ activate tfgpu20
    $ python detect_age.py --images examples/input/adrian.png --face face_detector --age age_detector
    $ python detect_age.py --images examples/input/
    $ python detect_age.py -i D:\iPhone_pictures\2011-09 -o tmp
    $ python detect_age.py -i D:\iPhone_pictures\2019-05\IMG_9842.JPG -o tmp
    $ python detect_age.py -i D:\iPhone_pictures\2019-05\IMG_9830.JPG -o tmp
"""
import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
from tqdm import tqdm


def detect_and_predict_age_img(args, image_path, faceNet, ageNet):
    """
    1枚の画像を、ssdで顔検出して、年齢クラスを予測するcnnで年齢予測する
    """
    # 年齢検出器が予測する年齢バケットのリストを定義
    # このクラス分けは作為的な気がする
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    # 入力画像を読み込み、画像用の入力ブロブを作成
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    # OpenCVでテンソル作成?
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # ネットワークを介してブロブを通過させ、顔検出を取得
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # 検出されたものをループ
    for i in range(0, detections.shape[2]):
        # 予測に関連する信頼度(すなわち，確率)を抽出
        confidence = detections[0, 0, i, 2]

        # 信頼度が最小信頼度よりも大きいことを確認することで，弱い検出をフィルタリング
        if confidence > args["confidence"]:
            # オブジェクトのバウンディングボックスの (x, y)座標を計算
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 顔のROIを抽出し，*顔のROIのみからブロブを作成
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # 年齢を予測し、対応する最大の確率で年齢のバケツを見つける
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()  # 予測スコア最大の年齢クラスを選択
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # 予測される年齢を表示
            text = "{}: {:.2f}%".format(age, ageConfidence * 100)
            print("[INFO] {}".format(text))

            # 顔のバウンディングボックスを，関連する予測年齢と一緒に描きます
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, nargs='*', help="paths to input image")
    ap.add_argument("-o", "--output_dir", default=r"examples\output", help="output dir path")
    ap.add_argument("-f", "--face", default="face_detector", help="path to face detector model directory")
    ap.add_argument("-a", "--age", default="age_detector", help="path to age detector model directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-is_s", "--is_show", action='store_const', const=True, default=False, help="image show flag")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    # シリアル化された顔検出器モデルをディスクからロード
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # ディスクからシリアル化された年齢検出器モデルをロード
    print("[INFO] loading age detector model...")
    prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
    weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
    ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    image_paths = glob.glob(os.path.join(args['images'][0], '*')) if os.path.isdir(args['images'][0]) else args['images']
    for image_path in tqdm(image_paths):
        # 年齢予測実行
        image = detect_and_predict_age_img(args, image_path, faceNet, ageNet)

        # 出力画像保存
        if args['output_dir'] is not None:
            cv2.imwrite(os.path.join(args['output_dir'], pathlib.Path(image_path).stem + '_detect_age.png'), image)

        # 元画像と出力画像を並べて表示
        if args['is_show']:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
