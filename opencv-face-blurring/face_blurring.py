# -*- coding: utf-8 -*-
"""
opencvのガウシアンブラーで顔だけぼかす
https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
"""
import numpy as np
import cv2


def anonymize_face_simple(image, factor=3.0):
    # 入力画像の空間寸法に基づいて、ぼかしカーネルのサイズを自動的に決定
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)

    # カーネルの幅が奇数であることを確認
    if kW % 2 == 0:
        kW -= 1

    # カーネルの高さが奇数であることを確認
    if kH % 2 == 0:
        kH -= 1

    # 計算されたカーネルサイズを用いて入力画像にガウスぼかしを適用
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks=3):
    # 入力画像をNxNのブロックに分割
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # x と y の両方向にブロックをループ
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # 現在のブロックの開始座標と終了座標 (x, y) を計算
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # NumPy配列のスライスを用いてROIを抽出し，ROIの平均値を計算し，元の画像のROI上にRGBの平均値を持つ矩形を描画
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)

    # ピクセル化されたぼやけた画像を返す
    return image
