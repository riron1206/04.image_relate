# USAGE
# python detect_age_video.py --face face_detector --age age_detector

# -*- coding: utf-8 -*-
"""
カメラのリアルタイム動画or動画ファイルを、ssdで顔検出して、年齢クラスを予測するcnnで年齢予測する
https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/

年齢クラスを予測するcnnは2015年のレガシーネットワーク(AlexNetっぽいの)
簡単にするため年齢回帰ではなく、5歳ごとに分けた年齢クラスをsoftmaxで予測する
https://talhassner.github.io/home/publication/2015_CVPR

Usage:
    $ activate tfgpu20
    $ python detect_age_video.py
    $ python detect_age_video.py -v None
    $ python detect_age_video.py -v D:\iPhone_pictures\2019-04\IMG_9303.MOV -o tmp -c 0.3 --is_rot90
    $ python detect_age_video.py -v D:\iPhone_pictures\2019-04\IMG_9304.MOV -o tmp --is_rot90
"""
import argparse
import glob
import os
import pathlib
import time
import traceback

import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from tqdm import tqdm


def detect_and_predict_age(args, frame, faceNet, ageNet):
    """
    動画のフレーム単位（画像1枚ごと）に、ssdで顔検出して、年齢クラスを予測するcnnで年齢予測する
    """
    # 年齢検出器が予測する年齢バケットのリストを定義
    # このクラス分けは作為的な気がする
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    # 結果リストを初期化
    results = []

    # フレームの寸法を取得し、そこからブロブを作成します
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # ネットワークを介してブロブを通過させ、顔検出を取得します
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
            face = frame[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # 年齢を予測し、対応する最大の確率で年齢のバケツを見つける
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()  # 予測スコア最大の年齢クラスを選択
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # 顔のバウンディングボックスの位置と年齢予測の両方からなる辞書を構築し、結果リストを更新
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    # 呼び出した関数に結果を返します
    return results


def pred_camera(args, faceNet, ageNet):
    """
    カメラのリアルタイム動画で予測実行
    """
    # ビデオストリームを初期化し, カメラセンサーがウォームアップ
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # ビデオストリームのフレームをループ
    while True:
        # スレッドされたビデオストリームからフレームを取得し、最大幅が 400 ピクセルになるようにサイズを変更
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # フレーム内の顔を検出し，フレーム内の各顔について，年齢を予測する
        results = detect_and_predict_age(args, frame, faceNet, ageNet)

        # 結果をループ
        for r in results:
            # 顔のバウンディングボックスを、予測された年齢と一緒に描画
            text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
            (startX, startY, endX, endY) = r["loc"]
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # 出力フレームを表示
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # qキーが押された場合、ループから抜け出す
        if key == ord("q"):
            break

    # 掃除をする
    cv2.destroyAllWindows()
    vs.stop()


def pred_video(args, faceNet, ageNet):
    """
    動画で実行
    動画ロード/保存は http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_video_display/py_video_display.html を参考にした
    """
    print("[INFO] starting video stream...")

    # 出力画像サイズ取得する
    cap = cv2.VideoCapture(args['video'])
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    print('h w:', h, w)
    cap.release()

    # 画像回転させるか
    out_h_w = (int(h), int(w)) if args['is_rot90'] else (int(w), int(h))

    # Define the codec and create VideoWriter object
    # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_video_display/py_video_display.html
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter((os.path.join(args['output_dir'], pathlib.Path(args['video']).stem + '_detect_age.mp4')),  # '_blur.avi'
                          fourcc,
                          20.0,  # フレームレート
                          out_h_w  # 画像サイズあってないと保存できない!!!!!
                          )
    cap = cv2.VideoCapture(args['video'])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_c in range(frame_count):
        # スレッドされた動画ストリームからフレームを取得し、最大幅が400ピクセルになるようにリサイズ
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)

        # フレーム内の顔を検出し，フレーム内の各顔について，年齢を予測する
        results = detect_and_predict_age(args, frame, faceNet, ageNet)
        # print(results)

        # 結果をループ
        for r in results:
            # 顔のバウンディングボックスを、予測された年齢と一緒に描画
            text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
            (startX, startY, endX, endY) = r["loc"]
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # args['is_rot90'] == True だと文字縦書きになる。。。
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # 縦動画の縦横が逆になってしまったので、２７０度回転  https://oliversi.com/2019/01/16/python-opencv-movie2/
        # out.write()の直前に書かないと描画されない。。。なんで?
        frame = np.rot90(frame, 3) if args['is_rot90'] else frame
        # write frame
        out.write(frame)

    # 掃除をする
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default=r"examples\mixkit-family-making-a-video-call-on-smartphone-4523.mp4", help="paths to input video")
    ap.add_argument("-o", "--output_dir", type=str, default=r"examples", help="output dir path")
    ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
    ap.add_argument("-a", "--age", default="age_detector", help="path to age detector model directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-is_r", "--is_rot90", action='store_const', const=True, default=False, help="image rot90 flag")
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

    if args['video'] == 'None':
        #  カメラを使ってリアルタイムで年齢予測
        pred_camera(args, faceNet, ageNet)
    else:
        # 動画ファイル年齢予測
        pred_video(args, faceNet, ageNet)
