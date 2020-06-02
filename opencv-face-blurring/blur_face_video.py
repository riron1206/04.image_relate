# -*- coding: utf-8 -*-
"""
カメラのリアルタイム動画or動画ファイルを、ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

Usage:
    $ activate tfgpu20
    $ python blur_face_video.py
    $ python blur_face_video.py -v None
    $ python blur_face_video.py -v D:\iPhone_pictures\2019-04\IMG_9303.MOV -o tmp --is_rot90
    $ python blur_face_video.py -v D:\iPhone_pictures\2019-04\IMG_9304.MOV -o tmp -m pixelated --is_rot90
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

from face_blurring import anonymize_face_pixelate
from face_blurring import anonymize_face_simple


def blur_face_camera(args, net):
    """
    カメラを使ってリアルタイムで、ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
    """
    # ビデオストリームを初期化し, カメラセンサーがウォームアップ
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # ビデオストリームのフレームをループ
    while True:
        # スレッドされた動画ストリームからフレームを取得し、最大幅が400ピクセルになるようにリサイズ
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # フレームの寸法を取得し、そこからブロブを作成
        (h, w) = frame.shape[:2]
        # OpenCVでテンソル作成?
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # ネットワークを介してブロブを通過させ、顔検出を取得
        net.setInput(blob)
        detections = net.forward()

        # 検出されたものをループ
        for i in range(0, detections.shape[2]):
            # 検出に関連する信頼度（すなわち，確率）を抽出
            confidence = detections[0, 0, i, 2]

            #  信頼度が最小信頼度よりも大きいことを確認することで，弱い検出をフィルタリング
            if confidence > args["confidence"]:
                # オブジェクトのバウンディングボックスの (x, y)座標を計算
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 顔のROIを抽出
                face = frame[startY:endY, startX:endX]

                # シンプルな顔のぼかし方を適用しているかどうかを確認
                if args["method"] == "simple":
                    face = anonymize_face_simple(face, factor=3.0)

                # そうでなければ、"ピクセル化された "顔の匿名化メソッドを適用
                else:
                    face = anonymize_face_pixelate(face, blocks=args["blocks"])

                # 出力画像にぼかした顔を保存
                frame[startY:endY, startX:endX] = face

        #  出力フレームを表示
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # qキーが押された場合、ループから抜け出す
        if key == ord("q"):
            break

    # 掃除をする
    cv2.destroyAllWindows()
    vs.stop()


def blur_face_video(args, net, is_rot90=True):
    """
    動画で、ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
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
    out_h_w = (int(h), int(w)) if is_rot90 else (int(w), int(h))

    # Define the codec and create VideoWriter object
    # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_video_display/py_video_display.html
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter((os.path.join(args['output_dir'], pathlib.Path(args['video']).stem + '_blur.mp4')),  # '_blur.avi'
                          fourcc,
                          20.0,  # フレームレート
                          out_h_w  # 画像サイズあってないと保存できない!!!!!
                          )
    cap = cv2.VideoCapture(args['video'])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_c in range(frame_count):
        try:
            # スレッドされた動画ストリームからフレームを取得し、最大幅が400ピクセルになるようにリサイズ
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=400)
            # フレームの寸法を取得し、そこからブロブを作成
            (h, w) = frame.shape[:2]
        except Exception:
            print(f_c)
            traceback.print_exc()
            break

        # OpenCVでテンソル作成?
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # ネットワークを介してブロブを通過させ、顔検出を取得
        net.setInput(blob)
        detections = net.forward()

        # 検出されたものをループ
        for i in range(0, detections.shape[2]):
            # 検出に関連する信頼度（すなわち，確率）を抽出
            confidence = detections[0, 0, i, 2]

            #  信頼度が最小信頼度よりも大きいことを確認することで，弱い検出をフィルタリング
            if confidence > args["confidence"]:
                # オブジェクトのバウンディングボックスの (x, y)座標を計算
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 顔のROIを抽出
                face = frame[startY:endY, startX:endX]

                # シンプルな顔のぼかし方を適用しているかどうかを確認
                if args["method"] == "simple":
                    face = anonymize_face_simple(face, factor=3.0)
                # そうでなければ、"ピクセル化された "顔の匿名化メソッドを適用
                else:
                    face = anonymize_face_pixelate(face, blocks=args["blocks"])

                # 出力画像にぼかした顔を保存
                frame[startY:endY, startX:endX] = face

        # 縦動画の縦横が逆になってしまったので、２７０度回転  https://oliversi.com/2019/01/16/python-opencv-movie2/
        if is_rot90:
            frame = np.rot90(frame, 3)

        # write the flipped frame
        # print(frame.shape)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 掃除をする
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def test_save_video(args):
    """
    動画ファイルコピーテスト
    http://rikoubou.hatenablog.com/entry/2019/01/15/174751
    """
    video = cv2.VideoCapture(args['video'])
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter((os.path.join(args['output_dir'], pathlib.Path(args['video']).stem + '_blur.mp4')),
                             fourcc,
                             20.0,  # フレームレート
                             (int(video.get(4)), int(video.get(3)))  # 画像サイズ
                             )

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = video.read()
        frame = np.rot90(frame, 3)  # 縦動画の縦横が逆になってしまったので、２７０度回転  https://oliversi.com/2019/01/16/python-opencv-movie2/
        # print(frame.shape)
        writer.write(frame)  # 画像を1フレーム分として書き込み

    writer.release()
    video.release()
    cv2.destroyAllWindows()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default=r"examples\mixkit-family-making-a-video-call-on-smartphone-4523.mp4", help="paths to input video")
    ap.add_argument("-o", "--output_dir", type=str, default=r"examples", help="output dir path")
    ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
    ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"], help="face blurring/anonymizing method")
    ap.add_argument("-b", "--blocks", type=int, default=20, help="# of blocks for the pixelated blurring method")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-is_r", "--is_rot90", action='store_const', const=True, default=False, help="image rot90 flag")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    # ディスクからシリアル化された顔検出器モデル(ssd)をロードする
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    if args['video'] == 'None':
        #  カメラを使ってリアルタイムでぼかし
        blur_face_camera(args, net)
    else:
        # 動画ファイルぼかす
        blur_face_video(args, net, is_rot90=args['is_rot90'])
        # 動画ファイル読み込みテスト用
        #test_save_video(args)
