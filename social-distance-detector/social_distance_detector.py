# -*- coding: utf-8 -*-
"""
OpenCV + yolo v3を使ったソーシャルディスタンス検出器
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

1. OpenCVのDNNモジュール(OpenCV 3.4.2以上が必要)と互換性があるYOLO v3(COCOデータセットで学習済み。80クラス検出可能)を使って「人」検出
2. 検出されたすべての人物間のペアワイズ距離を計算（重心のすべてのペア間のユークリッド距離を計算）
3. これらの距離に基づいて、2人の人物の間隔が50ピクセル未満かどうかを確認
→距離近い人のペアは赤枠で表示される

OpenCV v4.1.2ではcuda使えなかったのでgpu使えない。動画はめちゃ時間かかる

Usage:
    $ activate tfgpu20
    $ python social_distance_detector.py -i input/pedestrians.mp4
    $ python social_distance_detector.py -i input/person.jpg
    $ python social_distance_detector.py -i input/dog.jpg -l dog
    $ python social_distance_detector.py -i input/dog.jpg -l dog -o tmp --is_simple_yolo_pred
    $ python social_distance_detector.py -i IMG_1694.JPG -o tmp --is_simple_yolo_pred
"""
import argparse
import os
import pathlib

import cv2
import imutils
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance as dist

import social_distancing_config as config
from detection import detect_people


def load_yolo(cfg_dir=config.MODEL_PATH):
    # YOLOの重みとモデル構成へのパスを導出
    weightsPath = os.path.sep.join([cfg_dir, "yolov3.weights"])
    configPath = os.path.sep.join([cfg_dir, "yolov3.cfg"])

    # COCOCOデータセットで学習したYOLOオブジェクト検出器をロード (80クラス)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # GPUを使用するかどうかをチェック
    if config.USE_GPU:
        # バックエンドとターゲットに CUDA を設定
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # YOLOから*output*レイヤー名だけ取得
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln


def pred_yolo_v3_img(image_path: str, output_dir: str, coco_label_names=['person', 'dog']):
    """
    1画像について、YOLO v3(COCOデータセットで学習済み。80クラス検出可能)を使って、指定クラス検出
    Usage:
        import social_distance_detector
        img_path = r'IMG_5609.JPG'
        social_distance_detector.pred_yolo_v3_img(img_path, output_dir='tmp', coco_label_names=['person', 'dog'])
        social_distance_detector.pred_yolo_v3_img(img_path, output_dir='tmp', coco_label_names=[])
    """
    def get_random_color(seed=1):
        import random
        random.seed(seed)
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        return tuple([r, g, b])

    def _pred_class(image, net, ln, idx, color=(0, 0, 255)):
        # 検出実行
        results = detect_people(image, net, ln, personIdx=idx)
        # 結果をループ
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # バウンディングボックスとセントロイド座標を抽出し、アノテーションの色を初期化
            (startX, startY, endX, endY) = bbox
            # (cX, cY) = centroid
            # 外接箱を描く
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            text = f'{coco_label_name}: {round(prob*100, 1)}%'
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return image

    # 画像ロード
    image = cv2.imread(image_path)
    # yoloロード
    net, ln = load_yolo(cfg_dir=config.MODEL_PATH)

    # YOLOモデルが学習したCOCOクラスのラベルをロードします
    labels_path = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    labels = open(labels_path).read().strip().split("\n")

    # 指定なしなら全部のクラスで予測
    coco_label_names = labels if len(coco_label_names) == 0 else coco_label_names

    # クラスごとに検出
    pbar = tqdm(coco_label_names)
    for i, coco_label_name in enumerate(pbar):
        pbar.set_description(f'{coco_label_name}')
        idx = labels.index(coco_label_name)  # person は 0, dog は16
        # 検出
        image = _pred_class(image, net, ln, idx, color=get_random_color(seed=i))
    # 画像保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, str(pathlib.Path(image_path).stem) + '.png')
    cv2.imwrite(output_path, image)
    print(f"[INFO] save image: {output_path}")


def pred_social_distancing(input_file: str, output_dir: str, display=0, coco_label_name='person'):
    """ yoloでsocial_distancingの検出実行 """

    def _pred_frame(frame, idx):
        # 検出
        results = detect_people(frame, net, ln, personIdx=idx)
        # 最小のsocial distanceに違反するインデックスのセットを初期化するe
        violate = set()

        # 少なくとも2人の検出があることを確認します（ペアワイズ距離マップを計算するために必要です）
        if len(results) >= 2:
            # 結果からすべてのセントロイドを抽出し、すべてのペアのセントロイド間のユークリッド距離を計算する
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            # 距離行列の上三角形をループ
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # 2つのセントロイドのペア間の距離が設定されたピクセル数よりも小さいかどうかを確認
                    if D[i, j] < config.MIN_DISTANCE:
                        # 違反セットをセントロイドペアのインデックスで更新
                        violate.add(i)
                        violate.add(j)

        # 結果をループ
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # バウンディングボックスとセントロイド座標を抽出し、アノテーションの色を初期化
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            # インデックスペアが違反集合内に存在する場合, その色を更新
            if i in violate:
                color = (0, 0, 255)
            # (1)人物の周りに外接箱を描き、(2)人物のセントロイド座標を描く
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # 出力フレーム上に社会的距離の違反の総数を描画
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        return frame

    def _video(video_path, idx):
        # ビデオストリームと出力ビデオファイルへのポインタを初期化
        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(video_path if video_path else 0)
        writer = None
        # ビデオストリームのフレームをループ
        while True:
            # 次のフレームをファイルから読み込み
            (grabbed, frame) = vs.read()
            # フレームがつかめなかった場合は、ストリームの終わりに到達したことになります
            if not grabbed:
                break

            # 1frame検出
            # フレームのサイズを変更して、その中の人を検出します (人だけを検出します)
            frame = imutils.resize(frame, width=700)
            frame = _pred_frame(frame, idx)

            # 出力フレームが画面に表示されるかどうかを確認
            if display > 0:
                # 出力フレームを表示
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                # `q` キーが押された場合、ループから抜け出す
                if key == ord("q"):
                    break

            #  出力ビデオファイルのパスが指定されていて、ビデオライタが初期化されていない場合は、今すぐ初期化
            if output_path != "" and writer is None:
                # ビデオライターを初期化
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

            # video writer が None でない場合，フレームを出力ビデオファイルに書き込む
            if writer is not None:
                writer.write(frame)

        print(f"[INFO] save video: {output_path}")
        # 掃除をする
        vs.release()
        writer.release()
        cv2.destroyAllWindows()

    # YOLOモデルが学習したCOCOクラスのラベルをロードします
    labels_path = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    labels = open(labels_path).read().strip().split("\n")
    idx = labels.index(coco_label_name)  # personは 0

    # yolo v3ロード
    net, ln = load_yolo(cfg_dir=config.MODEL_PATH)

    if pathlib.Path(input_file).suffix.lower() in ['.jpg', '.png', '.jepg']:
        # 画像で検出実行
        image = cv2.imread(input_file)
        image = _pred_frame(image, idx)
        output_path = os.path.join(output_dir, str(pathlib.Path(input_file).stem) + '.png')
        cv2.imwrite(output_path, image)  # 画像保存
        print(f"[INFO] save image: {output_path}")
    else:
        # 動画で検出実行
        output_path = os.path.join(output_dir, str(pathlib.Path(input_file).stem) + '.mp4')
        _video(input_file, idx)


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str, default="input/pedestrians.mp4", help="path to (optional) input image/video file")
    ap.add_argument("-o", "--output_dir", type=str, default="output", help="path to (optional) output dir")
    ap.add_argument("-d", "--display", type=int, default=0, help="whether or not output frame should be displayed")
    ap.add_argument("-l", "--label_name_COCO", type=str, default='person', help="COCOのラベル名")
    ap.add_argument("-is_s", "--is_simple_yolo_pred", action='store_const', const=True, default=False, help="yoloで単純な矩形予測する場合")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()
    if args['is_simple_yolo_pred']:
        # yoloで単純な矩形予測
        pred_yolo_v3_img(args['input_file'], args['output_dir'], coco_label_names=[args['label_name_COCO']])
    else:
        # 社会的距離検出実行
        pred_social_distancing(args['input_file'], args['output_dir'], display=args['display'], coco_label_name=args['label_name_COCO'])
