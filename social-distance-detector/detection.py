import numpy as np
import cv2

# パラメータ書いているconfigファイルロード
from social_distancing_config import NMS_THRESH
from social_distancing_config import MIN_CONF


def detect_people(frame: np.ndarray, net: cv2.dnn_Net, ln: list, personIdx=0):
    """
    OpenCVのDNNモジュールでロードしたYOLO v3でビデオストリームの1フレームを検出
    1フレームなので、cv2.imread()でロードした画像でも検出可能
    デフォルトは「人」検出で固定
    Args:
        frame：ビデオファイルまたは直接ウェブカメラからのフレーム
        net：事前初期化および事前トレーニング済みのYOLOオブジェクト検出モデル
        ln：YOLO CNN出力レイヤー名
        personIdx：personクラスのid。COCOの別のクラスにしたい場合は「coco.names」確認して番号指定する
    Returns:
        [(<予測確率>, <検出用の境界ボックス座標>, <オブジェクトの重心>), (…), …]
    """
    # フレームの寸法を取得し，結果のリストを初期化
    (H, W) = frame.shape[:2]
    results = []

    # 入力フレームからブロブを構築し、YOLOオブジェクト検出器のフォワードパスを実行し、バウンディングボックスと関連する確率を与える
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # 検出された bounding box, centroids, confidences のリストをそれぞれ初期化
    boxes = []
    centroids = []
    confidences = []

    # 各レイヤーの出力をループ
    for output in layerOutputs:
        #  各検出器をループさせる
        for detection in output:
            # 現在のオブジェクト検出のクラスIDと信頼度（すなわち，確率）を抽出
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #  (1) 検出された物体が人であることを確認し、(2)最小信頼度が満たされていることを確認します
            if classID == personIdx and confidence > MIN_CONF:
                # YOLO は，実際にはバウンディングボックスの中心 (x, y) 座標を返し，その後にボックスの幅と高さを返すことに注意してください
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 中心(x, y)座標を使用して、バウンディングボックスの上隅と左隅を導出
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # バウンディングボックスの座標、セントロイド、コンフィデンスのリストを更新
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # 弱く重なり合うバウンディングボックスを抑制するために非最大化抑制を適用
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # 少なくとも1つの検出が存在することを確認
    if len(idxs) > 0:
        # 保持しているインデックスをループ
        for i in idxs.flatten():
            # バウンディングボックスの座標を抽出
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 結果リストを更新して, 人物予測確率, バウンディングボックス座標, セントロイドから構成されるようにします
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # 結果のリストを返す
    return results
