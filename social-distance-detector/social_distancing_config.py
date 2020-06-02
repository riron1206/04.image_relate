# YOLO ディレクトリへのベースパス
MODEL_PATH = "yolo-coco"

# 非最大抑圧を適用する際に，閾値と一緒に弱い検出をフィルタリングするための最小確率を初期化
MIN_CONF = 0.3
NMS_THRESH = 0.3

# NVIDIA CUDA GPUが使用されるべきかどうかを示すブール値
USE_GPU = False
# USE_GPU = True  # opencv v4.1.2 では cv2.dnn.DNN_BACKEND_CUDA がないから使えなかった

# 2人がお互いに近づける最小の安全な距離を定義します (ピクセル単位)
MIN_DISTANCE = 50
