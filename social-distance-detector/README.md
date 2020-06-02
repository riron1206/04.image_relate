# social-distance-detector
OpenCV + yolo v3を使ったソーシャルディスタンス検出器
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/ のほぼパクリ
- yolo v3の重みファイルが100MB以上でuploadできなかった。なので、上のリンクから重みファイル含めたファイル一式ダウンロードしないとだめ！！！！
- OpenCV v4.1.2ではcuda使えなかったのでgpu使えない。動画はめちゃ時間かかる

## Usage
```bash
$ activate tfgpu20
$ python social_distance_detector.py -i input/pedestrians.mp4  # 動画ファイルで検出
$ python social_distance_detector.py -i input/person.jpg       # 画像ファイルで検出
$ python social_distance_detector.py -i input/dog.jpg -l dog   # 予測クラスを「dog」にする
$ python social_distance_detector.py -i input/dog.jpg -l dog -o tmp --is_simple_yolo_pred  # yoloで単純な矩形予測する
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)