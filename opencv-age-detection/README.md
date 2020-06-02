# opencv-age-detection
ssdで顔検出して、年齢クラスを予測するcnnで年齢予測する
- https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/ のほぼパクリ
- 年齢予測のcnnモデルはAlexNetっぽいのなので精度超低い

## Usage
```bash
$ activate tfgpu20
$ python detect_age.py -i examples/input/ -o examples/output/     # 指定ディレクトリの画像全件について、顔の年齢クラスを予測
$ python detect_age_video.py -v None                              # カメラのリアルタイム動画について、顔の年齢クラスを予測
$ python detect_age_video.py -v D:\iPhone_pictures\2019-04\IMG_9304.MOV -o tmp --is_rot90   # 指定の動画ファイルについて、顔の年齢クラスを予測
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)