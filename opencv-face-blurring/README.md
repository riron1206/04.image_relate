# opencv-face-blurring
ssdで顔検出して、opencvのガウシアンブラーで顔だけぼかす
- https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/ のほぼパクリ

## Usage
```bash
$ activate tfgpu20
$ python blur_face.py -i examples/input/ -o examples/output/ -m simple     # 指定ディレクトリの画像全件について、顔の矩形範囲をぼかす
$ python blur_face.py -i examples/input/ -o examples/output/ -m pixelated  # 指定ディレクトリの画像全件について、顔の輪郭に沿ってぼかす
$ python blur_face_video.py -v None                                        # カメラのリアルタイム動画について、顔の矩形範囲をぼかす
$ python blur_face_video.py -v D:\iPhone_pictures\2019-04\IMG_9304.MOV -o tmp -m pixelated --is_rot90   # 指定の動画ファイルについて、顔の輪郭に沿ってぼかす
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)