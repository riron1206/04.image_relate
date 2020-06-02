# social-distance-detector
OpenCV + yolo v3���g�����\�[�V�����f�B�X�^���X���o��
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/ �̂قڃp�N��
- yolo v3�̏d�݃t�@�C����100MB�ȏ��upload�ł��Ȃ������B�Ȃ̂ŁA��̃����N����d�݃t�@�C���܂߂��t�@�C���ꎮ�_�E�����[�h���Ȃ��Ƃ��߁I�I�I�I
- OpenCV v4.1.2�ł�cuda�g���Ȃ������̂�gpu�g���Ȃ��B����͂߂��᎞�Ԃ�����

## Usage
```bash
$ activate tfgpu20
$ python social_distance_detector.py -i input/pedestrians.mp4  # ����t�@�C���Ō��o
$ python social_distance_detector.py -i input/person.jpg       # �摜�t�@�C���Ō��o
$ python social_distance_detector.py -i input/dog.jpg -l dog   # �\���N���X���udog�v�ɂ���
$ python social_distance_detector.py -i input/dog.jpg -l dog -o tmp --is_simple_yolo_pred  # yolo�ŒP���ȋ�`�\������
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)