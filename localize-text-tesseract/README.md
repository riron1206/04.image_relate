# localize-text-tesseract
LSTM�x�[�X��OCR�c�[���ŉ摜���琔��/�A���t�@�x�b�g�����o����
- https://www.pyimagesearch.com/2020/05/25/tesseract-ocr-text-localization-and-detection/ �̂قڃp�N��
- Windows�̏ꍇ https://github.com/UB-Mannheim/tesseract/wiki ����tesseract�̃C���X�g�[�����_�E�����[�h����ю��s���āAtesseract���C���X�g�[������K�v������

## Usage
```bash
localize_text_tesseract.py ��24�s�ړ������ pytesseract.pytesseract.tesseract_cmd ���C���X�g�[������tesseract��exe�t�@�C���̃p�X�ɕύX����
$ activate tfgpu20
$ python localize_text_tesseract.py -i apple_support.png -o output  # 1�摜�ɂ��āAOCR���o�B���o�摜��output�f�B���N�g���ɕۑ�
$ python localize_text_tesseract.py -i input -o tmp -c 50           # �w��f�B���N�g���̉摜�S���ɂ��āAOCR���o�B�\���X�R�A��50���傫���v�f���������o
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)