# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class TfAugment():

    def __init__(self, tf_dataset, params):
        self.tf_dataset = tf_dataset
        self.params = params

    def augment(self, X, y) -> tf.Tensor:
        """Some augmentation
        https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
        Args:
            X: 値0-1のImage
            y: ラベル
        Returns:
            Augmented image
        """
        # 90度ごとに回転
        # minval=0が回転なし、maxval=4が270度回転
        X = tf.image.rot90(X, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # 水平反転
        X = tf.image.random_flip_left_right(X)

        # 垂直反転
        X = tf.image.random_flip_up_down(X)

        # RGB画像の色相を調整
        X = tf.image.random_hue(X, self.params['random_hue'])

        # RGB画像の彩度を調整
        X = tf.image.random_saturation(X, self.params['random_saturation_min'], self.params['random_saturation_max'])

        # 画像の明るさを調整
        X = tf.image.random_brightness(X, self.params['random_brightness'])

        # 画像のコントラストを調整
        X = tf.image.random_contrast(X, self.params['random_contrast_min'], self.params['random_contrast_max'])

        # 与えられたサイズ(size)で画像をランダムにトリミング
        X = tf.image.random_crop(X, size=self.params['crop_size'], seed=None)

        # [0、1]の通常の範囲外の値を持つ画像になる可能性があるので0-1の範囲内に収める
        X = tf.clip_by_value(X, 0, 1)

        return X, y

    def main(self):
        """ tf.imageでaugmentした画像とラベルをtf.data型で返す """
        # num_parallel_calls=4：4コアで並列処理
        tf_dataset = self.tf_dataset.map(self.augment, num_parallel_calls=4)
        return tf_dataset


def plot_images(tf_dataset, n_images, samples_per_image, size=32):
    """tf.dataの画像表示"""
    output = np.zeros((size * n_images, size * samples_per_image, 3))

    row = 0
    for X, y in tf_dataset.repeat(samples_per_image).batch(n_images):
        output[:, row * size:(row + 1) * size] = np.vstack(X.numpy())
        row += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(output)
    plt.show()


if __name__ == '__main__':
    # cifar10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # 8画像だけ取ってtf.data型に変換
    tf_dataset = tf.data.Dataset.from_tensor_slices(((x_train[0:8] / 255).astype(np.float32), y_train[0:8]))
    print(tf_dataset)

    dic_params = {'minval': 0,
                  'maxval': 4,
                  'random_hue': 0.08,
                  'random_saturation_min': 0.6,
                  'random_saturation_max': 1.6,
                  'random_brightness': 0.05,
                  'random_contrast_min': 0.7,
                  'random_contrast_max': 1.3,
                  'crop_size': (28, 28, 3)}
    # tf.imageでaugment
    tf_dataset = TfAugment(tf_dataset, dic_params)
    tf_dataset = tf_dataset.main()
    # tf.dataの画像表示
    plot_images(tf_dataset, n_images=8, samples_per_image=10, size=28)