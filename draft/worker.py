import os
import sys
import logging

import keras
import keras.backend as K
import keras.callbacks as C
import tensorflow as tf

from toyosatomimi import Worker


def preprocess(dataset):
    x, y = dataset
    x = x.astype('float32') / 255 * 2 - 1
    return x, y


class MNISTTrainWorker(Worker):
    BASE_RESULT_PATH = 'results'

    def __init__(self, gpu_id):
        super().__init__(f'gpu_{gpu_id}')
        self.gpu_id = gpu_id
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = f'{gpu_id}'
        sess = tf.Session(config=config)
        K.set_session(sess)

        train_data, test_data = keras.datasets.mnist.load_data()
        self.train_data = preprocess(train_data)
        self.test_data = preprocess(test_data)


    def do(self, model_path, **kwargs):
        with keras.utils.generic_utils.CustomObjectScope({
            'sin': K.sin,
            'cos': K.cos,
        }):
            model = keras.models.load_model(model_path)
        result_path = os.path.join(self.BASE_RESULT_PATH, model.name)
        os.makedirs(result_path, exist_ok=True)

        model.fit(
            *self.train_data,
            epochs=50,
            validation_data=self.test_data,
            callbacks=[
                C.CSVLogger(os.path.join(result_path, 'log.csv')),
            ]
        )
        model.save(os.path.join(result_path, 'model.h5'))

        del model


def main():
    logging.basicConfig(level=logging.INFO)
    gpu_id = int(sys.argv[1])
    worker = MNISTTrainWorker(gpu_id)
    print(f'start worker:{gpu_id}')
    worker.run()


if __name__ == '__main__':
    main()
