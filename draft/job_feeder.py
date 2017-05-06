import itertools as it
import os
import logging

import keras.layers as L
import keras.backend as K
import keras.initializers as I
from keras.engine import Input, Model
import tensorflow as tf

from toyosatomimi import JobFeeder


def gen_model(act, width, init, model_name):
    if isinstance(init, float):
        initializer = I.RandomUniform(-init, init)
    else:
        initializer = init

    x = Input(shape=(28, 28))
    h = x
    h = L.Flatten()(h)
    h = L.Dense(width, kernel_initializer=initializer, bias_initializer='zeros')(h)
    h = L.Activation(act)(h)
    h = L.Dense(10, activation='softmax')(h)
    model = Model(x, h, name=model_name)
    return model


def job_generator():
    act_conds = [
        'sin',
        'cos',
        'tanh',
        'relu',
        'elu',
        'sigmoid',
        'softplus',
        'softsign',
        'softmax',
    ]
    width_conds = [
        64,
        128,
        256,
        512
    ]
    init_conds = [
        'glorot_uniform',
        0.05,  # default
        0.1,
        0.2,
        0.4,
        0.8,
    ]
    base_job_path = 'jobs'
    os.makedirs(base_job_path, exist_ok=True)
    for t in range(10):
        for act_name, width, init in it.product(act_conds, width_conds, init_conds):
            model_name = f'act_{act_name}-width_{width}-init_{init}-try_{t}'
            model_path = os.path.join(base_job_path, f'{model_name}.h5')

            act = {
                'sin': K.sin,
                'cos': K.cos,
            }.get(act_name, act_name)
            model = gen_model(act, width, init, model_name)
            model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
            model.save(model_path)

            yield {
                'act': act_name,
                'width': width,
                'init': init,
                'name': model_name,
                'model_path': model_path,
            }


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'
    sess = tf.Session(config=config)
    K.set_session(sess)
    logging.basicConfig(level=logging.INFO)
    feeder = JobFeeder()
    feeder.feed(job_generator())


if __name__ == '__main__':
    main()
