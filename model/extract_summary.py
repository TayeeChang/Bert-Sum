# -*- coding: utf-8 -*-
# @Time : 2021/9/24 14:20
# @Author : haojie zhang

# BertSum 摘要抽取 最终acc=0.93385

import json
import random
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_noam_decay
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense, Lambda, Input
from keras.models import Model
from utils import  *
from tqdm import tqdm


maxlen = 64
batch_size = 4
sequence_num = 4

warmup_steps=10000
learning_rate= 2e-3

# bert配置
config_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(fileName):
    D = []
    with open(fileName, encoding='utf-8') as infile:
        for line in infile:
            line = json.loads(line)
            text = line['text']
            D.append(text)
    return D


data = load_data('datasets/train.json')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_cls_ids, batch_labels = [], [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids, cls_ids, labels = [], [], [], []
            for i, items in enumerate(d):
                sentence = items['sentence']
                label = items['label']
                cls_ids.append(len(token_ids))
                token_id, segment_id = tokenizer.encode(sentence, maxlen=maxlen)
                if i % 2 == 1:
                    segment_id = [1] * len(token_id)
                token_ids += token_id
                segment_ids += segment_id
                labels.append([label])
                if len(cls_ids) == sequence_num:
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    batch_cls_ids.append(cls_ids)
                    batch_labels.append(labels)
                    token_ids, segment_ids, cls_ids, labels = [], [], [], []
                    if len(batch_token_ids) == self.batch_size:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)
                        batch_cls_ids = sequence_padding(batch_cls_ids)
                        batch_labels = sequence_padding(batch_labels)
                        yield [batch_token_ids, batch_segment_ids, batch_cls_ids], batch_labels
                        batch_token_ids, batch_segment_ids, batch_cls_ids, batch_labels = [], [], [], []


def extract_cls_token(inputs):
    output, cls_ids = inputs
    return batch_gather(output, cls_ids)


def custom_sparse_categorical_accuracy(y_true, y_pred):
    flatten_y_true = K.cast( K.reshape(y_true, (-1,1) ), K.floatx())
    flatten_y_pred = K.cast(K.reshape(y_pred, (-1, y_pred.shape[-1])), K.floatx())
    y_pred_labels = K.cast(K.argmax(flatten_y_pred, axis=-1), K.floatx())
    flatten_y_true = K.cast( K.reshape(flatten_y_true, (-1, )), K.floatx())
    return K.cast(K.equal(flatten_y_true,y_pred_labels), K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    # flatten y_true in case it's in shape (num_samples, 1) instead of (num_samples,)
    return K.cast(K.equal(K.flatten(y_true),
                          K.flatten(K.cast(K.argmax(y_pred, axis=-1), K.floatx()))),
                  K.floatx())


bert = build_transformer_model(
    config_path,
    checkpoint_path,
    return_keras_model=False,
)

cls_ids = Input(shape=(sequence_num,), name='cls_ids')

output = Lambda(extract_cls_token)([bert.model.output, cls_ids])
output = Dense(2,
        activation='softmax',
        # kernel_initializer=bert.initializer
)(output)

model = Model(bert.model.inputs + [cls_ids], output)

AdamNoam = extend_with_noam_decay(Adam, 'AdamNoam')
optimizer = AdamNoam(lr=learning_rate,
                  warmup_steps=warmup_steps)

model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=[sparse_categorical_accuracy],
)

# scheduler = BatchLearningRateScheduler(
#     WarmUpLearingRate(lr=learning_rate, warmup_steps=warmup_steps),
#     verbose=1)

idxs = list(range(len(data)))
random.shuffle(idxs)
data = [data[i] for i in idxs]

train_data = data[:int(len(idxs) * 0.8)]
valid_data = data[int(len(idxs) * 0.8):]


def evaluate(data):
    total, right = 0, 0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=-1)
        y_true = y_true[..., 0]
        total += np.prod(y_true.shape)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def on_batch_begin(self, batch, logs=None):

        lr = float(K.get_value(self.model.optimizer.llr))

        print('batch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (batch + 1, lr))


if __name__ == '__main__':

    evaluator = Evaluator()

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    model.load_weights('best_model.weights')
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        verbose=2,
        callbacks=[evaluator] # , scheduler
    )

    model.load_weights('best_model.weights')
    print(u'final valid acc: %05f\n' % (evaluate(valid_generator)))

else:

    model.load_weights('best_model.weights')
