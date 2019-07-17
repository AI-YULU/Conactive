import tensorflow as tf
import numpy as np
import os
import re
from tqdm import tqdm
class data_writer(object):
    def __init__(self,filename):
        self._writer = None
        self._filename=filename#output path
        self._writer=tf.python_io.TFRecordWriter(self._filename)
        self._written=0
    def write(self,sample_window,label):
        n_samples = len(sample_window.data)
        data = np.zeros((1,401),dtype=np.float32)
        data[0,0:n_samples] = sample_window.data
        example = tf.train.Example(features = tf.train.Features(feature={
                  'data':tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
                  'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        writer = self._writer
        writer.write(example.SerializeToString())
        self._written += 1
    def close(self):
        self._writer.close()
class data_reader(object):
    def __init__(self,path,config,shuffle=True):
        self._path = path
        self._shuffle = shuffle
        self._config = config
        self.win_size = config.win_size
        self._reader = tf.TFRecordReader()
    def read(self):
        filename_queue = self._filename_queue()
        _, serialized_example = self._reader.read(filename_queue)
        example = self._parse_example(serialized_example)
        return example
    def _filename_queue(self):
        fnames = []
        for root, dirs, files in os.walk(self._path):
            for f in files:
                if f.endswith(".tfrecords"):
                    fnames.append(os.path.join(root, f))
        fname_q = tf.train.string_input_producer(fnames,
                                                 shuffle=self._shuffle,
                                                 num_epochs=self._config.n_epochs)
        return fname_q                                         
        
    def _parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)})
        print(features)
        data = tf.decode_raw(features['data'], tf.float32)
        data = tf.reshape(data,[1,401])
        data = tf.transpose(data,[1,0])
        print("ddddddddddddddddddddddddddddddd")  
        features['data'] = data
        print(features)
        return features

class data_pipeline(object):
    def __init__(self,dataset_path,config,is_training):
        min_after_dequeue = 1000
        capacity = 1000+ 3 * config.batch_size
        batch_size = 100
        allow_smaller_final_batch=False
        if is_training:
            with tf.name_scope('inputs'):
                self._reader = data_reader(dataset_path, config=config)
                samples = self._reader.read()
                sample_input = samples['data']
                sample_label = samples["label"]
                print(samples['data'],samples['label'])
                print("#####################1")
                self.samples, self.labels = tf.train.shuffle_batch(
                    [sample_input, sample_label],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    allow_smaller_final_batch=False)
        elif not is_training:

            with tf.name_scope('validation_inputs'):
                self._reader = data_reader(dataset_path, config=config,shuffle=False)
                samples = self._reader.read()

                sample_input = samples["data"]
                sample_label = samples["label"]
                print("ccccccccccccccccccccccccccccccccccccccccccccccccc")
                self.samples, self.labels = tf.train.batch(
                    [sample_input, sample_label],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    num_threads=config.n_threads,
                    allow_smaller_final_batch=False)
        else:
            raise ValueError(
                "is_training flag is not defined, set True for training and False for testing")
