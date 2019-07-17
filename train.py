import argparse
import os
import numpy as np
import model
import dela_data as da
import config
import tensorflow as tf
import time
def main():
    tf.set_random_seed(1234)
    cfg = config.Config()
    event_pipeline = da.data_pipeline('rrsac/event',cfg,True)
    noise_pipeline = da.data_pipeline('rrsac/noise',cfg,True)
    event_samples = {
      'data':event_pipeline.samples,
      'label':event_pipeline.labels,}
    noise_samples = {
      'data':noise_pipeline.samples,
      'label':noise_pipeline.labels,}
    samples = {
      'data': tf.concat([event_samples['data'],noise_samples['data']],0),
      'label':tf.concat([event_samples['label'],noise_samples['label']],0)}
    cosac = model.cosac(inputs=samples,checkpoint_dir="output_rrsac6",is_training=True)
    time_start = time.time()

    cosac.train(
      learning_rate = cfg.learning_rate,
      resume = True,
      profiling=False,
      summary_step=1)
    time_end = time.time()
    train_time = time_end - time_start
    print("train time is {} S:".format(train_time))
if __name__ =='__main__':
    main()
