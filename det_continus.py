#!/usr/bin/env python
import os
import config as config
import shutil
from dela_data import data_pipeline
import model as md
import tensorflow as tf
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
def preprocess_stream(stream):
    stream = stream/max(stream)
    return stream
def main(args):
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    cfg = config.Config()
    cfg.batch_size = 1
    cfg.add = 1
    cfg.n_epochs = 1
    # Remove previous output directory
    '''if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    if args.plot:
        os.makedirs(os.path.join(args.output,"viz"))'''
    # data pipeline
    datapipeline = data_pipeline(args.dataset, config=cfg,is_training=False)
    samples = {"data":datapipeline.samples,
               "label":datapipeline.labels}
    #set up model and validation metrics
    model = md.get(args.model, samples, checkpoint_dir=args.checkpoint_dir,is_training=False)
    metrics = model.validation_metrics()
    max_windows = args.max_windows
    
    #run model
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        model.load(sess,args.step)
        print("detect using model at step {}".format(sess.run(model.global_step)))
        step = tf.train.global_step(sess, model.global_step)
        idx = 0
        time_start = time.time()
        pro = [0]*13608
        while True:
            try:
                # fetch
                to_fetch = [metrics,
                            samples['data'],
                            model.layers['class_prob'],
                            model.layers['class_prediction']]
                metrics_,sample,class_prob_,pred_label=sess.run(to_fetch)
                pro[idx+200] = class_prob_[0,1]
                idx +=1
                print("processed {} windows".format(idx))
                print(class_prob_,pred_label)
                mess = model.validation_metrics_message(metrics_)
                print ('{:03d} | '.format(idx)+mess)
                

            except KeyboardInterrupt:
                print ("Run time: ", time.time() - time_start)

            except tf.errors.OutOfRangeError:
                print ('Evaluation completed ({} epochs).'.format(cfg.n_epochs))
                break
        print ('joining data threads')
        m, s = divmod(time.time() - time_start, 60)
        print ("Prediction took {} min {} seconds".format(m,s))
        plt.plot(np.arange(0,68.036,0.005),pro,color="k")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.xlabel("Time/s",fontsize=20)
        #plt.ylabel("Probability of Event",fontsize=20)
        plt.show()
        coord.request_stop()
        coord.join(threads)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None,
                        help="path to tfrecords to analyze")
    parser.add_argument("--checkpoint_dir",type=str,default=None,
                        help="path to directory of chekpoints")
    parser.add_argument("--step",type=int,default=None,
                        help="step to load, if None the final step is loaded")
    parser.add_argument("--model",type=str,default="cosac",
                        help="model to load")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    args = parser.parse_args()

    main(args)
