import argparse
import time
import os
import numpy as np
import model as md
import dela_data as da
import config
import tensorflow as tf
import matplotlib.pyplot as plt

def main(args):
    if args.noise:
        summary_dir =  os.path.join(args.checkpoint_dir,"noise")
    if args.events:
        summary_dir =  os.path.join(args.checkpoint_dir,"events")
    while True:
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if args.eval_interval < 0 or ckpt:
            print ('Evaluating model')
            break
        print  ('Waiting for training job to save a checkpoint')
        time.sleep(args.eval_interval)

    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_epochs = 1
    cfg.add = 1
    coord = tf.train.Coordinator()
    while True:
        try:
            # data pipeline
            data_pipeline = da.data_pipeline(args.dataset, config=cfg,
                                            is_training=False)
            samples = {
              'data': data_pipeline.samples,
              'label': data_pipeline.labels,
               }
            model = md.get(args.model,samples,checkpoint_dir=args.checkpoint_dir,is_training=False,reuse=False)
            metrics = model.validation_metrics()
            summary_writer = tf.summary.FileWriter(summary_dir, None)

            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                tf.initialize_local_variables().run()
                #sess.run(tf.global_variables_initializer())
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                model.load(sess,args.step)
                print  ('Evaluating at step {}'.format(sess.run(model.global_step)))

                step = tf.train.global_step(sess, model.global_step)
                mean_metrics = {}
                for key in metrics:
                    mean_metrics[key] = 0
                n = 0
                m = 0
                pred = np.empty(2)
                true_labels = np.empty(1)
                print("7777777777777777777777777777777")
                while True:
                     try:
                         to_fetch = [metrics,
                                     model.layers['class_prediction'],
                                     model.layers['class_prob'],
			             samples["label"]]
                         metrics_,batch_prelabel, batch_pred, batch_true_label = sess.run(to_fetch)
                         pred = np.append(pred,batch_pred)
                         print(batch_prelabel,batch_true_label)
                         print(batch_pred)
                         trace = samples["data"]
                         if batch_prelabel ==batch_true_label:
                             m+=1
                         else:
                             print(sess.run(trace[0]))
                             print(batch_pred)
                             #plt.plot(sess.run(trace[0]))
                             #plt.show()
                         true_labels = np.append(true_labels,batch_true_label)

                         for key in metrics:
                                mean_metrics[key] += cfg.batch_size*metrics_[key]
                         n += cfg.batch_size

                         mess = model.validation_metrics_message(metrics_)
                         print ('{:03d} | '.format(n)+mess)
                     except KeyboardInterrupt:
                        print ('stopping evaluation')
                        break

                     except tf.errors.OutOfRangeError:
                        print ('Evaluation completed ({} epochs).'.format(cfg.n_epochs))
                        print ("{} windows seen".format(n))
                        break
            print('true = {} |  det_accuracy = {}'.format(m,m/n))
            break
            #tf.reset_default_graph()
            #print ('Sleeping for {}s'.format(args.eval_interval))
            #time.sleep(args.eval_interval)
        finally:
             print ('joining data threads')
             coord.request_stop()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default=None,
                        help='path to the recrords to evaluate')
    parser.add_argument('--model', type=str,default="cosac",
                        help='path to the recrords to evaluate')
    parser.add_argument('--checkpoint_dir',default='output_dir',
                        type=str, help='path to checkpoints directory')
    parser.add_argument('--step',type=int,default=None,
                        help='step to load')
    parser.add_argument('--eval_interval',type=int,default=10,
                        help='sleep time between evaluations')
    parser.add_argument('--save_summary',type=bool,default=True,
                        help='True to save summary in tensorboard')
    parser.add_argument('--noise', action='store_true',
                        help='pass this flag if evaluate acc on noise')
    parser.add_argument('--events', action='store_true',
                        help='pass this flag if evaluate acc on events')
    parser.set_defaults(profiling=False)

    args = parser.parse_args()
    main(args)
