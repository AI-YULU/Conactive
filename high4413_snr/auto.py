#calcute snr and chose 
import config as config
import shutil
import model as md
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import obspy
import os
from obspy import read
import math
from dela_data import data_writer
from tqdm import tqdm
import time
def preprocess_stream(stream):
    stream = stream.normalize()
    return stream
def main(args):
    #creat dir to store tfrecords
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    cfg = config.Config()
    cfg.batch_size = 1
    cfg.add = 1
    cfg.n_epochs = 1
    for root, dirs, files in os.walk('stream'):
	    a = len(files)
	    for file in files:
	        os.chdir("stream")
	        st = read(file,format="SAC")
	        tr = st[0]
	        data = tr.data
	        b = tr.stats.sac.t1
	        c = int(200*b)
                #calcut the signal
	        signal = 0
	        for i in range(200):
	            signal += data[c+i]*data[c+i]
	        signal /= 200
	        #calcute the noise
	        noise = 0
	        for i in range(200):
	            noise += data[c+1000+i]*data[c+1000+i]
	        noise /= 200
	        snr = 10*math.log(signal/noise,10)
	        stream = preprocess_stream(st)
	        win_gen = stream[0].slide(window_length=args.window_length,
                			   step=args.window_step,
				   include_partial_windows=False)
	        print(win_gen)
	        if args.max_windows is None:
	            total_time = stream[0].stats.endtime - stream[0].stats.starttime
	            max_windows = int(total_time / args.window_step)
	            print("total time {}, window_length {}, window_step {}, windows_num {}".
	        	  format(total_time, args.window_length, args.window_step, max_windows))
	        else:
	            max_windows = args.max_windows
	        start_time = time.time()
	        num_write = 0
	        print("#########################",file)
	        output_name = file.split(".pick")[0] + ".tfrecords"
	        output_path = os.path.join(args.output_dir, output_name)
	        writer = data_writer(output_path)
	        for idx, win in enumerate(win_gen):
                    #write name and path
                    writer.write(win,1)
                    num_write+=1
                    print(idx)
                    if args.plot:
                        trace = win
                        viz_dir = os.path.join(args.output_dir,"viz",stream_file.split(".pick")[0])
                        if not os.path.exists(viz_dir):
                            os.makedirs(viz_dir)
                        trace.plot(outfile=os.path.join(viz_dir,"window_{}.png".format(idx)))
                    if idx == max_windows:
                        break
	        print("number of windows written={}".format(num_write))
	        writer.close()
	        # data pipeline
	        args.dataset = output_path
	        datapipeline = data_pipeline(args.dataset, config=cfg,is_training=False)
	        samples = {"data":datapipeline.samples,
                	   "label":datapipeline.labels}
	        #set up model and validation metrics
	        model = md.get(args.model, samples, checkpoint_dir=args.checkpoint_dir,is_training=False)
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
                            to_fetch = [samples['data'],
					model.layers['class_prob'],
					model.layers['class_prediction']]
                            sample,class_prob_,pred_label=sess.run(to_fetch)
                            pro[idx] = class_prob_[0,1]
                            idx +=1
                            print("processed {} windows".format(idx))
                            print(class_prob_,pred_label)


                        except KeyboardInterrupt:
                            print ("processed {} windows, found {} events".format(idx+1,n_events))
                            print ("Run time: ", time.time() - time_start)

                        except tf.errors.OutOfRangeError:
                            print ('Evaluation completed ({} epochs).'.format(cfg.n_epochs))
                            break
                        print ('joining data threads')
                        m, s = divmod(time.time() - time_start, 60)
                        print ("Prediction took {} min {} seconds".format(m,s))
                        print(pro.index(max(pro)))
                        plt.plot(pro)
                        plt.show() 
	        print(file,snr)
	        os.chdir("..")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,default=None,
                       help="path to sac to convert")
    parser.add_argument("--window_length",type=int,default=2,
                        help="length of the window to analyze")
    parser.add_argument("--window_step",type=int,default=0.005,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to create")
    parser.add_argument("--checkpoint_dir",type=str,default="../output_dir",
                        help="path to directory of checkpoints")
    parser.add_argument("--output_dir",type=str,default="note10",
                        help="dir of predicted events")
    parser.add_argument("--plot",type=bool,default=False,
                        help="pass this flag to plot windows")
    parser.add_argument("--step",type=int,default=None,
                        help="step to load, if None the final step is loaded")
    args = parser.parse_args()

    main(args)
