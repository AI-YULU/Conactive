#creat continus tfrecords from continus sac
import os
import tensorflow as tf
import argparse
import obspy
import time
from tqdm import tqdm
from dela_data import data_writer
def preprocess_stream(stream):
    stream = stream.normalize()
    return stream
def main(args):
    #creat dir to store tfrecords
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #load stream
    stream_path = args.stream_path
    stream_file = os.path.split(stream_path)[-1]
    print("+loading stream {}".format(stream_file))
    stream = obspy.read(stream_path,format="SAC")
    print("+ preprocessing stream")
    stream = preprocess_stream(stream)
    # Create window generator
    print(stream)
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
    output_name = stream_file.split(".pick")[0] + ".tfrecords"
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
    parser.add_argument("--output_dir",type=str,default="output/continus_tfrecords",
                        help="dir of predicted events")
    parser.add_argument("--plot",type=bool,default=False,
                        help="pass this flag to plot windows")
    args = parser.parse_args()

    main(args)
