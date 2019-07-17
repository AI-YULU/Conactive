import obspy
import matplotlib.pyplot as plt
from dela_data import data_writer
import numpy as np
def preprocess_stream(stream):
    stream = stream/max(stream)
    return stream
syn = [0]*3402*4
st = obspy.read("JX.44136.10513.BHZ.sac.pick",format="SAC")
tr = st[0]
syn[0:3402]=tr.data
st1 = obspy.read("JX.44137.10513.BHZ.sac.pick",format="SAC")
tr = st1[0]
syn[3402:6804]= tr.data
st2 = obspy.read("JX.44141.10513.BHZ.sac.pick",format="SAC")
tr = st2[0]
syn[6804:3402*3] = tr.data
st3 = obspy.read("JX.44185.10513.BHZ.sac.pick",format="SAC")
tr = st3[0]
syn[3402*3:3402*4]=tr.data
plt.plot(syn[0:3402*4])
plt.show()
num_write = 0
syn = preprocess_stream(syn)
sta = [0]*3402*4
lta = [0]*3402*4
sta_lta = [0]*3402*4
for i in range(100,13608):
    for j in range(20):
        sta[i] += syn[i-j]*syn[i-j]
    for j in range(200):
        lta[i] += syn[i-j]*syn[i-j]
    sta_lta[i] = sta[i]/lta[i]
plt.plot(syn[0:3402*4])
plt.show()
plt.plot(sta_lta)
plt.show()
writer = data_writer("synback.tfrecords")
for i in range(13207):
    writer.write(syn[i:i+401],0)
    num_write+=1
    
print(num_write)
