import obspy
import matplotlib.pyplot as plt
from dela_data import data_writer
import numpy as np
import time
def preprocess_stream(stream):
    stream = stream/max(stream)
    return stream
syn = [0]*3402*4
st = obspy.read("JX.44154.10517.BHZ.sac.pick",format="SAC")
tr0 = st[0]
syn[0:3402]=tr0.data
st1 = obspy.read("JX.44155.10517.BHZ.sac.pick",format="SAC")
tr1 = st1[0]
syn[3402:6804]= tr1.data
st2 = obspy.read("JX.44156.10517.BHZ.sac.pick",format="SAC")
tr2 = st2[0]
syn[6804:3402*3] = tr2.data
st3 = obspy.read("JX.44158.10517.BHZ.sac.pick",format="SAC")
tr3 = st3[0]
t1 = int(200*tr0.stats.sac.t1)
syn[3402*3:3402*4]=tr3.data
#plt.plot(syn[0:3402*4])
#plt.show()
num_write = 0
syn = preprocess_stream(syn)
plt.plot(np.arange(0,68.036,0.005),syn[0:3402*4],color="k")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yticks([-1,-0.5,0,0.5,1])
#plt.xlabel("Time/s",fontsize=15)
#plt.ylabel("Amplitude",fontsize=15)
plt.show()
print(np.mean(syn),np.std(syn))
b = np.random.normal(0,0,len(syn))
syn = preprocess_stream(syn)
syn = syn+b
time_start = time.time()
m, s = divmod(time.time() - time_start, 60)
print ("Prediction took {} min {} seconds".format(m,s))
#plt.xlabel("Time/s",fontsize=20)
#plt.ylabel("Amplitude",fontsize=20)
writer = data_writer("syn3.tfrecords")
for i in range(0,13207):
    writer.write(syn[i:i+401],0)
    num_write+=1
    
print(num_write)
