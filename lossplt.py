import matplotlib.pyplot as plt
fp = open("loss.txt","r")
ff = fp.readlines()
a = [0]*30000
i = 0
for line in ff:
    line = line.rstrip('\n')
    a[i] = float(line)
    i+=1
plt.plot(a[:20000],color="k")
#plt.xlabel('train step',fontsize=15)
#plt.ylabel('loss function',fontsize=15)
plt.show()
