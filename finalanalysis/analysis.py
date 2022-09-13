import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data1=np.loadtxt('constantbatchservercpu.dat')
data2=np.loadtxt('constantbatchservergpu.dat')
data3=np.loadtxt('constantepochservercpu.dat')
data4=np.loadtxt('constantepochservergpu.dat')
data5=np.loadtxt('constantbatchlocalcpu.dat')
data6=np.loadtxt('constantepochlocalcpu.dat')


#Plotting Runtimes of constant batch sizes on various processing units


plotdata = pd.DataFrame({

    "Local GPU":[data6[0],data5[0],data5[1]],

    "Server CPU":[data3[0],data1[0],data1[1]],

    "Server GPU":[data4[0],data2[0],data2[1]]},

    index=['50','100','150'])

plotdata.plot(kind="bar",figsize=(15, 8))
plt.title('Runtimes of constant batch sizes on various Computing Engines')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Time in s')
plt.savefig('constantbatch.eps')
plt.show()


#Plotting Runtimes of constant epoch on various processing units


plotdata = pd.DataFrame({

    "Local GPU":[data6[0],data6[1],data6[2]],

    "Server CPU":[data3[0],data3[1],data3[2]],

    "Server GPU":[data4[0],data4[1],data4[2]]},

    index=['32','64','128'])

plotdata.plot(kind="bar",figsize=(15, 8))
plt.title('Runtimes of constant epoch on various Computing Engines')
plt.legend()
plt.xlabel('Batch Size')
plt.ylabel('Time in s')
plt.savefig('constantepoch.eps')
plt.show()
