# export TF_CPP_MIN_LOG_LEVEL=2 (Ignore warnings)
# 25 Sep 2017
# Daniel Kho
# Seungmin Lee

# GDO number
# initial variable:  w&b
# Learning rate
# Sample size  

             #  ====> shape of regression

from __future__ import print_function
from decimal import *
from tensorflow.contrib.learn.python import SKCompat

import time
import io
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') #Generate Image without window appear
import matplotlib.pyplot as plt
import tensorflow as tf

start = time.time() # Record performance time

filename_queue = tf.train.string_input_producer(["RFID_bathdata_random10000.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
#                   ID  BatchMainID     UserID  ProcCode        ProcSeqnum      Quantity        Good Number     Time    Location        newTime

record_defaults = [tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32)]

# Convert CSV records to tensors. Each column maps to one tensor.
ID, BatchMainID, UserID, ProcCode, ProcSeqnum, Quantity, Good_Number, Time_Original, Location, Time_Integer = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([ID, BatchMainID, UserID, ProcCode, ProcSeqnum, Quantity, Good_Number, Time_Original, Location, Time_Integer])

def gen_plot_all(cluster, centroid_values, num_clusters):

    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title('Number of Processes vs Time takes to finish Batches')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (days)')
    plt.legend()
    plt.show()
    colour=['ro','bo','go','ko','mo']
    colour_centroid=['rx','bx','gx','kx','mx']
    for i in xrange(num_clusters):
        plt.plot(centroid_values[i][0],centroid_values[i][1], colour_centroid[ i%(len(colour)) ], markersize=8)
    xaxis_all=[]
    yaxis_all=[]
    for j in xrange(num_clusters):
        xaxis, yaxis = cluster_reshape(cluster,num_clusters,j)
        xaxis_all.append(xaxis)
        yaxis_all.append(yaxis)
        plt.plot(xaxis, yaxis, colour[j])
        print('Plot',j,'cluster of colour',colour[j])
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf, xaxis_all, yaxis_all

def gen_plot(xaxis, yaxis, clusterNo):

    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title('Number of Processes vs Time takes to finish Batches')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (days)')
    plt.legend()
    plt.show()
    colour=['ro','bo','go','ko','mo']
    colour_reg=['r-','b-','g-','k-','m-']
    xaxis, yaxis = zip( *sorted(zip(xaxis, yaxis)) )
    plt.plot(xaxis, yaxis, colour[ clusterNo ])
    print("Each cluster length",len(xaxis),"on ",clusterNo)
    
    # Gradient Decent Optimization method
    w = tf.Variable(tf.random_uniform([1],0, 50), name = 'weight')
    b = tf.Variable(tf.random_normal([1],-10,10), name='bias1')
    xaxis = np.asarray(xaxis,dtype=np.float32)

    if (clusterNo == 0):
        #y = w*(tf.reciprocal(xaxis)) + b
        y = w*(tf.log(xaxis)) + b
    if (clusterNo == 1):
        plt.plot([5,5],[0,100],colour_reg[ clusterNo ])
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 2):
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 3):
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 4):
        y = w*(tf.reciprocal(xaxis)) + b

    loss = tf.reduce_mean(tf.square(y - yaxis), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(0.015)
    train = optimizer.minimize(loss)
    
    # Initialisation
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    learningNumber = 101
    #if (clusterNo != 1):
    for step in xrange(learningNumber):
        sess.run(train)
        if ( (step % 10) == 0):
            print(step,"w = ",sess.run(w), "b = ", sess.run(b),  "Loss = ", sess.run(loss))
        
    yOut = sess.run(y)
    #if (clusterNo != 1):
    plt.plot(xaxis, yOut, colour_reg[ clusterNo ])
    
    plt.axis([0,max(xaxis)+5,0,max(yaxis)+5])
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def gen_plot_eval(xaxis, yaxis, clusterNo, w, b):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title('Number of Processes vs Time takes to finish Batches')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (days)')
    plt.legend()
    plt.show()
    colour=['ro','bo','go','ko','mo']
    colour_reg=['r-','b-','g-','k-','m-']
    xaxis, yaxis = zip( *sorted(zip(xaxis, yaxis)) )
    plt.plot(xaxis, yaxis, colour[ clusterNo ])
    print("Each cluster length",len(xaxis),"on ",clusterNo)
    
    xaxis = np.asarray(xaxis,dtype=np.float32)
    
    if (clusterNo == 0):
        #y = w*(tf.reciprocal(xaxis)) + b
        y = w*(tf.log(xaxis)) + b
    if (clusterNo == 1):
        plt.plot([5,5],[0,100],colour_reg[ clusterNo ])
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 2):
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 3):
        y = w*(tf.reciprocal(xaxis)) + b
    if (clusterNo == 4):
        y = w*(tf.reciprocal(xaxis)) + b
    
    yOut = sess.run(y)
    if (clusterNo != 1):
        plt.plot(xaxis, yOut, colour_reg[ clusterNo ])
    
    plt.axis([0,max(xaxis)+5,0,max(yaxis)+5])
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def clustering(xaxis, yaxis, assignment_values, num_clusters):
    cluster = []
    for i in xrange(num_clusters):
        temp_array = []
        for j in xrange(len(xaxis)):
            if(assignment_values[j] == i):
                temp_array.append( [ xaxis[j], yaxis[j] ] )
        cluster.append(temp_array)       
    return cluster

def cluster_reshape(cluster, num_clusters, selected_clusterNo):
    temp = cluster[selected_clusterNo]
    xaxis = []
    yaxis = []
    for i in xrange(len(temp)):
        xaxis.append(temp[i][0])
        yaxis.append(temp[i][1])
    return xaxis, yaxis

sess = tf.InteractiveSession()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

BatchMainIDArray=[]
timeArray=[]
locationArray=[]
IDArray=[]
rowsArray=[]
loop=10000
# try 50 250 500 etc
# note computational cost for each sets
# mention data size too large. How to reduce comp. cost? Not fully utilized whole data size
for i in range(loop):
    if ( (i % 500) == 0):
        print(i,'read data over',loop)
    rows, IDNumber, BatchMainIDNumber, Time, location = sess.run([features, ID, BatchMainID, Time_Integer, Location])
    rowsArray = np.append(rowsArray, rows)
    IDArray = np.append(IDArray, IDNumber)
    BatchMainIDArray = np.append(BatchMainIDArray, BatchMainIDNumber)
    timeArray = np.append(timeArray, Time)
    locationArray = np.append(locationArray, location)

BatchMainIDArray = BatchMainIDArray.tolist()
timeArray = timeArray.tolist()

cmpBatchMainID = None
calledBatchMainID = []
noofProcArray=[]
diffTime=[]

for i,j in enumerate(BatchMainIDArray):
    if ( (i % 500) == 0):
        print(i,'classified data over',loop)
    cmpBatchMainID = j
    sameBatchMainID = []
    corrspTime=[]
    if (cmpBatchMainID not in calledBatchMainID):
        calledBatchMainID.append(j)
        for k,l in enumerate(BatchMainIDArray):
            if(cmpBatchMainID == l):
                sameBatchMainID.append(k)
                corrspTime.append(timeArray[k])
        corrspTime.sort()
        noofProcArray.append(len(sameBatchMainID))
        diffTime.append(corrspTime[len(corrspTime) - 1] - corrspTime[0])

print("Classified Data Length",len(noofProcArray))

# K-mean
num_clusters = 5

vector_values = []
for i in range(len(noofProcArray)):
    vector_values.append([noofProcArray[i],diffTime[i]])
    
# K means for evaluation
centroid_user_values = [[20.46451569, 11.56920528],
                       [5, 44.69249344],
                       [8.88530445, 8.81974125],
                       [3.50498343, 0.81601208],
                       [6.15176725, 3.04871345]]
w_values = [13.24903679,23.40257263,42.70797348,1.66092837,12.11023426]
b_values = [-28.31702423,38.09459686,2.82725811,0.34326613,0.47114417]
    
vectors = tf.constant(vector_values)
#centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors,seed=1),[0,0],[num_clusters,-1])) #10000,50000first seed=1, 10000,50000last seed=8, 10000random seed=8, 50000random seed=11 / first samples only
centroids = tf.constant(centroid_user_values)

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

#means = tf.concat([    # First samples only
#    tf.reduce_mean(
#        tf.gather(vectors,
#                  tf.reshape(
#                    tf.where(
#                      tf.equal(assignments, c)
#                    ),[1,-1])
#                 ),reduction_indices=[1])
#    for c in xrange(num_clusters)], 0)

#update_centroids = tf.assign(centroids, means)     # First samples only

# Gradient Decent Optimization method was below here:

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# K mean
num_k_steps = 10
#for step in xrange(num_k_steps):
    #_, centroid_values, assignment_values = sess.run([update_centroids,centroids,assignments])     # First samples only

centroid_values, assignment_values = sess.run([centroids,assignments])
print("Assignmnet_eval")
print(assignment_values)
print("Centroids")
print(centroid_values)
    
# coord
coord.request_stop()
coord.join(threads)

# Clustering
cluster = clustering(noofProcArray, diffTime, assignment_values, num_clusters)

# Prepare the plot with Gradient Decent optimizaion method
plot_buf, xaxis_all, yaxis_all = gen_plot_all(cluster, centroid_values, num_clusters)

# Prediction
predictNo = 100
predict_NoProc = np.random.randint( 1, max(max(xaxis_all)), size = predictNo )
print("Predict Input")
print(predict_NoProc)

predictCluster = []
predictPercentage = []
for i in xrange(predictNo):
    predictCountArray = []
    for j in xrange(num_clusters):
        predictCount = 0
        for k in xrange( len(xaxis_all[j]) ):
            if (xaxis_all[j][k] == predict_NoProc[i]):
                predictCount = predictCount + 1
        predictCountArray.append(predictCount)
    if ( max(predictCountArray) == 0):
        predictCluster.append(-1)
        predictPercentage.append(0.0)
    else:
        predictCluster.append( predictCountArray.index( max(predictCountArray) ) )
        predictPercentage.append( round( 100.0*max(predictCountArray)/sum(predictCountArray), 2) )
        
for i in xrange(predictNo): # Find approx prediction near -1s
    approx=2
    if (predictCluster[i]==-1):
        predictCountArray2 = [0]*num_clusters
        for j in range(predict_NoProc[i]-approx, predict_NoProc[i]+approx+1):
            for k in xrange(num_clusters):
                predictCount = 0
                for n in xrange( len(xaxis_all[k]) ):
                    if (xaxis_all[k][n] == j):
                        predictCount = predictCount + 1
                predictCountArray2[k] = predictCountArray2[k] + predictCount
        if ( max(predictCountArray2) != 0):
            predictCluster[i] = predictCountArray2.index( max(predictCountArray2) )
            predictPercentage[i] = round( 100.0*max(predictCountArray2)/sum(predictCountArray2), 2)
            
print("Predict Output")
print(predictCluster)
print("Predict Percentage")
print(predictPercentage)

# Continue ploting
subplot=[]
for i in xrange(num_clusters):
    print(len(xaxis_all[i]))
    if ( (len(xaxis_all[i])!=0) and (len(yaxis_all[i])!=0)  ):
        #subplot.append( gen_plot(xaxis_all[i], yaxis_all[i], i) )     # First samples only
        subplot.append( gen_plot_eval(xaxis_all[i], yaxis_all[i], i, w_values[i], b_values[i]) )
    else:
        #subplot.append( gen_plot(0, 0, i) )     # First samples only
        subplot.append( gen_plot_eval([0,0], [0.0,0.0], i, w_values[i], b_values[i]) )

# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
subImage_temp=[]
for i in xrange(num_clusters):
    subImage_temp.append( tf.image.decode_png(subplot[i].getvalue(), channels=4) )

# Add the batch dimension
image = tf.expand_dims(image, 0)
subImage=[]
for i in xrange(num_clusters):
    subImage.append( tf.expand_dims(subImage_temp[i], 0) )

# Add image summary
summary_op = tf.summary.image("Overall Plot", image, max_outputs=3)
summary_op_sub = []
for i in xrange(num_clusters):
    title = "Each Cluster Plot"+str(i)
    summary_op_sub.append( tf.summary.image(title, subImage[i], max_outputs=3) )

# Run
summary = tf.summary.merge_all()
summary = sess.run(summary_op)
summary_sub=[]
for i in xrange(num_clusters):
    summary_sub.append( sess.run(summary_op_sub[i]) )

# Write summary
writer = tf.summary.FileWriter('/notebooks/logs', sess.graph)
writer.add_summary(summary)
for i in xrange(num_clusters):
    writer.add_summary(summary_sub[i])
    
end = time.time()
print("Elapsed time ",end-start, "seconds")
































# export TF_CPP_MIN_LOG_LEVEL=2 (Ignore warnings)

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io


def gen_plot(xaxis, yaxis):
    
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title('Clench Force vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Clench Force (kg)')
    plt.legend()
    plt.show()

    plt.plot(xaxis, yaxis)
    
    # Gradient Decent Optimization method
    w = tf.Variable(tf.random_uniform([1],0, 50), name = 'weight')
    b = tf.Variable(tf.random_normal([1],-10,10), name = 'constant')
    xaxis = np.asarray(xaxis,dtype=np.float32)
    
    print(xaxis)

    y = w * xaxis + b

    loss = tf.reduce_mean(tf.square(y - forceArray), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(0.015)
    train = optimizer.minimize(loss)

    # Initialisation
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    learningNumber = 101
    for step in xrange(learningNumber):
        sess.run(train)
        if ( (step % 10) == 0):
            print(step,"w = ",sess.run(w), "b = ", sess.run(b),  "Loss = ", sess.run(loss))
            
    yOut = sess.run(y)        
            
    plt.plot(xaxis, yOut,'ro')
    plt.xlabel('Time(s)')
    plt.ylabel('Clench Force (kg)')
    plt.legend()
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

filename_queue = tf.train.string_input_producer(["decreasingForce.txt"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32)]

firstCol, secondCol, thirdCol, fourthCol = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([firstCol, secondCol, thirdCol, fourthCol])

EMGArray=[]
forceArray=[]

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

coord.request_stop()
coord.join(threads)

sampleRange = 1000#24206

for i in range(sampleRange):
    rows, firstCol1, secondCol1, thirdCol1, fourthCol1 = sess.run([features, firstCol, secondCol, thirdCol, fourthCol])
    
    EMGArray = np.append(EMGArray, secondCol1)
    forceArray = np.append(forceArray, thirdCol1)

timeArray=[]
#timeArray.np.asarray(timeArray, dtype = np.float32)
timeRate = 0.002024 # time(s) per sample

# Place zeros in the time array with the same number of sample range
for i in range(sampleRange):
    timeArray.append(0)

# Time at index 0 is placed to be 0.002024 like this to avoid multiplying by 0 in the for loop when the for loop range is between 0 and the sample range
timeArray[0] = timeRate 
for i in range(1, sampleRange):
    timeArray[i] = timeRate * i

    
    
    
    
    
    
plot_buf = gen_plot(timeArray, forceArray)
plot=[]
plot.append(plot_buf)
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
image = tf.expand_dims(image, 0)

# Add image summary
summary_op = tf.summary.image("Control Lab 1", image, max_outputs=3)

# Run
summary = tf.summary.merge_all()
summary = sess.run(summary_op)

# Write summary
writer = tf.summary.FileWriter('/notebooks/logs', sess.graph)
writer.add_summary(summary)
    
##########################################################################################################




# export TF_CPP_MIN_LOG_LEVEL=2 (Ignore warnings)
# 25 Sep 2017
# Daniel Kho
# Seungmin Lee

# GDO number
# initial variable:  w&b
# Learning rate
# Sample size  

             #  ====> shape of regression

from __future__ import print_function
from decimal import *
from tensorflow.contrib.learn.python import SKCompat

import time
import io
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') #Generate Image without window appear
import matplotlib.pyplot as plt
import tensorflow as tf

start = time.time() # Record performance time

filename_queue = tf.train.string_input_producer(["BuzzData.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.

record_defaults = [tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  tf.constant([1], dtype = tf.float32),
                  ]

# Convert CSV records to tensors. Each column maps to one tensor.
thickness, R, conductivity = tf.decode_csv(value, record_defaults=record_defaults)
#features = tf.stack([L, R, k])

def gen_plot(xaxis, yaxis):

    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title('Number of Processes vs Time takes to finish Batches')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (days)')
    plt.legend()
    plt.show()
  
    plt.plot(xaxis, yaxis)

    
    # Gradient Decent Optimization method
    w = tf.Variable(tf.random_uniform([1],0, 50), name = 'weight')
    b = tf.Variable(tf.random_normal([1],-10,10), name='bias1')
    xaxis = np.asarray(xaxis,dtype=np.float32)
    y = w*(tf.reciprocal(xaxis)) + b


    loss = tf.reduce_mean(tf.square(y - yaxis), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(0.00015)
    train = optimizer.minimize(loss)
    
    # Initialisation
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    learningNumber = 101
    for step in xrange(learningNumber):
        sess.run(train)
        if ( (step % 10) == 0):
            print(step,"w = ",sess.run(w), "b = ", sess.run(b),  "Loss = ", sess.run(loss))
        
    yOut = sess.run(y)
    plt.plot(xaxis, yOut)
    
    plt.axis([0,max(xaxis)+0.01,0,max(yaxis)+0.01])
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

sess = tf.InteractiveSession()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

thicknessArray=[]
conductivityArray=[]
locationArray=[]

loop=19

for i in range(loop):
    if ( (i % 1) == 0):
        print(i,'read data over',loop)
    L, k = sess.run([thickness, conductivity])
    thicknessArray = np.append(thicknessArray, L)
    conductivityArray = np.append(conductivityArray, k)

# coord
coord.request_stop()
coord.join(threads)
        
plot_buf = gen_plot(thicknessArray, conductivityArray)
plot=[]
plot.append(plot_buf)    
    
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

# Add the batch dimension
image = tf.expand_dims(image, 0)

# Add image summary
summary_op = tf.summary.image("Overall Plot", image, max_outputs=3)

# Run
summary = tf.summary.merge_all()
summary = sess.run(summary_op)

# Write summary
writer = tf.summary.FileWriter('/notebooks/logs', sess.graph)
writer.add_summary(summary)
    
end = time.time()
print("Elapsed time ",end-start, "seconds")





























