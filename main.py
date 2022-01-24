from network import *
from network_simple import *
from network_fixed import *
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt



# 1d to 2d, RGB to a tuple
def gridReshape(neurons,size):
    res = []
    list = np.array(neurons).reshape(size, size, 3)
    for set in list:
        row = []
        for col in set:
            row.append(tuple(col))
        res.append(row)
    return res

####################################
### 0. Make 20x3 Dataset
####################################

data = np.random.random((20,3))

list = []
for row in data:
    list.append(tuple(row))

plt.title("Input Data")
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
#data = [[(R, G, B), (R,G,B)], [(R, G, B), (R,G,B)]]
plt.imshow([list])
plt.show()
plt.imsave('./test.png',[list])


####################################
### 1. Implement a Kohonen network
### Please refer network.py
### Network(grid_x,grid_y)
####################################
network = Network(10,10)
network.data = data

# Check network plot before Training
res = gridReshape(network.neurons, 10) #100x3 to 10x10 pixels

plt.title("Network - Pre Training")
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.imshow(res)
fig = plt.figure()
plt.show()



####################################
### 2. Train a 10x10 network over 100 iterations
### Training data is a random set of 20 colours
### How long does this take?
### What does the map look like? (You will need to translate the weights of each node in the map to pixel data)
### What does this look like if you perform 200 & 500 iterations?
### You should end up with something that looks like below after 500 iterations:
####################################

#Train the 10x10 grid with 100 iteration
start = timer()
network.Train(100)
end = timer()
time = round((end - start),2)

res = gridReshape(network.neurons, 10)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("Post Training 100 Iteration, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()



#Train the 10x10 grid with 200 iteration
start = timer()
network.Train(200)
end = timer()
time = round((end - start),2)

res = gridReshape(network.neurons, 10)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("Post Training 200 Iteration, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()

#Train the 10x10 grid with 500 iteration
start = timer()
network.Train(500)
end = timer()
time = round((end - start),2)

res = gridReshape(network.neurons, 10)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("Post Training 500 Iteration, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()


####################################
### Try Simpler & Fixed Neighbour Network
####################################

# Simpler Network
# Train the 10x10 grid with 500 iteration
network_simple = Network_simple(10,10)
network_simple.data = data

start = timer()
network_simple.Train(500)
end = timer()
time = round((end - start),2)

res = gridReshape(network_simple.neurons, 10)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("500 itr, simpler network, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()

# Fixed Radius Network
# Train the 10x10 grid with 500 iteration
network_fixed = Network_fixed(10,10)
network_fixed.data = data

start = timer()
network_fixed.Train(500)
end = timer()
time = round((end - start),2)

res = gridReshape(network_fixed.neurons, 10)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("500 itr, fixed radius, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()


####################################
### 3. Train a 100x100 network over 1000 iterations
### This network will likely be significantly slower to train
### What could you do to improve performance?
### What does the network look like after 1000 iterations?
####################################
network = Network(100,100)
network.data = data

# Pre Training
res = gridReshape(network.neurons, 100)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("Pre Training")
plt.imshow(res, interpolation='nearest')
plt.show()

start = timer()
network.Train(1000)
end = timer()
time = round((end - start),2)

# Post Training
res = gridReshape(network.neurons, 100)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.title("Post Training 1000 Iteration, time= " + str(time))
plt.imshow(res, interpolation='nearest')
plt.show()
