import math
import numpy as np

class Network:
    neurons: list
    data: list

    alpha: float  # learning rate
    m: int
    n: int

    def __init__(self, m, n):
        self.T = 1  #Iteration No
        self.m = m  #grid x
        self.n = n  #grid y
        self.alpha = 0.1 #learning rate

        self.neurons = np.random.random((m*n,3)) # shape : grid x x grid y x 3
        self.radius = 0
        self.llambda = 0

    ####################################
    ### Calculating the Best Matching Unit (BMU)
    ####################################
    def Closest(self, point):
        minDist = float("inf")
        closest = []
        idx = 0
        for i, neuron in enumerate(self.neurons):
            dist = ((neuron[2] - point[0]) ** 2 + (neuron[1] - point[1]) ** 2 + (neuron[0] - point[2]) ** 2) ** 0.5
            if dist < minDist:
                minDist = dist
                closest = neuron
                idx = i
        return idx, closest


    ####################################
    ### Travel neighbours inside the radius scope
    ### Instead of calculating specific distance,
    ### this function treats the diagonal neighbours as distance 1
    ####################################
    def WithinRadius(self, bmuX, bmuY, bmuNode, radius):
        nodes = []

        for x in range(int(bmuX - radius), int(bmuX + radius + 1)):
            for y in range(int(bmuY - radius), int(bmuY + radius + 1)):
                if x < 0 or x >= self.m or y < 0 or y >= self.m:
                    pass
                else:
                    nodes.append(x*self.m + y)
        return nodes

        """# Wrong Approach
        for i, neuron in enumerate(self.neurons):
            dist = ((neuron[2] - bmuNode[2]) ** 2 + (neuron[1] - bmuNode[1]) ** 2 + (neuron[0] - bmuNode[0]) ** 2) ** 0.5
            if dist < radius:
                nodes.append([i,neuron])
        return nodes"""

    def getXY(self, i):
        x = math.floor(i/self.m)
        y = i - self.m*x - 1
        return x,y

    ####################################
    ### Iterate all input nodes
    ####################################
    def Update(self, t, T):

        # Calculate Radius
        radius_init = max(self.m, self.n) / 2
        llambda = self.T / math.log(radius_init)
        self.radius = radius_init * math.exp(-1*t*llambda)
        # Calculate Learning Rate
        alpha = self.alpha * math.exp(-1 * t /llambda)

        for point in self.data:

            closest = self.Closest(point)
            i, node = closest
            x,y = self.getXY(i)

            # lst - All Neighbour Nodes in Radius
            if(self.radius > 0):
                lst = self.WithinRadius(x,y, node, self.radius)
            else:
                lst = [i]

            # Update all Neighbour Nodes' Weight
            for eachnode in lst:

                # node : Closest node
                # point : each node in radius
                ### node_ = eachnode[1]
                node_idx = eachnode

                di = ((self.neurons[node_idx][2] - node[2]) ** 2 + (self.neurons[node_idx][1] - node[1]) ** 2 + (self.neurons[node_idx][0] - node[0]) ** 2) ** 0.5

                denominator =(2*(self.radius**2))
                if denominator <= 0:
                    denominator = np.finfo(float).eps # prevent /0 by epsilon smoothing

                # Theta - Influence of the node
                theta = math.exp(-1*di**2/denominator)

                #update
                self.neurons[node_idx][2] = self.neurons[node_idx][2] + alpha * theta * (point[0] - self.neurons[node_idx][2])
                self.neurons[node_idx][1] = self.neurons[node_idx][1] + alpha * theta * (point[1] - self.neurons[node_idx][1])
                self.neurons[node_idx][0] = self.neurons[node_idx][0] + alpha * theta * (point[2] - self.neurons[node_idx][0])


    ####################################
    ### Train - Iterate each epoch
    ####################################
    def Train(self, T):
        self.T = T
        for epoch in range(T):
            self.Update(epoch, T)
            #print("epoch: ", epoch)