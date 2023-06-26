import numpy as np
import os
import pandas as pd
import random 
import igraph as ig
import seaborn as sns
import datetime
import time
import matplotlib.pyplot as plt
from random import seed
from random import randrange

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, recall_score, f1_score

######################### Simulation ##############################
def simulation_data(num_core, N, feature_per_core, num_nodes, true_predictors, link, ba_m = 2, ba_power = 0.5, cov_base = 0.8, rate = 0.5, low = 0.1, up = 0.3, weight_a = 1, want_plot = False, epsilon=None):
    """
    Function: simulate dataset (X (N*num_nodes) and Y(N*1)), X is generated based on a BA graph and Y is generated as a 0-1 type data with link function.
              Specifically, first choose some 1 or 2 core node with high degree ranking, and then randomly select at most 'feature_per_core' adjacent nodes.
              The true features selection will expand untill it contain 'true_predictors' nodes.
    num_core: 1 core or 2 cores
    N: sample size
    link: link function with "logistic" and "abs"
    """
    
    max_dist = 0   # max distance of pairwise nodes when constructing two cores simulation dataset
    while ((max_dist < 4 and num_core == 2) or (num_core == 1) or (num_core>=3)):
        print("1")
        graph = ig.Graph.Barabasi(n=num_nodes, m = ba_m, power=ba_power, implementation='psumtree', outpref=False,
                            directed = False, zero_appeal= 0)
        dg = np.array(graph.degree())  
        dist = np.array(graph.shortest_paths())

        covariance = np.power(cov_base, dist)
    
        # select core nodes using certain criteria of differential information
        uu = np.argsort(-dg) # sorted index
        sorted_dg = dg[uu]   # sorted degree

        # averaged degree by three steps 
        avg3 = (sorted_dg[0:num_nodes-2] + sorted_dg[1:num_nodes-1] + sorted_dg[2:num_nodes])/3
        judge = (sorted_dg[2:num_nodes]-avg3[0:num_nodes-2])/avg3[0:num_nodes-2]

        # potential core nodes 
        index_core = np.where(-judge < 0.01)

        if index_core[0][0] <= 2:
            cores = uu[0:3]
        else:
            cores = uu[0:index_core[0][0]]     

        if num_core == 1:
            cores = np.array(random.sample(cores.tolist(),1)).tolist()  # randomly choose 1 core
            break
        elif num_core ==2:
            dist_cores = dist[list(cores),:][:, list(cores)]
            max_dist = dist_cores.max()
            c = []
            for ii in cores:
                for jj in cores:
                    if dist[ii, jj] == max_dist:
                        cores = np.array([ii,jj])
            cores = list(cores)
        else:
            cores = np.array(random.sample(uu[0:50].tolist(),num_core)).tolist()
            break
    print("FIND CORE")
    # select true predictors by choosing alternatively including neighboring nodes
    pool = cores
    predictors = cores
    uu = cores
    k = 0
    while len(predictors) < true_predictors:
        if k>0:
            feature_per_core = int(feature_per_core*0.8)+1
        uu = []
        k += 1
        
        for i in cores:
            i_neighbor = (np.where(dist[i,:]==1)[0]).tolist()
            pool = pool + i_neighbor
            u = i_neighbor[0:min(feature_per_core,len(i_neighbor))]
            uu = uu + u
            predictors = list(predictors)
            predictors = predictors + u
        pool = (np.unique(pool)).tolist()
        predictors = (np.unique(predictors)).tolist()
        if (len(predictors)) < (len(cores) * feature_per_core):
            add_num = (len(cores) * feature_per_core) - len(predictors)
            z = []
            for j in pool:
                if j not in predictors:
                    z.append(j)
            add_sample = random.sample(z, min(add_num, len(z)))
            predictors = predictors + add_sample
        predictors = np.unique(predictors)
        cores = uu
    predictors = predictors[0:true_predictors]
    
    print("find true predictors")
    # generate feature matrix X
    mean = np.zeros(num_nodes)
    X = np.random.multivariate_normal(mean, covariance, N)

    # genereate coefficient beta for Y = link_function(X * beta)
    beta0 = np.array([random.uniform(low, up) for i in range(len(predictors)+1)])
    negative_index = np.random.randint(len(predictors)+1, size=int((len(predictors)+1)*rate))
    beta0[negative_index] = beta0[negative_index] * (-1)

    Z = np.dot(np.column_stack((np.ones((N,1)),X[:,predictors])), beta0)
    
    # add some noise to X*beta
    if epsilon:
        noise = np.random.normal(0, epsilon, N)
        Z += noise
    
    # generate y with different link function
    if link == 'logistic':
        y = 1 / (weight_a * (1 + np.exp(- Z + np.median(Z))))
        y = (y >= 0.5/weight_a) + 0

    elif link == 'abs':
        y = abs(Z - np.median(Z))
        y = (y >= np.median(y)) + 0
    
    if want_plot == True:
        graph.vs["size"] = 2
        graph.vs["color"] = "grey"
        graph.vs[list(predictors)]["color"] ="red"
        graph.vs[list(predictors)]["size"] = 4

        graph.es["color"] = ["red" if (edge.source in predictors and \
                                edge.target in predictors) else "black" \
                        for edge in graph.es]

        graph.es["width"] = [1.5 if (edge.source in predictors and \
                                edge.target in predictors) else 0.3 \
                        for edge in graph.es]

        # 用Igraph内置函数绘图
        layout = graph.layout('fr')   # fr drl kk random 
        ig_plot = ig.plot(graph,layout=layout)

    return (X, y, covariance, beta0, predictors, dist, graph, ig_plot)
