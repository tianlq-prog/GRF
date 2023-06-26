################################## select only one core #################################
from logging import NOTSET
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


#################### Tool #######################
def heatmap(x, plotname, path = None, name = None):
    sns.set()
    ax = sns.heatmap(x)
    if path:
        plt.savefig(os.path.join(path, name))
    plt.title(plotname)
    plt.show()


####### Build Random Forest ########
def rf(X, y, n_estimators = None, max_depth = None, true_predictors = None, has_split=False, X_train=None,X_test=None):
    """"
    generate random forest using sklearn
    data: a combination of X and Y
    num_nodes: number of features
    """
    N = X.shape[0]
    num_nodes = X.shape[1]
    data= pd.DataFrame(np.concatenate((X, y.reshape([N, 1])), axis=1))
    
    feature = data.iloc[:,0:num_nodes]
    target = data.iloc[:,num_nodes]
    
    # standardscale of feature X
    # scale = StandardScaler()
    # scale.fit(feature)
    # feature = scale.transform(feature)

    if n_estimators == None:
        # find best n_estimators
        score_lt = []
        time_lt = []

        # grid search for proper n_estimators
        aa = 100
        bb = 1000
        for i in range(aa, bb, 100):
            start = time.time()
            rfc = RandomForestClassifier(n_estimators=i
                                        , random_state=90)
            end = time.time()
            time_lt.append(end-start)
            score = cross_val_score(rfc, feature, target, cv=10).mean()
            score_lt.append(score)
            print("###finding n_estimator with param:", i)
        score_max = max(score_lt)
        print('max_score：{}'.format(score_max),
            'number_of_trees：{}'.format(aa + score_lt.index(score_max) * 100))
        n_estimators = aa + score_lt.index(score_max) * 100

        # plot the curve of n_estimators
        x = np.arange(aa, bb, 100)
        
        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        l1 = plt.plot(x, score_lt, 'r-')
        ax1.set_ylabel('score')

        ax2 = ax1.twinx()  # this is the important function
        l2 = plt.plot(x, time_lt, 'b-')
        ax2.set_ylabel('time')
        plt.title("cv score and time for n_estimators")
        plt.show()
        
        rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=90)
        # grid search for proper max_depth
        param_grid = {'max_depth':np.arange(5,25,2)}
        GS = GridSearchCV(rfc, param_grid, cv=10)
        GS.fit(feature, target)

        best_param = GS.best_params_
        best_score = GS.best_score_
        print(best_param, best_score)
        max_depth = int(best_param["max_depth"])
        print("The best parameter for max_depth is: ", max_depth)
    

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    #clf.fit(feature, target)
    #predict_results = clf.predict(feature)
    #print(accuracy_score(predict_results, target))
    #conf_mat = confusion_matrix(target, predict)
    #print(conf_mat)
    #print(classification_report(target, predict))
    if has_split==False:
        X_train, X_test = split_train_test(X, y)
    l_test = X_test.shape[0]   # samle size in testing dataset

    clf.fit(X_train.iloc[:,0:num_nodes],X_train.iloc[:,num_nodes])
    acc = (clf.predict(X_test.iloc[:, 0:num_nodes]) == X_test.iloc[:,num_nodes]).sum()/l_test  # the acc on testing set
    imp = clf.feature_importances_

    roc = 0; prauc = 0
    if true_predictors is not None:
        roc = roc_auc_score(true_predictors, imp) # roc of graph-based random forest
        prauc = average_precision_score(true_predictors, imp)  # prauc of rf

    return(imp, acc, roc, prauc, n_estimators, max_depth)

"""
# run rf without searching for proper parameters
def run_rf(data, num_nodes,n_estimators, max_depth):
    feature = data.iloc[:,0:num_nodes]
    target = data.iloc[:,num_nodes]
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(feature, target)
    print("The forest using trainning data is set up")
    return(clf)
"""

def get_neighbor(i, thres, dist):
    """
    Get the neighboring node index set under certain criteria
    i: get the neighbor of node i
    thres: get neighbor in 'thres' hops 
    dist: distance matrix of the graph
    """
    if dist.shape[0] == dist.shape[1]:
        i_neighbor = np.where(dist[i,]<=thres)[0]
        i_neighbor = list(filter(lambda x: dist[i,x]>0, i_neighbor))
    else:
        cores = list(np.unique(dist[list(np.where(dist[:,0]==i)),1]))
        i_neighbor = cores
        for i in range((thres-1)):
            for c in cores:
                i_neighbor = i_neighbor + (list(np.unique(dist[list(np.where(dist[:,0]==c)),1])))
        i_neighbor = list(np.unique(i_neighbor))
        if i in i_neighbor:
            i_neighbor.remove(i)
    
    return(i_neighbor)

 


def split_train_test(X, y):
    """
    split the dataset into trainning and testing
    """
    # standardscale of feature X
    scale = StandardScaler()
    scale.fit(X)
    X = scale.transform(X)
    
    N = X.shape[0]  # sample size
    X_df = pd.DataFrame(np.concatenate((X, y.reshape([N, 1])), axis=1))

    # split into training and testing
    idx = [i for i in range(N)]
    random.shuffle(idx)
    idx_train = idx[0:int(N*0.7)]   # train set idx
    idx_test = idx[int(N*0.7):]     # test set idx
    l_train = len(idx_train)
    l_test = len(idx_test)
    X_train = X_df.iloc[idx_train,:]
    X_test = X_df.iloc[idx_test,:]
    return(X_train, X_test)

def graph_rf(X, y, iter_near_size, dist, true_predictors = None, has_split=False, X_train=None,X_test=None, pretrain_tree = 500, num_tree = 500):
    """ 
    iter_near_size: length of hops for potential splitting nodes
    dist: distance matrix of graph 
    true_preditors: works when simulation
    """
    num_nodes = X.shape[1]
    ######## use part of r code #########
    import rpy2.robjects as ro
    import rpy2.robjects as robjects
    import rpy2.robjects
    import rpy2.robjects.numpy2ri
    from collections import Counter
    rpy2.robjects.numpy2ri.activate()

    Xr = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    ro.r.assign("X", Xr)

    yr = ro.r.matrix(y, nrow=len(y), ncol=1)
    ro.r.assign("y", yr)
    
    nr = ro.r.matrix(pretrain_tree)
    ro.r.assign("ntree", nr)
 
    rpy2.robjects.numpy2ri.activate()
    robjects.r(
               '''
               f <- function(x,y, num_nodes){
               library(randomForest)
               
               y <- as.factor(1*(y>0))
               r <- randomForest(x,y,ntree = ntree, maxnode = 2, keepForest = T)
               z = r$forest$bestvar[1,]
               return(z)
               }
              '''
               )
    split_rf = robjects.r['f'](X,y,num_nodes)
    split_rf =np.array(split_rf).tolist()
    result = Counter(split_rf)  # the number of occur 
    count = [result[i] for i in range(num_nodes)]
    count = np.array(count)

    if has_split==False:
        X_train, X_test = split_train_test(X, y)
    l_test = X_test.shape[0]   # samle size in testing dataset

    non_zero_count = np.where(count!=0)[0]
    imp = np.zeros([num_nodes])
    y_pred = np.zeros([l_test,2])
    for i in non_zero_count:
        neighbor = get_neighbor(i, iter_near_size, dist) + [i]
        clf = RandomForestClassifier(n_estimators=int(count[i]*(num_tree/pretrain_tree))) # the number of trees = count[i]
        # train the model using training dataset
        clf.fit(X_train.iloc[:,neighbor], X_train.iloc[:,num_nodes])
        #imp[neighbor] = imp[neighbor] + clf.feature_importances_/clf.feature_importances_.sum()*count[i]
        imp[neighbor] = imp[neighbor] + clf.feature_importances_*count[i]
        y_pred = y_pred + clf.predict_proba(X_test.iloc[:,neighbor]) * count[i]
    # aggregated prediction y
    y_pred_res = (y_pred[:,1] > y_pred[:,0]).astype('uint8')
    # aggregated importance
    #imp = imp/count.sum()   # ensure the sum of importance equals 1
    #imp = (imp-np.min(imp))/(np.max(imp)-np.min(imp))  # scale the importance to 0~1
    acc = (y_pred_res == X_test.iloc[:,num_nodes]).sum()/l_test
    
    roc = 0; prauc = 0
    if true_predictors is not None:
        roc = roc_auc_score(true_predictors, imp) # roc of graph-based random forest
        prauc = average_precision_score(true_predictors, imp)  # prauc of rf
    
    return(imp, acc, roc, prauc)

def sub_graph(imp, dist, thres):
    import igraph
    idx_r = np.argsort(-imp)
    dist_new = dist[idx_r[0:thres],:][:,idx_r[0:thres]]
    g = igraph.Graph.Adjacency((dist_new == 1).tolist(), mode = "undirected")
    pp = igraph.plot(g)    
    l=[]
    for ll in g.components():
        l.append(len(ll))
    l = np.array(l)
    idx_l = np.where(l==np.max(l))[0][0]
    max_count = len(g.components()[idx_l])   # size of the max cluster
    idx1 = g.components()[idx_l]
    dist_sub = np.array(g.shortest_paths())
    dist_sub[np.where(abs(dist_sub)>1000)] = 0
    
    density = g.density()
    n_connected = len(g.components())
    mean_dist = np.mean(dist_sub)
    mean_dist_in_largest = np.mean(dist_sub[:, idx1][idx1,:])
    
    
    return(density, n_connected, max_count, mean_dist, mean_dist_in_largest, pp)


def get_neighbor_with_ls(i, thres, dist):
    cores = list(np.unique(dist[list(np.where(dist[:,0]==i)),1]))
    neighbor = cores
    for i in range((thres-1)):
        for c in cores:
            neighbor = neighbor + (list(np.unique(dist[list(np.where(dist[:,0]==c)),1])))
    neighbor = list(np.unique(neighbor))
    if i in neighbor:
        neighbor.remove(i)
    return(neighbor)