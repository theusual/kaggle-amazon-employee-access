"""
Original code by Miroslaw Horbal
Modified by Luca Massaroon

changes:

1) An option to start from a set of predictors
2) An option to immediately compute the final solution without further feature selection
3) Multiprocessor, automatically choses the best number of jobs for maximum computation speed
4) Introduced a small_change variable fixing the minimum change in a model to be acceptable in order to avoid overfitting
5) Features with less than 3 cases are clustered together in a rare group
6) After inserting a new variable it checks if the already present variabiles are still meaningful to be enclosed in the model (pruning)
7) As for as cross validation, it fixes test_size=.15 and it uses median not mean to average the cross validation results
8) It prints out only significative model changes, history of the model, best C value
9) Randomized start, final CV score saved with the filename
"""

from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
import multiprocessing
import random

def group_data(data, degree=3, cutoff = 1, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indexes]])
    for z in range(len(new_data)):
        counts = dict()
        useful = dict()
        for item in new_data[z]:
            if item in counts:
                counts[item] += 1
                if counts[item] > cutoff:
                    useful[item] = 1
            else:
                counts[item] = 1
        for j in range(len(new_data[z])):
            if not new_data[z][j] in useful:
                new_data[z][j] = 0
    return np.array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indexes.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def evaluation(y_true, y_pred):
    return metrics.auc_score(y_true, y_pred[:,1])

# This loop essentially from Paul's starter code

def validation_worker(args):
    X, y, model, j, SEED = args
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.15, 
                                       random_state = j*SEED)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:,1]
    auc = metrics.auc_score(y_cv, preds)
    return auc  

def cv_loop(X, y, model, N, pool, SEED): 
    instructions = [(X, y, model, i, SEED) for i in range(N)]
    pooled_auc = pool.map(validation_worker, instructions)        
    return np.median(np.array(pooled_auc))
    
def main(train='train.csv', test='test2.csv', submit='submission', initial_solution=list(), finalize=False):    
    global SEED
    global N_JOBS
    
    SEED = random.randint(0,2500)
    print "Random seed is:",SEED
    N_JOBS = max(1,min(10, multiprocessing.cpu_count()-1))
    N = 10 # Cross validation parameter
    small_change = 0.00005 # set the smallest acceptable change in the model performance
    
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))

    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp1 = group_data(all_data, degree=2, cutoff=2) 
    dt1 = group_data(all_data, degree=3, cutoff=2)
    dz1 = group_data(all_data, degree=4, cutoff=2)
    dp2 = group_data(all_data, degree=5, cutoff=2)
    dp3 = group_data(all_data, degree=6, cutoff=2) 

    y = np.array(train_data.ACTION)
    X = all_data[:num_train]
    X_2  = dp1[:num_train]
    X_3  = dt1[:num_train]
    X_4  = dz1[:num_train]
    X_5  = dp2[:num_train]
    X_6  = dp3[:num_train]

    X_test = all_data[num_train:]
    X_test_2 = dp1[num_train:]
    X_test_3 = dt1[num_train:]
    X_test_4 = dz1[num_train:]
    X_test_5 = dp2[num_train:]
    X_test_6 = dp3[num_train:]

    X_train_all = np.hstack((X, X_2, X_3, X_4, X_5, X_6))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6))
    num_features = X_train_all.shape[1]
    
    model = linear_model.LogisticRegression()
    
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    
    print "Performing greedy feature selection..."
    
    p = multiprocessing.Pool(N_JOBS)
    good_features = set(initial_solution)
    
    if len(good_features) == 0:
        score_hist = []
    else:
        feats = list(good_features)
        Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
        score_hist = [(cv_loop(Xt, y, model, N, p, SEED),-1)] 
    
    if finalize:
        good_features = initial_solution
        print "Final features: %s" % sorted(list(good_features))
    else:
        # Greedy feature selection loop
        maxscore = 0
        while len(score_hist) < 2 or (score_hist[-1][0] - score_hist[-2][0]) > small_change:
            scores = []
            for f in range(len(Xts)):
                if f not in good_features:
                    feats = list(good_features) + [f]
                    Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                    score = cv_loop(Xt, y, model, N, p, SEED)
                    scores.append((score, f))
                    if score > maxscore:
                        print "Feature: %i Mean AUC: %f" % (f, score)
                        maxscore = score
            good_features.add(sorted(scores)[-1][1])
            score_hist.append(sorted(scores)[-1])
            print "Current features: %s" % sorted(list(good_features))
            if len(good_features) > 2 :
                print "Pruning..."
                to_be_removed = None 
                gain = 0
                baseline = score_hist[-1][0]
                for f,target in enumerate(good_features):
                    feats = list(good_features)
                    feats.remove(target)
                    Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                    score = cv_loop(Xt, y, model, N, p, SEED)
                    if score > baseline and (score-baseline) > gain:
                        gain = score-baseline
                        to_be_removed = target
                        print "Removing %i will improve AUC by %f" % (target, gain)
                if to_be_removed:
                    good_features.discard(to_be_removed)
                    score_hist.append((baseline+gain,target*-1))
                    print "Current features: %s" % sorted(list(good_features))
        
        # Remove last added feature from good_features
        # good_features.remove(score_hist[-1][1])
        good_features = sorted(list(good_features))
        print "Selected features %s" % good_features
        print "History:",score_hist
      
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N, p, SEED)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    bestscore = sorted(score_hist)[-1][0]
    print "Best C value: %f" % (bestC)   
    
    model.C = bestC # Specifies the best strength of the regularization. 
    
    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit+str(bestscore)+'.csv', preds)
    p.terminate() 
    
if __name__ == "__main__":
    args = { 'train':  'train.csv',
             'test':   'test.csv',
             'submit': 'logistic_regression_pred_submission_', 
             'initial_solution': [],
             'finalize': False
             }
    main(**args)
    
