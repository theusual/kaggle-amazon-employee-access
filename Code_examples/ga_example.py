__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '04-07-2013'

import numpy as np
import pandas as pd
from numpy import array, hstack, mean, copy
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from pylab import find

from data_utils import OneHotEncoder, group_data
from Code_examples.genetic_algorithm import GeneticAlgorithm, Individual


random = np.random

### Parameters ### 
SEED = 421  
N = 4       # How many cross validation splits to use
N_JOBS = 4  # How many CPUs to use in cross_val_score

# See docstrings for GeneticAlgorithm to get an idea of what these values mean
ga_args = {'keep': 25,
           'population_size': 50,
           'selection_split': 0.8}
n_generations = 5

# This is used in the mutate helper function
mutation_rate = 1e-2

### Initialization ###
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = np.vstack((data.ix[:,1:-1], test_data.ix[:,1:-1]))
all_data = all_data - all_data.min(0) + 1

num_train = np.shape(data)[0]
num_test = np.shape(test_data)[0]

X = all_data[:num_train]
X_test = all_data[num_train:]
y = array(data.ACTION)

dp = group_data(all_data, degree=2)
dt = group_data(all_data, degree=3)

X = np.hstack((X, dp[:num_train], dt[:num_train]))
X_test = np.hstack((X_test, dp[num_train:], dt[num_train:]))

# Encode the data and keep it in a list for easy access during feature selection
OHs = [OneHotEncoder(X[:,[i]]) for i in range(X.shape[1])]
Xts = [o[0] for o in OHs]
Xts_test = [OneHotEncoder(X_test[:,[i]], OHs[i][1])[0] for i in range(X.shape[1])]

# Reassign the predict method for LogisticRegression so it returns probabilities
# This is primarily so it plays nice with cross_val_score
new_predict = lambda M, x: M.predict_proba(x)[:,1]
linear_model.LogisticRegression.predict = new_predict
model = linear_model.LogisticRegression()

# Initialize cross validation split
cv = cross_validation.StratifiedShuffleSplit(y, random_state=SEED, n_iter=N)

# Initialize the genetic algorithm
ga = GeneticAlgorithm(**ga_args)

# Two helper methods that take a binary index of features (eg [True, False, True])
# and returns those indices == True from Xts
getX = lambda gne: sparse.hstack([Xts[i] for i in find(gne)]).tocsr()
getX_test = lambda gne: sparse.hstack([Xts_test[i] for i in find(gne)]).tocsr()

### Create the Gene class as required by the GeneticAlgorithm ###
def mutate(gene, mutation_rate=1e-2):
    """
    Mutation method. Randomly flips the bit of a gene segment at a rate of
    mutation rate
    """
    sel = random.rand(len(gene)) <= mutation_rate
    gene[sel] = -gene[sel]
    return gene
    
class Gene(Individual):
    """
    Gene class used for feature selection
    Implements a fitness function and reproduce function as required by the
    Individual base class
    """
    def fitness(self):
        """
        The fitness of an group of features is equal to 1 - mean(cross val scores)
        """
        cv_args = {'X': getX(self.gene),
                   'y': y,
                   'score_func': metrics.auc_score,
                   'cv': cv,
                   'n_jobs': N_JOBS}
        cv_scores = cross_validation.cross_val_score(model, **cv_args)
        return 1 - mean(cv_scores)
        
    def reproduce(self, other, n_times=1, mutation_rate=mutation_rate):
        """
        Takes another Gene and randomly swaps the genetic material between
        this gene and other gene at some cut point n_times. Afterwords, it
        mutates the resulting genetic information and creates two new Gene 
        objects as children
        
        The high level description:
            copy the genes from self and other as g1, g2
            do n_times:
                randomly generate integer j along the length of the gene
                set g1 to g1[:j] + g2[j:]
                set g2 to g2[:j] + g1[j:]
            mutate g1 and g2
            return [Gene(g1), Gene(g2)]
        """
        lg = len(self.gene)
        g1 = copy(self.gene)
        g2 = copy(other.gene) 
        for i in xrange(n_times):
            j = random.randint(lg)
            g1 = hstack((g1[:j], g2[j:]))
            g2 = hstack((g2[:j], g1[j:]))
        g1 = mutate(g1, mutation_rate)
        g2 = mutate(g2, mutation_rate)
        return [Gene(g1), Gene(g2)]

# Do the feature selection
print "Creating the initial gene pool... be patient" 
n_genes = ga_args['population_size']
start_genes = (random.rand(n_genes, len(Xts)) > 0.5).astype(bool)
start_genes = sorted([Gene(g) for g in start_genes])

print 'Running the genetic algorithm...'
gene_pool = ga.evolve(start_genes, n_generations)
