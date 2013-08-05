__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '04-07-2013'

from numpy import array, inf, mean, all, ceil
from random import sample

import numpy.random as random 
from itertools import combinations

def random_pairs(C, n):
    """
    Generate n random distinct pairs of items from the indexed collection C
    
    Inputs:
        C: A collection with length and indexing (list-like)
        n: the number of pairs to generate (int >= 0)
    """
    lenC = len(C)
    for i in xrange(n):
        p = random.randint(lenC)
        q = random.randint(lenC)
        if p != q:
            yield (C[p], C[q])
        
class GeneticAlgorithm(object):
    """
    This class runs a genetic algorithm (GA) for maintaining a population of 
    individuals, selecting fit individuals for reproduction, creating new
    generations, and maintaining the fittest individual overall
    """
    def __init__(self, population_size=100, keep=25, selection_split=.8):
        """
        Inputs:
            population_size: The maximum size of the gene pool at any given
                             generation. Must be an int > 0
            keep: The number of individuals to keep for reproduction at each
                  generation. Must be an int >= 0
            slection_split: The fraction of the best individuals to keep 
                            when selecting for reproduction. 
                            Must be a float on the interval [0, 1]
                            0.8 selection split means:
                                80% of the kept individuals will be the fittest
                                in the gene pool
                                20% of the kept individuals will be randomly 
                                selected from the remaining population   
                            For keep = 10, selection_split = 0.8:
                            the top 8 fittest individuals are selected for
                            reproduction and 2 others are randomly selected
        """
        self.keep = keep
        self.selection_split = selection_split
        self.population_size = population_size
        self.best = NoGeneIndividual()
        self.total_gen = 0
    
    def __repr__(self):
        return "GeneticAlgorithm(" + \
                "keep=%i, " % (self.keep) + \
                "population_size=%i, " % (self.population_size) + \
                "selection_split=%0.2f)" % (self.selection_split) 
    
    def step(self, gene_pool):
        """
        Take a one-generation step forward in the GA. The resulting population
        will contain '.keep' individuals from the previous generation and
        approximately '.population_size - .keep' new individuals
        
        Inputs:
            gene_pool: A sorted list of Individual-like objects in the current 
                       gene pool. If the list is not sorted then there are no
                       guaruntees that the resulting generation will keep the
                       fittest individuals
        
        Returns:
            children: A sorted list of Individual-like object of the next 
                      generation. Exactly '.keep' of these individuals will be
                      from the previous generation. Since a set is used to
                      maintain the population internally, any two individuals
                      that have the same hash will be considered identical 
                      and only one will be kept. 
                       
        """
        keepN = int(ceil(self.keep * self.selection_split))
        upper_tier = gene_pool[:keepN]
        lower_tier = sample(gene_pool[keepN:], self.keep - keepN)
        gene_pool = lower_tier + upper_tier
        children = set([])
        Nchild = self.population_size - self.keep
        for gene1, gene2 in random_pairs(gene_pool, Nchild):
            children = children.union(gene1.reproduce(gene2))
        children = children.union(gene_pool)
        children = sorted(list(children))
        if children[0] < self.best:
            self.best = children[0]
        return children
    
    def evolve(self, gene_pool, n_gen=1):
        """
        Take n_gen steps of the GA on a gene_pool and return a history of n_gen
        generations
        
        Inputs:
            gene_pool: The same requirements as for the step function. See step
                       for more details
            n_gen: The number of generations to step forward with the GA
        
        Returns:
            A list of the n_gen gene_pools returned during the course of evolution
        """
        msg = lambda j, m, b: "Generation %i fitness: %f - best: %f" % (j, m, b)
        history = []
        for i in range(n_gen):
            children = self.step(gene_pool)
            history.append(children)
            self.total_gen += 1
            gene_pool = children
            mean_score = mean([c.fitness_score for c in children])
            print msg(self.total_gen, mean_score, self.best.fitness_score)
        return history

class IndividualMixin(object):
    """
    Base mixin for all individuals used in the genetic algorithm
    
    Provides functionality for printing, comparing individuals by fitness, 
    computing a hash value, testing equality, and accessing genetic information
    
    To use this mixin the user must provide an __init__ method that sets 
    the following attributes:
        _gene: an immutable, hashable tuple-like object 
        fitness_score: a numeric value that assigns some score to the individual
                       a lower score implies a fitter individual 
    """
    def __repr__(self):
        return "Fitness: %f" % self.fitness_score
        
    def __cmp__(self, other):
        """
        Comparison between two individuals for maintaining an ordering is done
        on the fitness_score level
        """
        return cmp(self.fitness_score, other.fitness_score)
    
    def __eq__(self, other):
        """
        Testing two individuals are equal is done on the gene level
        
        Note that self.gene returns a numpy.array 
        """
        return all(self.gene == other.gene)
    
    def __hash__(self):
        """
        The hash of an individual is equal to the hash of the genetic information
        
        The hash is important for determining uniqueness when maintaining a set
        of individuals in a genetic pool
        """
        return self._gene.__hash__()
    
    @property
    def gene(self):
        """
        Accessor method for the _gene attribute returning the gene as
        an numpy.array
        """
        return array(self._gene)

class NoGeneIndividual(IndividualMixin):
    """
    Atomic unit describing an individual with no genetic information and
    infinite fitness (high fitness is bad, low fitness is good)
    
    This class only has one use, as a placeholder in the GeneticAlgorithm
    class
    """
    def __init__(self):
        self._gene = None
        self.fitness_score = inf
        
class Individual(IndividualMixin):
    """
    Base class for an individual with genetic information and some fitness
    
    This is meant to be inherited or overrided for user defined individuals 
    as it cannot be used without a 'reproduce' function 
    """
    def __init__(self, gene, fitness_score=None):
        """
        Initializes the _gene and fitness_score attributes
        
        Inputs: 
            gene: a tuple-like object whose individual components can be hashed
            fitness_score: a numeric value assigning fitness, if fitness_score
                           is omitted then the individuals fitness() function
                           is called to compute this value
        """
        self._gene = tuple(gene)
        if fitness_score is None:
            self.fitness_score = self.fitness()
        else:
            self.fitness_score = fitness_score

    def fitness(self):
        """
        Not implemented. User defined implementations should return a
        numeric value for fitness and shouldn't take any other inputs. 
        A lower value for fitness indicates a stronger individual 
        """
        raise NotImplementedError('fitness not implemented')
    
    def reproduce(self, other):
        """
        Not implemented. User defined implementations should take another
        individual as input and combine the genetic information between 
        the two to produce offspring. This function should return an interable
        of offspring, even if one offspring is produced 
        """
        raise NotImplementedError('reproduce not implemented')
