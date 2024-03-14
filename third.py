#!/usr/bin/env python3

from sklearn import datasets
import numpy as np
import math
from first_itds_modules import entropy

################################ . 1 . ################################
print ( "--------------------part1--------------------")
def pmf_estimator(samples):
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/n

def pmf_multivariate(data_matrix):
    rows, columns = data_matrix.shape       #returns the number of rows and columns of the data_matrix
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)  #the parameter axis=0 allows to count the unique rows
    return unique_rows_array, pmf_vector/rows       #To obtain the probability, the count must be normalized to the total count of samples

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target
#print(data_matrix)
first_feature = 0
second_feature = 1
third_feature = 2
forth_feature = 3
discretized_data_matrix = (10*data_matrix[:, 0:4]).astype(int)

first_feature_samples = np.transpose((discretized_data_matrix[:, 0]))
second_feature_samples = np.transpose((discretized_data_matrix[:, 1]))
third_feature_samples = np.transpose((discretized_data_matrix[:, 2]))
forth_feature_samples = np.transpose((discretized_data_matrix[:, 3]))
full_feature_samples = discretized_data_matrix
#print(forth_feature_samples)
first_unq_values,first_pmf = pmf_estimator(first_feature_samples)
print("PMF of featuer 1 :\n")
print(first_pmf, "\n\n")
second_unq_values,second_pmf = pmf_estimator(second_feature_samples)
print("PMF of featuer 2 :\n")
print(second_pmf, "\n\n")
third_unq_values,third_pmf = pmf_estimator(third_feature_samples)
print("PMF of featuer 3 :\n")
print(third_pmf, "\n\n")
forth_unq_values,forth_pmf = pmf_estimator(forth_feature_samples)
print("PMF of featuer 4 :\n")
print(forth_pmf, "\n\n")
full_unq_rows,full_pmf = pmf_multivariate(full_feature_samples)
print("PMF of all featuers :\n")
print(full_pmf)
################################ . 2 . ################################
print ( "--------------------part2--------------------")
print("entropy of first feature = ",entropy(first_pmf))
print("entropy of second feature = ",entropy(second_pmf))
print("entropy of third feature = ",entropy(third_pmf))
print("entropy of forth feature = ",entropy(forth_pmf))
print("entropy of all features = ",entropy(full_pmf))

################################ . 3 . ################################
print ( "--------------------part3--------------------")
def mutualInfo(data_samples,f1,f2): # Mutual Information func 
    mutinf=0 # final result
    selected_data_samples = np.transpose(np.vstack((data_samples[:, f1], data_samples[:, f2]))) # all rows with just 2 columns for 2 wanted features
    full_unq_rows_np,jntpmf = pmf_multivariate(selected_data_samples) # full_unq_rows_np: all unique row with feature 1 and feature 2. jntpmf: joint PMF
    full_unq_rows=full_unq_rows_np.tolist() # convert np.array to python list to get profit from ".index()" method
    feature_samples1 = np.transpose((discretized_data_matrix[:, f1])) # all values of feature 1
    feature_samples2 = np.transpose((discretized_data_matrix[:, f2])) # all values of feature 2
    unq_values1,margpmf1 = pmf_estimator(feature_samples1) # unq_values1 : all unique values of feature 1. margpmf1: marginal pmf of feature 1
    unq_values2,margpmf2 = pmf_estimator(feature_samples2) # unq_values2 : all unique values of feature 2. margpmf2: marginal pmf of feature 2
    # compute mutual information
    for i in range(0, len(margpmf1)): # for outer sum
        innerSum=0
        margp1=margpmf1[i] # marginal probabiliti of each value of feature 1
        for j in range(0, len(margpmf2)): # for inner sum
            margp2=margpmf2[j] # marginal probabiliti of each value of feature 2
            if [unq_values1[i],unq_values2[j]] in full_unq_rows: # if for that specific value of feature 1 and feature2 we has the joint probability 
                value_index = full_unq_rows.index([unq_values1[i],unq_values2[j]]) # find that row
                jntp=jntpmf[value_index] # and take its value which is the joint probabinility of that two value
            else: # esle
                jntp=0 # the joint probability is 0
            if jntp !=0:
                innerSum+=(jntp)*math.log((jntp)/((margp1)*(margp2)),2) # calculate
        mutinf+=innerSum
    return mutinf

print("mutual info btw features 1 & 2 = ",mutualInfo(discretized_data_matrix,0,1))
print("mutual info btw features 2 & 3 = ",mutualInfo(discretized_data_matrix,1,2))
print("mutual info btw features 3 & 4 = ",mutualInfo(discretized_data_matrix,2,3))
print("mutual info btw features 1 & 3 = ",mutualInfo(discretized_data_matrix,0,2))
print("mutual info btw features 1 & 4 = ",mutualInfo(discretized_data_matrix,0,3))
print("mutual info btw features 2 & 4 = ",mutualInfo(discretized_data_matrix,1,3))