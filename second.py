#!/usr/bin/env python3

import math
from first_itds_modules import entropy
import numpy as np
from scipy.integrate import quad
from scipy import stats
from sklearn.neighbors import KernelDensity

############################ . 1 . ############################
print("--------------part1--------------")
def pmf_estimator(samples): # estimates the pmf of a descrete random variable
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/n

vector_length = 1500 
discrete_list = [10, 20, 30, 40]
discrete_probabilities = [0.4, 0.1, 0.2, 0.3] #real pmf
discrete_samples = np.random.choice(discrete_list, p=discrete_probabilities, size=vector_length) #generate sample set based on the pmf
#print("generated samples = ",discrete_samples)
discrete_list_estimated, pmf_estimated = pmf_estimator(discrete_samples) # estimate pmf based on the sample set
#print("estimated PMF = ",pmf_estimated)
ent = entropy(discrete_probabilities) # compute the entropy based on real pmf
#print ("entropy of real PMF =",ent)
est_ent = entropy(pmf_estimated) # compute the entropy based on estimated pmf
print("entropy of estimated PMF =",est_ent)
print ("difference",ent - est_ent) # difference

############################ . 2 . ############################
#differential entropy of original pdf

def integrand(x,pdf): #here pdf is gaussian
    return pdf(x)*math.log(pdf(x)) # math.log without base is ln

def differential_entropy(pdf,a,b): #a,b are min and max of support set
    return - quad(integrand, a, b, args=(pdf))[0]

#differential entropy of estimated pdf with samples
def estimated_integrand(x, probability_calculator, pdf): #here calculete the prob of x based on the kde object(estimated pdf) / probability_calculator simply pass x to pdf ang get the probability of x
    return probability_calculator(x,pdf)*math.log(probability_calculator(x,pdf))

def estimated_differential_entropy(probability_calculator, a, b, pdf): #pdf is the kde object(estimated pdf) and probability_calculator is the func that calculate prob of x based on estimated pdf
    return - quad(estimated_integrand, a, b, args=(probability_calculator, pdf))[0]

############################ . 3 . ############################
print("--------------part3--------------")
def sapmle_generator(mu,std): #to generate sample set from a gaussian pdf
    continuous_samples = np.random.normal(loc=mu, scale=std, size=1500) 
    return continuous_samples

def kde_estimator(continuous_samples): # to estimate the pdf(kde object)
    #set the parameters of kde
    cont_samp_len = len(continuous_samples)
    cont_samp_std = np.std(continuous_samples)
    optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
    bandwidthKDE = optimal_bandwidth  
    kernelFunction = 'gaussian'  
    #do the estimation
    kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(continuous_samples.reshape(-1, 1))
    return kde_object # the estimated pdf

def probability_calculator(x,kde_object): #estimate the probability of x based on estimated pdf(kde object)
    kde_LogDensity_estimate = kde_object.score_samples([[x]]) # get the result (logarithmic) (array)
    kde_estimate = np.exp(kde_LogDensity_estimate) # delete the logarithm (still an array)
    return kde_estimate[0] #return the probability of that specific x(input)


mu = 5 # mean
std = 10 # standard derivation
gaussian_pdf = stats.norm(mu, std).pdf # the real pdf
continuous_samples=sapmle_generator(5,10) #generate samples based on mu and std
print ("generated continuous samples = ",continuous_samples)
cont_samp_min = min(continuous_samples) # min of support set (for integral)
cont_samp_max = max(continuous_samples) # maximum of support set (for integral)
diff_ent = differential_entropy(gaussian_pdf,cont_samp_min,cont_samp_max) # real differential entropy
print ("differential entropy = ", diff_ent)
kde_object=kde_estimator(continuous_samples) #estimate pdf based on sample set
est_diff_ent = estimated_differential_entropy(probability_calculator,cont_samp_min,cont_samp_max,kde_object) #differential entropy based on estimated pdf
print ("estimated differential entropy = ", est_diff_ent)
print ("difference = ",est_diff_ent-diff_ent) # difference
#print(quad(gaussian_pdf, -9999, 9999))