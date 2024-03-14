#!/usr/bin/env python3

from sklearn import datasets
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats

#################################### . 1 . ####################################

def kde_estimator_multivar(x,continuous_samples): # to estimate the pdf
    #set the parameters of kde
    cont_samp_len = len(continuous_samples)
    cont_samp_std = np.std(continuous_samples)
    optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
    bandwidthKDE = optimal_bandwidth  
    kernelFunction = 'gaussian'  
    #do the estimation
    kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(continuous_samples)
    #estimate the probability of x
    kde_LogDensity_estimate = kde_object.score_samples([x]) # get the result (logarithmic) (array)
    kde_estimate = np.exp(kde_LogDensity_estimate) # delete the logarithm (still an array)
    return kde_estimate[0] #return the probability of that specific x(input)

def pmf_estimator(samples):
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/n

def bayes_classifier(train_set,train_class_lbl,test_set):
    unique_classes,class_lbl_pmf = pmf_estimator(train_class_lbl) # pmf of class lables and unique class lables
    tr_sets = [None] * len(unique_classes) 
    # divide training set to 3 part (3 = number of classes) based on thair classes and put them in tr_sets[][]
    for i in range(0,len(unique_classes)):
        tr_set_temp=[]
        for j in range(0,len(train_set)):
            if train_class_lbl[j]==i:
                tr_set_temp.append(train_set[j])
        tr_sets[i]=tr_set_temp
    
    estimated_class_vector=[]
    for i in range(0,len(test_set)): # for each row of test set (75 times)
        m=[] # probability of being that row in each class
        for j in range(0,len(unique_classes)): # for each class (3 times) (likelihood)
            m.append(kde_estimator_multivar(test_set[i],tr_sets[j])*class_lbl_pmf[j]) # compute for each class (posterior probability=max(likelihood*prior probability))
        estimated_c=m.index(max(m)) # find the max 
        #print(estimated_c)
        estimated_class_vector.append(estimated_c) # set that class as the estimated class for that row
    #print (estimated_class_vector)
    return estimated_class_vector

#################################### . 2 . ####################################

def kde_estimator_univar(x,continuous_samples): # to estimate the pdf
    #set the parameters of kde
    cont_samp_len = len(continuous_samples)
    cont_samp_std = np.std(continuous_samples)
    optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
    bandwidthKDE = optimal_bandwidth  
    kernelFunction = 'gaussian'  
    #do the estimation
    kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(continuous_samples.reshape(-1, 1))
    #estimate the probability of x
    kde_LogDensity_estimate = kde_object.score_samples([[x]]) # get the result (logarithmic) (array)
    kde_estimate = np.exp(kde_LogDensity_estimate) # delete the logarithm (still an array)
    return kde_estimate[0] #return the probability of that specific x(input)

def bayes_naive_classifier(train_set,train_class_lbl,test_set):
    unique_classes,class_lbl_pmf = pmf_estimator(train_class_lbl)
    tr_sets = [None] * len(unique_classes)
    # divide training set to 3 part (3 = number of classes) based on thair classes and put them in tr_sets[][]
    for i in range(0,len(unique_classes)):
        tr_set_temp=[]
        for j in range(0,len(train_set)):
            if train_class_lbl[j]==i:
                tr_set_temp.append(train_set[j])
        tr_sets[i]=tr_set_temp
   

    tr_sets_univar = [None] * len(unique_classes)
    #convert the multivariate to multiple univariate and put them to tr_sets_univar[][]
    for i in range(0,len(unique_classes)):
        tr_set_temp=np.transpose(np.array(tr_sets[i]))
        tr_sets_univar[i]=tr_set_temp
        #print (tr_sets_univar[i])

    estimated_class_vector= [None] * len(test_set)

    for i in range(0,len(test_set)): # for each row (75 times)
        classes_prob=[None] * len(tr_sets_univar) # likelihood (multivariate conditional probability)
        for k in range(0,len(tr_sets_univar)): # for each class (3 times)
            features_prob= [None] * len(test_set[i]) # univariate conditional probability
            for j in range(0,len(test_set[i])): # for each feature (4 times)
                features_prob[j]=kde_estimator_univar(test_set[i][j],tr_sets_univar[k][j]) # prob of each feature in each class 
            for j in range(0,len(features_prob)): # multiply all the probabilities to gether (for the product in the formula)
                if j==0:
                    classes_prob[k]=features_prob[0]
                else:
                    classes_prob[k] *= features_prob[j]
            #print(classes_prob[k])
        for k in range(0,len(tr_sets_univar)):
            classes_prob[k] *= class_lbl_pmf[k] # multiply the resault of product with probability of the realted class (=posterior probability)
        estimated_c=classes_prob.index(max(classes_prob)) # find the maximum
        estimated_class_vector[i]=estimated_c # set that class as the estimated class for that row
    
    return estimated_class_vector

#################################### . 3 . ####################################

def bayes_naive_classifier_gaussian(train_set,train_class_lbl,test_set):
    unique_classes,class_lbl_pmf = pmf_estimator(train_class_lbl)
    mu=[]
    std=[]
    tr_sets = [None] * len(unique_classes)
    # divide training set to 3 part (3 = nomber of classes) based on thair classes and put them in tr_sets[][]
    for i in range(0,len(unique_classes)):
        tr_set_temp=[]
        for j in range(0,len(train_set)):
            if train_class_lbl[j]==i:
                tr_set_temp.append(train_set[j])
        tr_sets[i]=tr_set_temp
    
    #estimate mean and standard derivation of each feature in each class
    for i in range(0,len(unique_classes)):
        mu_temp = [None] * len(train_set[0])
        std_temp = [None] * len(train_set[0])
        for j in range(0,len(tr_sets[i][0])):
            tr_set_temp=np.array(tr_sets[i])
            mu_temp[j], std_temp[j]= tr_set_temp[:, j].mean(), tr_set_temp[:, j].std()
        mu.append(mu_temp)
        std.append(std_temp)
    #print (mu)
    #print(std)
    estimated_class_vector= [None] * len(test_set)

    for i in range(0,len(test_set)): # for each row (75 times)
        classes_prob=[None] * len(unique_classes) # likelihood (multivariate conditional probability)
        for k in range(0,len(unique_classes)): # for each class (3 times)
            features_prob= [None] * len(test_set[i]) # univariate conditional probability
            for j in range(0,len(test_set[i])): # for each feature (4 times)
                mutemp = mu[k][j] # put its mean to mutemp
                stdtemp = std[k][j] # put its standard derivation to stdtemp
                gaussian_pdf = stats.norm(mutemp, stdtemp).pdf # create the gaussian distribution based on given mu and std
                features_prob[j]=gaussian_pdf(test_set[i][j]) # prob of each feature for each class
            for j in range(0,len(features_prob)):  # multiply all the probabilities to gether (for the product in the formula)
                if j==0:
                    classes_prob[k]=features_prob[0]
                else:
                    classes_prob[k] *= features_prob[j]
        for k in range(0,len(unique_classes)):
            classes_prob[k] *= class_lbl_pmf[k] # multiply the resault of product with probability of the realted class (for posterior probability)
        estimated_c=classes_prob.index(max(classes_prob)) # find the maximum
        estimated_class_vector[i]=estimated_c # set that class as the estimated class for that row
    
    return estimated_class_vector

#################################### . 4 . ####################################

def preparation(): #create training & test, datasets & class lables
    iris = datasets.load_iris()
    data_matrix, class_vector = iris.data, iris.target

    first_class_rows=[]
    first_class_lbl=[]
    second_class_rows=[]
    second_class_lbl=[]
    third_class_rows=[]
    third_class_lbl=[]
    #separate data_matrix class_vector base on different classes
    for i in range (0,len(class_vector)):
        if class_vector[i]==0:
            first_class_rows.append(data_matrix[i])
            first_class_lbl.append(class_vector[i])
        elif class_vector[i]==1:
            second_class_rows.append(data_matrix[i])
            second_class_lbl.append(class_vector[i])
        elif class_vector[i]==2:
            third_class_rows.append(data_matrix[i])
            third_class_lbl.append(class_vector[i])
            
    # put half of each in training set and other half to test set (class1)
    first_class_tr_set=np.array(first_class_rows[:25])
    first_class_tr_lbl=np.array(first_class_lbl[:25])
    first_class_tst_set=np.array(first_class_rows[25:])
    first_class_tst_lbl=np.array(first_class_lbl[25:])

    # put half of each in training set and other half to test set (class2)
    second_class_tr_set=np.array(second_class_rows[:25])
    second_class_tr_lbl=np.array(second_class_lbl[:25])
    second_class_tst_set=np.array(second_class_rows[25:])
    second_class_tst_lbl=np.array(second_class_lbl[25:])

    # put half of each in training set and other half to test set (class3)
    third_class_tr_set=np.array(third_class_rows[:25])
    third_class_tr_lbl=np.array(third_class_lbl[:25])
    third_class_tst_set=np.array(third_class_rows[25:])
    third_class_tst_lbl=np.array(third_class_lbl[25:])
    
    #concatenate all classes an create a unified test set and a unified training set
    test_set=np.concatenate((first_class_tst_set, second_class_tst_set,third_class_tst_set), axis=0) # test set
    train_set=np.concatenate((first_class_tr_set, second_class_tr_set,third_class_tr_set), axis=0) # training set
    train_class_lbl=np.concatenate((first_class_tr_lbl, second_class_tr_lbl,third_class_tr_lbl), axis=0) # class label of training set
    test_class_lbl=np.concatenate((first_class_tst_lbl, second_class_tst_lbl,third_class_tst_lbl), axis=0) # class label of testtraining set

    return test_set,train_set,train_class_lbl,test_class_lbl

def accuracy(real_class_lbl,estimated_class_lbl):
    err=0 # number of errors
    for i in range(0, len(real_class_lbl)): # compare each element of real_class_lbl with estimated_class_lbl
        if real_class_lbl[i]!= estimated_class_lbl[i]:
            err+=1
    return (len(real_class_lbl)-err)/len(real_class_lbl) # result = percentage of accuracy

test_set,train_set,train_class_lbl,test_class_lbl = preparation()

print("accuracy of general bayes =","%.2f" % (accuracy(bayes_classifier(train_set,train_class_lbl,test_set),test_class_lbl)*100),"%")
print("accuracy of naive bayes =","%.2f" % (accuracy(bayes_naive_classifier(train_set,train_class_lbl,test_set),test_class_lbl)*100),"%")
print("accuracy of naive gaussian bayes =","%.2f" % (accuracy(bayes_naive_classifier_gaussian(train_set,train_class_lbl,test_set),test_class_lbl)*100),"%")