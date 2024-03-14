#!/usr/bin/env python3

import math

def entropy(pmf): # entropy func
    ent=0 # final result
    for i in range(0, len(pmf)): # for all elements inside pmf
        ent+=pmf[i]*math.log(1/pmf[i],2) # calculate and sum with previous results
    return ent

def jointEntropy(jntpmf): # Joint entropy func
    jntent=0 # final result
    for i in range(0, len(jntpmf)): # for outer sum
        innerSum=0
        for j in range(0, len(jntpmf[i])): # For inner sum
            innerSum+=jntpmf[i][j]*math.log(1/jntpmf[i][j],2) # calculate
        jntent+=innerSum
    return jntent

def conditionalEntropy(margpmfy,jntpmf): # Conditional entropy func
    condent=0 # final result
    for i in range(0, len(jntpmf)): # for outer sum
        innerSum=0
        for j in range(0, len(jntpmf[i])): # for inner sum
            innerSum+=jntpmf[i][j]*math.log(margpmfy[j]/jntpmf[i][j],2) # calculate
        condent+=innerSum
    return condent

def mutualInfo(margpmfx,margpmfy,jntpmf): # Mutual entropy func 
    mutinf=0 # final result
    for i in range(0, len(jntpmf)): # for outer sum
        innerSum=0
        for j in range(0, len(jntpmf[i])): # for inner sum
            innerSum+=(jntpmf[i][j])*math.log((jntpmf[i][j])/((margpmfx[i])*(margpmfy[j])),2) # calculate
        mutinf+=innerSum
    return mutinf

def NormalConditionalEntropy(margpmfx,margpmfy,jntpmf): # Normal Conditional Entropy func
    condent=conditionalEntropy(margpmfy,jntpmf) # calculate conditional entropy
    entx=entropy(margpmfx) # calculate entropy of x
    nrmcondent=condent/entx # normalize
    return nrmcondent

def NormalJointEntropy(margpmfx,margpmfy,jntpmf): # Normal Joint Entropy func
    jntent=jointEntropy(jntpmf) # calculate Mutual information
    entx=entropy(margpmfx) # calculate entropy of x
    enty=entropy(margpmfy) # calculate entropy of y
    nrmjntent=1-(jntent/(entx+enty)) # normalize
    return nrmjntent

def NormalMutualInfo(margpmfx,margpmfy,jntpmf): # Normal Mutual Info func
    nrmjntent=NormalJointEntropy(margpmfx,margpmfy,jntpmf) # calculate Normal Joint Entropy
    nrmmutinf=(1/nrmjntent)-1 # normalize
    return nrmmutinf

