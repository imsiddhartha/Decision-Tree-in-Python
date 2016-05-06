from math import log
import operator
import random
import numpy as np
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
def entropy(data):
    entries = len(data)
    labels = {}
    for feat in data:
        label = feat[-1]
        if label not in labels.keys():
      	  labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        probability = float(labels[key])/entries
        entropy -= probability * log(probability,2)
    return entropy
    
def split(data, axis, val):
    newData = []
    for feat in data:
        if feat[axis] == val:
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis+1:])
            newData.append(reducedFeat)
    return newData
    
def choose(data):
    features = len(data[0]) - 1
    max_entropy = entropy(data)
    max_ig = 0.0;
    max_attri = -1
    for i in range(features):
        featList = [ex[i] for ex in data]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            newData = split(data, i, value)
            probability = len(newData)/float(len(data))
            newEntropy += probability * entropy(newData)
        infoGain = max_entropy - newEntropy
        if (infoGain > max_ig):
            max_ig = infoGain
            max_attri = i
    return max_attri
    
def majority(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def tree(data,labels):
    classList = [ex[-1] for ex in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majority(classList)
    max_attri = choose(data)
    max_attri_Label = labels[max_attri]
    theTree = {max_attri_Label:{}}
    #print max_attri_Label
    del(labels[max_attri])
    attri_vals = [ex[max_attri] for ex in data]
    uniqueVals = set(attri_vals)
    for value in uniqueVals:
        subLabels = labels[:]
        theTree[max_attri_Label][value] = tree(split(data, max_attri, value),subLabels)
    return theTree                
    
def read_ip(fname):
	with open (fname) as f:
		data=[]
		labels=[]
		for line in f:
			l=line.split(",")
			data.append(l)
	return data
def split_file(f):
	leng=len(f)//2
	dt=[]
	lab=[]
	tst=[]
	tstlab=[]
	count=0
	for line in f:
		#print line
		if count <= leng:
			dt.append(line)
			lab.append(line[len(line)-1])
		else:
			tst.append(line)
			tstlab.append(line[len(line)-1])
		count=count+1	
	return dt,lab,tst,tstlab				
def testing(t,test):

	if type(t) is dict:
		for key,val in t.iteritems():
			for sec,val1 in val.iteritems():
				#print sec,val1
				for word in test:
					if word not in val:
						continue
					
					else:
						return testing(val[word],test)	
	else:
		return t					

filedata=read_ip('car.data')

avg_accu=0
avg_pre=0
avg_recall=0

for i in range(5):
	print i+1,"fold :"
	con_mat=np.zeros((4,4))
	random.shuffle(filedata)
	data,labels,testdata,testlabels=split_file(filedata)

	t=tree(data,labels)
	#print t
	correct=0
	incorrect=0
	j=0
	for test in testdata:
		res=testing(t,test[:-1])
		if res==testlabels[j]:
			correct+=1
			if res=='unacc\n':
				con_mat[0][0]+=1
			if res=='acc\n':
				con_mat[1][1]+=1
			if res=='good\n':
				con_mat[2][2]+=1
			if res=='vgood\n':
				con_mat[3][3]+=1			
		else:
			incorrect+=1
			if testlabels[j]=='unacc\n' and res=='acc\n':
				con_mat[0][1]+=1
			if testlabels[j]=='unacc\n' and res=='good\n':
				con_mat[0][2]+=1
			if testlabels[j]=='unacc\n' and res=='vgood\n':
				con_mat[0][3]+=1
				
			if testlabels[j]=='acc\n' and res=='unacc\n':
				con_mat[1][0]+=1
			if testlabels[j]=='acc\n' and res=='good\n':
				con_mat[1][2]+=1
			if testlabels[j]=='acc\n' and res=='vgood\n':
				con_mat[1][3]+=1
				
			if testlabels[j]=='good\n' and res=='unacc\n':
				con_mat[2][0]+=1
			if testlabels[j]=='good\n' and res=='unacc\n':
				con_mat[2][1]+=1
			if testlabels[j]=='good\n' and res=='vgood\n':
				con_mat[2][3]+=1
				
			if testlabels[j]=='vgood\n' and res=='unacc\n':
				con_mat[3][0]+=1
			if testlabels[j]=='vgood\n' and res=='acc\n':
				con_mat[3][1]+=1
			if testlabels[j]=='vgood\n' and res=='good\n':
				con_mat[3][2]+=1												
		j+=1
		#print res
	avg_accu+=correct*1.0/(incorrect+correct)
	temp=0
	temp1=0
	sum_matrix=(sum(con_mat[0,:])+sum(con_mat[1,:])+sum(con_mat[2,:])+sum(con_mat[3,:]))
	#print sum_matrix
	for j in range(4):
		
		temp+=(con_mat[j][j]/sum(con_mat[j,:]))
		temp1+=(con_mat[j][j]/sum_matrix)
	#print temp/4,temp1
	avg_recall+=temp1
	avg_pre+=(temp/4)	
	#print correct,incorrect	
	#print correct*1.0/(incorrect+correct)
	print con_mat
	print "Accuracy:",correct*1.0/(incorrect+correct)
	print "Recall :",temp1
	print "Precision:",temp/4

print "\n" 		
print "Avg Accuracy:",avg_accu/5
print "Avg Recall :",avg_recall/5
print "Avg Precision:",avg_pre/5	
