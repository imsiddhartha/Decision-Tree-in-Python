from math import log
import operator
import numpy as np
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
			data.append(l[:])
			labels.append(l[len(l)-1])
	return data,labels		
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

data,labels=read_ip('hayes-roth.data')
testdata,l=read_ip('hayes-roth.test')
#print len(data),len(testdata)
#print testdata[0],l[0]
#print testdata[1],l[1]
#print testdata[2],l[2]

t=tree(data,labels)

#print t
correct=0
incorrect=0
i=0
con_mat=np.zeros((3,3))
for test in testdata:
	res=testing(t,test[:-1])
	#print res
	if res==l[i]:
		correct+=1
		if res=="1\n":
			con_mat[0][0]+=1
		if res=="2\n":
			con_mat[1][1]+=1
		if res=="3\n":
			con_mat[2][2]+=1
			
	else:
		incorrect+=1
		if l[i]=="1\n" and res=="2\n":
			con_mat[0][1]+=1
		if l[i]=="1\n" and res=="3\n":
			con_mat[0][2]+=1	
		
		if l[i]=="2\n" and res=="1\n":
			con_mat[1][0]+=1
		if l[i]=="2\n" and res=="3\n":
			con_mat[1][2]+=1
			
		if l[i]=="3\n" and res=="1\n":
			con_mat[0][1]+=1
		if l[i]=="3\n" and res=="2\n":
			con_mat[0][2]+=1	
			
	#print i,res
	i+=1
#print correct,incorrect	
print "Confusion Matrix:"
print con_mat
print "Accuracy:",correct*1.0/(incorrect+correct)
