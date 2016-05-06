from sklearn import svm
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
from matplotlib import pyplot as plt
import Image
import numpy as np
from sklearn.decomposition import PCA
from random import shuffle

arr=[]
label=[]
def read_ip():
	count=0
	with open('train.txt') as f:
		for line in f: # read rest of lines
			
			l=[]
			count=count+1
			#x=line.rfind(" ",0,len(line))
			#l1=line[:x]
			l1=line[:].split()
			#print l1
			for u in l1:
				#print u
				l.append(float(u))
		
	    		arr.append(l)
	    	#print arr
	    	#print count
	    	
	    	f.close()
	    	return count
def read_label():    	
	with open('label.txt') as f:
		for line in f: 			# read rest of lines
			label.append(int(line))


total=read_ip()
read_label()
con_mat11=np.zeros((2,2))		#for k=10,kernal==rbf
con_mat12=np.zeros((2,2))		#for k=10,kernal==linear

con_mat21=np.zeros((2,2))		#for k=100,kernal==rbf
con_mat22=np.zeros((2,2))		#for k=100,kernal==linear

avg_acc11=0
avg_acc12=0
avg_acc21=0
avg_acc22=0

for kk in range(5):
	data=[]
	lab=[]
	
	index_shuf = range(total)
	shuffle(index_shuf)

	for ind in index_shuf:
		data.append(arr[ind])
		lab.append(label[ind])
	
	#print total
	traindata= np.array(data[1:total//2+1],float)
	#print traindata.shape
	testdata= np.array(data[total//2:total+1],float)
	#print testdata.shape
	trainlabels=np.array(lab[1:total//2+1],float)
	#print trainlabels.shape
	testlabels=np.array(lab[total//2:total],float)
	#print testlabels.shape
	#print data
	#print label

	k=10

	pca = PCA(n_components=k)
	pca.fit(traindata)
	red_mat=pca.transform(traindata)
	red_mat2=pca.transform(testdata)
	#print red_mat.shape,red_mat2.shape

	clf = svm.SVC()
	#clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(red_mat,trainlabels)

	#dec = clf.decision_function([[1]])
	#print dec.shape[1]

	res=clf.predict(red_mat2)

	correct=0
	incorrect=0
	#print res
	#print "For k=",k
	#print "RBF Kernel"
	for i in range(len(res)):
		if res[i]==testlabels[i]:
			correct=correct+1
			if res[i]==-1:
				con_mat11[0][0]+=1
			else:
				con_mat11[1][1]+=1	
		else:
			if testlabels[i]==-1:			#predicted wrong i.e actual label -1 but predicted 1
				con_mat11[0][1]+=1
			else:
				con_mat11[1][0]+=1
			incorrect=incorrect+1
	#print correct,incorrect
	avg_acc11+=correct*1.0/(incorrect+correct)
	#print "Accuracy:",correct*1.0/(incorrect+correct)


	lin_clf = svm.LinearSVC()
	lin_clf.fit(traindata,trainlabels)

	#dec = clf.decision_function([[1]])
	#print dec.shape[1]

	res=lin_clf.predict(testdata)
	correct=0
	incorrect=0
	#print res
	#print "Linear Kernel"
	for i in range(len(res)):
		if res[i]==testlabels[i]:
			correct=correct+1
			if res[i]==-1:
				con_mat12[0][0]+=1
			else:
				con_mat12[1][1]+=1				
		else:
			if testlabels[i]==-1:			#predicted wrong i.e actual label -1 but predicted 1
				con_mat12[0][1]+=1
			else:
				con_mat12[1][0]+=1
			incorrect=incorrect+1
	#print correct,incorrect
	avg_acc12+=correct*1.0/(incorrect+correct)
	#print "Accuracy:",correct*1.0/(incorrect+correct)


	k=100
	pca = PCA(n_components=k)
	pca.fit(traindata)
	red_mat=pca.transform(traindata)
	red_mat2=pca.transform(testdata)
	#print red_mat.shape,red_mat2.shape

	clf = svm.SVC()
	#clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(red_mat,trainlabels)

	#dec = clf.decision_function([[1]])
	#print dec.shape[1]

	res=clf.predict(red_mat2)

	correct=0
	incorrect=0
	#print res
	#print "For k=",k
	#print "RBF Kernel"
	for i in range(len(res)):
		if res[i]==testlabels[i]:
			correct=correct+1
			if res[i]==-1:
				con_mat21[0][0]+=1
			else:
				con_mat21[1][1]+=1	
		else:
			if testlabels[i]==-1:			#predicted wrong i.e actual label -1 but predicted 1
				con_mat21[0][1]+=1
			else:
				con_mat21[1][0]+=1
			incorrect=incorrect+1
	#print correct,incorrect
	avg_acc21+=correct*1.0/(incorrect+correct)
	#print "Accuracy:",correct*1.0/(incorrect+correct)


	lin_clf = svm.LinearSVC()
	lin_clf.fit(traindata,trainlabels)

	#dec = clf.decision_function([[1]])
	#print dec.shape[1]

	res=lin_clf.predict(testdata)
	correct=0
	incorrect=0
	#print res
	#print "Linear Kernel"
	for i in range(len(res)):
		if res[i]==testlabels[i]:
			correct=correct+1
			if res[i]==-1:
				con_mat22[0][0]+=1
			else:
				con_mat22[1][1]+=1	
		else:
			if testlabels[i]==-1:			#predicted wrong i.e actual label -1 but predicted 1
				con_mat22[0][1]+=1
			else:
				con_mat22[1][0]+=1
			incorrect=incorrect+1
	#print correct,incorrect
	avg_acc22+=correct*1.0/(incorrect+correct)
	#print "Accuracy:",correct*1.0/(incorrect+correct)

print "For k==10"
print "Kernel==rbf"
print con_mat11
print avg_acc11/5.0
print "Kernel==linear"
print con_mat12
print avg_acc12/5.0

print "For k==100"
print "Kernel==rbf"
print con_mat21
print avg_acc21/5.0
print "Kernel==linear"
print con_mat22
print avg_acc22/5.0
print "Done"

