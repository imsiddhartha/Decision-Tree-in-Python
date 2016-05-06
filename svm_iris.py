from sklearn import svm
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
from matplotlib import pyplot as plt
import Image
import numpy as np
arr=[]
label=[]
def read_ip():
	count=0
	
	with open('iris.data') as f:
		for line in f: # read rest of lines
			
			l=[]
			count=count+1
			x=line.rfind(",",0,len(line))
			#l1=line[:x]
			l1=line[:x].split(',')
			nm=line[:].split(',')
			#print nm[4]
			label.append(nm[4])
			#print l1
			for u in l1:
				#print u
				l.append(float(u))
		
	    		arr.append(l)
	    	#print arr
	    	#print count
	    	f.close()
	    	return count	   
def pca(s,d,k,flag):
	E, V = eigh(s)				#E eigen values and V is eigen vectors	
	key = argsort(E)[::-1][:k]		#key will have indices of array
    	E, V = E[key], V[:, key]
    	V=V.transpose()
    	#print "dim of eigen vector ",V.shape
    	#print V
    	m = V[:,:]				#pc1 vs pc2
    	
    	U = np.dot(m,d.transpose())			# U is projection matrix
    	#print "dim of Modified Data ",U.shape
    	return U,V


total=read_ip()    	    	    	
#data= np.array(arr,float)

con_mat11=np.zeros((3,3))		#for k=10,kernal==rbf
con_mat12=np.zeros((3,3))		#for k=10,kernal==linear

avg_acc11=0
avg_acc12=0

#print total
qaut=total//6
#print qaut
traindata= np.array(arr[0:qaut],float)
traindata= np.concatenate((traindata,np.array(arr[2*qaut:3*qaut],float)),axis=0)
traindata= np.concatenate((traindata,np.array(arr[4*qaut:5*qaut],float)),axis=0)
#print traindata.shape
testdata= np.array(arr[qaut:2*qaut],float)
testdata= np.concatenate((testdata,np.array(arr[3*qaut:4*qaut],float)),axis=0)
testdata= np.concatenate((testdata,np.array(arr[5*qaut:6*qaut],float)),axis=0)
#print testdata.shape

trainlabels=np.array(label[0:qaut])
trainlabels=np.concatenate((trainlabels,np.array(label[2*qaut:3*qaut])),axis=0)
trainlabels=np.concatenate((trainlabels,np.array(label[4*qaut:5*qaut])),axis=0)

#print trainlabels[0],trainlabels[1]
#print trainlabels.shape

testlabels=np.array(label[qaut:2*qaut])
testlabels=np.concatenate((testlabels,np.array(label[3*qaut:4*qaut])),axis=0)
testlabels=np.concatenate((testlabels,np.array(label[5*qaut:6*qaut])),axis=0)

#print testlabels[0],testlabels[1]
#print testlabels.shape


#print data
mean_mat=mean(traindata,0)
mean_mat2=mean(testdata,0)
#print mean_mat

temp=traindata-mean_mat
temp2=testdata-mean_mat2
#print temp

scatter_mat= np.dot(temp.transpose(),temp)
scatter_mat2=np.dot(temp2.transpose(),temp2)
#print "Dim of scatter_mat"
#print scatter_mat.shape,scatter_mat2.shape

k=2
red_mat,V=pca(scatter_mat,traindata,k,0)
red_mat2,V2=pca(scatter_mat2,testdata,k,0)
#print red_mat
#print "Dim of Reduced_mat"
#print red_mat.shape,red_mat2.shape

clf = svm.SVC()
clf.fit(red_mat.transpose(),trainlabels)

res=clf.predict(red_mat2.transpose())

correct=0
incoreect=0
#print res
print "RBF Kernel"
for i in range(len(res)):
	if res[i]==testlabels[i]:
		correct=correct+1
		if res[i]=='Iris-setosa\n':
			con_mat11[0][0]+=1
		if res[i]=='Iris-versicolor\n':
			con_mat11[1][1]+=1	
		if res[i]=='Iris-virginica\n':
			con_mat11[2][2]+=1
	else:
		incoreect=incoreect+1
		if res[i]=='Iris-setosa\n' and testlabels[i]=='Iris-versicolor\n':
			con_mat11[0][1]+=1
		if res[i]=='Iris-setosa\n' and testlabels[i]=='Iris-virginica\n':
			con_mat11[0][2]+=1
		
		if res[i]=='Iris-versicolor\n' and testlabels[i]=='Iris-setosa\n':
			con_mat11[1][0]+=1
		if res[i]=='Iris-versicolor\n' and testlabels[i]=='Iris-virginica\n':
			con_mat11[1][2]+=1	
			
		if res[i]=='Iris-virginica\n' and testlabels[i]=='Iris-setosa\n':
			con_mat11[2][0]+=1
		if res[i]=='Iris-virginica\n' and testlabels[i]=='Iris-versicolor\n':
			con_mat11[2][1]+=1				
		
print correct,incoreect
print "Accuracy:",correct*1.0/(incoreect+correct)
print con_mat11

lin_clf = svm.LinearSVC()
lin_clf.fit(traindata,trainlabels)

#dec = clf.decision_function([[1]])
#print dec.shape[1]

res=lin_clf.predict(testdata)
correct=0
incoreect=0
#print res

for i in range(len(res)):
	if res[i]==testlabels[i]:
		correct=correct+1
		if res[i]=='Iris-setosa\n':
			con_mat12[0][0]+=1
		if res[i]=='Iris-versicolor\n':
			con_mat12[1][1]+=1	
		if res[i]=='Iris-virginica\n':
			con_mat12[2][2]+=1
	else:
		incoreect=incoreect+1
		if res[i]=='Iris-setosa\n' and testlabels[i]=='Iris-versicolor\n':
			con_mat12[0][1]+=1
		if res[i]=='Iris-setosa\n' and testlabels[i]=='Iris-virginica\n':
			con_mat12[0][2]+=1
		
		if res[i]=='Iris-versicolor\n' and testlabels[i]=='Iris-setosa\n':
			con_mat12[1][0]+=1
		if res[i]=='Iris-versicolor\n' and testlabels[i]=='Iris-virginica\n':
			con_mat12[1][2]+=1	
			
		if res[i]=='Iris-virginica\n' and testlabels[i]=='Iris-setosa\n':
			con_mat12[2][0]+=1
		if res[i]=='Iris-virginica\n' and testlabels[i]=='Iris-versicolor\n':
			con_mat12[2][1]+=1
print "Linear Kernel"
print correct,incoreect
print "Accuracy:",correct*1.0/(incoreect+correct)	
print con_mat12
print "Done"

