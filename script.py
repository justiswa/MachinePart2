import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def getUniqueY(y):
    uniqueY = np.unique(y) #returns vector of unique results
    
    return uniqueY

def findSigma(input, mewVector):
    numRows = input.shape[0] #read the var name its pretty self explanatory
    sigma = np.zeros((input.shape[1],input.shape[1])) #make sigma be Column number x Column number
    
    for i in range(0,numRows): # for every row in the input
       value1 = input[i] - mewVector
       value1 = value1.reshape(value1.size,1) #getting first value and reshapeing to (2,1)
       
       value2 = input[i] - mewVector
       value2 = value2.reshape(value2.size,1)#getting second value and reshaping to (2,1)
       
       sigma += np.matmul(value1, np.transpose(value2))#matmulling the values
       
    sigma = np.divide(sigma,numRows)  #some good ol' fashioned matrix divison?
    return sigma


def  Gaussian(mean,sigma,X):
    #Inputs
    #mean is vector of column means
    #sigma is a matrix
    #x is data
    D = X.shape[0]
    value1 = X-mean
    value1 = value1.reshape(value1.size,1)
    
    first = 1/(pow((2*pi),(D/2))*pow(float(det(sigma)),(1/2)))
    ePart = -1*np.matmul(np.matmul(np.transpose(value1),inv(sigma)),value1)/2
    
    
    
    return first*np.exp(ePart)
    
def findClasses(X,y):#splits up primary vector based on result
   #
   #Returns a dirctionary of str(y) as the key value is a matrix of the correct rows from X
    #first find number of unique result types in Y
    uniqueY = getUniqueY(y) #returns vector of unique results
    
    #Now, get all indices of each result type
    i = 0
    resultIndices = {}
    #print(uniqueY)
    for i in uniqueY:
        resultIndices[str(int(i))] = []
    i = 0
    
    while(i < np.shape(y)[0]):#iterate through entire result vector
        #store i-value in matching point in resultIndicies
        value = int(y[i])
        index = i
        
        resultIndices[str(value)]+=[index]
        #print(type(value))
        i=i+1
    for i in uniqueY:
       resultIndices[str(int(i))] = np.take(X,resultIndices[str(int(i))],axis=0)
         
    return resultIndices,uniqueY#returning the split X values, and unique Y values
    

def findMew(input):#finds the mean of each column in a matrix you pass to it
    numCol = np.shape(input)[1] #number of colums
    i = 0 #counter variable
    Mew = [] #initializing random Mew vector
    input= np.transpose(input)
    while(i < numCol): #for each column
        Mew = np.append(Mew,np.mean(input[i]) )#the output of Mew[i] is the mean of column i
        i = i+1 #increment i
    #print("Mew Vector = ", Mew)
    return Mew

def ldaLearn(X,y):
    mewVector = findMew(X) #get the mew vector for X
    sigma = findSigma(X, mewVector) #find the sigma matrix for X
    splitClasses,uniqueY = findClasses(X, y) #use findClasses to split up matrix
    numColumns = X.shape[1]
    
    means = np.ones((uniqueY.shape[0],numColumns))
    covmat = sigma
    
    for i in uniqueY:
       means[int(i-1)] = findMew(splitClasses[str(int(i))])
        
    
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
   
    return means,covmat

def qdaLearn(X,y):
    numColumns = X.shape[1]
    splitClasses,uniqueY = findClasses(X, y) #use findClasses to split up matrix
    means = np.ones((uniqueY.shape[0],numColumns))
   
    for i in uniqueY:
       means[int(i-1)] = findMew(splitClasses[str(int(i))])
    
    covmats =[]
    for i in uniqueY:
        covmats.append(findSigma(splitClasses[str(int(i))],means[int(i-1)]))
    
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    
    
    uniqueY = getUniqueY(ytest) #fetch uniquey
    ypred = []
    
    for j in range(0,Xtest.shape[0]):
        minimum = sys.maxsize#minimum value
        
        for i in uniqueY:
            val2 = (Xtest[j] - means[int(i-1)])
            val1 = np.transpose(val2)
            val3 = np.matmul(val1,inv(covmat))
            val4 = np.matmul(val3,val2)
            
            
            if(val4 < minimum):
                minimum = val4
                predClass = int(i)
        
        ypred.append(predClass)
    numCorrect =0
    #print(ypred)
    ypred = np.array(ypred)
    ypred = ypred.reshape(len(ypred),1)
    
    
    for i in range(0,len(ypred)):
        if(ypred[i][0]==int(ytest[i][0])):
                numCorrect+=1
    acc = numCorrect/ytest.shape[0] *100
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    acc = None
    ypred = []
    
    counts = np.unique(ytest, return_counts=True)[1] #number of times each unique item appears
    
    
    X,uniqueY = findClasses(Xtest,ytest)
    numY = uniqueY.shape[0]
    
    for j in range(0,Xtest.shape[0]):
        maximum = -sys.maxsize-1
        for i in uniqueY:
            val1 = Gaussian(means[int(i-1)],covmats[int(i-1)],Xtest[j])
            if(val1 > maximum):
                maximum = val1
                predClass = int(i)
        ypred.append(predClass)
    numCorrect =0   
    ypred = np.array(ypred)
    ypred = ypred.reshape(len(ypred),1)
    
    for i in range(0,len(ypred)):
        if(ypred[i][0]==int(ytest[i][0])):
                numCorrect+=1
    acc = numCorrect/ytest.shape[0] *100
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred
#PART 1 ENDS HERE-------------------------------------------------------
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
    
    val1 = np.matmul(np.transpose(X),X)
    val1 = np.matmul(inv(val1),np.transpose(X))        
    w = np.matmul(val1,y)     
            
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # Ridge Estimate of W PartCSlides1 Page 20
    val1 = np.matmul(np.transpose(X),X)
    val1 = val1 + (lambd*np.identity(X.shape[1]))
    val1 = np.matmul(inv(val1),np.transpose(X))        
    w = np.matmul(val1,y)

	# slides partc1, #20
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    mse = 0
    numY = ytest.shape[0]
    
    for i in range(0,numY):
        wtx = np.dot(np.transpose(w),Xtest[i])
        
        val2 = (ytest[i][0] - wtx)[0]
        
        mse+=pow(val2,2)
    mse= mse/numY
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
    
    # IMPLEMENT THIS METHOD   
    
    squarer = lambda wj: wj ** 2
    square_func = np.vectorize(squarer)
    L2 = lambd * sqrt(np.sum(square_func(w))) / 2
    error = 0
    for i in range(0,X.shape[0]):
        error += np.square((y[i] - np.matmul(np.transpose(w), X[i]))) + L2
        
    error /= 2
    
    error_grad = np.zeros(w.shape)
   
    for j in range(0,len(w)):
        run_sum = 0
        for i in range(0,X.shape[0]):
            val = np.matmul(np.transpose(w),X[i])
            val = val - y[i]
            val = val *X[i,j]
            run_sum+= (lambd*w[j])
        error_grad[j]+=run_sum

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    
    N = x.shape[0]
    Xp = np.power(x, 0)
    Xp=Xp.reshape(N,1)
    
    for i in range(1,p+1):
        newVal = np.power(x,i)
        newVal = newVal.reshape(newVal.shape[0],1)
        Xp = np.hstack((Xp,newVal))
        
        
    
    return Xp

# Main script
# Problem 1
#This was added so it would run on my computer       
"""
inp = open('sample.pickle', 'rb')
str_inp = inp.read().decode()
modified_file = str_inp.replace('\r\n', '\n')
inp.close()

out = open('sample.pickle', 'wb')
out.write(modified_file.encode())
out.close() 
        

"""

# load the sample data 
                                
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))




# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))

plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('QDA')

plt.show()

#Part 1 Ends Here
"""

#This part was added so it would run on my computer
inp = open('diabetes.pickle', 'rb')
str_inp = inp.read().decode()
modified_file = str_inp.replace('\r\n', '\n')
inp.close()

out = open('diabetes.pickle', 'wb')
out.write(modified_file.encode())
out.close() 
"""
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')






# add intercept

X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_TD = testOLERegression(w,X,y)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_TD = testOLERegression(w_i,X_i,y)
print("TRAIN DATA")

print('MSE without intercept '+str(mle_TD))
print('MSE with intercept '+str(mle_i_TD))
print("TEST DATA")
print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
mintest = sys.maxsize
mintrain = sys.maxsize
opLamTest = 0
opLamTrain = 0
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    #mins vvv
    if(mses3_train[i] < mintrain):
        mintrain = mses3_train[i]
        opLamTrain = lambd
    if(mses3[i] < mintest):
        mintest = mses3[i]
        opLamTest = lambd
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
print("OpTrain = " , opLamTrain)
print("OpTest = " , opLamTest)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = opLamTest # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)



fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
