rm(list=ls())
library('MASS')
library('rpart')
source('DecisionTree.R')
source('Adaboost.R')

mytrain = read.csv('myfeature.csv')


traindata = mytrain[which(mytrain$outcome != -1),-c(1,2)]
testdata = mytrain[which(mytrain$outcome == -1),-c(1,2)]
weak_cf = DecisionTree(traindata,0.6,1000,testdata)
weak_cf_train = weak_cf[[1]]
weak_cf_test = weak_cf[[2]]
weak_cf_train = weak_cf_train-1
weak_cf_test = weak_cf_test-1


save(weak_cf_train,file = 'weak_cf_train.Rdata')
save(weak_cf_test,file = 'weak_cf_test.Rdata')

outcome = traindata$outcome
myalpha = Adaboost(weak_cf_train,outcome)
weak_cf_test = weak_cf_test
mypred1 = weak_cf_test%*%myalpha
mypred1 = mypred1/1000
mypred1 = scale(mypred1)
mypred1 = cbind(mytrain[which(mytrain$outcome == -1),],mypred1)
mypred1 = mypred1[,c(2,418)]


rownames(mypred1) <- NULL
colnames(mypred1) = c('bidder_id','prediction')
write.csv(mypred1,'mypred.csv',row.names=F)



