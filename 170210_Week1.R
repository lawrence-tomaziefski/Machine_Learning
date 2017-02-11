require(kernlab)
data("spam")

require(caret)
require(e1071)

##Data Splitting

in_train = createDataPartition(y = spam$type, p = .75, list = FALSE)
training = spam[in_train, ]
testing = spam[-in_train, ]
dim(training)

#SPAM Example: Fit a model

set.seed(32343)
modelFit <- train(type ~., data= training, method="glm")
modelFit

modelFit <- train(type ~.,data=training, method="glm")
modelFit$finalModel

predictions <- predict(modelFit,newdata=testing)
predictions

confusionMatrix(predictions,testing$type)


# SPAM Example: K-fold

set.seed(32323)
folds = createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=TRUE)
sapply(folds,length)

folds[[1]][1:10]

#SPAM Example: Return test

set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=FALSE)
sapply(folds,length)

folds[[1]][1:10]
#SPAM Example: Resampling

set.seed(32323)
folds <- createResample(y=spam$type,times=10,
                        list=TRUE)
sapply(folds,length)
folds[[1]][1:10]
#SPAM Example: Time Slices

set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices(y=tme,initialWindow=20,
                          horizon=10)
names(folds)
folds$train[[1]]
folds$test[[1]]

###Plotting predictiors

require(ISLR)
data(Wage)
summary(Wage)


#Get training/test sets

inTrain = createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)

qq <- qplot(age,wage,colour=education,data=training)
qq +  geom_smooth(method='lm',formula=y~x)

require(Hmisc)
cutWage <- cut2(training$wage,g=3)
table(cutWage)

p1 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot"))
p1

require(ggthemes)
p2 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot","jitter"))
grid.arrange(p1,p2,ncol=2)

####Week 2 Quiz
install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
data(AlzheimerDisease)



#2 Make a plot of the outcome (CompressiveStrength) versus the index of the samples. 
#Color by each of the variables in the data set (you may find the cut2() function in the 
#Hmisc package useful for turning continuous covariates into factors). 
#What do you notice in these plots?

data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

plot(training$CompressiveStrength, col = cut2(training$CompressiveStrength))

#Make a histogram and confirm the SuperPlasticizer variable is skewed. 
#Normally you might use the log transform
#to try to make the data more symmetric. Why would that be a poor choice for this variable?


hist(training$Superplasticizer)

#4 Find all the predictor variables in the training set that begin with IL. 
#Perform principal components on these variables with the preProcess() function 
#from the caret package. Calculate the number of principal components needed to capture 
#90% of the variance. How many are there?

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

adData_names = colnames(adData)

il = grep("IL",adData_names)

require(dplyr)

training = select(training, il)
training = select(training, -13)

preProcess(training, method = "pca", thresh = .9)


#5 Create a training data set consisting of only the predictors with variable names beginning 
#with IL and the diagnosis. Build two predictive models, one using the predictors as they 
#are and one using PCA with principal components explaining 80% of the variance in the 
#predictors. Use method="glm" in the train function.

#What is the accuracy of each method in the test set? Which is more accurate?
set.seed(3433)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

#PCA method

training = select(training, 1, il[1:12])
testing = select(testing, 1, il[1:12])

pre_proc = preProcess(training[,-1], method = "pca", thresh = .8)
train_pca = predict(pre_proc, training[,-1])
model_fit_pca = train(train_pca, training$diagnosis, method = "glm")

test_pca = predict(pre_proc, testing[,-1])
confusionMatrix(testing$diagnosis, predict(model_fit_pca,test_pca))

model_fit = train(diagnosis ~., data= training, method="glm")

predictions <- predict(model_fit,newdata=testing)

confusionMatrix(predictions,testing$diagnosis)

###train(training[,-1], training$diagnosis, method = "glm", 
      ###preProcess = "pca", trControl = trainControl(preProcOptions=list(thresh=0.8)))
