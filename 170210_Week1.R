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


#########Quiz 3

#1.
#1. Subset the data to a training set and testing set based on the Case variable in the data set.
#2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and 
#default caret settings.
#3. In the final model what would be the final model prediction for cases with the following variable values:

require(caret)
require(AppliedPredictiveModeling)
data(segmentationOriginal)


set.seed(125)

intrain = createDataPartition(y = segmentationOriginal$Class, p = .75, list = FALSE)
training = segmentationOriginal[intrain,]
testing = segmentationOriginal[-intrain,]


mod_fit_tree = train(Class ~., method = "rpart", data = training)

print(mod_fit_tree$finalModel)
plot(mod_fit_tree$finalModel, uniform = TRUE)
text(mod_fit_tree$finalModel, use.n = TRUE, all = TRUE, cex = .8)



predict(mod_fit_tree, newdata = testing)


#5. 

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

set.seed(33833)

vowel.train$y = as.factor(vowel.train$y)
vowel_rf = randomForest(y ~.,data = vowel.train, importance = FALSE)

order(varImp(vowel_rf), decreasing = TRUE)



#################Quiz 4

#1.
#Load the vowel.train and vowel.test data sets:
#Set the variable y to be a factor variable in both the training and test set. Then
#set the seed to 33833. Fit (1) a random forest predictor relating the factor
#variable y to the remaining variables and (2) a boosted predictor using the
#"gbm" method. Fit these both with the train() command in the caret package.
#What are the accuracies for the two approaches on the test data set? What is
#the accuracy among the test set samples where the two methods agree?

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

set.seed(33833)

vowel.train$y = as.factor(vowel.train$y)
vowel.test$y = as.factor(vowel.test$y)


vowel_rf = train(y ~., method = "rf", data = vowel.train)
vowel_gbm = train(y ~., method = "gbm", data = vowel.train, verbose = FALSE)

prediction_vowel_rf = predict(vowel_rf, vowel.test)
prediction_vowel_gbm = predict(vowel_gbm, vowel.test)

combined_data = data.frame(prediction_vowel_rf, prediction_vowel_gbm, y=vowel.test$y)

combined_fit = train(y ~.,method="rf",data=combined_data)
combined_pred_test = predict(combined_fit, newdata = vowel.test)

c1 = confusionMatrix(prediction_vowel_rf, vowel.test$y)$overall['Accuracy']
c2 = confusionMatrix(prediction_vowel_gbm, vowel.test$y)$overall['Accuracy']
c3 = confusionMatrix(combined_pred_test, combined_data$y)$overall['Accuracy']

combinedValData <- data.frame(gbm_pred = gbm_pred_val,rf_pred = rf_pred_val)
# run the comb.fit on the combined validation data
comb_pred_val <- predict(combined_fit,combinedValData)

confusionMatrix(prediction_vowel_rf, vowel.test$y)
confusionMatrix(comb_pred_val, combinedValData)$overall['Accuracy']


# Question 2 --------------------------------------------------------------

rm(list = ls())
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
model_rf = train(diagnosis ~ ., method = 'rf', data = training)
model_gbm = train(diagnosis ~ ., method = 'gbm', data = training)
model_lda = train(diagnosis ~ ., method = 'lda', data = training)

pred_rf = predict(model_rf, training)
pred_gbm = predict(model_gbm, training)
pred_lda = predict(model_lda, training)

comb_data = data.frame(rf = pred_rf, gbm = pred_gbm, lda = pred_lda, diagnosis = training$diagnosis)
model_comb = train(diagnosis ~ ., method = 'rf', data = comb_data)

pred_rf_test = predict(model_rf, testing)
pred_gbm_test = predict(model_gbm, testing)
pred_lda_test = predict(model_lda, testing)
comb_data_test = data.frame(rf = pred_rf_test, gbm = pred_gbm_test, lda = pred_lda_test, diagnosis = testing$diagnosis)
pred_comb_test = predict(model_comb, comb_data_test)

accuracy_rf = sum(pred_rf_test == testing$diagnosis) / length(pred_rf_test)
accuracy_gbm = sum(pred_gbm_test == testing$diagnosis) / length(pred_gbm_test)
accuracy_lda = sum(pred_lda_test == testing$diagnosis) / length(pred_lda_test)

accuracy_comb = sum(pred_comb_test == comb_data_test$diagnosis) / length(pred_comb_test)

# Accuracy Results:
#   RF  : 0.7683
#   GBM : 0.7927
#   LDA : 0.7683
#   COMB: 0.7927

# So, the final answer is:
# Stacked Accuracy: 0.79 is better than random forests and lda and the same as boosting



# Question 3 --------------------------------------------------------------

rm(list = ls())
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
model = train(CompressiveStrength ~ ., method = 'lasso', data = training)
plot(model$finalModel)

# The plot is hard to understand.  I choose 'Cement' as the variable since it spent the most
# time away from zero...  (I am not sure this is the correct way to interprit this plot)


# Question 4 --------------------------------------------------------------

dat = read.csv("gaData.csv")
training = dat[year(dat$date) == 2011,]
tstrain = ts(training$visitsTumblr)

remdata = dat[year(dat$date) > 2011,]
tsrem = ts(remdata$visitsTumblr)

model = bats(tstrain)

pred <- forecast(model, h=length(tsrem),level=c(95))

accuracy(pred, remdata$visitsTumblr)
acc = sum(remdata$visitsTumblr <= pred$upper) / nrow(remdata)

# Result was 0.9617

# Question 5 --------------------------------------------------------------

rm(list = ls())
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(325)
model = svm(CompressiveStrength ~ ., data = training)
model

pred = predict(model, testing)
RMSE = sqrt(sum((pred - testing$CompressiveStrength)^2))

predins = predict(model, training)
RMSEins = sqrt(sum((predins - training$CompressiveStrength)^2))

# RMSE = 107.4401, this does not match any of the options...
# It did however match the value of 11543.39 which is the MSE not the RMSE

