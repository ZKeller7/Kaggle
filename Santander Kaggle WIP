## script for Santander Customer Satisfaction Kaggle competition
## currently ~300/1100 competitors, or top 27%
library(caret)
library(xgboost)
library(pROC)
library(DMwR)

#importing data
test <- read.csv("E:/Downloads/Santander/test.csv", header=TRUE, stringsAsFactors = FALSE)
train <- read.csv("E:/Downloads/Santander/train.csv", header=TRUE, stringsAsFactors = FALSE)

#stripping IDs - dont need in train, saving from test for submission file
train <- train[,-1]
testID <- test[,1]
test <- test[,-1]

#showing correlations between features
hm <- cor(train)

#rm vars with no values
train <- train[,-which(colnames(train) %in% rownames(hm)[which(is.na(hm[,1]))])]
test <- test[,-which(colnames(test) %in% rownames(hm)[which(is.na(hm[,1]))])]

#now trying to show correlations between features
hm <- cor(train)

#separating features above the correlation cutoff
uh <- findCorrelation(hm, cutoff = .9999, verbose = TRUE, names = TRUE)

#removing those features from consideration
train <- train[,-which(colnames(train) %in% uh)]
test <- test[,-which(colnames(test) %in% uh)]


#########MAIN FLOW
#setting seed for reproducibility
set.seed(8)
#sampling data for split between validation and training sets
q1 <- which(train$TARGET == 1)
q2 <- which(train$TARGET == 0)
q1 <- sample(q1, length(q1) * .5)
q2 <- sample(q2, length(q2) * .5)
q <- unique(c(q1, q2))

#q <- sample(nrow(train), nrow(train) * .8)

#setting label index
LabelIndex <- ncol(train)

#splitting
ValLabs <- factor(train[,LabelIndex][-q])
hTrainLabs <- factor(train[,LabelIndex][q])

#removing target feature labels and converting to data.matrix for xgboost
Validation <- train[-q,-LabelIndex]
Training <- train[q,-LabelIndex]

#showing distribution of target feature
table(hTrainLabs)

#upsampling training data - significantly better results 
#vs no upsampling and vs downsampling
xgTraining <- upSample(data.matrix(Training), hTrainLabs)
TrainLabs <- xgTraining[,ncol(xgTraining)]
xgTraining <- data.matrix(xgTraining[,-LabelIndex])
levels(TrainLabs) <- c(0,1)
##########MAIN FLOW END

#setting xgb cross validation/training parameters
e <- .025
s <- .7
c <- .7
g <- 5
b <- .5
t <- 1000
    
#placing previously defined parameters and others in parameter list
ParamSet <- list(objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = e,
                 max_depth           = 4,
                 subsample           = s,
                 colsample_bytree    = c,
                 num_parallel_tree   = 1,
                 max_delta_step      = 0,
                 stratified          = T,
                 gamma               = g, 
                 base_score          = b,
                 nthread             = 8)



#cross validating nrounds with given parameter set
crossval <- xgb.cv(params            = ParamSet,
                   data              = data.matrix(Training),
                   nrounds           = t+500,
                   nfold             = 10,
                   label             = as.integer(sapply(hTrainLabs, as.character)),
                   metrics           = "auc",
                   stratified        = T,
                   verbose           = T,
                   maximize          = TRUE,
                   early_stop_round  = 10)

mods <- list(NA)
for (i in 1:250) {
#builing optimized model
boost.mod <- xgboost(params              = ParamSet, 
                     data                = xgTraining,
                     label               = as.numeric(sapply(TrainLabs, as.character)),
                     nrounds             = 250, #from cross validation
                     verbose             = F,
                     maximize            = TRUE,
                     stratified          = T)
       
#predicting onto validation data
Predictions <- predict(boost.mod, data.matrix(Validation))
#check on prediction distribution
hist(Predictions)

#resetting validation labels to match predictions
levels(ValLabs) <- c("0","1")

#rounding predictions to binary to use with confusion matrix
BinaryPreds <- round(Predictions, digits = 0)

#building and displaying confusion matrix
CM.mod <- confusionMatrix(BinaryPreds, ValLabs)
CM.mod
print(CM.mod$overall[1])
print(CM.mod$byClass[8])
print("##")
mods[[i]] <- boost.mod
}


imp <- xgb.importance(dimnames(xgTraining)[[2]], model = boost.mod)
library(Ckmeans.id.dp)
xgb.plot.importance(imp[1:20,])

trim <- imp$Feature[which(imp$Gain > .000005)]
xgTraining <- cbind(xgTraining, xgTraining[,trim])


#printing and writing test predications for submission
test2 <- data.matrix(test)
testpreds <- predict(boost.mod, test2)
submission <- as.data.frame(cbind(testID,testpreds))
colnames(submission) <- c("ID", "TARGET")
submission$ID <- as.integer(submission$ID)
hist(submission$TARGET)
table(round(submission$TARGET))
write.csv(submission, file = "E:/Downloads/Santander/43.csv", row.names = FALSE)
save(boost.mod, file = "E:/Downloads/Santander/mod41.RData")


