#load packages
library(bitops)
library(caret)
library(class)
library(e1071)
library(ggplot2)
library(jsonlite)
library(SnowballC)
library(tm)

#import data
cook <- fromJSON("train.json", flatten = TRUE)

#check data imported correctly
View(cook)

#setting seed
set.seed(74)

#re-code target feature as a factor
cook$cuisine <- factor(cook$cuisine)
str(cook$cuisine)

#load explanatory text data into a corpus
cook_corpus <- VCorpus(VectorSource(cook$ingredients))

#convert corpus to a document term matrix
#control operations are TOLOWER, REMOVENUMBERS, STOPWORDS, REMOVEPUNCTUATION and STEMMING
cook_DTM <- DocumentTermMatrix(cook_corpus, control = list(
  tolower = TRUE, 
  removeNumbers = TRUE, 
  stopwords = TRUE, 
  removePunctuation = TRUE, 
  stemming = TRUE)
)

#draw random sample (vector of indices)
q <- sample(39774, size = 39774*.8)
str(q)

#split dtm into testing and training sections
cook_train <- cook_DTM[q,]
cook_test <- cook_DTM[-q,]

#draw correpsonding labels for testing and training data from the raw data
cook_training_labels <- cook[q,]$cuisine
cook_test_labels <- cook[-q,]$cuisine

#filtered my DTM to only include words repeated more than twice
#this will increase specificity BUT it comes at the cost of lowering
#the impact from very unique words 

#returning vector of all words repeated at least twice
cookfreqwords <- findFreqTerms(x = cook_train, 2)

#applying the frequent words vector to the training and testing sets to refine them
cookfreqtrain <- cook_train[,cookfreqwords]
cookfreqtest <- cook_test[,cookfreqwords]

#short function to convert counting the number of word occurences, as it is listed in the DTm,
#to a binary Yes/No
convert_counts <- function(x) {x <- ifelse(x >0, "Yes", "No")}

#applying the convert function to the training and testing sets
cook_train <- apply(cookfreqtrain, FUN = convert_counts, MARGIN = 2)
cook_test <- apply(cookfreqtest, FUN = convert_counts, MARGIN = 2)

#creating Naive Bayes model
NBmodel <- naiveBayes(cook_train, cook_training_labels)

#predicting
preds <- predict(NBmodel, cook_test)

#confusion matrix 
confusionMatrix(preds, cook_test_labels)
