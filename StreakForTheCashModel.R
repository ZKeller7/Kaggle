library(XML)
library(sampling)
library(dplyr)
library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(scales)

#setting column names for raw data output
MyColNames <- c("Time", "Side_One", "Sport", "Score_One", "Side_Two",
                "Vote_One", "Score_Two", "Status", "Vote_Two", "Warmth", "ID")
#creating function to transpose this specific data scrape into useable rows in 
#the order used above
makevec <- function(x) {
  r <- 1
  q <- vector('numeric')
  T <- ifelse(is.null(ncol(x)), 1, ncol(x))
  P <- ifelse(is.null(nrow(x)), 1, nrow(x))
  for (y in 1:T) {
    for (i in 1:P) {
      if (!is.na(x[i,y])) {
        if (x[i,y] != "") {
          q[r] <- x[i,y]
          r <- r + 1
        }
      }
    }
  }
  q <- as.data.frame(t(q))
  q[r] <- paste(sample(0:9, 10, replace = TRUE), collapse = "")
  return(q)
}

#code for scraping from ESPN
#setting base URL, will add to it to go to each specific date page
baseURL <- "http://streak.espn.go.com/en/entry?date=201501"
#1:500 lets us go from Jan 1 2015 to the present
for (URLAdd in 1:500) {
  LoopedURL <- paste0(baseURL, URLAdd) #adding day increment to base URL
  #reading table
  kappa <- readHTMLTable(doc = LoopedURL,
                         as.data.frame = TRUE,
                         stringsAsFactors = FALSE)
  #dropping noise
  kappa <- kappa[-length(kappa)]
  #dropping table headers
  if (length(kappa) < 2) {
    next
  } else {
    #building output
    InterimStreakData <- lapply(kappa, makevec)
    InterimStreakData <- do.call(rbind,
                                 Filter(function(x) length(x) == 11,
                                 InterimStreakData))
    ifelse(URLAdd == 1,
           BigStreakData <- InterimStreakData,
           BigStreakData <- data.frame(rbind(InterimStreakData, BigStreakData)))
  }
}

#adding names to scraped data
names(BigStreakData) <- MyColNames
#dropping unwanted columns
FinalStreakData <- unique(BigStreakData[,1:10])
sdata <- FinalStreakData[, c(-1,-8)]

#factoring Sport
sdata$Sport <- factor(sdata$Sport)
#cleaning votes columns
sdata$Vote_One <- gsub("%", "", sdata$Vote_One)
sdata$Vote_Two <- gsub("%", "", sdata$Vote_Two)

#converting score and votes to numerics
sdata$Score_One <- as.numeric(sdata$Score_One)
sdata$Score_Two <- as.numeric(sdata$Score_Two)
sdata$Vote_One <- as.numeric(sdata$Vote_One)
sdata$Vote_Two <- as.numeric(sdata$Vote_Two)

#catching winner
sdata$T1Wins <- ifelse(sdata$Score_One > sdata$Score_Two, TRUE, FALSE)

#catching prediction correct or not
sdata$VotedT1 <- ifelse(sdata$Vote_One >= sdata$Vote_Two, TRUE, FALSE)
sdata$VotersRight <- ifelse(sdata$T1Wins == sdata$VotedT1, TRUE, FALSE)

#printing table of how often voters were right
prop.table(table(sdata$VotersRight))

#function to bin strength of preference for team one
binner <- function(x) {
  if (x < 15) {
    "One-Sided Against"} 
  else if (x < 35 && x >= 15) {
    "Moderate Against"} 
  else if (x < 50 && x >= 35) {
    "Mild Against"}
  else if (x < 65 && x >= 50) {
    "Mild For"} 
  else if (x < 85 && x >= 65) {
    "Moderate For"}
  else if (x >= 85) {
    "One-Sided For"}
}
#applying function to data
sdata$StrengthofVoteforT1 <- sapply(sdata$Vote_One, binner)

#catching type of bet - janky but functional
sdata$Type <- ifelse(
  grepl("@", substr(sdata$Side_One,1,3), ignore.case = TRUE) == TRUE, #first test
      "Heads Up", #if true
        ifelse(grepl("Wins By|Draw", sdata$Side_One) == TRUE | grepl("Wins By|Draw", sdata$Side_Two) == TRUE, #if false
            "Wins By", #if true
              ifelse(grepl("Both|Either", sdata$Side_One) == TRUE | grepl("Both|Either", sdata$Side_Two) == TRUE, #if false
                "Parley", #if true
                  "Prop")) #if false
) #if false

#catching home field
sdata$T1Home <- ifelse(grepl("@", substr(sdata$Side_One,1,3),ignore.case = TRUE) == TRUE,
                       TRUE,
                       FALSE)

#creating custom function to find team win loss records from their 
#names field from the table, then return a ratio of the record
RecordPull <- function(x) {
  #records are enclosed in parentheses - we will search for those
  x <- gsub("[\\(\\)]", "", regmatches(x, gregexpr("\\(.*?\\)", x))[[1]])
  #pulling record
  x <- as.vector(na.omit(as.numeric(unlist(strsplit(unlist(x), "[^0-9]+")))))
  WP <- vector('numeric')
  if (length(x) == 0) {return(NA)}
  #calucate winning percentage
  if (length(x) == 2) {
    WP <- x[1]/sum(x)
  } else if (length(x) == 3) {
    WP <- (x[1] + .5 * x[3])/sum(x)
  }
  if (length(x) == 1) {
    return(NA)
  }
  return(WP)
}

#applying function to data
sdata$Side_OneRecord <- sapply(sdata$Side_One, RecordPull)
sdata$side_TwoRecord <- sapply(sdata$Side_Two, RecordPull)

#create function to bin record disparity
recbin <- function(x,y) {
  if (is.na(x) | is.na(y) == TRUE) {return(NA)}
  x <- x + .01
  y <- y + .01
  if ((x/y) >= 1.25) {
    "Strong T1 Advantage"
  }
  else if ((x/y) >= 1 && (x/y) < 1.25) {
    "Moderate T1 Advantage"
  }
  else if ((x/y) >= .75 && (x/y) < 1) {
    "Moderate T1 Disadvantage"
  } 
  else if ((x/y) < .75) {
    "Strong T1 Disadvantage"
  }
}

#applying function
sdata$RecordDisparity <- mapply(recbin, sdata$Side_OneRecord, sdata$side_TwoRecord)
sdata[is.na(sdata)] <- "None"

#removing small sample sizes
mod.data <- sdata[which(sdata$Sport %in% names(which(table(sdata$Sport) > 100))),]
table(mod.data$Sport)

#factoring sport
mod.data$Sport <- factor(mod.data$Sport)
#changing votersright from boolean to binary
mod.data$VotersRight <- ifelse(mod.data$VotersRight == T, 1, 0)

mod.data <- mod.data[order(mod.data$Sport),]
rownames(mod.data) <- c(1:nrow(mod.data))

#dropping columns that would cause target leakage
mod.data <- mod.data[,-c(1,3,4,6,9)]

#factoring appropriate columns
mod.data[,7] <- factor(mod.data[,7])
mod.data[,8] <- factor(mod.data[,8])
mod.data[,12] <- factor(mod.data[,12])

#if no record is available, we will substitute a neutral record
#we want to do this because win/loss record should be treated
#as a continous variable
mod.data[which(mod.data[,10] == "None"),10] <- .5
mod.data[which(mod.data[,11] == "None"),11] <- .5
mod.data[,10:11] <- sapply(mod.data[,10:11], as.numeric)

#moving target feature to the end
mod.data <- cbind(mod.data[,c(1:5,7:12)], mod.data[,6])
colnames(mod.data)[12] <- "VotersRight"

#checking structure
str(mod.data)

#setting seed for reproducibility
set.seed(7)

#splitting into training and validation sets
q <- sample(nrow(mod.data), nrow(mod.data) * .8)
Train <- mod.data[q,]
Validate <- mod.data[-q,]

#saving feature labels and removing them from training and validation
ValLabs <- Validate$VotersRight
Validate <- data.matrix(Validate[,-12])
ValLabs <- factor(ValLabs, levels = c(0,1))

TrainLabs <- Train$VotersRight
Train <- data.matrix(Train[,-12])
TrainLabs <- factor(TrainLabs, levels = c(0,1))

#creating control grid for testing XGBoost hyperparameters
ControlGrid <- expand.grid(nrounds   = 250, #tuning this later
                           eta       = c(.001,.005,.01),
                           max_depth = c(2,3,4)) #first guesses

#creating control parameter list for training hyperparameters
Control <- trainControl(method          = "cv",
                        number          = 10,
                        returnData      = FALSE,
                        returnResamp    = "all",
                        classProbs      = TRUE,
                        summaryFunction = twoClassSummary)

levels(TrainLabs) <- c("o", "l") #the train function doesn't like numeric labels
#grid searching hyperparameters
TrainHyper = train(
  x = Train,
  y = TrainLabs,
  trControl = Control,
  tuneGrid = ControlGrid,
  method = "xgbTree")

#plotting results
ggplot(TrainHyper$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")

#setting xgb cross validation/training parameters
#placing previously defined parameters and others in parameter list
ParamSet <- list(objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = .01,
                 max_depth           = 3,
                 nthread             = 8)

levels(TrainLabs) <- c("0", "1")
#cross validating nrounds with given parameter set
crossval <- xgb.cv(params            = ParamSet,
                   data              = Train,
                   nrounds           = 500,
                   nfold             = 10,
                   label             = as.integer(sapply(TrainLabs, as.character)),
                   metrics           = "auc",
                   stratified        = T,
                   verbose           = T,
                   maximize          = TRUE,
                   early_stop_round  = 10)

#saving optimal number of trees
trees <- which(crossval$test.auc.mean == max(crossval$test.auc.mean))
print(trees)
#plotting auc vs trees
ggplot(data = crossval, aes(x = seq_along(crossval$test.auc.mean), y = crossval$test.auc.mean)) + geom_line()

levels(TrainLabs) <- c(0,1)
#builing optimized model
boost.mod <- xgboost(params              = ParamSet, 
                     data                = Train,
                     label               = as.numeric(sapply(TrainLabs, as.character)),
                     nrounds             = trees, #from cross validation
                     verbose             = F,
                     maximize            = TRUE,
                     stratified          = T)

#predicting onto validation data
Predictions <- predict(boost.mod, Validate)

#check on prediction distribution
hist(Predictions)

#rounding predictions to binary to use with confusion matrix
BinaryPreds <- round(Predictions, digits = 0)

#building and displaying confusion matrix
CM.mod <- confusionMatrix(BinaryPreds, ValLabs)
CM.mod
