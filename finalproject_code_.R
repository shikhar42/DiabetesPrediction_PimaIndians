##packages install

#install.packages("dplyr")
#install.packages("caret")
#install.packages("plotROC")
#install.packages("InformationValue")
#install.packages("gmodels")
#install.packages("pROC")
#install.packages("class")
#install.packages("highcharter")
#install.packages("precrec")
#install.packages("corrplot")
#install.packages("C50")

##importing libraries
library(dplyr)
library(caret)
library(plotROC)
library(InformationValue)
library(gmodels)
library(pROC)
library(class)
library(highcharter)
library(precrec)
library(corrplot)
library(C50)

##to maintain consistency
set.seed(5689) 
##data load
diabetes <- read.csv(file = "diabetes.csv", header = TRUE) 
##converting our outcome variable to factor
diabetes$Outcome <- as.factor(diabetes$Outcome) 
##EDA##
##descriptive statistic
summary(diabetes)
prop.table(table(diabetes$Outcome))
##Plotting correlations between predictor variables
correlat <- cor(diabetes[, setdiff(names(diabetes), 'Outcome')])
corrplot(correlat)
##plotting outcome variable

outvar <- diabetes %>% group_by(Outcome)%>% count(Outcome)

ggplot(data= outvar, aes(x = Outcome,y = n)) +geom_bar(stat='identity',colour="white", fill = "#FFA07A") 
  

#split data into train/test
nrows <- NROW(diabetes)
                        # fix random value
index <- sample(1:nrows, 0.7 * nrows)   # shuffle and divide

diabetes_train <- diabetes[index,]                  
diabetes_test <- diabetes[-index,]

###decision trees#####


##training model of decision tree
decision_model <- C5.0(diabetes_train[-9], diabetes_train$Outcome)
summary(decision_model)
##predicting on test data
decision_predict <- predict(decision_model, diabetes_test)
##finding model accuracy
CrossTable(diabetes_test$Outcome, decision_predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
mean(diabetes_test$Outcome == decision_predict)
sensitivity_dt <- 71/(71+19)
specificity_dt <- 97/(97+44)
##improving model performance
grid_c50 <- expand.grid(.model = "tree",
                        .trials = c(10, 20, 30, 40),
                        .winnow = "FALSE")
ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 10)
m_c50 <- train(Outcome ~ ., data = diabetes_train, method = "C5.0",
               metric = "Kappa", trControl = ctrl,
               tuneGrid = grid_c50)


##predicting on test data
decision_predict_boost <- predict(m_c50, diabetes_test)
##finding model accuracy
CrossTable(diabetes_test$Outcome, decision_predict_boost,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))


mean(diabetes_test$Outcome == decision_predict_boost) 
sensitivity_dtb <- 57/(57+33)
specificity_dtb <- 119/(119+22)
#Standardize predictors 
diabetes$Pregnancies <- scale(diabetes$Pregnancies, scale=TRUE, center = TRUE)
diabetes$Glucose <- scale(diabetes$Glucose, scale=TRUE, center = TRUE)
diabetes$BloodPressure <- scale(diabetes$BloodPressure, scale = TRUE, center = TRUE)
diabetes$SkinThickness <- scale(diabetes$SkinThickness, scale = TRUE, center = TRUE)
diabetes$Insulin <- scale(diabetes$Insulin, scale = TRUE, center = TRUE)
diabetes$BMI <- scale(diabetes$BMI, scale = TRUE, center = TRUE)
diabetes$DiabetesPedigreeFunction <- scale(diabetes$DiabetesPedigreeFunction, scale = TRUE, center = TRUE)
diabetes$Age <- scale(diabetes$Age, scale = TRUE, center = TRUE)

#KNN CLASSIFICATION ####
acc_test <- numeric() 
##for loop from k = 1 to 30 to find maximum accuracy of algorithm
for(i in 1:30){
  
  predict <- knn(train=diabetes_train[,-9], test=diabetes_test[,-9], cl=diabetes_train[,9], k=i, prob=T)
  
  acc_test <- c(acc_test,mean(predict==diabetes_test[,9]))
  
}

acc <- data.frame(k= seq(1,30), cnt = acc_test)

opt_k <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of k is", opt_k$k, "(accuracy :", opt_k$cnt,") in KNN")
##plot of accuracies using highcharter
hchart(acc, 'line', hcaes(k, cnt)) %>%
  
  hc_title(text = "Accuracy With Varying K (KNN)") %>%
  
  hc_subtitle(text = sub) %>%
  
  hc_add_theme(hc_theme_google()) %>%
  
  hc_xAxis(title = list(text = "Number of Neighbors(k)")) %>%
  
  hc_yAxis(title = list(text = "Accuracy"))

##creation of model using optimized value of k
pre_knn <- knn(train = diabetes_train[,-9], test = diabetes_test[,-9], cl = diabetes_train[,9], k=opt_k$k, prob=T)
##finding model accuracy
CrossTable(diabetes_test$Outcome, pre_knn,prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                                                  dnn = c('actual default', 'predicted default'))
mean(diabetes_test[,9] == pre_knn)

sensitivity_knn <- 48/(48+42)
specificity_knn <- 129/(129+12)
##logistic regression

#Train model using glm
logistic_dmodel <- glm(Outcome ~., family=binomial(link="logit"), data=diabetes_train)
summary(logistic_dmodel)

#Testing model accuracy on test data
pred_model1 <- predict(logistic_dmodel, newdata = diabetes_test, type = "response")
predictd1 <- ifelse(pred_model1 > 0.5, 1,0)

pred_table1 <- table(predicted = predictd1, actual = diabetes_test$Outcome)
pred_table1
mean(predictd1 == diabetes_test$Outcome)

#Confusion Matrix
CrossTable(diabetes_test$Outcome, predictd1,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

#Sensitivity
sensitivity(diabetes_test$Outcome, pred_model1, threshold = 0.5)

#Specificity: Percentage of 0s predicted as 0s
specificity(diabetes_test$Outcome, pred_model1, threshold = 0.5)

#Improve logistic regression model accuracy
optCutOff1 <- optimalCutoff(diabetes_test$Outcome, pred_model1)[1] 
y_pred_num1 <- ifelse(pred_model1 > optCutOff1, 1, 0)
y_predicted1 <- factor(y_pred_num1, levels=c(0, 1))
y_observed1 <- diabetes_test$Outcome
mean(y_predicted1 == y_observed1)


#Plot AUROC curve
plotROC(diabetes_test$Outcome, pred_model1)

precrec_obj <- evalmod(scores = pred_model1, labels = diabetes_test$Outcome)
autoplot(precrec_obj)

