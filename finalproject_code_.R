##packages install

#install.packages("dplyr")
#install.packages("caret")
#install.packages("InformationValue")
#install.packages("gmodels")
#install.packages("pROC")
#install.packages("class")
#install.packages("ggplot2")
#install.packages("highcharter")
#install.packages("corrplot")
#install.packages("C50")
## NOTE: two formerly loaded plotting packages were removed -- they were
## only used for plots that duplicated the pROC analysis below.

##importing libraries
library(dplyr)
library(caret)
library(InformationValue)
library(gmodels)
library(pROC)
library(class)
library(ggplot2)
library(highcharter)
library(corrplot)
library(C50)

##to maintain consistency
set.seed(5689)
##data load (relative path -- place diabetes.csv next to this script;
##see README.md for the data source)
diabetes <- read.csv(file = "diabetes.csv", header = TRUE)
##converting our outcome variable to factor
diabetes$Outcome <- as.factor(diabetes$Outcome)

##column names (used everywhere instead of positional column indices)
predictor_cols <- setdiff(names(diabetes), "Outcome")
##columns where 0 is physiologically impossible and really means "missing"
zero_as_missing_cols <- c("Glucose", "BloodPressure", "SkinThickness",
                          "Insulin", "BMI")

##EDA##
##descriptive statistic -- RAW data, before any cleaning. The zeros in
##the summary below for Glucose / BloodPressure / SkinThickness /
##Insulin / BMI are not real measurements: 0 is physiologically
##impossible for those columns and encodes "missing" (counted explicitly
##as a data-quality issue just below). A summary of the CLEANED data
##follows the train/test split further down.
cat("== EDA summary: raw data (before cleaning) ==\n")
summary(diabetes)
prop.table(table(diabetes$Outcome))
##data-quality issue, counted explicitly: how many "missing" zeros each
##of the zero-as-missing columns contains. These become NA and are
##imputed with training-set medians after the split.
zero_counts <- sapply(diabetes[, zero_as_missing_cols],
                      function(x) sum(x == 0))
cat("Zeros per zero-as-missing column (data-quality issue):\n")
print(zero_counts)
##majority-class baseline: predicting the majority class (Outcome = 0) for
##everyone already achieves this accuracy, so every model below must beat it
majority_baseline <- max(prop.table(table(diabetes$Outcome)))
majority_baseline # approx. 0.651 for this dataset
##Plotting correlations between predictor variables
correlat <- cor(diabetes[, predictor_cols])
corrplot(correlat)
##plotting outcome variable

outvar <- diabetes %>% group_by(Outcome) %>% count(Outcome)

ggplot(data = outvar, aes(x = Outcome, y = n)) +
  geom_bar(stat = 'identity', colour = "white", fill = "#FFA07A")


## ---- helper: sensitivity / specificity computed from a confusion table ----
## (previously these were typed in by hand as hard-coded fractions)
confusion_stats <- function(actual, predicted, positive = "1") {
  actual <- factor(actual, levels = c("0", "1"))
  predicted <- factor(predicted, levels = c("0", "1"))
  tp <- sum(actual == positive & predicted == positive)
  tn <- sum(actual != positive & predicted != positive)
  fp <- sum(actual != positive & predicted == positive)
  fn <- sum(actual == positive & predicted != positive)
  c(accuracy = (tp + tn) / (tp + tn + fp + fn),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp))
}

## ---- helper: model accuracy vs the majority-class baseline, computed ------
## majority_baseline (printed in the EDA section, ~0.651 on the full data)
## was previously only eyeballed against each printed accuracy. This
## computes the comparison for a model scored on the test set: the
## test-set majority baseline (predict the test set's majority class for
## everyone) and accuracy MINUS that baseline, printed as one line.
report_vs_baseline <- function(actual, predicted, model) {
  acc <- mean(as.character(actual) == as.character(predicted))
  baseline <- max(prop.table(table(factor(actual, levels = c("0", "1")))))
  cat(sprintf(paste0("%s: test accuracy = %.4f | majority baseline ",
                     "(test set) = %.4f | accuracy - baseline = %+.4f\n"),
              model, acc, baseline, acc - baseline))
  invisible(acc - baseline)
}

## ---- helper: preprocessing learned from the TRAINING split only -----------
## 1) zeros in zero_as_missing_cols become NA
## 2) NAs are imputed with the training-set medians
## 3) scaled copies are produced with training-set means / sds, so the test
##    set is transformed with training statistics (never its own)
preprocess_split <- function(train_df, test_df) {
  train_med <- train_df
  test_med <- test_df
  medians <- sapply(zero_as_missing_cols, function(col) {
    x <- train_df[[col]]
    x[x == 0] <- NA
    median(x, na.rm = TRUE)
  })
  for (col in zero_as_missing_cols) {
    tr <- train_df[[col]]; tr[tr == 0] <- NA
    te <- test_df[[col]];  te[te == 0] <- NA
    tr[is.na(tr)] <- medians[[col]]
    te[is.na(te)] <- medians[[col]]
    train_med[[col]] <- tr
    test_med[[col]] <- te
  }
  ## scale after splitting, using training-set statistics for both sets
  train_scaled <- train_med
  test_scaled <- test_med
  centers <- sapply(train_med[, predictor_cols], mean)
  scales <- sapply(train_med[, predictor_cols], sd)
  train_scaled[, predictor_cols] <- scale(train_med[, predictor_cols],
                                          center = TRUE, scale = TRUE)
  test_scaled[, predictor_cols] <- scale(test_med[, predictor_cols],
                                         center = centers, scale = scales)
  list(train = train_med, test = test_med,
       train_scaled = train_scaled, test_scaled = test_scaled,
       medians = medians)
}

#split data into train/test -- stratified on Outcome, so both sets keep the
#class proportions of the full data (previously a plain random sample)
index <- createDataPartition(diabetes$Outcome, p = 0.7, list = FALSE)

diabetes_train_raw <- diabetes[index, ]
diabetes_test_raw <- diabetes[-index, ]

prep <- preprocess_split(diabetes_train_raw, diabetes_test_raw)
diabetes_train <- prep$train            # imputed, unscaled (trees / glm)
diabetes_test <- prep$test
diabetes_train_scaled <- prep$train_scaled  # imputed + scaled (KNN)
diabetes_test_scaled <- prep$test_scaled

##descriptive statistic -- CLEANED data (train + test rows after
##preprocessing): the zeros counted in the EDA section above were set to
##NA and imputed with the TRAINING-set medians (test rows use the same
##training medians, so no test information leaks into the cleaning).
##Note the minimums of Glucose / BloodPressure / SkinThickness / Insulin
##/ BMI are now all > 0 -- no impossible zeros remain.
cat("== Summary: cleaned data (after zero -> NA -> median imputation) ==\n")
summary(rbind(diabetes_train, diabetes_test))

###decision trees#####


##training model of decision tree
decision_model <- C5.0(diabetes_train[, predictor_cols], diabetes_train$Outcome)
summary(decision_model)
##predicting on test data
decision_predict <- predict(decision_model, diabetes_test)
##finding model accuracy
CrossTable(diabetes_test$Outcome, decision_predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Outcome', 'predicted Outcome'))
mean(diabetes_test$Outcome == decision_predict)
confusion_stats(diabetes_test$Outcome, decision_predict)
report_vs_baseline(diabetes_test$Outcome, decision_predict,
                   "Decision tree (untuned C5.0)")
##improving model performance (tuning by repeated CV on the training data)
grid_c50 <- expand.grid(.model = "tree",
                        .trials = c(10, 20, 30, 40),
                        .winnow = FALSE)
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
           dnn = c('actual Outcome', 'predicted Outcome'))


mean(diabetes_test$Outcome == decision_predict_boost)
confusion_stats(diabetes_test$Outcome, decision_predict_boost)
report_vs_baseline(diabetes_test$Outcome, decision_predict_boost,
                   "Decision tree (C5.0, CV-tuned)")

#KNN CLASSIFICATION ####
##k is chosen by cross-validation on the TRAINING data only (previously the
##loop scored every k on the test set and picked the best test accuracy).
##KNN uses the scaled splits, scaled with training-set statistics -- in the
##old script diabetes_train had been built before scaling, so KNN silently
##ran on unscaled data.
ctrl_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
m_knn <- train(Outcome ~ ., data = diabetes_train_scaled, method = "knn",
               trControl = ctrl_knn,
               tuneGrid = expand.grid(k = seq(1, 30)))

acc <- data.frame(k = m_knn$results$k, cnt = m_knn$results$Accuracy)
opt_k <- m_knn$bestTune$k
sub <- paste("Optimal number of k is", opt_k,
             "(cross-validated training accuracy :",
             round(max(m_knn$results$Accuracy), 4), ") in KNN")
##plot of cross-validated accuracies using highcharter
hchart(acc, 'line', hcaes(k, cnt)) %>%

  hc_title(text = "Cross-Validated Accuracy With Varying K (KNN)") %>%

  hc_subtitle(text = sub) %>%

  hc_add_theme(hc_theme_google()) %>%

  hc_xAxis(title = list(text = "Number of Neighbors(k)")) %>%

  hc_yAxis(title = list(text = "Accuracy"))

##creation of model using the CV-chosen value of k, scored on the test set once
pre_knn <- knn(train = diabetes_train_scaled[, predictor_cols],
               test = diabetes_test_scaled[, predictor_cols],
               cl = diabetes_train_scaled$Outcome, k = opt_k, prob = TRUE)
##finding model accuracy
CrossTable(diabetes_test$Outcome, pre_knn, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Outcome', 'predicted Outcome'))
mean(diabetes_test$Outcome == pre_knn)
confusion_stats(diabetes_test$Outcome, pre_knn)
report_vs_baseline(diabetes_test$Outcome, pre_knn,
                   paste0("KNN (k = ", opt_k, ", CV-chosen)"))

##logistic regression

#Train model using glm
logistic_dmodel <- glm(Outcome ~ ., family = binomial(link = "logit"), data = diabetes_train)
summary(logistic_dmodel)

##choose the probability cutoff on the TRAINING data only, using
##out-of-fold (10-fold CV) predicted probabilities -- previously
##optimalCutoff() was run on the test set, which leaks the test labels
##into the model selection.
folds <- createFolds(diabetes_train$Outcome, k = 10, list = TRUE)
oof_pred <- numeric(nrow(diabetes_train))
for (f in folds) {
  fold_fit <- glm(Outcome ~ ., family = binomial(link = "logit"),
                  data = diabetes_train[-f, ])
  oof_pred[f] <- predict(fold_fit, newdata = diabetes_train[f, ],
                         type = "response")
}
optCutOff <- optimalCutoff(diabetes_train$Outcome, oof_pred)[1]
optCutOff

#Score the test set once, with the training-chosen cutoff
pred_model1 <- predict(logistic_dmodel, newdata = diabetes_test, type = "response")
predictd1 <- ifelse(pred_model1 > optCutOff, 1, 0)

pred_table1 <- table(predicted = predictd1, actual = diabetes_test$Outcome)
pred_table1
mean(predictd1 == diabetes_test$Outcome)

#Confusion Matrix
CrossTable(diabetes_test$Outcome, predictd1,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Outcome', 'predicted Outcome'))

#Sensitivity and specificity, computed in code from the confusion counts
confusion_stats(diabetes_test$Outcome, predictd1)
report_vs_baseline(diabetes_test$Outcome, predictd1,
                   "Logistic regression")

##AUC with a 95% confidence interval (pROC), on the test set, reported
##against the majority-class baseline printed above
roc_logistic <- roc(diabetes_test$Outcome, pred_model1)
auc_logistic <- auc(roc_logistic)
auc_logistic_ci <- ci.auc(roc_logistic)
auc_logistic
auc_logistic_ci
plot(roc_logistic)

## ---- stability check: repeated stratified splits -------------------------
## The single split above is seed-dependent, so the pipeline is repeated
## over several seeds. Each replication is FULLY INDEPENDENT: every seed
## makes its own stratified split, learns its own imputation medians and
## scaling statistics from its own training data (via preprocess_split),
## selects its own k by cross-validation on that seed's training data,
## and derives its own logistic cutoff from that seed's out-of-fold
## training predictions. Nothing -- in particular not k or the cutoff --
## is carried over from the first split or shared between seeds, and
## each seed's test set is scored exactly once. (An earlier version of
## this loop reused the first split's opt_k and optCutOff for every
## seed, so the runs were not independent replications.)
repeat_seeds <- c(5689, 123, 2024, 42, 777)
repeat_results <- do.call(rbind, lapply(repeat_seeds, function(s) {
  set.seed(s)
  idx <- createDataPartition(diabetes$Outcome, p = 0.7, list = FALSE)
  p <- preprocess_split(diabetes[idx, ], diabetes[-idx, ])

  ## this seed's own k, chosen by repeated CV on this seed's scaled
  ## training data only (same tuning setup as the main KNN section)
  ctrl_s <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  m_knn_s <- train(Outcome ~ ., data = p$train_scaled, method = "knn",
                   trControl = ctrl_s,
                   tuneGrid = expand.grid(k = seq(1, 30)))
  k_s <- m_knn_s$bestTune$k

  ## this seed's own cutoff, from this seed's out-of-fold (10-fold CV)
  ## training predictions (same procedure as the main logistic section)
  folds_s <- createFolds(p$train$Outcome, k = 10, list = TRUE)
  oof_s <- numeric(nrow(p$train))
  for (f in folds_s) {
    fold_fit_s <- glm(Outcome ~ ., family = binomial(link = "logit"),
                      data = p$train[-f, ])
    oof_s[f] <- predict(fold_fit_s, newdata = p$train[f, ],
                        type = "response")
  }
  cutoff_s <- optimalCutoff(p$train$Outcome, oof_s)[1]

  ## score this seed's test set once, with this seed's k and cutoff
  fit <- glm(Outcome ~ ., family = binomial(link = "logit"), data = p$train)
  probs <- predict(fit, newdata = p$test, type = "response")
  preds <- ifelse(probs > cutoff_s, 1, 0)
  knn_pred <- knn(train = p$train_scaled[, predictor_cols],
                  test = p$test_scaled[, predictor_cols],
                  cl = p$train_scaled$Outcome, k = k_s)
  ## this seed's own test-set majority baseline, and each model's
  ## accuracy minus that baseline (computed, not eyeballed)
  test_baseline <- max(prop.table(table(p$test$Outcome)))
  logistic_acc <- mean(preds == p$test$Outcome)
  knn_acc <- mean(knn_pred == p$test$Outcome)
  data.frame(seed = s,
             k = k_s,
             cutoff = cutoff_s,
             logistic_accuracy = logistic_acc,
             logistic_auc = as.numeric(auc(roc(p$test$Outcome, probs,
                                               quiet = TRUE))),
             knn_accuracy = knn_acc,
             test_majority_baseline = test_baseline,
             logistic_acc_minus_baseline = logistic_acc - test_baseline,
             knn_acc_minus_baseline = knn_acc - test_baseline)
}))
repeat_results
##mean and spread across seeds
apply(repeat_results[, c("logistic_accuracy", "logistic_auc",
                         "knn_accuracy",
                         "logistic_acc_minus_baseline",
                         "knn_acc_minus_baseline")], 2,
      function(x) c(mean = mean(x), sd = sd(x)))
