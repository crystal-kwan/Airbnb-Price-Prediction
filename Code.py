#set seed
set.seed(123)

#import libraries
library(caTools)
library(randomForest)
library(caret)
library(GGally)
library(e1071)
library(plyr)
library(readr)
library(dplyr)
library(ggplot2)
library(repr)

#read data
mainData = read.csv("Airbnb Data_Final.csv")

#create new column of index
mainData$index = 1:nrow(mainData)

#take log of mainData's variable "price"
mainData$price = log(mainData$price) 

mainData$neighbourhood_cleansed = as.factor(mainData$neighbourhood_cleansed)
mainData$property_type = as.factor(mainData$property_type)
mainData$room_type = as.factor(mainData$room_type)
mainData$bed_type = as.factor(mainData$bed_type)
mainData$cancellation_policy = as.factor(mainData$cancellation_policy)
mainData$amenities <- NULL
mainData$transit <- NULL

#split data into training and test sets (8:2)
split = sample.split(mainData$price, SplitRatio=0.8)
training_set = subset(mainData, split == TRUE)
test_set = subset(mainData, split == FALSE)

#______________________________________________________________________________________
#Model 1: Linear Regression
linear_model = lm(price ~. -index, data=training_set)
summary(linear_model)

#evaluation metrics function
eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 6))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 6))
  cat("Adjusted R-squared: ", adj_r2, "  |  ") #Adjusted R-squared
  cat("RMSE: " ,as.character(round(sqrt(sum(resids2)/N), 6))) #RMSE
}

#predictions on training set, R-Squared / RSME
train_prediction = predict(linear_model, newdata = training_set)
eval_metrics(linear_model, training_set, train_prediction, target = 'price')

#predictions on test set, R-Squared / RSME
test_prediction = predict(linear_model, newdata = test_set)
eval_metrics(linear_model, test_set, test_prediction, target = 'price')

#EXAMPLE TO PREDICT SPECIFIC PRICE FOR SPECIFIC INDEX (Linear Regression)
listing1 = mainData[which(mainData$index == '1'),] #Name can be anything, key is "index"
listing1 #Show variables summary of that particular index

cat("Predicted: ", exp(predict(linear_model, listing1)), "Actual: ", exp(listing1$price)) #Print out Predicted Price and Actual Price



#______________________________________________________________________________________
#Model 2: Ridge Regression

#Scaling the numeric features
cols = c('mtr','bus','host_is_superhost','host_has_profile_pic','host_identity_verified','neighbourhood_cleansed','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','TV','Internet','Air.conditioning','Kitchen','Washer','Elevator','Essentials','Wheelchair.accessible','security_deposit','cleaning_fee','number_of_reviews','cancellation_policy')

pre_proc_val <- preProcess(training_set[,cols], method = c("center", "scale"))

training_set[,cols] = predict(pre_proc_val, training_set[,cols])
test_set[,cols] = predict(pre_proc_val, test_set[,cols])

summary(training_set)


#Regularization to overcome large coefficients, thus solving over-fitting
cols_reg = c('mtr','bus','host_is_superhost','host_has_profile_pic','host_identity_verified','neighbourhood_cleansed','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','TV','Internet','Air.conditioning','Kitchen','Washer','Elevator','Essentials','Wheelchair.accessible','security_deposit','cleaning_fee','number_of_reviews','cancellation_policy','price')

dummies <- dummyVars(price ~ ., data = mainData[,cols_reg])

train_dummies = predict(dummies, newdata = training_set[,cols_reg])

test_dummies = predict(dummies, newdata = test_set[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))


library(glmnet)

x = as.matrix(train_dummies)
y_train = training_set$price

x_test = as.matrix(test_dummies)
y_test = test_set$price

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

summary(ridge_reg)

#Automate the task of finding the optimal lambda value using the cv.glmnet() function
cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

best_ridge= glmnet(x, y_train, alpha = 0, family = 'gaussian', lambda = optimal_lambda)
coef(best_ridge)

# Create a model performance evaluation matrix 
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on training set
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, training_set)

# Prediction and evaluation on test set
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test_set)


#______________________________________________________________________________________
#Model 3: Lasso Regression

#Find the best best cross-validated lambda
lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best


#Train Lasso model
lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

best_lasso= glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)
coef(best_ridge)

#Evaluate thhis model based on our model performance evaluation matrix
predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, training_set)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test_set)



#______________________________________________________________________________________
#Model 3: Random Forest

rf_model_train <- train(price~.,
                         data=training_set,  
                         method="rf",              
                         nodesize= 10,              # 10 data-points/node. Speeds modeling
                         ntree =500,               # Default 500. Reduced to speed up modeling
                         trControl=trainControl(method="repeatedcv", number=2,repeats=1),  # cross-validation strategy
                         tuneGrid = expand.grid(mtry = c(26))
)

rf_model_test <- train(price~., 
                  data=test_set,  
                  method="rf",  
                  nodesize= 10,              # 10 data-points/node. Speeds modeling
                  ntree =500,               # Default 500. Reduced to speed up modeling
                  trControl=trainControl(method="repeatedcv", number=2,repeats=1),  # cross-validation strategy
                  tuneGrid = expand.grid(mtry = c(26))
)

rf_model_train #Show RMSE and Rsquared of Training Set
plot(varImp(rf_model_train)) #Plot Importance Variable of Training Set

rf_model_test #Show RMSE and Rsquared of Training Set
plot(varImp(rf_model_test)) #Plot Importance Variable of Training Set

#EXAMPLE TO PREDICT SPECIFIC PRICE FOR SPECIFIC INDEX (Random Forest Regression)
listing2 = mainData[which(mainData$index == '2'),] #Name can be anything, key is "index"
listing2 #Show variables summary of that particular index

cat("Predicted: ", exp(predict(rf_model_train, listing1)), "Actual: ", exp(listing1$price)) #Print out Predicted Price and Actual Price


############################################################################

#random forest model
trctrl = trainControl(method = "none")
RFModel = train(price~., data = training_set, method="rf", nbagg = 50, trControl = trctrl, importance = TRUE)

#Summary of Model
(RFModel)
plot(varImp(RFModel))

#Random Forest Accuracy (Training Set)
rf_prediction_train = predict(RFModel, newdata = training_set)
training_set$predict_p <- predict(RFModel, newdata = training_set)
train_actual_p <- training_set$price
train_predict_p <-training_set$predict_p
train_rss <- sum((train_predict_p-train_actual_p)^2)
train_tss <- sum((train_actual_p - mean(train_actual_p))^2)
train_rsq <- 1 - train_rss/train_tss
train_rsq

#Random Forest Accuracy (Test Set)
rf_prediction_test = predict(RFModel, newdata = test_set)
test_set$predict_p <- predict(RFModel, newdata = test_set)
test_actual_p <- test_set$price
test_predict_p <-test_set$predict_p
test_rss <- sum((test_predict_p-test_actual_p)^2)
test_tss <- sum((test_actual_p - mean(test_actual_p))^2)
test_rsq <- 1 - test_rss/test_tss
test_rsq


#______________________________________________________________
#Visualization


library(glmnet)
## Loading required package: Matrix
## Loaded glmnet 4.0-2
# Prepare glmnet input as matrix of predictors and response var as vector
varmtx <- model.matrix(price~.-1, data=mainData)
response <- mainData$price
# alpha=0 means ridge regression. 
ridge <- glmnet(scale(varmtx), response, alpha=0)
# Cross validation to find the optimal lambda penalization
cv.ridge <- cv.glmnet(varmtx, response, alpha=0)
lbs_fun <- function(fit, offset_x=1, ...) {
  L <- length(fit$lambda)
  x <- log(fit$lambda[L])+ offset_x
  y <- fit$beta[, L]
  labs <- names(y)
  text(x, y, labels=labs, ...)
}
plot(ridge, xvar = "lambda", label=T)
lbs_fun(ridge)
abline(v=cv.ridge$lambda.min, col = "red", lty=2)
abline(v=cv.ridge$lambda.1se, col="blue", lty=2)




#Option B: Building Lasso regression

# alpha=1 means lasso regression. 
lasso <- glmnet(scale(varmtx), response, alpha=1)

# Cross validation to find the optimal lambda penalization
cv.lasso <- cv.glmnet(varmtx, response, alpha=1)


plot(lasso, xvar = "lambda", label=T)
lbs_fun(ridge, offset_x = -2)
abline(v=cv.lasso$lambda.min, col = "red", lty=2)
abline(v=cv.lasso$lambda.1se, col="blue", lty=2)




# Prepare glmnet input as matrix of predictors and response var as vector
varmtx <- model.matrix(price~.-1, data=mainData)
response <- mainData$price


# alpha=0 means ridge regression. 
ridge2 <- glmnet(scale(varmtx), response, alpha=0)

# Cross validation to find the optimal lambda penalization
cv.ridge2 <- cv.glmnet(varmtx, response, alpha=0)



# alpha=1 means lasso regression. 
lasso2 <- glmnet(scale(varmtx), response, alpha=1)

# Cross validation to find the optimal lambda penalization
cv.lasso2 <- cv.glmnet(varmtx, response, alpha=1)
par(mfrow=c(1,2))
par(mar=c(4,2,6,2))

# Plot Ridge Model
plot(ridge2, xvar = "lambda", label=T)
lbs_fun(ridge2, offset_x = 1)
abline(v=cv.ridge2$lambda.min, col = "red", lty=2)
abline(v=cv.ridge2$lambda.1se, col="blue", lty=2)
title("Ridge (with co-linearity)", line=2.5)

# Plot Lasso Model

plot(lasso2, xvar = "lambda", label=T)
lbs_fun(lasso2, offset_x = 1)
abline(v=cv.lasso2$lambda.min, col = "red", lty=2)
abline(v=cv.lasso2$lambda.1se, col="blue", lty=2)
title("Lasso (with co-linearity)", line=2.5)
