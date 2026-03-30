# Loading packages: ------------------------------------------------------------

library(tidymodels)
library(readr)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(here)
library(GGally)
library(rpart)
library(rpart.plot)
library(sf)
library(spatialsample)
library(tidysdm)
library(stringr)
library(janitor)
library(purrr)
library(readxl)
library(VIM)
library(mice)
library(parallel)
library(doParallel)
library(brulee)
library(glmnet)

# Parallel processing setup: ---------------------------------------------------

cores <- parallel::detectCores()
cl <- makePSOCKcluster(cores - 2)
registerDoParallel(cl)

# Cleaning data: ---------------------------------------------------------------

listings <- read_excel("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/AirBnB Listings.xlsx")

# Checking for duplicate values: 
listings %>% janitor::get_dupes()
  # There are no duplicate values found in the data

# Converting key factors: 
listings <- listings %>%
  mutate(
    property_type = as.factor(property_type),
    room_type = as.factor(room_type),
    neighbourhood_cleansed = as.factor(neighbourhood_cleansed),
    across(where(is.integer), as.numeric),
    across(where(~ all(.x %in% c("t", "f", NA))), ~ as.numeric(.x == "t"))
  )

# Coding host_acceptance_rate and host_response_rate as numeric variables: 
listings <- listings %>% 
  mutate(host_acceptance_rate = as.numeric(na_if(host_acceptance_rate, "N/A")))
 
listings <- listings %>% 
  mutate(host_response_rate = as.numeric(host_acceptance_rate))
  # They have been stored as characters instead of numeric values

# Coding host response time as an ordered factor: 
listings$host_response_time <- ordered(listings$host_response_time, levels = c("a few days or more", "within a day", "within a few hours", "within an hour"))

# Cleaning price variable and generating log_price:
listings$price <- as.numeric(gsub("[$,]", "", listings$price))
listings$log_price <- log(listings$price + 1)

# Creating other informative variables: 
listings$desc_length <- str_length(listings$description)
listings$host_days   <- as.numeric(listings$last_scraped - listings$host_since)
listings <- listings %>% mutate(amenities_count = map_int(str_split(amenities, ","), length))

# Extract bathrooms from text where numeric column is NA:
listings$bathsfromtxt <- as.numeric(str_extract(listings$bathrooms_text, "[0-9]+\\.?[0-9]*"))
listings <- listings %>%
  filter(complete.cases(bathsfromtxt)) %>%
  mutate(bathrooms = ifelse(is.na(bathrooms), bathsfromtxt, bathrooms))

# Removing unnecessary columns: 

useful_columns <- c("id", "accommodates", "desc_length", "host_days", "host_response_rate",
                    "host_response_time", "host_acceptance_rate", "host_is_superhost",
                    "host_identity_verified", "host_has_profile_pic", "latitude", "longitude",
                    "property_type", "room_type", "bedrooms", "amenities_count",
                    "log_price", "minimum_nights", "maximum_nights",  "availability_365", "number_of_reviews",
                    "number_of_reviews_l30d", "review_scores_rating", "estimated_occupancy_l365d",
                    "reviews_per_month", "instant_bookable", "calculated_host_listings_count",
                    "neighbourhood_cleansed", "bathrooms")

listings_small <- listings %>% 
  select(useful_columns) 

# Dropping rows with high proportion of missing values: 

listings_small <- listings_small %>% 
  filter(!rowSums(is.na(across(useful_columns))) >= 6)

# Dropping missing values for key variables: 
listings_small <- listings_small %>% drop_na(estimated_occupancy_l365d) 
listings_small <- listings_small %>% drop_na(host_is_superhost)
  # I have dropped observations where the outcome and treatment variables contain missing values
  # Imputing them would bias estimates 


# Model preparation  -----------------------------------------------------------

set.seed(72)

data_split <- initial_split(listings_small, prop = 0.75)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Defining formula: 
formula <- estimated_occupancy_l365d ~
  host_acceptance_rate + host_is_superhost + host_identity_verified +
  host_has_profile_pic + latitude + longitude + property_type + bedrooms +
  amenities_count + log_price + minimum_nights + maximum_nights +
  availability_365 + number_of_reviews + review_scores_rating +
  reviews_per_month + instant_bookable + host_days + neighbourhood_cleansed +
  calculated_host_listings_count + desc_length + accommodates +
  host_response_rate + host_response_time + room_type + number_of_reviews_l30d

# Data imputation: 

impute_recipe <- recipe(formula, data = train_data) %>%
  step_novel(all_nominal_predictors()) %>%  # handle unseen factor levels
  step_other(all_nominal_predictors(), threshold = 0.1) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

learned_recipe <- prep(impute_recipe, training = train_data)
train_imputed  <- bake(learned_recipe, new_data = NULL)
test_imputed   <- bake(learned_recipe, new_data = test_data)

# Cross-validation setup:

cv_split    <- vfold_cv(train_imputed, strata = estimated_occupancy_l365d, v = 10)
reg_metrics <- metric_set(yardstick::rmse, yardstick::mae)

# Features recipe (applied on top of already-imputed data) 

features_recipe <- recipe(formula, data = train_imputed) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%                       # drop zero-variance cols
  step_corr(all_predictors(), threshold = 0.85)


# Extracting the processed model matrix from the features recipe 

prepped_features <- prep(features_recipe, training = train_imputed)
train_processed  <- bake(prepped_features, new_data = NULL)
test_processed   <- bake(prepped_features, new_data = test_imputed)

# Prediction - LASSO model --------------------------

lasso_model <- linear_reg(mixture = 1, penalty = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lasso_workflow <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(features_recipe)

# Parameter grid: 

lasso_param_set  <- extract_parameter_set_dials(lasso_workflow) %>%
  update(penalty = penalty(range = c(-5, 0)))
lasso_param_grid <- grid_space_filling(lasso_param_set, size = 30)

# Tuning hyperparameters via 10-fold cross-validation: 

lasso_tuned <- lasso_workflow %>%
  tune_grid(
    resamples = cv_split,
    grid      = lasso_param_grid,
    control   = control_grid(save_pred = TRUE, save_workflow = TRUE),
    metrics   = reg_metrics
  )

print(
  collect_metrics(lasso_tuned) %>%
    filter(.metric == "rmse") %>%
    arrange(mean) %>%
    head(10)
)

# Tuning curve plot
autoplot(lasso_tuned)

lasso_best <- select_best(lasso_tuned, metric = "rmse")

# Using lasso model on test data: 
final_lasso <- lasso_workflow %>%
  finalize_workflow(lasso_best) %>%
  fit(data = train_imputed)

lasso_predict <- predict(final_lasso, new_data = test_imputed) %>%
  bind_cols(test_imputed %>% select(estimated_occupancy_l365d))

# Performance metrics:

lasso_RMSE <- lasso_predict %>% yardstick::rmse(truth = estimated_occupancy_l365d, estimate = .pred)
lasso_MAE  <- lasso_predict %>% yardstick::mae(truth  = estimated_occupancy_l365d, estimate = .pred)
lasso_R2   <- lasso_predict %>% yardstick::rsq(truth  = estimated_occupancy_l365d, estimate = .pred)

lasso_RMSE
lasso_MAE
lasso_R2

# Variable importance: 

library(vip)

lasso_vip <- final_lasso %>%
  extract_fit_parsnip() %>%
  vip::vip(
    num_features = 20,
    aesthetics   = list(fill = "steelblue")
  ) +
  labs(
    title = "LASSO – Variable Importance (Top 20)",
    x     = "Importance (|coefficient|)"
  )

lasso_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_lasso.png", plot = lasso_vip, width = 8, height = 6, dpi = 300)


# Prediction - Adaptive LASSO: -------------------------------------------------

X_train <- train_processed %>% select(-estimated_occupancy_l365d) %>% as.matrix()
Y_train <- train_processed$estimated_occupancy_l365d
X_test  <- test_processed  %>% select(-estimated_occupancy_l365d) %>% as.matrix()
Y_test  <- test_processed$estimated_occupancy_l365d

# Getting the glmnet model matrix column names:

model_cols <- colnames(X_train)

# Extracting LASSO coefficients for those exact columns:

lasso_coef_matrix <- coef(
  extract_fit_engine(final_lasso),         # extract underlying glmnet object
  s = lasso_best$penalty                   # using the best lambda from earlier
)

# Converting to a named vector and dropping the intercept:

lasso_coef_vec <- as.vector(lasso_coef_matrix)[-1]          # drop intercept value
names(lasso_coef_vec) <- rownames(lasso_coef_matrix)[-1]    # drop intercept name

lasso_coef_aligned <- lasso_coef_vec[model_cols]
# This step aligns weights to model matrix columns (some may have been dropped by step_corr/step_zv):
# I only keep weights for columns that survived into the model matrix


# Calculating the adaptive weights: 

gamma <- 1
adaptive_weights <- 1 / (abs(lasso_coef_aligned) + 1e-6)

# Fitting adaptive LASSO via 10-fold cross-validation 

set.seed(72)
adaptive_fit <- cv.glmnet(
  x              = X_train,
  y              = Y_train,
  alpha          = 1,
  nfolds         = 10,
  penalty.factor = as.vector(adaptive_weights)   # must be unnamed vector
)

# Best lambda:

best_lambda_ada <- adaptive_fit$lambda.min

# Predictions on test set

adaptive_preds <- as.vector(predict(adaptive_fit, newx = X_test, s = "lambda.min"))

adaptive_predict <- tibble(
  .pred                     = adaptive_preds,
  estimated_occupancy_l365d = Y_test
)

# Performance metrics: 

adaptive_RMSE <- adaptive_predict %>% 
  yardstick::rmse(truth = estimated_occupancy_l365d, estimate = .pred)
adaptive_MAE  <- adaptive_predict %>% 
  yardstick::mae(truth  = estimated_occupancy_l365d, estimate = .pred)
adaptive_R2   <- adaptive_predict %>% 
  yardstick::rsq(truth  = estimated_occupancy_l365d, estimate = .pred)

adaptive_RMSE
adaptive_MAE
adaptive_R2

# Most importance variables: 

adaptive_coefs <- coef(adaptive_fit, s = "lambda.min")
adaptive_coef_df <- tibble(
  variable   = rownames(adaptive_coefs)[-1],
  importance = abs(as.vector(adaptive_coefs)[-1])
) %>%
  filter(importance > 0) %>%
  arrange(desc(importance)) %>%
  slice_head(n = 20)

adaptive_vip <- adaptive_coef_df %>%
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Adaptive LASSO - Variable Importance",
       x = NULL,
       y = "Importance (|coefficient|)")

adaptive_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_adaptive.png", plot = adaptive_vip, width = 8, height = 6, dpi = 300)


# Prediction - Random forest model: --------------------------------------------

rf_model <- rand_forest(
  mtry = tune(), 
  trees = 400, 
  min_n = tune()) %>%
  set_engine("ranger", 
             num.threads = parallel::detectCores() - 2, 
             importance = "permutation") %>%
  set_mode("regression")

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>%
  add_recipe(features_recipe)

# Restricting number of variables to randomly sample at each split: 
rf_param_set <- rf_workflow  %>% 
  extract_parameter_set_dials() %>% 
  update(
    mtry = mtry(c(2, 10)))

# Parameter grid: 

rf_param_grid <- grid_space_filling(rf_param_set, size = 10)
rf_param_grid
  # I have restricted the parameter grid size to 10 as increasing beyond 10 is too commputationally expensive for me 

# Tuning hyperparameters via 10-fold cross-validation: 

rf_tuned <- rf_workflow %>% 
  tune_grid(resamples = cv_split, grid = rf_param_grid, metrics = reg_metrics)

rf_best <- select_best(rf_tuned, metric = "rmse")

final_rf <- rf_workflow %>%
  finalize_workflow(rf_best) %>%
  fit(data = train_imputed)

rf_predict <- predict(final_rf, new_data = test_imputed) %>%
  bind_cols(test_imputed %>% select(estimated_occupancy_l365d))

# Performance metrics: 

RF_RMSE <- rf_predict %>% yardstick::rmse(truth = estimated_occupancy_l365d, estimate = .pred)
RF_RMSE
  # The RMSE of the random forest model is 25.6, indicating that it performed better than the LASSO model
RF_MAE <- rf_predict %>% yardstick::mae(truth = estimated_occupancy_l365d, estimate = .pred)
RF_R2 <- rf_predict %>% yardstick::rsq(truth = estimated_occupancy_l365d, estimate = .pred)


# Most important variables: 

rf_vip <- final_rf %>%
  extract_fit_parsnip() %>%
  vip::vip(num_features = 20,
           aesthetics = list(fill = "steelblue")) +
  labs(title = "Random Forest - Variable Importance",
       x = "Importance (|coefficient|)")
rf_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_RF.png", plot = rf_vip, width = 8, height = 6, dpi = 300)


# Prediction - Bagging: --------------------------------------------------------

n_predictors <- ncol(train_processed) - 1  # not using the outcome column as a predictor

bag_model <- rand_forest(
  mtry = n_predictors , 
  trees = 400, 
  min_n = tune()) %>%
  set_engine("ranger", 
             num.threads = parallel::detectCores() - 2,
             importance = "permutation") %>%
  set_mode("regression")

bag_workflow <- workflow() %>% 
  add_model(bag_model) %>%
  add_recipe(features_recipe)

# Parameter grid: 

bag_param_set <- bag_workflow  %>% extract_parameter_set_dials()
bag_param_grid <- grid_space_filling(bag_param_set, size = 10)
  # I have restricted the parameter grid size to 10 as increasing beyond 10 is too commputationally expensive for me 

# Tuning via 10-fold cross-validation: 

bag_tuned <- bag_workflow %>% 
  tune_grid(resamples = cv_split, grid = bag_param_grid, metrics = reg_metrics)

bag_best <- select_best(bag_tuned, metric = "rmse")

final_bag <- bag_workflow %>%
  finalize_workflow(bag_best) %>%
  fit(data = train_imputed)

bag_predict <- predict(final_bag, new_data = test_imputed) %>%
  bind_cols(test_imputed %>% select(estimated_occupancy_l365d))

# Performance metrics: 

bag_RMSE <- bag_predict %>% yardstick::rmse(truth = estimated_occupancy_l365d, estimate = .pred)
bag_RMSE
  # It appears that the RMSE of bagged trees is 26.3, which is marginally worse than the standard RF model
bag_MAE <- bag_predict %>% yardstick::mae(truth = estimated_occupancy_l365d, estimate = .pred)
bag_R2 <- bag_predict %>% yardstick::rsq(truth = estimated_occupancy_l365d, estimate = .pred)


# Most important variables: 

bag_vip <- final_bag %>%
  extract_fit_parsnip() %>%
  vip::vip(num_features = 20,
           aesthetics = list(fill = "steelblue")) +
  labs(title = "Bagging - Variable Importance",
       x = "Importance (Mean Decrease in MSE)")

bag_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_bagging.png", plot = bag_vip, width = 8, height = 6, dpi = 300)


# Prediction - Pruned Decision Tree: -------------------------------------------

# Defining model: 
tree_model <- decision_tree(cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
# Here, cost_complexity (cp) is the tuning parameter:

tree_workflow <- workflow() %>%
  add_model(tree_model) %>%
  add_recipe(features_recipe)

# Parameter grid for cp: 

tree_param_set  <- tree_workflow %>% extract_parameter_set_dials() %>%
  update(cost_complexity = cost_complexity(c(-4, -1)))
# The update searches for cp in range (1e-4, 1e-1) on the log scale
tree_param_grid <- grid_space_filling(tree_param_set, size = 10)

# Tuning via 10-fold cross-validation: 

tree_tuned <- tree_workflow %>%
  tune_grid(
    resamples = cv_split,
    grid      = tree_param_grid,
    control   = control_grid(save_pred = TRUE, save_workflow = TRUE),
    metrics   = reg_metrics
  )

collect_metrics(tree_tuned)

# Tuning curve plot: 
autoplot(tree_tuned)

# Select best cp by RMSE:
tree_best <- select_best(tree_tuned, metric = "rmse")
tree_best

# Fit final pruned tree on full training data:
final_tree <- tree_workflow %>%
  finalize_workflow(tree_best) %>%
  fit(data = train_imputed)

# Visualise the pruned tree:
final_tree %>%
  extract_fit_engine() %>%
  rpart.plot(type = 4, extra = 101, roundint = FALSE,
             main = "Pruned Decision Tree")

# Evaluate on test data:
tree_predict <- predict(final_tree, new_data = test_imputed) %>%
  mutate( .pred_rescaled = exp(.pred) - 1) %>%
  bind_cols(test_data %>% select(estimated_occupancy_l365d))

# Performance metrics: 

tree_RMSE <- tree_predict %>% yardstick::rmse(truth = estimated_occupancy_l365d, estimate = .pred)
tree_MAE <- tree_predict %>% yardstick::mae(truth = estimated_occupancy_l365d, estimate = .pred)
tree_R2 <- tree_predict %>% yardstick::rsq(truth = estimated_occupancy_l365d, estimate = .pred)

# Most important variables: 

tree_vip <- final_tree %>%
  extract_fit_parsnip() %>%
  vip::vip(num_features = 20,
           aesthetics = list(fill = "steelblue")) +
  labs(title = "Pruned Tree - Variable Importance",
       x = "Importance (MSE Reduction)")

tree_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_tree.png", plot = tree_vip, width = 8, height = 6, dpi = 300)



# Prediction - Feedforward Neural Network: -------------------------------------

rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))

cl <- makeCluster(detectCores() - 2)

clusterExport(cl, varlist = c("train_imputed", "formula"))
clusterEvalQ(cl, library(nnet))

seeds  <- c(1, 2, 3, 4)
models <- parLapply(cl, seeds, function(s) {
  set.seed(s)
  nnet(formula, data = train_imputed, size = 10,
       linout = TRUE, maxit = 500, decay = 0.01, trace = FALSE)
})

target   <- "estimated_occupancy_l365d"
features <- setdiff(names(train_imputed), target)

results <- lapply(seq_along(models), function(i) {
  preds <- predict(models[[i]], test_imputed[, features])
  r     <- rmse(test_imputed[[target]], preds)
  cat(sprintf("Seed %d | RMSE: %.4f\n", seeds[i], r))
  list(model = models[[i]], rmse = r, preds = preds)
})

best_idx   <- which.min(sapply(results, `[[`, "rmse"))
best_model <- results[[best_idx]]$model
best_preds <- results[[best_idx]]$preds

NN_rmse <- rmse(test_imputed[[target]], best_preds)
NN_mae  <- mean(abs(test_imputed[[target]] - best_preds))
NN_r2   <- 1 - sum((test_imputed[[target]] - best_preds)^2) /
  sum((test_imputed[[target]] - mean(test_imputed[[target]]))^2)


# Most important variables: 

library(vip)

pred_wrapper <- function(object, newdata) {
  as.vector(predict(object, newdata = newdata))
}

# Generate permutation-based variable importance
set.seed(72)

nn_vip <- vip(
  best_model,
  method      = "permute",
  train       = train_imputed[, features],
  target      = train_imputed[[target]],
  metric      = "rmse",
  pred_wrapper = pred_wrapper,
  num_features = 20,
  aesthetics  = list(fill = "steelblue")
) +
  labs(title = "FNN - Variable Importance (Permutation)",
       x     = "Importance (Increase in RMSE)")

nn_vip

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/VIP_FNN.png", plot = nn_vip, width = 8, height = 6, dpi = 300)



# Results = Prediction:  --------------------------------------------------
## Results Table: --------------------------------------------------------------

prediction_results_table_1 <- tibble(
  Method     = c("LASSO", "Adaptive LASSO",  "Random forest", "Bagged Trees", "Pruned Tree", "Feedforward Neural Network"),
  RMSE       = c(
    lasso_RMSE$.estimate, adaptive_RMSE$.estimate, RF_RMSE$.estimate, bag_RMSE$.estimate, tree_RMSE$.estimate , NN_rmse),
  MAE        = c(lasso_MAE$.estimate, adaptive_MAE$.estimate , RF_MAE$.estimate, bag_MAE$.estimate, tree_MAE$.estimate, NN_mae),
  R2         = c(lasso_R2$.estimate, adaptive_R2$.estimate, RF_R2$.estimate, bag_R2$.estimate, tree_R2$.estimate, NN_r2)
  
)
print(prediction_results_table_1)



## Interpretation of predictions: ----------------------------------------------

# Build pred_vars_values to exactly match train_imputed structure
pred_vars_values <- train_imputed[1, features]  

# Overwriting with median/mode values
for (col in features) {
  if (is.numeric(train_imputed[[col]])) {
    pred_vars_values[[col]] <- median(train_imputed[[col]], na.rm = TRUE)
  } else {
    pred_vars_values[[col]] <- names(sort(table(train_imputed[[col]]), decreasing = TRUE))[1]
  }
}


# Generating predictions: 

# LASSO: 

newpred_lasso <- predict(final_lasso, new_data = pred_vars_values)

# Adaptive LASSO: 

pred_vars_processed <- 
  bake(prepped_features, new_data = pred_vars_values %>%
                              bind_cols(tibble(estimated_occupancy_l365d = 0))) %>% 
  select(-estimated_occupancy_l365d) %>% as.matrix()
newpred_adaptive <- as.vector(predict(adaptive_fit, newx = pred_vars_processed, s = "lambda.min"))

# Random Forest: 

newpred_RF <- predict(final_rf, new_data = pred_vars_values)

# Bagging: 

newpred_bagging <- predict(final_bag, new_data = pred_vars_values)

# Pruned tree: 

newpred_prunedtree <- predict(final_tree, new_data = pred_vars_values)

# Feedforward Neural Network: 

newpred_FNN <- as.vector(predict(best_model, newdata = pred_vars_values))


# Results table of predictions using median/modal values: 

prediction_results_table_2 <- tibble(
  Method     = c("LASSO", "Adaptive LASSO",  "Random forest", "Bagged Trees", "Pruned Tree", "Feedforward Neural Network"),
  Estimate   = c(newpred_lasso$.pred, newpred_adaptive[1], newpred_RF$.pred, newpred_bagging$.pred, newpred_prunedtree$.pred, newpred_FNN[1])
)
print(prediction_results_table_2)


# Causal Analysis - Model preparation ------------------------------------------

# Defining regression formula: 

regression_formula <- estimated_occupancy_l365d ~  host_is_superhost+ 
  host_acceptance_rate+ host_identity_verified+ host_has_profile_pic+ latitude+ 
  longitude+ property_type+ bedrooms+ amenities_count+ log_price+ minimum_nights+
  maximum_nights+ availability_365+ number_of_reviews + neighbourhood_cleansed +
  review_scores_rating+ reviews_per_month+ instant_bookable + 
  host_days+ calculated_host_listings_count+ desc_length

covariates <- c("host_acceptance_rate", "host_identity_verified", "host_has_profile_pic",
                "latitude", "longitude", "property_type", "bedrooms", "amenities_count",
                "log_price", "minimum_nights", "maximum_nights", "availability_365",
                "number_of_reviews", "review_scores_rating", "reviews_per_month",
                "instant_bookable", "host_days", "calculated_host_listings_count",
                "desc_length", "neighbourhood_cleansed")

# Imputing missing data on full dataset: 

impute_causal <- recipe(regression_formula, data = listings_small) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_knn(all_numeric_predictors())

learned_recipe2 <- prep(impute_causal, training = listings_small)

# Applying to full listings dataset: 

listings_fullimputed <- bake(learned_recipe2, new_data = NULL)

# Checking scale of outcome variable: 

summary(listings_fullimputed$estimated_occupancy_l365d)

# Saving labels for neighborhood for later estimates of CATE:

neighbourhood_labels <- listings_fullimputed$neighbourhood_cleansed
property_labels <- listings_fullimputed$property_type

# Reducing number of levels in property_type:

property_counts <- table(property_labels)
property_props  <- property_counts / sum(property_counts)

property_labels_collapsed <- ifelse(
  property_props[property_labels] < 0.05,
  "Other",
  as.character(property_labels)
) %>% as.factor()

levels(property_labels_collapsed)

# Reducing number of levels in neighbourhood_cleansed:

neighbourhood_counts <- table(neighbourhood_labels)
property_props  <- property_counts / sum(neighbourhood_counts)

neighbourhood_labels_collapsed <- ifelse(
  property_props[neighbourhood_labels] < 0.05,
  "Other",
  as.character(neighbourhood_labels)
) %>% as.factor()

levels(neighbourhood_labels_collapsed)


# Updating covariates list to match dummy-expanded column names:
all_cols      <- names(listings_fullimputed)
covariate_cols <- setdiff(all_cols, c("estimated_occupancy_l365d", "host_is_superhost", "id"))


# Causal Analysis - Naive OLS: -------------------------------------------------

OLS_model <- lm(regression_formula, data = listings_fullimputed)
OLS_summary <- summary(OLS_model)
treat_coef  <- coef(OLS_summary)["host_is_superhost", ]
treat_coef 
  # Using the naive OLS model, we find that being a superhost increases estimated occupancy levels by approximately 26.87 days across the year 

# Causal Analysis - Metalearners: ----------------------------------------------

# Setting up: 

Y <- listings_fullimputed$estimated_occupancy_l365d 
D <- listings_fullimputed$host_is_superhost 
X <- listings_fullimputed %>%
  select(all_of(covariates)) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

# Scaling X:  

X_min  <- apply(X, 2, min)
X_max  <- apply(X, 2, max)

# Protecting against zero-range columns (constant after dummy encoding):
X_range <- X_max - X_min
X_range[X_range == 0] <- 1   # avoid division by zero
X_norm  <- as_tibble(scale(X, center = X_min, scale = X_range))

# Scaling Y: 

mean_Y <- mean(Y)
sd_Y   <- sd(Y)
Y_norm <- (Y - mean_Y) / sd_Y

# Setting up generic function to fit an MLP model with specific parameter set: 

fit_mlp <- function(X, y) {
  set.seed(39)
  brulee_mlp(
    x            = X,
    y            = y,
    hidden_units = c(32, 32),  #2 layers
    dropout      = 0.3, # dropout rate
    epochs       = 300, # number of iterations
    learn_rate   = 0.005, # learning rate
    activation   = c("relu"), #activation function - ReLu (better for vanishing gradients)
    stop_iter    = 10
  )
}

# Predict and return predictions transformed to original scale
pred_mlp <- function(model, predictors, mean_y, std_y) {
  predict(model, new_data = predictors)[[".pred"]] * std_y + mean_y
}

# Fit brulee MLP for propensity score 
fit_mlp_ps <- function(X, D) {
  set.seed(72)
  brulee_mlp(
    x            = X,
    y            = as.factor(D),
    hidden_units = c(64, 64),
    dropout      = 0.3,
    epochs       = 200,
    learn_rate   = 0.01,
    activation   = c("relu")
  )
}

clip01 <- function(p, eps = 1e-3) pmin(pmax(p, eps), 1 - eps)
# This function clips parameters to range (0.001, 0.999)
# This ensures that there are no division by zero errors later on

## S-Learner: ------------------------------------------------------------------ 

s_mod  <- fit_mlp(cbind(X_norm, D = D), Y_norm)

mu_s0 <- pred_mlp(s_mod, cbind(X_norm, D = 0), mean_Y, sd_Y)
mu_s1 <- pred_mlp(s_mod, cbind(X_norm, D = 1), mean_Y, sd_Y)

cate_s <-  mu_s1 - mu_s0
ATE_s <- mean(cate_s)
ATE_s
  # According to the S-Learner, being a superhost increased the estimated occupancy by 19.52 days across the year   


## T-Learner: ------------------------------------------------------------------   

idx_0 <- which(D == 0)
idx_1 <- which(D == 1)

t_mod0 <- fit_mlp(X_norm[idx_0, ], Y_norm[idx_0])
t_mod1 <- fit_mlp(X_norm[idx_1, ], Y_norm[idx_1])

mu1_t  <- pred_mlp(t_mod1, X_norm, mean_Y, sd_Y)
mu0_t  <- pred_mlp(t_mod0, X_norm, mean_Y, sd_Y)

cate_t <- mu1_t - mu0_t
ATE_t <- mean(cate_t)
ATE_t
  # According to the T-Learner, being a superhost increased the estimated occupancy by ~62 days across the year
  # This estimate of the ATE is greater than that of the S-Learner model 


## X-Learner: ------------------------------------------------------------------ 


Delta_1 <- Y[idx_1] - mu0_t[idx_1]
Delta_0 <- mu1_t[idx_0] - Y[idx_0]

x_mod0 <- fit_mlp(X_norm[idx_0, ], Delta_0)
x_mod1 <- fit_mlp(X_norm[idx_1, ], Delta_1)

tau0_x  <- pred_mlp(x_mod0, X_norm, 0, 1)
tau1_x  <- pred_mlp(x_mod1, X_norm, 0, 1)

# Fit the model for propensity scores 
ps_mod <- fit_mlp_ps(X_norm, D)

# Generating predictions for D
ps_predictions <- predict(ps_mod, new_data = X_norm,
                          type = "prob")
ehat  <- clip01(ps_predictions$.pred_1)
  # clip probability D=1 to (0.001, 0.999) range


cate_x <- ehat * tau0_x + (1 - ehat) * tau1_x
ATE_x <- mean(cate_x)
  # The X-Learner estimates that being a superhost increased the estimated_occupancy by 34.84 days across the year 
  # This estimate is less than the T-learner estimate but greater than the S-learner estimate 
SE_x <- sd(cate_x)

# Causal Analysis - AIPW: ------------------------------------------------------

n <- nrow(X)
# X was the matrix of covariates as a subset of the data
# X was created in the "Metalearners" section  

# N of folds
K <- 5

# Creating vector of zeroes of size N
fold_id <- numeric(n)

fold_id[D == 1] <- sample(rep(1:K, length.out = sum(D == 1)))
fold_id[D == 0] <- sample(rep(1:K, length.out = sum(D == 0)))

# Creating vectors of size N to hold our estimates for e, mu0 and mu1 (filled with NAs)
ehat_cf  <- rep(NA_real_, n)
mu0hat   <- rep(NA_real_, n)
mu1hat   <- rep(NA_real_, n)

for (k in 1:K) {
  cat(sprintf("  fold %d / %d\n", k, K))
  idx_te <- which(fold_id == k)
  idx_tr <- which(fold_id != k)
  
  # create train and test splits
  train_X <- X[idx_tr, ]
  test_X <- X[idx_te, ]
  train_Y <- Y[idx_tr]
  train_D <- D[idx_tr]
  
  train_X_min <- apply(train_X,2, min)
  train_X_max <- apply(train_X,2, max)
  train_X_range <- train_X_max - train_X_min
  train_X_range[train_X_range == 0] <- 1   # guard zero-range columns
  
  train_X_scaled  <- as_tibble(scale(train_X, center = train_X_min,
                                     scale = train_X_range))
  
  test_X_scaled  <- as_tibble(scale(test_X, center = train_X_min,
                                    scale = train_X_range))
  
  mean_y_train <- mean(train_Y)
  sd_y_train   <- sd(train_Y)
  train_Y_norm <- (train_Y - mean_y_train) / sd_y_train
  
  # fit a model for propensity score estimation
  ps_mod_cf <- fit_mlp_ps(train_X_scaled, train_D)
  # generate predictions of probability of D=0 and D=1
  pred_ps_mod_cf <- predict(ps_mod_cf, new_data = test_X_scaled,
                            type = "prob")
  
  # save the estimates of e=P_hat(D=1)
  ehat_cf[idx_te] <- pred_ps_mod_cf$.pred_1
  
  # fit the model for D=1 subset
  mod_cf1 <- fit_mlp(train_X_scaled[train_D == 1, ], train_Y_norm[train_D == 1])
  # fit the model for D=0 subset
  mod_cf0 <- fit_mlp(train_X_scaled[train_D == 0, ], train_Y_norm[train_D == 0])
  
  # generate predictions on test for mu1 and mu0, return them in original scale
  mu1hat[idx_te] <- pred_mlp(mod_cf1, test_X_scaled, mean_y_train, sd_y_train)
  mu0hat[idx_te] <- pred_mlp(mod_cf0, test_X_scaled, mean_y_train, sd_y_train)
}

# Check predictions are on the right scale
summary(mu1hat)
summary(mu0hat)
summary(ehat_cf_clipped)


# Clipping probability estimates to (0.001, 0.999) range
ehat_cf_clipped  <- clip01(ehat_cf)

# Clipping mu0hat and mu1hat to a valid range: 
Y_min <- min(Y)
Y_max <- max(Y)

mu0hat <- pmin(pmax(mu0hat, Y_min), Y_max)
mu1hat <- pmin(pmax(mu1hat, Y_min), Y_max)
  # I do this to ensure that there are no absurd values being predicted by the MLP 
  # Extreme values beyond the range of Y can contaminate the estimate of the ATE


psi_aipw <- (mu1hat - mu0hat) +
  D * (Y - mu1hat) / ehat_cf_clipped -
  (1 - D) * (Y - mu0hat) / (1 - ehat_cf_clipped)

tau_aipw <- mean(psi_aipw)
  # The AIPW estimator estimates an ATE of []
  # The impact of being assigned superhost status is that it improves occupancy by [] days across the year

se_aipw  <- sd(psi_aipw) / sqrt(n)

# Plotting the overlap assumption for AIPW: 

p <- tibble(propensity_score=ehat_cf_clipped, treatment=as.factor(D)) %>% 
  ggplot(aes(propensity_score, fill=treatment)) +
  geom_histogram()

ggsave("C:/Users/abhik/OneDrive/Desktop/Year 3/EC349/Assignment/EC349 Assignment/Overlap_plot_1.png", plot = p, width = 8, height = 6, dpi = 300)


# Results - Causal Analysis: ---------------------------------------------------
## Results Table: ---------------------------------------------------------------
causal_results_table <- tibble(
  Method     = c("Naive OLS", "S-Learner", "T-Learner", "X-Learner", "AIPW"),
  ATE        = c(treat_coef["Estimate"], ATE_s, ATE_t, ATE_x, tau_aipw),
  SE         = c(treat_coef["Std. Error"], "invalid standard errors", "invalid standard errors", "invalid standard errors", se_aipw)
)
print(causal_results_table)

## ATE across different property types: -----------------------------------

# S-Learner CATE by property type:
cate_s_property <- tibble(
  property_type = property_labels_collapsed,
  cate                   = cate_s
) %>%
  group_by(property_type) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "S-Learner")

# T-Learner CATE by property:
cate_t_property <- tibble(
  property_type = property_labels_collapsed,
  cate                   = cate_t
) %>%
  group_by(property_type) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "T-Learner")

# X-Learner CATE by property:
cate_x_property <- tibble(
  property_type = property_labels_collapsed,
  cate                   = cate_x
) %>%
  group_by(property_type) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "X-Learner")

# AIPW CATE by neighbourhood:
cate_aipw_property <- tibble(
  property_type = property_labels_collapsed,
  cate                   = psi_aipw
) %>%
  group_by(property_type) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "AIPW")

# Combining all estimators into one dataframe:
cate_all_property <- bind_rows(
  cate_s_property,
  cate_t_property,
  cate_x_property,
  cate_aipw_property
)

print(cate_all_property, n = 40)

# Pivot to wide format for cleaner presentation
cate_wide_property <- cate_all_property %>%
  select(property_type, estimator, CATE, CI_lower, CI_upper) %>%
  mutate(CATE_CI = sprintf("%.1f (%.1f, %.1f)", CATE, CI_lower, CI_upper)) %>%
  select(property_type, estimator, CATE_CI) %>%
  pivot_wider(
    names_from  = estimator,
    values_from = CATE_CI
  ) %>%
  arrange(desc(`AIPW`))

print(cate_wide_property)


## ATE across different neighbourhoods: ----------------------------------------

# S-Learner CATE by neighbourhood:
cate_s_neighbourhood <- tibble(
  neighbourhood_cleansed = neighbourhood_labels_collapsed,
  cate                   = cate_s
) %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "S-Learner")

# T-Learner CATE by neighbourhood:
cate_t_neighbourhood <- tibble(
  neighbourhood_cleansed = neighbourhood_labels_collapsed,
  cate                   = cate_t
) %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "T-Learner")

# X-Learner CATE by neighbourhood:
cate_x_neighbourhood <- tibble(
  neighbourhood_cleansed = neighbourhood_labels_collapsed,
  cate                   = cate_x
) %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "X-Learner")

# AIPW CATE by neighbourhood:
cate_aipw_neighbourhood <- tibble(
  neighbourhood_cleansed = neighbourhood_labels_collapsed,
  cate                   = psi_aipw
) %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(
    CATE     = mean(cate),
    SE       = sd(cate) / sqrt(n()),
    n        = n(),
    CI_lower = CATE - 1.96 * SE,
    CI_upper = CATE + 1.96 * SE
  ) %>%
  arrange(desc(CATE)) %>%
  mutate(estimator = "AIPW")

# Combining all estimators into one dataframe:
cate_all_neighbourhood <- bind_rows(
  cate_s_neighbourhood,
  cate_t_neighbourhood,
  cate_x_neighbourhood,
  cate_aipw_neighbourhood
)

# Presenting results:
cate_wide <- cate_all_neighbourhood %>%
  select(neighbourhood_cleansed, estimator, CATE, CI_lower, CI_upper) %>%
  mutate(CATE_CI = sprintf("%.1f (%.1f, %.1f)", CATE, CI_lower, CI_upper)) %>%
  select(neighbourhood_cleansed, estimator, CATE_CI) %>%
  pivot_wider(
    names_from  = estimator,
    values_from = CATE_CI
  ) %>%
  arrange(desc(`AIPW`))

print(cate_wide)

# Reverting back to single processing: -----------------------------------------

stopCluster(cl)
registerDoSEQ()


