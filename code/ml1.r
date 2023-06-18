# Data ####

data(credit_data, package='modeldata')

credit <- tibble::as_tibble(credit_data)

head(credit_data)
head(credit)

library(dplyr)

set.seed(456)
fake_customers <- credit |> 
    select(-Status) |> 
    slice_sample(n=10)

fake_customers


# split the data

library(rsample)

set.seed(82)
data_split <- initial_split(
    credit,
    prop=0.8,
    strata='Status'
)

data_split

train <- training(data_split)
test <- testing(data_split)

train
test

# EDA ####

library(ggplot2)

ggplot(train, aes(x=Status)) + geom_bar()
ggplot(train, aes(x=Age, y=Amount, color=Status))  + 
    geom_point()

# Feature Engineering ####

train |> count(Home)

# train |> select(Home) |> head(n=15) |> View()

train |> 
    head(n=15) |> 
    model.matrix( ~ Home, data=_)


library(recipes)

rec1 <- recipe(Status ~ ., data=credit) |> 
    themis::step_downsample(Status, under_ratio=1.2) |> 
    # remove inputs with near zero variance
    step_nzv(all_predictors()) |> 
    step_filter_missing(all_predictors(), threshold=0.5) |> 
    step_impute_knn(all_numeric_predictors()) |> 
    step_unknown(
        all_nominal_predictors(),
        new_level='missing'
    ) |> 
    step_normalize(all_numeric_predictors()) |> 
    # step_center() |> step_scale()
    step_other(all_nominal_predictors(), other='misc') |> 
    step_novel(
        all_nominal_predictors(), new_level='unseen'
    ) |> 
    step_dummy(
        all_nominal_predictors(),
        one_hot=TRUE
    )

rec1
rec1 |> prep()
rec1 |> prep() |> bake(new_data=NULL)
# rec1 |> prep() |> bake(new_data=NULL) |> View()

# Define the Model ####

library(parsnip)

linear_reg()
linear_reg() |> set_engine('lm')
linear_reg() |> set_engine('glmnet')
linear_reg() |> set_engine('stan')
linear_reg() |> set_engine('keras')
linear_reg() |> set_engine('spark')

rand_forest()
rand_forest() |> set_mode('regression')
rand_forest() |> set_mode('classification')
rand_forest() |> set_mode('classification') |> 
    set_engine('spark')

rand_forest(mode='classification') |> set_engine('spark')


boost_tree(mode='classification') |> set_engine('xgboost')


spec1 <- boost_tree(
    mode='classification',
    trees=100,
    tree_depth=4
) |> 
    set_engine('xgboost')
spec1

# workflows ####

library(workflows)

workflow(preprocessor=rec1, spec=spec1)
flow1 <- workflow() |> 
    add_recipe(rec1) |> 
    add_model(spec1)

# Fit the Model ####

fit1 <- fit(flow1, data=train)
fit1
fit1 |> summary()
fit1 |> extract_fit_engine() |> vip::vip()

spec2 <- boost_tree(
    mode='classification',
    trees=400, 
    tree_depth=4
)

flow2 <- flow1 |> 
    update_model(spec2)
flow2

fit2 <- fit(flow2, data=train)

fit2 |> extract_fit_engine() |> vip::vip()

# Evaluate the Model ####

# regression:
# - root mean squared error
# - mean absolute error

# classification
# - accuracy
# - log loss
# - AUC

library(yardstick)

loss_fn <- metric_set(roc_auc, mn_log_loss, accuracy)
loss_fn <- metric_set(roc_auc)
loss_fn

# Cross-Validation ####

# https://twitter.com/chrisalbon/status/1664318913470169088

set.seed(43)
theCV <- vfold_cv(
    data=train,
    v=5,
    repeats=1,
    strata='Status'
)
theCV
theCV$splits[[1]]
data_split
theCV$splits[[1]] |> assessment()
theCV$splits[[1]] |> testing()
testing


library(tune)

val1 <- fit_resamples(
    flow1,
    resamples=theCV,
    metrics=loss_fn
)

val2 <- fit_resamples(
    flow2,
    resamples=theCV,
    metrics=loss_fn
)

val1
val1$.metrics[[1]]
val1$.metrics[[2]]

val1 |> collect_metrics()
val2 |> collect_metrics()

# Tuning ####

spec3 <- boost_tree(
    mode='classification',
    trees=tune(),
    tree_depth=tune(),
    sample_size=0.8
)
spec2
spec3

flow3 <- flow2 |> 
    update_model(spec3)
flow3

library(dials)

flow3 |> extract_parameter_set_dials()
flow3 |> extract_parameter_set_dials() |> 
    pull(object)

params3 <- flow3 |> 
    extract_parameter_set_dials() |> 
    update(
        trees=trees(c(25, 500)),
        tree_depth=tree_depth(c(2, 6))
    )
params3 |> pull(object)


set.seed(24)
grid3 <- grid_latin_hypercube(params3, size=100)
grid3

grid3 |> 
    group_nest(tree_depth) |> 
    mutate(data=purrr::map(data, \(x) arrange(x, trees))) |> 
    pull(data)

library(tictoc)

library(parallel)
library(doFuture)

registerDoFuture()
cl <- makeCluster(5)
plan(cluster, workers=cl)

options(tidymodels.dark=TRUE)
tic(msg='tuning 3')
tuned3 <- tune_grid(
    flow3,
    resamples=theCV,
    grid=grid3,
    metrics=loss_fn,
    control=control_grid(
        verbose=TRUE,
        allow_par=TRUE
    )
)
toc(log=TRUE)

tuned3
tuned3$.metrics[[1]]

tuned3 |> collect_metrics()

tuned3 |> autoplot()

tuned3 |> show_best(n=10)

tuned3 |> select_best()
tuned3 |> select_by_one_std_err(trees, tree_depth)
best_params3 <- tuned3 |> 
    select_by_one_std_err(trees, tree_depth)

mod3 <- flow3 |> finalize_workflow(parameters=best_params3)
mod3

fit3 <- mod3 |> fit(data=train)
fit3


preds <- predict(fit3, new_data=test)
preds

preds3_prob <- predict(fit3, new_data=test, type='prob')
preds3_prob

combined3 <- test |> 
    select(Status) |> 
    bind_cols(preds3_prob)
combined3

roc_auc(combined3, truth=Status, .pred_bad)

accuracy(
    test |> 
        select(Status) |> 
        bind_cols(preds),
    Status, .pred_class
)

last3 <- last_fit(
    object=mod3,
    split=data_split
)
last3
last3$.metrics[[1]]

# Fit Model on All Data ####

final_fit <- fit(mod3, data=credit)

fake_customers[1, ]

predict(final_fit, new_data=fake_customers[1, ])
predict(final_fit, new_data=fake_customers[1, ], type='prob')


new_customer <- tibble::tibble(
    Seniority=12,
    Home='owner',
    Time=12,
    Marital='married',
    Records='no',
    Job='circus clown',
    Expenses=50,
    Income=200,
    Assets=0,
    Debt=150,
    Age=50,
    Amount=2400,
    Price=3000
)
predict(final_fit, new_data=new_customer, type='prob')

predict(final_fit, new_data=fake_customers, type='prob')
?modeldata::credit_data

# Deploy Model ####

library(vetiver)

small_fit <- final_fit |> butcher::butcher()

saveRDS(small_fit, 'scoring/small_fit.rds')

new_customer |> jsonlite::toJSON()

# https://rstudio.lander.ai/s/d5de88c86696831d6b7ba/p/2ac0a587/predict?new_data=%5B%7B%22Seniority%22%3A12%2C%22Home%22%3A%22owner%22%2C%22Time%22%3A12%2C%22Marital%22%3A%22married%22%2C%22Records%22%3A%22no%22%2C%22Job%22%3A%22circus%20clown%22%2C%22Expenses%22%3A50%2C%22Income%22%3A200%2C%22Assets%22%3A0%2C%22Debt%22%3A150%2C%22Age%22%3A50%2C%22Amount%22%3A2400%2C%22Price%22%3A3000%7D%5D%20

# Digital Ocean Referral Code: https://m.do.co/c/72a2f8fbbfc9