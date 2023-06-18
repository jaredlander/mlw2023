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

train |> select(Home) |> head(n=15) |> View()

train |> 
    head(n=15) |> 
    model.matrix( ~ Home, data=_)


library(recipes)

recipe(Status ~ ., data=credit) |> 
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
    