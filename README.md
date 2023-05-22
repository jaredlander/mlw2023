
- <a href="#machine-learning-in-r" id="toc-machine-learning-in-r">Machine
  Learning in R</a>
  - <a href="#setup" id="toc-setup">Setup</a>
  - <a href="#git" id="toc-git">Git</a>
  - <a href="#docker" id="toc-docker">Docker</a>
  - <a href="#codespaces" id="toc-codespaces">Codespaces</a>
  - <a href="#code" id="toc-code">Code</a>
  - <a href="#workshop-plan" id="toc-workshop-plan">Workshop Plan</a>
    - <a href="#preparing-data-for-the-modeling-process"
      id="toc-preparing-data-for-the-modeling-process">Preparing Data for the
      Modeling Process</a>
    - <a href="#eda-and-feature-engineering"
      id="toc-eda-and-feature-engineering">EDA and Feature Engineering</a>
    - <a href="#model-fitting-and-parameter-tuning"
      id="toc-model-fitting-and-parameter-tuning">Model Fitting and Parameter
      Tuning</a>
    - <a href="#deploying-the-model-into-production"
      id="toc-deploying-the-model-into-production">Deploying the Model into
      Production</a>

<!-- README.md is generated from README.Rmd. Please edit that file -->

# Machine Learning in R

<!-- badges: start -->
<!-- badges: end -->

Thanks for attending my sessions at [Machine Learning Week
2023](https://www.predictiveanalyticsworld.com/machinelearningweek/).
This repo will hold code we write in [Machine Learning in
R](https://www.predictiveanalyticsworld.com/machinelearningweek/workshops/machine-learning-with-r/)
workshop.

## Setup

For this course you need a recent version of R. Anything greater than
4.0 is good but 4.3 is even better. I also highly recommend using your
IDE/code editor of choice. Most people use either
[RStudio](https://www.rstudio.com/products/rstudio/) or [VS
Code](https://code.visualstudio.com/) with [R Language
Extensions](https://code.visualstudio.com/docs/languages/r).

After you have R and your favorite editor installed, you should install
the packages needed today with the following line of code.

``` r
install.packages(c(
  'here', 'markdown', 'rmarkdown', 'knitr', 'tidyverse', 'ggthemes', 'ggridges', 
  'tidymodels', 'coefplot', 'glmnet', 'xgboost', 'vip', 'DiagrammeR', 'here', 
  'DBI', 'themis', 'vetiver', 'fable', 'tsibble', 'echarts4r', 'leaflet', 
  'leafgl', 'leafem', 'tictoc'
))
```

## Git

If you are comfortable with git, you can clone this repo and have the
project structure.

``` sh
git clone https://github.com/jaredlander/mlw2023.git
```

## Docker

If you are having trouble installing R or the packages, but are
comfortable with Docker, you can pull the Docker image using the
following command in your terminal.

``` sh
docker pull jaredlander/r_ml_workshop:4.3.0
```

You can run the container with the following command which will also
mount a folder as a volume for you to use.

``` sh
docker run -it --rm --name rstudio_ml -e PASSWORD=password -e ROOT=true -p 8787:8787 -v $PWD/workshop:/home/rstudio/workshop  jaredlander/r_ml_workshop:4.3.0
```

## Codespaces

The Docker image should work natively in [GitHub
Codespaces](https://github.com/features/codespaces) so you can run a
remote instance of VS Code with all the packages ready to go. You can
theoretically even launch RStudio from within the VS Code instance,
though I haven’t figured that out yet.

## Code

Throughout the class I will be pushing code to this repo in case you
need to catch up. Most, if not all, will be in the `code` folder.

## Workshop Plan

Modern statistics has become almost synonymous with machine learning, a
collection of techniques that utilize today’s incredible computing
power. A combination of supervised learning (regression-like models) and
unsupervised learning (clustering), the field is supported by theory,
yet relies upon intelligent programming for implementation.

In this training session we will work through the entire process of
training a machine learning model in R. Starting with the scaffolding of
cross-validation, onto exploratory data analysis, feature engineering,
model specification, parameter tuning and model selection. We then take
the finished model and deploy it as an API in a Docker container for
production use.

We will make extensive use the `{tidymodels}` framework of R packages.

### Preparing Data for the Modeling Process

The first step in a modeling project is setting up the evaluation loop
in order to properly define a model’s performance. To accomplish this we
will learn the following tasks:

1.  Load Data
2.  Create train and test sets from the data using the `{rsample}`
    package
3.  Create cross-validation set from the train set using the `{rsample}`
    package
4.  Define model evaluation metrics such as RMSE and logloss using the
    `{yardstick}` package

### EDA and Feature Engineering

Before we can fit a model we must first understand the model by
performing exploratory data analysis. After that we prepare the data
through feature engineering, also called preprocessing and data munging.
The primary steps we will learn include:

1.  Perform summary EDA with `{dplyr}`
2.  Visualize the data with `{ggplot2}`
3.  Balance the data with the `{themis}` package
4.  Impute or otherwise mark missing data with the `{recipes}` package
5.  Perform data transformations with the `{recipes}` package
    1.  Numeric centering and scaling
    2.  Collapse noisy categorical data
    3.  Handle new categorical values
    4.  Convert categorical data into dummy (or indicator) variables

### Model Fitting and Parameter Tuning

Now we can begin fitting models. This involves defining the type of
model, such as a penalized regression, random forest or boosted tree.
This has been simplified thanks to the parsnip and workflows packages.
Modern machine learning has essentially become an excercise in
brute-forcing over tuning parameters, which we will do by combining the
dials and tune package with the previously created cross-validation set.

1.  Define the model structure with the `{parsnip}` package
2.  Set tuning parameter candidates with the `{dials}` package
3.  Iterate over the tuning parameter candidates using the `{tune}`
    package to perform cross-validation
4.  Identify the best model fit with the `{yardstick}` package

### Deploying the Model into Production

After we build various machine learning models we need to make them
accessible to others. We use the plumber package to expose our model as
a REST API that can be hosted in a Docker container.

1.  Make predictions using the `{workflows}` package
2.  Convert the model to an API using the `{plumber}` package
3.  Bundle the model object and API code into a Docker container
4.  Serve that container and use curl to make perform predictions
