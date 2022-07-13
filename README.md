# Deploying a Machine Learning Model on Heroku with FastAPI

This is a project about deoplying a machine learning model using Census data on Heroku with FastAPI

### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate census_dev
```

### Cookie cutter
In order to save time from writing boilerplate code cookie cutter is used. Just run the cookiecutter and enter the required information, and a new component will be created including the `conda.yml` file, the `MLproject` file as well as the script. You can then modify these as needed, instead of starting from scratch. For example:

```bash
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called ``basic_cleaning`` under the directory ``src`` with the following structure:

```bash
> ls src/basic_cleaning/
conda.yml  MLproject  run.py
```

You can now modify the script (``run.py``), the conda environment (``conda.yml``) and the project definition 
(``MLproject``) as you please.

The script ``run.py`` will receive the input parameters ``parameter1``, ``parameter2``,
``parameter3`` and it will be called like:

```bash
> mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
```

### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``upload`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=upload
```
If you want to run the ``upload`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=upload,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> gradient_boosting_classifier -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.gradient_boosting_classifier.n_estimators=10 etl.min_price=50"
```

To split the data and upload to wandb

```bash
> mlflow run . -P steps=data_split
```

To train the model

```bash
> mlflow run . -P steps=train_gradient_boosting
```

To check the model performance in different slices of the data

```bash
> mlflow run . -P steps=slice_performance_check
```
To run unit tests on model

```bash
> mlflow run . -P steps=model_check
```

To create the api locally

```bash
> mlflow run . -P steps=api_creation
```

### CI/CD

python-app.yaml is used for continuous integration. With every push flake8 and all the unit tests are checked using pytest. The model then delpoyed in Heroku automitacally.
