# Meal Prep Time - Exercise
Contains a **Jupyter Notebook** with a preliminary analysis of the data provided in addition to a Docker-based **Web App** which
processes the provided .csv files, trains a model based on parameters specified by the user, and returns performance metrics for
the specified model *(uses python/flask)*. A **summary of notes** is also provided.

## Running the Jupyter Notebook:
cd into the **meal-prep-time** repository and run *jupyter notebook* from your console.
Navigate to **notebook.ipynb** from the GUI and follow the script accordingly.

## Running the Docker container:
cd into the **meal-prep-time** repository, then cd into the **app** folder.
With Docker running, run *docker-compose build* followed by *docker-compose up*.

### Testing:
With the container started, open a new console instance and enter:

*curl localhost:8000/model/type*

where *type* can be specified as **xgbr** (for *XGBRegressor*) or **rfr** (for *RandomForestRegressor*).

If successful, the above request will return a string which contains the model type, r2 score,
parameter configurations, and optimized parameters based on GridSearchCV.

Once finished, the container can be stopped with *ctrl-c*.
