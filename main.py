# We use Joblib to basically save the trained model to a file so that we can use it later without retraining.
# the program checks if a saved model file exists. If it does, it loads the model from that file using Joblib hence skipping the training process.
# If the file doesn't exist, it trains and saves the model for future use.

import os
import joblib
import pandas as pd 
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


MODEL_FILE = "model.pkl"        # in capslock cause its gonna be constant.
PIPELINE_FILE = "pipeline.pkl"   # we're pickling the pipeline file as well because the test data (or inference data) needs to be processed as well as it'll also have raw features.

def build_pipeline(num_attribs, cat_attribs):
    # 6. Pipeline for Numerical columns:
    num_pipeline = Pipeline([
            ("imputer",SimpleImputer()),
            ("standardizer", StandardScaler())
    ])

    # 7. Pipeline for categorical columns:
    cat_pipeline = Pipeline([
            ("encoder",OneHotEncoder(handle_unknown="ignore"))   # handle unknown handles all unknown values thrown at it by ignoring them
            # What 'handle_unknown' actually does is that when the transformer is trained on categories but during testing/production if a new category appears so then it'll not give error, it'll just ignore that and encode a '0' in all current existing categories that its trained on. 
    ])

    # 8. Building the full pipeline:
    full_pipeline = ColumnTransformer([
            ("num",num_pipeline, num_attributes),  # num pipeline will be applied to num_attributes which is list of columns
            ("cat",cat_pipeline, cat_attributes)   # cat pipeline will be applied to cat_attributes which is list of columns
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets train the model.

    # 1. Loading the dataset
    housing_data = pd.read_csv("House_Price_Prediction_Model/housing.csv")

    # 2. Create a stratified test set
    housing_data['income_cat'] = pd.cut(housing_data['median_income'],bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing_data, housing_data['income_cat']):
        housing_data.loc[test_index].drop("income_cat",axis=1).to_csv('House_Price_Prediction_Model/input.csv',index=False)
        housing = housing_data.loc[train_index].drop("income_cat",axis=1)

    housing_features = housing.drop("median_house_value",axis=1)
    housing_labels = housing['median_house_value'].copy()

    num_attributes = housing_features.drop("ocean_proximity",axis=1).columns.tolist() 
    cat_attributes = ['ocean_proximity']    

    pipeline = build_pipeline(num_attributes,cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    # ok so here in the line 63, I've just created a pipeline object that takes our numerical and categorical data, then in the next line I've made 'housing_prepared', which uses the earlier line by giving it the housing_features and telling that whatever fitting you'll perform, the transformation needs to be done into this housing_features. Then this final output is stored in the 'housing_prepared' variable.


    # now we'll train the model, since the pipeline's built.

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    # now since both are done, we'll pickle both the model and the pipeline using joblib.

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained, Congratulations!")


else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('House_Price_Prediction_Model/input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data["median_house_value"] = predictions
    input_data.to_csv("House_Price_Prediction_Model/output.csv", index=False)
    print("Inference complete. Results saved to output.csv")




    # accuracy of our predictions:
        # for this we'll have to take out the label and then our predictions, right?
        # for labels we'll take out 'median_house_value' column of the input-copy csv file.
        # for predictions we'll take out 'median_house_value' column of the output csv file.

    df_metrics_label = pd.read_csv('House_Price_Prediction_Model/input.csv')
    df_metrics_label = df_metrics_label["median_house_value"]

    df_metrics_prediction = pd.read_csv('House_Price_Prediction_Model/output.csv')
    df_metrics_prediction = df_metrics_prediction["median_house_value"]


    rmse = root_mean_squared_error(df_metrics_label,df_metrics_prediction)
    print("How far off the predicted price is on average (in '$'): ", rmse)
    



    # firstly we created input.csv using the test set data which will be used for testing then we'll store the predicted 
    # outcomes from input.csv in output.csv then we'll need to see how accurate we are, which'll be measured by comparing 
    # the two files. 



    # We can use another models here, neural networks, SVMs, etc.
    # We could also fine-tune our model (i.e by tweaking hyperparams):
        # RF-regressor has hyperparams such as: n_estimators, criterion, etc.
        # We could find the best hyperparams for an algorithm by using the sklearn class called 'GridSearchCV'. or another one is 'RandomizedSearchCV'.

    

    # Cross-Validation to improve the model's performance.