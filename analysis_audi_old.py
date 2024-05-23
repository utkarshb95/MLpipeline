# import necessary libraries
import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None

def data_load():
    # use glob to get all the csv files in the folder
    path = "data/audi_etron/"
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    dfs = []
    # loop over the list of csv files
    for i, f in enumerate(csv_files):
        drive = int(f.split("Drive")[1].replace(".csv", ""))
        # read the csv file
        df = pd.read_csv(f)
        df["exp_no"] = f
        df["drive_no"] = drive
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df[df["R1L First Name"] != "Test"]
    return df

def preprocessing(df):
    selection = [
        "R1L Weight (lbs)",
        "Outside Temp (C)",
        "Cabin Temp (C)",
        "R1L First Name",
        "R1L Sex",
        "exp_no",
        "drive_no"
    ]
    target = "R1L Target OTS (-3=verycold to 3=verywarm)"

    df = df[selection + [target]]
    df = df.dropna()

    df["male"] = df["R1L Sex"].apply(lambda x: 1 if x == "Male" else 0)
    df["female"] = df["R1L Sex"].apply(lambda x: 1 if x == "Female" else 0)
    sub_df = df.groupby("exp_no").nunique()[target].reset_index()
    change_setpoint = sub_df[sub_df[target] > 1]["exp_no"].values
    other = sub_df[sub_df[target] == 1]["exp_no"].values
    
    return df, change_setpoint, other

def cross_validate(df, features, target, change_setpoint, other, verbose=0):
    accs = []
    for test_exp in np.concatenate((change_setpoint, other)):
        sub_df = df[df["exp_no"].isin(np.concatenate((change_setpoint, other)))]
        model = LogisticRegression(random_state=0, class_weight="balanced")
        pipe = make_pipeline(StandardScaler(), model)
        test_set = True
        if test_set:
            pipe.fit(
                sub_df[~sub_df["exp_no"].isin([test_exp])][features],
                sub_df[~sub_df["exp_no"].isin([test_exp])][target],
            )
        else:
            pipe.fit(sub_df[features], sub_df[target])

        if test_set:
            test_target = df[df["exp_no"].isin([test_exp])][target]
            test_prediction = pipe.predict(df[df["exp_no"].isin([test_exp])][features])
        else:
            test_target = df[target]
            test_prediction = pipe.predict(df[features])
        if verbose:
            print(
                "Accuracy score: ",
                accuracy_score(test_target, test_prediction),
                "Experiment no: ",
                test_exp,
            )
            print("Predictions: ", test_prediction)
            print("Target: ", test_target.values)
        accs.append(accuracy_score(test_target, test_prediction))
    print(features)
    print("Average accuracy: ", np.mean(accs))
    print("Rounded acc: ", len([i for i in accs if i > 0.3]) / len(accs))


def train(df, features, target, change_setpoint, other, test_exclude=0, verbose=0):

    sub_df = df[df["exp_no"].isin(np.concatenate((change_setpoint, other)))]
    model = LogisticRegression(random_state=0, class_weight="balanced")
    pipe = make_pipeline(StandardScaler(), model)

    if test_exclude != 0:
        print("Training on drives: ", set(sub_df[~sub_df["drive_no"].isin([test_exclude])]["drive_no"]))
        pipe.fit(
            sub_df[~sub_df["drive_no"].isin([test_exclude])][features],
            sub_df[~sub_df["drive_no"].isin([test_exclude])][target],
        )
    else:
        print("Training on drives: ", set(sub_df["drive_no"]))
        pipe.fit(sub_df[features], sub_df[target])
    
    test_target = df[target]
    test_prediction = pipe.predict(df[features])
    score = accuracy_score(test_target, test_prediction)
    print("Features: ", features)
    print("Training accuracy: {:.2f}%".format(score*100))
    return pipe

def test(df, features, target, pipe, test_exp, verbose=0):

    test_target = df[df["drive_no"].isin([test_exp])][target]
    test_prediction = pipe.predict(df[df["drive_no"].isin([test_exp])][features])
    acc = accuracy_score(test_target, test_prediction)
    # we need to import scaler object for test
        
    if verbose:
        print("Test accuracy on drive {} is {:.2f}%.".format(test_exp, acc)   
        )
        # print("Predictions: ", test_prediction)
        # print("Target: ", test_target.values)
    return acc

# def init():
#     model_path = Model.get_model_path(model_name="sklearn_logistic_regression.pkl")
#     model = joblib.load(model_path)

def main():
    # Load and process data
    df = data_load()
    df, change_setpoint, other = preprocessing(df)

    # Performance for different feature combinations (cross validation)
    features = ["male", "female", "R1L Weight (lbs)"]
    target = "R1L Target OTS (-3=verycold to 3=verywarm)"
    cross_validate(df, features, target, change_setpoint, other)
    features = ["male", "female", "R1L Weight (lbs)", "Outside Temp (C)"]
    cross_validate(df, features, target, change_setpoint, other, verbose=0)
    features = ["male", "female", "R1L Weight (lbs)", "Cabin Temp (C)"]
    cross_validate(df, features, target, change_setpoint, other, verbose=0)
    features = ["male", "female", "R1L Weight (lbs)", "scaled_cabin_temp"]
    # cross_validate(df, features, target, change_setpoint, other, verbose=0)
    # features = [
    #     "male",
    #     "female",
    #     "R1L Weight (lbs)",
    #     "scaled_cabin_temp",
    #     "Outside Temp (C)",
    # ]
    # cross_validate(df, features, target, change_setpoint, other, verbose=0)

    # Train on whole dataset
    features = ["male", "female", "R1L Weight (lbs)", "Outside Temp (C)", "Cabin Temp (C)"]
    target = "R1L Target OTS (-3=verycold to 3=verywarm)"
    pipe = train(df, features, target, change_setpoint, other, test_exclude=6)

    # Test on a single drive
    acc = test(df, features, target, pipe, test_exp=6, verbose=1)

     # Save Model
    # model_name = "sklearn_logistic_regression.pkl"
    # joblib.dump(value=acc, filename=model_name)

main()