from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

class Tester:
    def __init__(self, trained_pipeline, session_id='drive_no'):
        self.trained_pipeline = trained_pipeline
        self.session_id = session_id

    # Inference on drives specified in test_exclude for testing
    def test(self, df, features, target, test_exclude=0, verbose=0):    
        if test_exclude != 0:
            test_df = df[df[self.session_id].isin([int(test_exclude)])]
            if verbose:
                print("Testing on drive: ", set(test_df[self.session_id]))
        else:
            print("No test session provided. Testing on the latest session.")
            test_exclude = df[self.session_id].tail(1)
            test_df = df[df[self.session_id].isin([int(test_exclude)])]
            if verbose:
                print("Testing on drive: ", set(test_df[self.session_id]))

        # Exclude the specified column if it exists using isin
        input_columns = test_df[features].loc[:, ~test_df[features].columns.isin([self.session_id])]

        # Evaluate model performance
        test_prediction = self.trained_pipeline.predict(input_columns)
        mse = mean_squared_error(test_df[target], test_prediction)
        r2 = r2_score(test_df[target], test_prediction)
        if verbose:
            print("Test results\n")
            print("Mean Squared Error:", mse)
            print("R-squared:", r2)

        return mse, r2