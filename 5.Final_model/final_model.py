import pandas as pd 
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class saas_prediction:
    @staticmethod
    def ref_LinearRegression(X_scaled,Y,feature_names):
        # RFE using Linear Regression
        rfe=RFE(estimator=LinearRegression(),n_features_to_select=20,step=10)
        X_rfe=rfe.fit_transform(X_scaled,Y)

        selected_features=np.array(feature_names)[rfe.support_]
        print("\nSelected Features:")
        for feature in selected_features:
            print(feature)

        # final pridiction model
        final_model=DecisionTreeRegressor(random_state=0)
        final_model.fit(X_rfe,Y)  

        # training score
        train_pred=final_model.predict(X_rfe)
        train_r2=r2_score(Y,train_pred)
        print("\nTraining R² Score :", train_r2)

        return  rfe, final_model, selected_features
        
    @staticmethod
    def main():

        #  load dataset
        dataset = pd.read_csv(r"D:\python\Machine Learning and Data Science Capstone Project\2.Data preprocessing\cleaned_saas_sales.csv")

        #input/output 
        X=dataset.drop("Profit",axis=1)
        Y=dataset["Profit"]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        
        # categorical columns 
        cat_cols = [
            'Country',
            'City',
            'Region',
            'Subregion',
            'Customer',
            'Industry',
            'Segment',
            'Product'
        ]

        # one hot encoding 
        preprocessor=ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown='ignore',
                        sparse_output=False),cat_cols)],remainder="passthrough",force_int_remainder_cols=False)
        X_encoded=preprocessor.fit_transform(X)
        feature_names=preprocessor.get_feature_names_out()
        print("Encoded Shape:",X_encoded.shape)

        # Scalling
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X_encoded)

        # Train model
        rfe, final_model, selected_features=saas_prediction.ref_LinearRegression(X_scaled,Y,feature_names)

        
        # save everything
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("scaler", scaler),
            ("rfe", rfe),
            ("model", final_model)
        ])

        joblib.dump(pipeline, "saas_pipeline.pkl")
        print("\nModel Saved Successfully")















