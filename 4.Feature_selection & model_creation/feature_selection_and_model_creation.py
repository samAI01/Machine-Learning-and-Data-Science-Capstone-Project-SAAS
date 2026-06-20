import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


class RFE_reg():

    # =========================
    # R2 Prediction
    # =========================
    @staticmethod
    def r2_prediction(regressor, X_test, y_test):
        y_pred = regressor.predict(X_test)
        return r2_score(y_test, y_pred)

    # =========================
    # Models
    # =========================
    @staticmethod
    def Linear(X_train, y_train, X_test, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return RFE_reg.r2_prediction(model, X_test, y_test)

    @staticmethod
    def linear_svr_model(X_train, y_train, X_test, y_test):
        model = LinearSVR(random_state=0, max_iter=20000, tol=1e-3)
        model.fit(X_train, y_train)
        return RFE_reg.r2_prediction(model, X_test, y_test)

    @staticmethod
    def svm_NL(X_train, y_train, X_test, y_test):
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        return RFE_reg.r2_prediction(model, X_test, y_test)

    @staticmethod
    def Decision(X_train, y_train, X_test, y_test):
        model = DecisionTreeRegressor(random_state=0)
        model.fit(X_train, y_train)
        return RFE_reg.r2_prediction(model, X_test, y_test)

    @staticmethod
    def random(X_train, y_train, X_test, y_test):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return RFE_reg.r2_prediction(model, X_test, y_test)

    # =========================
    # Result DataFrame
    # =========================
    @staticmethod
    def rfe_regression(acclin, accsvml, accsvmnl, accdes, accrf):

        df = pd.DataFrame(
            index=[
                'RFE_LinearRegression',
                'RFE_LinearSVR',
                'RFE_DecisionTree',
                'RFE_RandomForest'
            ],
            columns=['Linear', 'LinearSVR', 'SVMnl', 'Decision', 'Random']
        )

        for i, idx in enumerate(df.index):
            df.loc[idx, 'Linear'] = acclin[i]
            df.loc[idx, 'LinearSVR'] = accsvml[i]
            df.loc[idx, 'SVMnl'] = accsvmnl[i]
            df.loc[idx, 'Decision'] = accdes[i]
            df.loc[idx, 'Random'] = accrf[i]

        return df

    # =========================
    # MAIN FUNCTION
    # =========================
    @staticmethod
    def Result():

        # -------------------------
        # Load dataset
        # -------------------------
        dataset = pd.read_csv(
            r"D:\python\Machine Learning and Data Science Capstone Project\2.Data preprocessing\cleaned_saas_sales.csv"
        )

        X = dataset.drop("Profit", axis=1)
        y = dataset["Profit"]

        # -------------------------
        # Train-test split
        # -------------------------
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        # -------------------------
        # One-hot encoding
        # -------------------------
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        cat_cols = [
            'Country', 'City', 'Region', 'Subregion',
            'Customer', 'Industry', 'Segment', 'Product'
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    cat_cols
                )
            ],
            remainder='passthrough'
        )

        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        feature_names = preprocessor.get_feature_names_out()

        # -------------------------
        # Scaling (IMPORTANT: before RFE)
        # -------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------------
        # Models used in RFE
        # -------------------------
        rfemodellist = [
            LinearRegression(),
            LinearSVR(random_state=0, max_iter=20000, tol=1e-3),
            DecisionTreeRegressor(random_state=0),
            RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        ]

        acclin, accsvml, accsvmnl, accdes, accrf = [], [], [], [], []

        # -------------------------
        # RFE LOOP
        # -------------------------
        for model in rfemodellist:

            print("\n" + "=" * 60)
            print(f"Running RFE using {model.__class__.__name__}")
            print("=" * 60)

            rfe = RFE(
                estimator=model,
                n_features_to_select=20,
                step=10
            )

            # ✔ FIX: RFE applied on SCALED data
            X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
            X_test_rfe = rfe.transform(X_test_scaled)

            print("X_train_rfe shape:", X_train_rfe.shape)
            print("X_test_rfe shape :", X_test_rfe.shape)
            selected_features = feature_names[rfe.support_]

            print("\nSelected Features:")
            for f in selected_features:
                print(f)

            # -------------------------
            # ✔ FIX: use RFE OUTPUT here
            # -------------------------
            acclin.append(
                RFE_reg.Linear(X_train_rfe, y_train, X_test_rfe, y_test)
            )

            accsvml.append(
                RFE_reg.linear_svr_model(X_train_rfe, y_train, X_test_rfe, y_test)
            )

            accsvmnl.append(
                RFE_reg.svm_NL(X_train_rfe, y_train, X_test_rfe, y_test)
            )

            accdes.append(
                RFE_reg.Decision(X_train_rfe, y_train, X_test_rfe, y_test)
            )

            accrf.append(
                RFE_reg.random(X_train_rfe, y_train, X_test_rfe, y_test)
            )

        # -------------------------
        # Results table
        # -------------------------
        result = RFE_reg.rfe_regression(
            acclin, accsvml, accsvmnl, accdes, accrf
        )

        print("\nFINAL RESULTS:\n")
        print(result)

        # -------------------------
        # Best model
        # -------------------------
        best_idx = result.stack().idxmax()

        print("\nBest RFE Method :", best_idx[0])
        print("Best Model :", best_idx[1])
        print("Best R2 :", result.loc[best_idx])

        return result