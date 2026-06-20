import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class SAAS_SALES:

    def __init__(self):

        # Load dataset
        self.dataset = pd.read_csv(
            r"D:\python\Machine Learning and Data Science Capstone Project\1.Data collection\SaaS-Sales.csv"
        )

        # Drop unnecessary columns
        drop_cols = [
            'Row ID',
            'Order ID',
            'Contact Name',
            'Customer ID',
            'License',
            'Date Key'
        ]

        self.dataset = self.dataset.drop(columns=drop_cols)

        # Convert date column
        self.dataset['Order Date'] = pd.to_datetime(
            self.dataset['Order Date']
        )

        # Numeric columns
        num_cols = ['Sales', 'Quantity', 'Discount', 'Profit']

        # Fill numeric missing values
        num_imputer = SimpleImputer(strategy='median')

        self.dataset[num_cols] = num_imputer.fit_transform(
            self.dataset[num_cols]
        )

        # Categorical columns
        cat_cols = self.dataset.select_dtypes(
            include=['object', 'string']
        ).columns

        # Fill categorical missing values
        for col in cat_cols:
            self.dataset[col] = self.dataset[col].fillna(
                self.dataset[col].mode()[0]
            )

        # Feature engineering
        self.dataset['Year'] = self.dataset['Order Date'].dt.year
        self.dataset['Month'] = self.dataset['Order Date'].dt.month
        self.dataset['Day'] = self.dataset['Order Date'].dt.day

        # Drop original date column
        self.dataset = self.dataset.drop(columns=['Order Date'])

    # Method to return processed dataset
    def get_data(self):
        return self.dataset