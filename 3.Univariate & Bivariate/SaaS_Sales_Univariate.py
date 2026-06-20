import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Univariate:

    # Load Dataset
    dataset = pd.read_csv(
        r"D:\python\Machine Learning and Data Science Capstone Project\2.Data preprocessing\cleaned_saas_sales.csv"
    )

    # ==================================================
    # Frequency Table for Categorical Variables
    # ==================================================
    @staticmethod
    def freq_table(dataset):

        categorical = dataset.select_dtypes(
            include=["object", "string"]
        ).columns

        result = {}

        for columnname in categorical:

            freq = dataset[columnname].value_counts()

            freqTable = pd.DataFrame({
                "Unique_Values": freq.index,
                "Frequency": freq.values,
                "Relative Frequency": freq.values / len(dataset),
                "Cumulative Frequency": freq.cumsum(),
                "Percentage": (freq.values / len(dataset)) * 100
            })

            result[columnname] = freqTable

        return result

    # ==================================================
    # Descriptive Statistics for Numerical Variables
    # ==================================================
    @staticmethod
    def descriptive_statistics(dataset):

        numeric = dataset.select_dtypes(
            include=["float64", "int64"]
        ).columns

        descriptive = pd.DataFrame(
            index=[
                "Mean",
                "Median",
                "Mode",
                "Q1:25%",
                "Q2:50%",
                "Q3:75%",
                "IQR",
                "1.5 Rule",
                "Lower Limit",
                "Upper Limit",
                "Outlier Count",
                "Standard Deviation",
                "Minimum",
                "Maximum",
                "Frequency Counts"
            ],
            columns=numeric
        )

        for cols in numeric:

            # Mean
            descriptive.loc["Mean", cols] = dataset[cols].mean()

            # Median
            descriptive.loc["Median", cols] = dataset[cols].median()

            # Mode
            descriptive.loc["Mode", cols] = dataset[cols].mode()[0]

            # Quartiles
            descriptive.loc["Q1:25%", cols] = dataset[cols].quantile(0.25)

            descriptive.loc["Q2:50%", cols] = dataset[cols].quantile(0.50)

            descriptive.loc["Q3:75%", cols] = dataset[cols].quantile(0.75)

            # IQR
            descriptive.loc["IQR", cols] = (
                descriptive.loc["Q3:75%", cols]
                - descriptive.loc["Q1:25%", cols]
            )

            # 1.5 Rule
            descriptive.loc["1.5 Rule", cols] = (
                1.5 * descriptive.loc["IQR", cols]
            )

            # Lower Limit
            descriptive.loc["Lower Limit", cols] = (
                descriptive.loc["Q1:25%", cols]
                - descriptive.loc["1.5 Rule", cols]
            )

            # Upper Limit
            descriptive.loc["Upper Limit", cols] = (
                descriptive.loc["Q3:75%", cols]
                + descriptive.loc["1.5 Rule", cols]
            )

            # Outliers
            lower = descriptive.loc["Lower Limit", cols]

            upper = descriptive.loc["Upper Limit", cols]

            outliers = dataset[
                (dataset[cols] < lower) |
                (dataset[cols] > upper)
            ][cols]

            descriptive.loc["Outlier Count", cols] = len(outliers)

            # Standard Deviation
            descriptive.loc["Standard Deviation", cols] = (
                dataset[cols].std()
            )

            # Minimum
            descriptive.loc["Minimum", cols] = dataset[cols].min()

            # Maximum
            descriptive.loc["Maximum", cols] = dataset[cols].max()

            # Frequency Counts
            descriptive.loc["Frequency Counts", cols] = str(
                dataset[cols].value_counts().head().to_dict()
            )

        return descriptive

    # ==================================================
    # Visualization Plots
    # ==================================================
    @staticmethod
    def plots(dataset):

        numeric = dataset.select_dtypes(
            include=["float64", "int64"]
        ).columns

        for cols in numeric:

            plt.figure(figsize=(16, 4))

            # Histogram
            plt.subplot(1, 3, 1)
            sns.histplot(dataset[cols], kde=False)
            plt.title(f"Histogram - {cols}")

            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(x=dataset[cols])
            plt.title(f"Boxplot - {cols}")

            # KDE Plot
            plt.subplot(1, 3, 3)
            sns.kdeplot(dataset[cols], fill=True)
            plt.title(f"KDE Plot - {cols}")

            # Adjust Layout
            plt.tight_layout()

            # Show Plot
            plt.show()
