import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.formula.api import ols

class Bivariate:
    pass
    dataset = pd.read_csv(
    r"D:\python\Machine Learning and Data Science Capstone Project\2.Data preprocessing\cleaned_saas_sales.csv"
)
    # Load Dataset
    @staticmethod 
    def corr(dataset):   
        num_cols=dataset.select_dtypes(include=["int64","float64"]).columns 
        corr_matrix=dataset[num_cols].corr()
        print(corr_matrix)
        
        #heatmap
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f")
        plt.title("correlation matrix")
        plt.show()
    
        # Pair Plot
        num_cols = dataset.select_dtypes(include=["int64","float64"]).columns
        sns.pairplot(dataset[num_cols])
    
        # scatter plot 
        x_col="Sales"
        y_col="Profit"
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=dataset,x=x_col,y=y_col)
        plt.title(f"{x_col} vs {y_col}")
        plt.show()
    
        # boxplot
        cat_cols="Country"
        num_cols="Sales"
        plt.figure(figsize=(10,5))
        sns.boxplot(data=dataset,x=cat_cols,y=num_cols)
        plt.xticks(rotation=90)
        plt.title(f"{cat_cols} vs {num_cols}")
        plt.show()


    # Variance inflation factor
    @staticmethod 
    def calc_vif(x):
        vif=pd.DataFrame()
        vif["variables"]=x.columns
        vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
        return vif

    # independent sample-unpaired test
    @staticmethod
    def test(dataset):
        usa_profit=dataset[dataset["Country"]=="United States"]["Profit"].dropna()
        germany_profit=dataset[dataset["Country"]=="Germany"]["Profit"].dropna()
        t_stat,p_value=ttest_ind(usa_profit,germany_profit,equal_var=False)
        if p_value < 0.05:
            print("Reject Null Hypothesis")
            print("t_stat",t_stat,"p_value",p_value)
        else:
            print("Fail to Reject Null Hypothesis")
        return t_stat,p_value                       

    # ANOVA (Analysis of Variance)
    @staticmethod
    def anova(dataset):
        usa_profit=dataset[dataset["Country"]=="United States"]["Profit"]
        germany_profit=dataset[dataset["Country"]=="Germany"]["Profit"]
        ireland_profit=dataset[dataset["Country"]=="Ireland"]["Profit"]
        f_stat,p_value=f_oneway(usa_profit,germany_profit,ireland_profit)
        if p_value < 0.05:
           print("At least one group mean differs significantly")
        else:
           print("No significant difference among group means")
        return f_stat,p_value

    # One-Way ANOVA
    @staticmethod
    def anova_one_way(dataset):
        df=pd.DataFrame(dataset)
        model=ols("Profit ~ C(Country)",data=df).fit()
        anova_table=sm.stats.anova_lm(model,typ=2)
        return anova_table

