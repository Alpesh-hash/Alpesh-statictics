"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
#from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

df = pd.read_csv("data.csv")
print(df)



print(df)
def plot_relational_plot(df):
    fig, ax = plt.subplots()
    plt.savefig('relational_plot.png')
    #alpesh
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots()
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    fig, ax = plt.subplots()
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    mean =
    stddev =
    skew =
    excess_kurtosis =
    return mean, stddev, skew, excess_kurtosis 


def preprocessing(df):
    # You should preprocess your data in this function and
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.
    
    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Convert object columns to numeric, forcing errors to NaN for proper cleaning
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # 3. Remove rows and columns with any NaN values
    df = df.dropna(axis=0)  # Remove rows with NaN values
    df = df.dropna(axis=1)  # Remove columns with NaN values

    # 4. Rename columns for clarity, indicating temperature and uncertainty
    df.rename(columns={
        'Monthly Anomaly': 'Monthly Temperature Anomaly',
        'Monthly Unc.': 'Monthly Temperature Uncertainty',
        'Annual Anomaly': 'Annual Temperature Anomaly',
        'Annual Unc': 'Annual Temperature Uncertainty',
        'Five-Year Anomaly': '5-Year Temperature Anomaly',
        'Five-Year Unc.': '5-Year Temperature Uncertainty',
        'Ten-Year Anomaly': '10-Year Temperature Anomaly',
        'Ten-Year Unc.': '10-Year Temperature Uncertainty',
        'Twenty-Year Anomaly': '20-Year Temperature Anomaly',
        'Twenty-year Unc.': '20-Year Temperature Uncertainty'
    }, inplace=True)

    # 5. Display quick insights
    print("\n--- Data Head ---")
    print(df.head())
    print("\n--- Data Tail ---")
    print(df.tail())
    print("\n--- Data Description ---")
    print(df.describe())
    print("\n--- Data Correlation ---")
    print(df.corr())


    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    print('The data was right/left/not skewed and platy/meso/leptokurtic.')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = '<your chosen column for analysis>'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return 


if __name__ == '__main__':
    main() 
