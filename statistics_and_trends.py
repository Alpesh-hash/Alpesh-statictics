"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates and saves a relational
    plot showing temperature anomalies over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    sns.lineplot(
        data=df, x='Year', y='Monthly Temperature Anomaly',
        marker='o', linewidth=1.5, color='red', ax=ax
    )
    ax.set_title('Temperature Anomalies Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Monthly Temperature Anomaly (°C)', fontsize=14)
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Creates and saves a categorical plot showing the average temperature
    anomaly per month.
    """
    if 'Month' in df.columns and 'Monthly Temperature Anomaly' in df.columns:
        df['Month Name'] = df['Month'].apply(
            lambda x: [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October',
                'November', 'December'
            ][int(x) - 1] if pd.notnull(x) and 1 <= int(x) <= 12 else 'Unknown'
        )

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.barplot(
            data=df, x='Month Name', y='Monthly Temperature Anomaly',
            ci=None, palette='coolwarm', order=[
                'January', 'February', 'March', 'April', 'May',
                'June', 'July', 'August', 'September',
                'October', 'November', 'December'
            ]
        )
        plt.xticks(rotation=45)
        plt.title('Average Monthly Temperature Anomaly', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Temperature Anomaly (°C)', fontsize=14)
        plt.tight_layout()
        plt.savefig('categorical_plot.png')
        plt.show()
    else:
        print("Required columns for the categorical plot are not available.")


def plot_statistical_plot(df):
    """
    Creates and saves a combined statistical plot using:
    - A box plot to visualize data distribution
    - A corner plot to show pairwise relationships
    """
    numerical_columns = [
        'Monthly Temperature Anomaly',
        'Annual Temperature Anomaly',
        '5-Year Temperature Anomaly',
        '10-Year Temperature Anomaly',
        '20-Year Temperature Anomaly'
    ]

    # Filter only the columns that exist in the DataFrame
    existing_columns = [col for col in numerical_columns if col in df.columns]

    if existing_columns:
        data = df[existing_columns].dropna()
        print(
            f"Generating box plot and corner plot for columns: "
            f"{existing_columns}"
       )
        # 1. Generate a Box Plot (Statistical Plot Requirement)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        sns.boxplot(
            data=data,
            ax=ax,
            palette='coolwarm'
        )

        ax.set_title('Distribution of Temperature Anomalies', fontsize=16)
        ax.set_xlabel('Temperature Anomaly Type', fontsize=14)
        ax.set_ylabel('Temperature Anomaly (°C)', fontsize=14)
        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.savefig('statistical_box_plot.png')
        plt.show()
        print("Statistical box plot saved as 'statistical_box_plot.png'.")

        # 2. Generate a Corner Plot
        corner_fig = corner(
            data, labels=existing_columns, show_titles=True,
            title_fmt=".2f", color='red'
        )

        corner_fig.savefig('statistical_corner_plot.png')
        plt.close(corner_fig)
        print("Corner plot saved as 'statistical_corner_plot.png'.")

    else:
        print("No valid numerical columns available for the statistical plot.")
    return


def statistical_analysis(df, col: str):
    """
    Calculates and returns the four main
    statistical moments for a given column.
    """
    if col in df.columns:
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        mean = np.mean(data)
        stddev = np.std(data, ddof=1)
        skew = ss.skew(data, nan_policy='omit')
        excess_kurtosis = ss.kurtosis(data, nan_policy='omit')

        print(f"Statistical Analysis for '{col}':")
        print(f"Mean = {mean:.2f}")
        print(f"Standard Deviation = {stddev:.2f}")
        print(f"Skewness = {skew:.2f}")
        print(f"Excess Kurtosis = {excess_kurtosis:.2f}")

        return mean, stddev, skew, excess_kurtosis
    print(f"Column '{col}' not found in the DataFrame.")
    return None, None, None, None


def preprocessing(df):
    """
    Preprocesses the DataFrame by cleaning and formatting data.
    """
    df.columns = df.columns.str.strip()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)

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

    print("\n--- Data Head ---")
    print(df.head())
    print("\n--- Data Description ---")
    print(df.describe())
    print("\n--- Data Correlation ---")
    print(df.corr())

    return df


def writing(moments, col):
    """
    Provides a descriptive summary of the statistical analysis results.
    """
    if moments and all(m is not None for m in moments):
        mean, stddev, skew, excess_kurtosis = moments

        # Print statistical moments in a multiline format
        print(f'For the attribute "{col}":')
        print(
            f'Mean = {mean:.2f}, Std Dev = {stddev:.2f}, '
            f'Skewness = {skew:.2f}, '
            f'Excess Kurtosis = {excess_kurtosis:.2f}.'
        )

        # Use parentheses for multiline ternary operations
        skew_type = (
            'right-skewed' if skew > 0 else
            'left-skewed' if skew < 0 else
            'not skewed'
        )

        kurtosis_type = (
            'platykurtic' if excess_kurtosis < 0 else
            'leptokurtic' if excess_kurtosis > 0 else
            'mesokurtic'
        )

        print(f'The data is {skew_type} and {kurtosis_type}.')
    else:
        print(f"Invalid or missing moments for column '{col}'.")


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Monthly Temperature Anomaly'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
