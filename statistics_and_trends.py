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
    Creates and saves a relational plot showing temperature anomalies over time.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing the data.
        
    Returns:
        None
    """
    # Create a figure and axis using subplots (as per template)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Create a line plot showing Year vs. Monthly Temperature Anomaly
    sns.lineplot(data=df, x='Year', y='Monthly Temperature Anomaly', 
                 marker='o', linewidth=1.5, color='red', ax=ax)
    
    # Add labels and title
    ax.set_title('Temperature Anomalies Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Monthly Temperature Anomaly (°C)', fontsize=14)
    
    # Improve layout and save the plot
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()
    
    return

    


def plot_categorical_plot(df):
    
    """
    Creates and saves a categorical plot showing the average temperature anomaly per month.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing the data.
        
    Returns:
        None
    """
    # Convert the "Month" column from numbers to month names for categorical plotting
    if 'Month' in df.columns and 'Monthly Temperature Anomaly' in df.columns:
        df['Month Name'] = df['Month'].apply(lambda x: 
            ['January', 'February', 'March', 'April', 'May', 
             'June', 'July', 'August', 'September', 'October', 
             'November', 'December'][int(x)-1] if pd.notnull(x) and 1 <= int(x) <= 12 else 'Unknown')

        # Set plot style and size
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Create a bar plot of average temperature anomaly by month
        sns.barplot(data=df, x='Month Name', y='Monthly Temperature Anomaly', 
                    ci=None, palette='coolwarm', order=[
                        'January', 'February', 'March', 'April', 'May', 
                        'June', 'July', 'August', 'September', 'October', 
                        'November', 'December'
                    ])
        
        # Enhance the plot with labels and title
        plt.xticks(rotation=45)
        plt.title('Average Monthly Temperature Anomaly', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Temperature Anomaly (°C)', fontsize=14)
        
        # Save the plot to a file
        plt.tight_layout()
        plt.savefig('categorical_plot.png')
        plt.show()
    
    else:
        print("Required columns for the categorical plot are not available in the DataFrame.")
    
    return




def plot_statistical_plot(df):
    """
    Creates and saves a statistical plot using a box plot to visualize
    the distribution of temperature anomaly data.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing the data.
        
    Returns:
        None
    """
    # Select relevant numerical columns for the box plot
    numerical_columns = [
        'Monthly Temperature Anomaly',
        'Annual Temperature Anomaly',
        '5-Year Temperature Anomaly',
        '10-Year Temperature Anomaly',
        '20-Year Temperature Anomaly'
    ]
    
    # Filter the DataFrame to include only existing columns
    existing_columns = [col for col in numerical_columns if col in df.columns]
    
    if existing_columns:
        print(f"Generating statistical box plot for columns: {existing_columns}")
        
        # Create a figure and axis using subplots (as per template)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a box plot for the existing columns
        sns.boxplot(data=df[existing_columns], ax=ax, palette='coolwarm')
        
        # Enhance the plot with labels and title
        ax.set_title('Distribution of Temperature Anomalies', fontsize=16)
        ax.set_xlabel('Temperature Anomaly Type', fontsize=14)
        ax.set_ylabel('Temperature Anomaly (°C)', fontsize=14)
        plt.xticks(rotation=30)
        
        # Save the plot to a file
        plt.tight_layout()
        plt.savefig('statistical_plot.png')
        plt.show()
        print("Statistical box plot saved as 'statistical_plot.png'.")
    
    else:
        print("No valid numerical columns available for the statistical plot.")
    
    return




def statistical_analysis(df, col: str):
    """
    Calculates and returns the four main statistical moments for a given column.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing the data.
        col (str): The name of the column to analyze.
        
    Returns:
        mean (float): The mean of the column.
        stddev (float): The standard deviation of the column.
        skew (float): The skewness of the column.
        excess_kurtosis (float): The excess kurtosis of the column.
    """
    if col in df.columns:
        # Calculate the four main statistical moments
        mean = df[col].mean()
        stddev = df[col].std()
        skew = ss.skew(df[col], nan_policy='omit')
        excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')  # Excess kurtosis = kurtosis - 3
        
        print(f"Statistical Analysis for '{col}':")
        print(f"Mean = {mean:.2f}")
        print(f"Standard Deviation = {stddev:.2f}")
        print(f"Skewness = {skew:.2f}")
        print(f"Excess Kurtosis = {excess_kurtosis:.2f}")
        
        return mean, stddev, skew, excess_kurtosis
    else:
        print(f"Column '{col}' not found in the DataFrame.")
        return None, None, None, None


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
    """
    Provides a descriptive summary of the statistical analysis results.
    
    Args:
        moments (tuple): The statistical moments (mean, stddev, skewness, excess kurtosis).
        col (str): The name of the column analyzed.
        
    Returns:
        None
    """
    if moments and all(m is not None for m in moments):
        mean, stddev, skew, excess_kurtosis = moments
        
        # Display the calculated statistical moments
        print(f'For the attribute "{col}":')
        print(f'Mean = {mean:.2f}, '
              f'Standard Deviation = {stddev:.2f}, '
              f'Skewness = {skew:.2f}, '
              f'Excess Kurtosis = {excess_kurtosis:.2f}.')
        
        # Determine skewness type
        if skew > 0:
            skew_type = 'right-skewed'
        elif skew < 0:
            skew_type = 'left-skewed'
        else:
            skew_type = 'not skewed'
        
        # Determine kurtosis type
        if excess_kurtosis < 0:
            kurtosis_type = 'platykurtic (flat distribution)'
        elif excess_kurtosis > 0:
            kurtosis_type = 'leptokurtic (peaked distribution)'
        else:
            kurtosis_type = 'mesokurtic (normal distribution)'
        
        # Print the interpretation of skewness and kurtosis
        print(f'The data is {skew_type} and {kurtosis_type}.')
    
    else:
        print(f"Invalid or missing moments for column '{col}'.")
    
    return



def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Monthly Temperature Anomaly' 
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return 


if __name__ == '__main__':
    main() 
