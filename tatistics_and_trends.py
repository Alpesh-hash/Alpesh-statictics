warning: in the working copy of 'statistics_and_trends.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/statistics_and_trends.py b/statistics_and_trends.py[m
[1mindex 3b5572c..2584582 100644[m
[1m--- a/statistics_and_trends.py[m
[1m+++ b/statistics_and_trends.py[m
[36m@@ -13,9 +13,11 @@[m [mimport numpy as np[m
 import pandas as pd[m
 import scipy.stats as ss[m
 import seaborn as sns[m
[32m+[m
[32m+[m
 def plot_relational_plot(df):[m
     """[m
[31m-    Creates and saves a relational [m
[32m+[m[32m    Creates and saves a relational[m
     plot showing temperature anomalies over time.[m
     """[m
     fig, ax = plt.subplots(figsize=(10, 6))[m
[36m@@ -30,6 +32,8 @@[m [mdef plot_relational_plot(df):[m
     plt.tight_layout()[m
     plt.savefig('relational_plot.png')[m
     plt.show()[m
[32m+[m
[32m+[m
 def plot_categorical_plot(df):[m
     """[m
     Creates and saves a categorical plot showing the average temperature[m
[36m@@ -63,6 +67,8 @@[m [mdef plot_categorical_plot(df):[m
         plt.show()[m
     else:[m
         print("Required columns for the categorical plot are not available.")[m
[32m+[m
[32m+[m
 def plot_statistical_plot(df):[m
     """[m
     Creates and saves a combined statistical plot using:[m
[36m@@ -76,52 +82,54 @@[m [mdef plot_statistical_plot(df):[m
         '10-Year Temperature Anomaly',[m
         '20-Year Temperature Anomaly'[m
     ][m
[31m-    [m
[32m+[m
     # Filter only the columns that exist in the DataFrame[m
     existing_columns = [col for col in numerical_columns if col in df.columns][m
[31m-    [m
[32m+[m
     if existing_columns:[m
         data = df[existing_columns].dropna()[m
         print([m
     f"Generating box plot and corner plot for columns: "[m
     f"{existing_columns}"[m
[31m-)    [m
[32m+[m[32m)[m
         # 1. Generate a Box Plot (Statistical Plot Requirement)[m
         fig, ax = plt.subplots(figsize=(10, 6))[m
         sns.set_theme(style="whitegrid")[m
[31m-        [m
[32m+[m
         sns.boxplot([m
             data=data,[m
             ax=ax,[m
             palette='coolwarm'[m
         )[m
[31m-        [m
[32m+[m
         ax.set_title('Distribution of Temperature Anomalies', fontsize=16)[m
         ax.set_xlabel('Temperature Anomaly Type', fontsize=14)[m
         ax.set_ylabel('Temperature Anomaly (°C)', fontsize=14)[m
         plt.xticks(rotation=30)[m
[31m-        [m
[32m+[m
         plt.tight_layout()[m
         plt.savefig('statistical_box_plot.png')[m
         plt.show()[m
         print("Statistical box plot saved as 'statistical_box_plot.png'.")[m
[31m-        [m
[31m-        # 2. Generate a Corner Plot [m
[32m+[m
[32m+[m[32m        # 2. Generate a Corner Plot[m
         corner_fig = corner([m
             data, labels=existing_columns, show_titles=True,[m
             title_fmt=".2f", color='red'[m
         )[m
[31m-        [m
[32m+[m
         corner_fig.savefig('statistical_corner_plot.png')[m
         plt.close(corner_fig)[m
         print("Corner plot saved as 'statistical_corner_plot.png'.")[m
[31m-    [m
[32m+[m
     else:[m
         print("No valid numerical columns available for the statistical plot.")[m
     return[m
[32m+[m
[32m+[m
 def statistical_analysis(df, col: str):[m
     """[m
[31m-    Calculates and returns the four main [m
[32m+[m[32m    Calculates and returns the four main[m
     statistical moments for a given column.[m
     """[m
     if col in df.columns:[m
[36m@@ -130,22 +138,24 @@[m [mdef statistical_analysis(df, col: str):[m
         stddev = np.std(data, ddof=1)[m
         skew = ss.skew(data, nan_policy='omit')[m
         excess_kurtosis = ss.kurtosis(data, nan_policy='omit')[m
[31m-        [m
[32m+[m
         print(f"Statistical Analysis for '{col}':")[m
         print(f"Mean = {mean:.2f}")[m
         print(f"Standard Deviation = {stddev:.2f}")[m
         print(f"Skewness = {skew:.2f}")[m
         print(f"Excess Kurtosis = {excess_kurtosis:.2f}")[m
[31m-        [m
[32m+[m
         return mean, stddev, skew, excess_kurtosis[m
     print(f"Column '{col}' not found in the DataFrame.")[m
     return None, None, None, None[m
[32m+[m
[32m+[m
 def preprocessing(df):[m
     """[m
     Preprocesses the DataFrame by cleaning and formatting data.[m
     """[m
     df.columns = df.columns.str.strip()[m
[31m-    [m
[32m+[m
     for column in df.columns:[m
         if df[column].dtype == 'object':[m
             df[column] = pd.to_numeric(df[column], errors='coerce')[m
[36m@@ -175,13 +185,15 @@[m [mdef preprocessing(df):[m
     print(df.corr())[m
 [m
     return df[m
[32m+[m
[32m+[m
 def writing(moments, col):[m
     """[m
     Provides a descriptive summary of the statistical analysis results.[m
     """[m
     if moments and all(m is not None for m in moments):[m
         mean, stddev, skew, excess_kurtosis = moments[m
[31m-        [m
[32m+[m
         # Print statistical moments in a multiline format[m
         print(f'For the attribute "{col}":')[m
         print([m
[36m@@ -202,10 +214,12 @@[m [mdef writing(moments, col):[m
             'leptokurtic' if excess_kurtosis > 0 else[m
             'mesokurtic'[m
         )[m
[31m-        [m
[32m+[m
         print(f'The data is {skew_type} and {kurtosis_type}.')[m
     else:[m
         print(f"Invalid or missing moments for column '{col}'.")[m
[32m+[m
[32m+[m
 def main():[m
     df = pd.read_csv('data.csv')[m
     df = preprocessing(df)[m
[36m@@ -215,6 +229,8 @@[m [mdef main():[m
     plot_categorical_plot(df)[m
     moments = statistical_analysis(df, col)[m
     writing(moments, col)[m
[32m+[m
[32m+[m
 if __name__ == '__main__':[m
     main()[m
 [m
