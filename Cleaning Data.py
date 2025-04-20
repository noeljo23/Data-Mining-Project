import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#########################################
# PART 1: Define File Paths and Load Files
#########################################

# Map file keys to (file path, encoding) tuples.
files = {
    "vital_deaths": (r"C:\Users\NJ\Desktop\DFG HACK\Deaths, by cause, Chapter V Mental and behavioural disorders F00 to F99.csv", 'utf-8'),
    "health_services": (r"C:\Users\NJ\Desktop\DFG HACK\Mental Health and Substance Use Health Services - mental-health.csv", 'latin1'),
    "dashboard": (r"C:\Users\NJ\Desktop\DFG HACK\Mental health in Canada dashboard.csv", 'utf-8'),
    "indicators_by_pop": (r"C:\Users\NJ\Desktop\DFG HACK\Mental health indicators by population group.csv", 'utf-8'),
    "indicators": (r"C:\Users\NJ\Desktop\DFG HACK\Mental health indicators.csv", 'utf-8'),
    "perceived_socio": (r"C:\Users\NJ\Desktop\DFG HACK\Perceived mental health, by gender and other selected sociodemographic characteristics.csv", 'utf-8'),
    "perceived_prov": (r"C:\Users\NJ\Desktop\DFG HACK\Perceived mental health, by gender and province.csv", 'utf-8')
}

# Initialize the dictionary.
dfs = {}
for key, (path, encoding) in files.items():
    try:
        dfs[key] = pd.read_csv(path, encoding=encoding, low_memory=False)
        print(f"Loaded {key} from {path} with shape: {dfs[key].shape}")
    except Exception as e:
        print(f"Error loading {key} from {path}: {e}")

#########################################
# PART 2: Enhanced Date Column Standardization
#########################################


def ensure_date_column(df):
    """
    Ensure a unified date column exists named 'ref_date'. 
    Check common names such as 'ref_date', 'year', 'period', or 'date'.
    Renames the first found column to 'ref_date'.
    """
    possible_date_cols = ['ref_date', 'year', 'period', 'date']
    for col in possible_date_cols:
        if col in df.columns:
            if col != 'ref_date':
                df.rename(columns={col: 'ref_date'}, inplace=True)
            return df
    print("Warning: No recognized date column in DataFrame. Columns available:",
          df.columns.tolist())
    return df

#########################################
# PART 3: File-Specific Cleaning Functions
#########################################

# 1. Vital Deaths File


def clean_vital_deaths(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    if 'cause of death (icd-10)' in df.columns:
        df.rename(
            columns={'cause of death (icd-10)': 'cause_of_death'}, inplace=True)
    return df


if "vital_deaths" in dfs:
    dfs["vital_deaths"] = clean_vital_deaths(dfs["vital_deaths"])

# 2. Health Services File


def clean_health_services(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    return df


if "health_services" in dfs:
    dfs["health_services"] = clean_health_services(dfs["health_services"])

# 3. Dashboard File


def clean_dashboard(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    return df


if "dashboard" in dfs:
    dfs["dashboard"] = clean_dashboard(dfs["dashboard"])

# 4. Indicators by Population Group File


def clean_indicators_by_pop(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    if 'population group' in df.columns:
        df.rename(
            columns={'population group': 'population_group'}, inplace=True)
    return df


if "indicators_by_pop" in dfs:
    dfs["indicators_by_pop"] = clean_indicators_by_pop(
        dfs["indicators_by_pop"])

# 5. Indicators File


def clean_indicators(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    return df


if "indicators" in dfs:
    dfs["indicators"] = clean_indicators(dfs["indicators"])

# 6. Perceived Mental Health by Gender & Sociodemographic File


def clean_perceived_socio(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    return df


if "perceived_socio" in dfs:
    dfs["perceived_socio"] = clean_perceived_socio(dfs["perceived_socio"])

# 7. Perceived Mental Health by Gender & Province File


def clean_perceived_prov(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    elif 'province' in df.columns:
        df['province'] = df['province'].str.strip().str.lower()
        df.rename(columns={'province': 'geo'}, inplace=True)
    return df


if "perceived_prov" in dfs:
    dfs["perceived_prov"] = clean_perceived_prov(dfs["perceived_prov"])

#########################################
# PART 4: Apply Value Adjustment (For Files with Scalar Metadata)
#########################################


def get_multiplier(scalar_factor):
    sf = scalar_factor.lower().strip()
    if sf == 'thousands':
        return 1000
    elif sf == 'millions':
        return 1000000
    elif sf == 'units':
        return 1
    return 1


def adjust_value(row):
    multiplier = get_multiplier(row['scalar_factor'])
    adjusted = row['value'] * multiplier
    adjusted = round(adjusted, int(
        row['decimals']) if pd.notnull(row['decimals']) else 0)
    return adjusted


# Only apply to the "indicators" file if it has the required metadata.
if "indicators" in dfs and all(col in dfs["indicators"].columns for col in ['scalar_factor', 'decimals']):
    dfs["indicators"]["adjusted_value"] = dfs["indicators"].apply(
        adjust_value, axis=1)
    if 'characteristics' in dfs["indicators"].columns:
        print("Adjusted values in indicators file:")
        for idx, row in dfs["indicators"].head().iterrows():
            print(
                f"Row {idx}: {row['characteristics']}, Adjusted Value: {row['adjusted_value']}")

#########################################
# PART 5: Data Transformation & Merging for Analysis
#########################################

# Instead of using the dashboard file (which lacks ref_date and geo),
# we use the "indicators" file to select the measurement for
# "major depressive episode, life".

if "indicators" in dfs:
    indicators_df = dfs["indicators"]
    print("Indicators columns:", indicators_df.columns.tolist())
    # Filter the indicators file for rows where the indicator matches.
    # (Assume that the indicator info is stored in a column named 'indicators'.)
    mh_overall = indicators_df[indicators_df['indicators'].str.lower(
    ) == 'major depressive episode, life']
else:
    print("Indicators file not available.")
    mh_overall = pd.DataFrame()

# For the vital deaths file, filter for the cause of death.
if "vital_deaths" in dfs:
    vital_df = dfs["vital_deaths"]
    vital_df = vital_df[vital_df['cause_of_death'].str.lower() ==
                        'chapter v: mental and behavioural disorders [f01-f99]']
else:
    print("Vital deaths file not available.")
    vital_df = pd.DataFrame()

# Debug prints to check columns in the filtered DataFrames.
print("mh_overall columns:", mh_overall.columns.tolist())
print("vital_df columns:", vital_df.columns.tolist())

# Prepare both DataFrames for merging using 'ref_date' and 'geo'.
if not mh_overall.empty and all(col in mh_overall.columns for col in ['ref_date', 'geo']):
    mh_overall = mh_overall[['ref_date', 'geo', 'value']].rename(
        columns={'value': 'mh_depressive_count'})
else:
    print("Indicators data missing required 'ref_date' or 'geo' columns.")

if not vital_df.empty and all(col in vital_df.columns for col in ['ref_date', 'geo']):
    vital_df = vital_df[['ref_date', 'geo', 'value']].rename(
        columns={'value': 'mh_mortality'})
else:
    print("Vital deaths data missing required 'ref_date' or 'geo' columns.")

try:
    merged_data = pd.merge(mh_overall, vital_df, on=[
                           'ref_date', 'geo'], how='inner')
    print("Merged Data Sample:")
    print(merged_data.head())
except KeyError as e:
    print("Error merging data. Missing key(s):", e)
    merged_data = pd.DataFrame()

#########################################
# PART 6: Analysis and Visualization
#########################################

if not merged_data.empty:
    # Trend Plot: Plot time series of the mental health indicator and mortality.
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['ref_date'], merged_data['mh_depressive_count'],
             marker='o', label='Major Depressive Episodes (Count)')
    plt.plot(merged_data['ref_date'], merged_data['mh_mortality'],
             marker='s', label='Mental Health Mortality')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Trends: Major Depressive Episodes vs. Mental Health Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Correlation Analysis: Pearson correlation.
    if len(merged_data) > 2:
        corr_coef, p_value = pearsonr(
            merged_data['mh_depressive_count'], merged_data['mh_mortality'])
        print(f"Pearson Correlation: {corr_coef:.3f}, p-value: {p_value:.3f}")
    else:
        print("Not enough data points for correlation analysis.")

    # Scatter Plot with Linear Fit.
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_data['mh_depressive_count'], merged_data['mh_mortality'],
                color='blue', label='Data Points')
    slope, intercept = np.polyfit(
        merged_data['mh_depressive_count'], merged_data['mh_mortality'], 1)
    x_vals = np.array([merged_data['mh_depressive_count'].min(),
                      merged_data['mh_depressive_count'].max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red',
             label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
    plt.xlabel('Major Depressive Episodes (Count)')
    plt.ylabel('Mental Health Mortality')
    plt.title('Scatter Plot: Depressive Episodes vs. Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Merged data is empty. Skipping analysis and visualization.")
