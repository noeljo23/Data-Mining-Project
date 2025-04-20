import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#########################################
# PART 1: File Paths and Loading
#########################################

files = {
    "indicators": (r"C:\Users\NJ\Desktop\DFG HACK\Mental health indicators.csv", 'utf-8'),
    "vital_deaths": (r"C:\Users\NJ\Desktop\DFG HACK\Deaths, by cause, Chapter V Mental and behavioural disorders F00 to F99.csv", 'utf-8')
}

dfs = {}
for key, (path, encoding) in files.items():
    try:
        dfs[key] = pd.read_csv(path, encoding=encoding, low_memory=False)
        print(f"Loaded {key} with shape: {dfs[key].shape}")
    except Exception as e:
        print(f"Error loading {key} from {path}: {e}")

#########################################
# PART 2: Standardize and Clean
#########################################


def ensure_date_column(df):
    # Rename year/period/date to 'ref_date'
    possible_date_cols = ['ref_date', 'year', 'period', 'date']
    for col in possible_date_cols:
        if col in df.columns:
            if col != 'ref_date':
                df.rename(columns={col: 'ref_date'}, inplace=True)
            break
    return df


def clean_indicators(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    # Convert ref_date and value to numeric if present
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # Convert geo to lower if present
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    return df


def clean_vital_deaths(df):
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_date_column(df)
    if 'ref_date' in df.columns:
        df['ref_date'] = pd.to_numeric(df['ref_date'], errors='coerce')
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'geo' in df.columns:
        df['geo'] = df['geo'].str.strip().str.lower()
    # Rename cause-of-death column if needed
    if 'cause of death (icd-10)' in df.columns:
        df.rename(
            columns={'cause of death (icd-10)': 'cause_of_death'}, inplace=True)
    return df


if 'indicators' in dfs:
    dfs['indicators'] = clean_indicators(dfs['indicators'])
if 'vital_deaths' in dfs:
    dfs['vital_deaths'] = clean_vital_deaths(dfs['vital_deaths'])

#########################################
# PART 3: Filter for Annual, Comparable Measures
#########################################

### A. MENTAL HEALTH INDICATORS ###

if 'indicators' in dfs:
    mh_df = dfs['indicators'].copy()
    # Adjust these strings to match your columns
    # e.g. 'indicators' might actually be the column name you see in the unique list
    # We'll filter for "Major depressive episode, 12 months"
    # and pick 'Number of persons' for 'Characteristics' if you have that,
    # or 'Percent' if that's your measure type.

    # Example filter for "Major depressive episode, 12 months" + "number of persons" + "total, 15 years and over" + "total gender" + "canada"
    cond_indicator = mh_df['indicators'].str.lower().eq(
        'major depressive episode, 12 months')
    cond_charac = mh_df['characteristics'].str.lower().eq('number of persons')
    cond_age = mh_df['age group'].str.lower().eq('total, 15 years and over')
    cond_gender = mh_df['gender'].str.lower().eq('total, gender of person')
    cond_geo = mh_df['geo'].str.lower().eq('canada')

    mh_df_filtered = mh_df[cond_indicator & cond_charac &
                           cond_age & cond_gender & cond_geo].copy()
    mh_df_filtered.rename(columns={'value': 'depression_count'}, inplace=True)
else:
    mh_df_filtered = pd.DataFrame()

print("Filtered mental health data (Major depressive episode, 12 months):",
      mh_df_filtered.shape)

### B. MENTAL HEALTH–RELATED DEATHS ###

if 'vital_deaths' in dfs:
    vd_df = dfs['vital_deaths'].copy()
    # Example filter: cause_of_death = 'chapter v: mental and behavioural disorders [f01-f99]'
    # For the entire population: 'age group'='total, all ages', 'sex'='both sexes', 'geo'='canada'
    cond_cod = vd_df['cause_of_death'].str.lower().eq(
        'chapter v: mental and behavioural disorders [f01-f99]')
    cond_age = vd_df['age group'].str.lower().eq('total, all ages')
    cond_sex = vd_df['sex'].str.lower().eq('both sexes')
    cond_geo = vd_df['geo'].str.lower().eq('canada')

    vd_df_filtered = vd_df[cond_cod & cond_age & cond_sex & cond_geo].copy()
    vd_df_filtered.rename(columns={'value': 'mh_deaths'}, inplace=True)
else:
    vd_df_filtered = pd.DataFrame()

print("Filtered vital deaths data:", vd_df_filtered.shape)

#########################################
# PART 4: Merge and Possibly Aggregate (If Multiple Rows per Year)
#########################################

# If each table has one row per year, we're good. If multiple rows exist per year, sum them or pick the one that matches 'Characteristics=Number of persons'.
# Then merge on [ref_date].

# Example: If there are duplicates by 'ref_date', group by year.
if not mh_df_filtered.empty:
    # Group in case multiple sub-subcategories exist in the data
    mh_yearly = mh_df_filtered.groupby('ref_date', as_index=False)[
        'depression_count'].sum()
else:
    mh_yearly = pd.DataFrame()

if not vd_df_filtered.empty:
    vd_yearly = vd_df_filtered.groupby('ref_date', as_index=False)[
        'mh_deaths'].sum()
else:
    vd_yearly = pd.DataFrame()

merged_data = pd.merge(mh_yearly, vd_yearly, on='ref_date', how='inner')

print("Merged data sample:")
print(merged_data.head())

#########################################
# PART 5: Analysis & Visualization
#########################################

if not merged_data.empty:
    # 1. Plot Trend Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['ref_date'], merged_data['depression_count'],
             'o-', label='Major Depressive Episodes (12 months)')
    plt.plot(merged_data['ref_date'], merged_data['mh_deaths'],
             's-', label='Mental Health Mortality')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Major Depressive Episodes vs. MH Mortality (Annual)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Correlation
    if len(merged_data) > 2:
        corr_coef, p_value = pearsonr(
            merged_data['depression_count'], merged_data['mh_deaths'])
        print(f"Pearson Correlation: {corr_coef:.3f}, p-value: {p_value:.3f}")
    else:
        print("Not enough data points for correlation analysis.")

    # 3. Scatter Plot with Fit
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_data['depression_count'],
                merged_data['mh_deaths'], c='blue', label='Data Points')
    slope, intercept = np.polyfit(
        merged_data['depression_count'], merged_data['mh_deaths'], 1)
    x_vals = np.array([merged_data['depression_count'].min(),
                      merged_data['depression_count'].max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red',
             label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
    plt.xlabel('Major Depressive Episodes (12 months, Count)')
    plt.ylabel('Mental Health Mortality (Count)')
    plt.title('Scatter: Major Depressive Episodes vs. MH Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No merged data available for analysis.")
