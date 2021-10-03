# Analysis-of-COVID-19-and-Its-Related-Effects
"""
Created on Wed Sep 30 20:39:36 2020

    The purpose of this program is to extract and transform data in order to prepare for analysis and visualization of the following variables:    
        
        New Covid-19 Cases
        Unemployment Rate (%)
        Workplace Mobility Change from Baseline (%)
        Residential Mobility Change from Baseline (%)
        Retail Mobility Change from Baseline (%)
        Oil Price (WTI) **national level only**

@authors: Zach Bannon, Sean BoxenBaum, Austin York
"""

# Establishes input information for local files.
inpath = 'C:\\Users\\zachb\\Documents\\Texas Tech University\\MS Data Science\\ISQS 6339\\Group Project\\' # Local file path. Update to your desired file location.
oil_price_file = 'PET_PRI_SPT_S1_D.xls' # File name for oil price data.
mobility_file = '2020_US_Region_Mobility_Report.csv' # File name for mobility data.
cases_file = 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv' # File name for US Covid data.

# Establishes input information for scraped files.
fips_url = 'https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt' # URL path for FIPS code data.
st_abbr_url = 'https://worldpopulationreview.com/static/states/abbr-name.csv' # URL path for state abbreviations data.
unemployment_url = 'https://www.bls.gov/web/metro/laucntycur14.txt' # URL path for unemployment data

# Establishes output file information for analysis and visualization.
outpath = 'C:\\Users\\zachb\\Documents\\Texas Tech University\\MS Data Science\\ISQS 6339\\Group Project\\Test\\'
state_output_file = 'StateData.csv'
national_output_file = 'NationalData.csv'
state_corr_matrix_file = 'StateCorrelationMatrix.csv'
natl_corr_matrix_file = 'NationalCorrelationMatrix.csv'
state_heatmap = 'StateHeatmap.png'
natl_heatmap = 'NationalHeatmap.png'

# Establishes output file information for scraped data.
unemployment_output_file = 'UnemploymentScrapedData.csv'
fips_output_file = 'FIPS_ScrapedData.csv'
st_abbr_output_file = 'StateAbbreviationsScrapedData.csv'

# Imports necessary libraries.
import pandas as pd # Necessary for data cleaning and transformation
import requests as r # Necessary for web scrapes.
import io # Necessary for web scrapes.
import calendar # Necessary in order to convert abbreviated month name to month number.
import seaborn as sn # Necessary in order to produce correlation matrix and heat map.

# Shows max columns for viewing dataframes.
pd.set_option('display.max_columns', None)

# Reads in data sources and creates data frames for local files.
oil_price_df = pd.read_excel(inpath + oil_price_file, sheet_name='Data 1', skiprows=2)
df_mobility = pd.read_csv(inpath + mobility_file)
cases_df = pd.read_csv(inpath + cases_file)

# Performs web scrape for unemployment data and creates dataframe.
unemployment_res = r.get(unemployment_url)
unemployment_df = pd.read_csv(io.StringIO(unemployment_res.text), delimiter='|', header=None, skiprows=6, names=[
    'LAUS Code',
    'State FIPS Code',
    'County FIPS Code',
    'Area Title',
    'Period',
    'Labor Force',
    'Employed',
    'Unemployed',
    'Unemployment Rate (%)'
    ])

# Creates .csv record of scraped unemployment data.
unemployment_df.to_csv(outpath + unemployment_output_file, index=False)

# Creates dictionary of abbreviated month names to month number.
# Code to create dictionary discovered while searching for solution on
# Stack Overflow at:
# https://stackoverflow.com/questions/3418050/month-name-to-month-number-and-vice-versa-in-python/21938128.
# Credit to user David Z.
month_df = pd.DataFrame({v:k for k, v in enumerate(calendar.month_abbr)}.items(),
                        columns=['Abbreviated_Month', 'Month_Number'])

# Creates reference data frame for FIPS information.
fips_res = r.get(fips_url)
fips_df = pd.read_csv(io.StringIO(fips_res.text), header=None, dtype='object')
fips_df.columns = ['State_Abbreviation', 'State FIPS Code', 'County FIPS Code', 'County', 'H']

# Creates .csv record of scraped FIPS code data.
fips_df.to_csv(outpath + fips_output_file, index=False)

# Creates reference data frame for state name / abbreviations info.
st_abbr_res = r.get(st_abbr_url)
st_abbr_df = pd.read_csv(io.StringIO(st_abbr_res.text), header=None)
st_abbr_df.columns = ['State_Abbreviation', 'State_Name']

# Creates .csv record of scraped state abbreviation data.
st_abbr_df.to_csv(outpath + st_abbr_output_file, header=None)

# Merges the two reference data frames regarding state/FIPS into one data frame.
# Final reference data frame contains state abbreviation, state FIPS code, and state name.
state_fips_df = fips_df.merge(st_abbr_df, how='inner', left_on='State_Abbreviation', right_on='State_Abbreviation')
state_fips_df = state_fips_df.drop(['County FIPS Code', 'County', 'H'], axis=1)
state_fips_df.drop_duplicates(inplace=True)
state_fips_df.reset_index(drop=True, inplace=True)

###########################################################################################################################

# Oil Price data cleaning/munging

# Renames column for WTI price per barrel, and drops Brent price column.
oil_price_df.rename(columns={'Cushing, OK WTI Spot Price FOB (Dollars per Barrel)':'WTI Price per Barrel'}, inplace=True)
oil_price_df.drop(columns=['Europe Brent Spot Price FOB (Dollars per Barrel)'], inplace=True)

# Encodes columns for month and year.
oil_price_df['Month'] = pd.DatetimeIndex(oil_price_df['Date']).month
oil_price_df['Year'] = pd.DatetimeIndex(oil_price_df['Date']).year

# Ensures only 2020 data is included in oil_price_df
oil_price_df = oil_price_df[oil_price_df['Year'] == 2020]

# Cleans null values due to holidays by filling with previous recorded value (fill forward).
oil_price_df.fillna(method='ffill', inplace=True)

# Creates dataframe for average monthly WTI price per barrel.
oil_monthly_avg_df = pd.DataFrame(oil_price_df[[
    'Year',
    'Month',
    'WTI Price per Barrel'
    ]].groupby([
        'Year',
        'Month']).mean().reset_index())

# Encodes state FIPS code column in order to prepare with merging of other datasets.
# Value assigned will be 00 to not represent a specific area, rather the nation as a whole.
oil_monthly_avg_df['State FIPS Code'] = '00'

###########################################################################################################################

# Unemployment data cleaning/munging

# Renames columns for unemployment data frame, then drops rows that contain no data as well as the last two rows that just contain notes.
unemployment_df.drop(unemployment_df.tail(5).index, inplace=True)

# Trims spaces from data.
unemployment_df = unemployment_df.replace(' ', '', regex=True)

# Replaces non-numeric values with zero in columns that should have numeric data types.
unemployment_df['Labor Force'] = unemployment_df['Labor Force'].replace('-', '', regex=True)
unemployment_df['Labor Force'] = unemployment_df['Labor Force'].replace(',', '', regex=True)
unemployment_df['Employed'] = unemployment_df['Employed'].replace('-', '', regex=True)
unemployment_df['Employed'] = unemployment_df['Employed'].replace(',', '', regex=True)
unemployment_df['Unemployed'] = unemployment_df['Unemployed'].replace('-', '', regex=True)
unemployment_df['Unemployed'] = unemployment_df['Unemployed'].replace(',', '', regex=True)
unemployment_df['Unemployment Rate (%)'] = unemployment_df['Unemployment Rate (%)'].replace('-', '', regex=True)

# Removes preliminary notation from period column where found.
unemployment_df['Period'] = unemployment_df['Period'].str.rstrip('(p)')

# Encodes column for year and abbreviated month name based on period value.
unemployment_df['Year'] = ('20' + unemployment_df['Period'].str[-2:]).apply(pd.to_numeric)
unemployment_df['Abbreviated_Month'] = unemployment_df['Period'].str[0:3]

# Merges unemployment_df with month_df to add in month number column.
unemployment_df = unemployment_df.merge(month_df, how='inner', left_on='Abbreviated_Month', right_on='Abbreviated_Month')

# Ensures data types are correct for merge.
unemployment_df[['Month_Number']] = unemployment_df[['Month_Number']].apply(pd.to_numeric)
# unemployment_df[['State FIPS Code']] = unemployment_df[['State FIPS Code']].apply(pd.to_numeric)
unemployment_df[['State FIPS Code']] = unemployment_df[['State FIPS Code']].astype(int)

# Changes data types for numeric values to be analyzed.
unemployment_df[['Labor Force', 'Employed', 'Unemployed', 'Unemployment Rate (%)']] = unemployment_df[['Labor Force', 'Employed', 'Unemployed', 'Unemployment Rate (%)']].apply(pd.to_numeric)

# Creates dataframe to contain average values by LAUS code to be used to
# populate null values in numeric columns.
avg_county_unemployment_df = pd.DataFrame(unemployment_df[[
    'LAUS Code',
    'Labor Force',
    'Employed',
    'Unemployed'
    ]].groupby([
        'LAUS Code']).mean().reset_index())

# Adds unemployment rate column to avg_county_unemployment_df.
avg_county_unemployment_df['Unemployment Rate (%)'] = (avg_county_unemployment_df['Unemployed'] / avg_county_unemployment_df['Labor Force']) * 100

# Iterrates through unemployment data frame and assigns null values the
# average value for the corresponding LAUS code.
for index, row in unemployment_df.iterrows():
    if pd.isna(unemployment_df.at[index, 'Labor Force']):
        unemployment_df.at[index, 'Labor Force'] = avg_county_unemployment_df[
                'Labor Force'
                ].astype(int)[avg_county_unemployment_df['LAUS Code'] == row['LAUS Code']]
    if pd.isna(unemployment_df.at[index, 'Employed']):
        unemployment_df.at[index, 'Employed'] = avg_county_unemployment_df[
            'Employed'
            ].astype(int)[avg_county_unemployment_df['LAUS Code'] == row['LAUS Code']]
    if pd.isna(unemployment_df.at[index, 'Unemployed']):
        unemployment_df.at[index, 'Unemployed'] = avg_county_unemployment_df[
            'Unemployed'
            ].astype(int)[avg_county_unemployment_df['LAUS Code'] == row['LAUS Code']]
    if pd.isna(unemployment_df.at[index, 'Unemployment Rate (%)']):
        unemployment_df.at[index, 'Unemployment Rate (%)'] = avg_county_unemployment_df[
            'Unemployment Rate (%)'
            ][avg_county_unemployment_df['LAUS Code'] == row['LAUS Code']]

# Creates dataframe to contain totals by state/month.
state_unemployment_df = pd.DataFrame(unemployment_df[[
    'State FIPS Code',
    'Year',
    'Month_Number',
    'Labor Force',
    'Employed',
    'Unemployed'
    ]].groupby([
        'State FIPS Code',
        'Year',
        'Month_Number']).sum().reset_index())

# Encodes column for unemployment rate in state_unemployment_df.
state_unemployment_df['Unemployment Rate (%)'] = (state_unemployment_df['Unemployed'] / state_unemployment_df['Labor Force']) * 100

# Creates dataframe to contain national totals by month.
natl_unemployment_df = pd.DataFrame(unemployment_df[[
    'Year',
    'Month_Number',
    'Labor Force',
    'Employed',
    'Unemployed'
    ]].groupby([
        'Year',
        'Month_Number']).sum().reset_index())

# Encodes column for unemployment rate in natl_unemployment_df.
natl_unemployment_df['Unemployment Rate (%)'] = (natl_unemployment_df['Unemployed'] / natl_unemployment_df['Labor Force']) * 100

# Creates column for state FIPS code that will be left blank in preparation to append records into one dataset.
natl_unemployment_df['State FIPS Code'] = '00'

# Creates final unemployment dataframe ready to be merged with other datasets for analysis.
final_unemployment_df = state_unemployment_df.append(natl_unemployment_df, ignore_index=True)

# Removes 2019 data from final_unemployment_df and natl_unemployment_df
final_unemployment_df = final_unemployment_df[final_unemployment_df['Year'] != 2019]
natl_unemployment_df = natl_unemployment_df[natl_unemployment_df['Year'] != 2019]

###########################################################################################################################

# Sean

# Munge covid cases data
# Add column with only month number
cases_df['month']=pd.DatetimeIndex(cases_df['submission_date']).month

# Compute new cases per month
new_cases_df=cases_df.groupby(['state', 'month'])[['new_case']].sum().reset_index()

# Merge with fips
new_cases_df=new_cases_df.merge(state_fips_df, how='inner', left_on='state', right_on='State_Abbreviation')
new_cases_df.drop(['State_Abbreviation', 'State_Name'], axis=1, inplace=True)

# Ensures data types are correct for merge.
new_cases_df['month'] = new_cases_df['month'].apply(pd.to_numeric)
new_cases_df['State FIPS Code'] = new_cases_df['State FIPS Code'].astype(int)

# Calculate national new cases per month
new_cases_US_df=new_cases_df.groupby('month').sum().reset_index()

# Fill in missing columns
new_cases_US_df['state']='US'
new_cases_US_df['State FIPS Code']='00'

###########################################################################################################################

#Clean the mobility data

# Encodes column for month.
df_mobility['Month'] = pd.DatetimeIndex(df_mobility['date']).month

# Creates the national dataframe
df_mobility_national = df_mobility.head(n=207)

# Renames column for state FIPS code.
df_mobility_national = df_mobility_national.rename (columns={'sub_region_1': 'state_fips'})

# Populates state FIPS code with 00 to represent the nation as opposed to a single state.
df_mobility_national['state_fips'].fillna('00', inplace=True)

# Groups national data by month.
df_mobility_national = df_mobility_national.groupby(['state_fips', 'Month']).mean().reset_index()

# Changes data type of national dataframe to object.
convert_dict = {'state_fips': object}
df_mobility_national = df_mobility_national.astype(convert_dict)

# Removes county level data leaving only state totals.
df_mobility = df_mobility[df_mobility['sub_region_2'].isnull()]

# Merge local mobility dataframe to add fips data
df_mobility = df_mobility.merge(state_fips_df, how='left', left_on='sub_region_1', right_on='State_Name')

# Groups the the local data by month.
df_mobility = df_mobility.groupby(['State FIPS Code', 'Month']).mean().reset_index()

# Ensures data types are correct for merge.
df_mobility['Month'] = df_mobility['Month'].apply(pd.to_numeric)
df_mobility['State FIPS Code'] = df_mobility['State FIPS Code'].astype(int)

###########################################################################################################################

# Merges the unemployment, new cases, and mobility dataframes into one containing monthly state data.

# Merges new cases with unemployment.
state_df = new_cases_df.merge(final_unemployment_df, how='inner', left_on=['State FIPS Code', 'month'], right_on=['State FIPS Code', 'Month_Number'])

state_df.head()

# Merges mobility into final data frame.
state_df = state_df.merge(df_mobility, how='inner', left_on=['State FIPS Code', 'month'], right_on=['State FIPS Code', 'Month'])

# Assigns final column order, dropping unneeded columns.
state_df = state_df[[
    'month',
    'state',
    'State FIPS Code',
    'new_case',
    'Unemployment Rate (%)',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'retail_and_recreation_percent_change_from_baseline'
    ]]

# Renames columns.
state_df.rename(columns={'state':'State', 'month':'Month'}, inplace=True)

# Writes state data to output file.
state_df.to_csv(outpath + state_output_file, index=False)

###########################################################################################################################

# Merges the unemployment, new cases, mobility, and oil price dataframes into one containing monthly national data.

# Merges national unemployment data with oil price data.
natl_df = natl_unemployment_df.merge(oil_monthly_avg_df, how='inner', left_on=['State FIPS Code', 'Month_Number', 'Year'], right_on=['State FIPS Code', 'Month', 'Year'])

# Merges in national cases.
natl_df = natl_df.merge(new_cases_US_df, how='inner', left_on=['State FIPS Code', 'Month_Number'], right_on=['State FIPS Code', 'month'])

# Merges in national mobility.
natl_df = natl_df.merge(df_mobility_national, how='inner', left_on=['State FIPS Code', 'Month_Number'], right_on=['state_fips', 'Month'])

# Renames columns.
natl_df.rename(columns={'Month_x': 'Month'}, inplace=True)

# Assigns final column order, dropping unneeded columns.
natl_df = natl_df[[
    'Month',
    'new_case',
    'Unemployment Rate (%)',
    'WTI Price per Barrel',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'retail_and_recreation_percent_change_from_baseline'
    ]]

# Writes national data to output file.
natl_df.to_csv(outpath + national_output_file, index=False)

###########################################################################################################################

# Correlation matrices and heat maps.

# Creates state correlation matrix.
state_corr  = state_df.corr()

# Creates state heatmap
ax = sn.heatmap(
    state_corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sn.diverging_palette(20, 220, n=200),
    square=True
    )
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
    )

# Assigns state heatmap to variable.
state_heatmap_plot = ax.get_figure()

# Outputs state correlation matrix to .csv.
state_corr.to_csv(outpath+state_corr_matrix_file)

# Outputs state heatmap to .png.
state_heatmap_plot.savefig(outpath + state_heatmap, bbox_inches='tight')

# Creates national correlation matrix.
natl_corr = natl_df.corr()

# Creates national heatmap.
ax = sn.heatmap(
    natl_corr,
    vmin=-1, vmax=1, center=0,
    cmap=sn.diverging_palette(20, 220, n=200),
    square=True
    )
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
    )

# Assigns national heatmap to variable.
natl_heatmap_plot = ax.get_figure()

# Outputs national correlation matrix to .csv.
natl_corr.to_csv(outpath + natl_corr_matrix_file)

# Outputs national heatmap to .png.
natl_heatmap_plot.savefig(outpath + natl_heatmap, bbox_inches='tight')
