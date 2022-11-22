# Standard imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pp
import seaborn as sns
from sklearn.decomposition import PCA

# Reading the file
df = pd.read_csv(
    'C:/Users/sigsp/OneDrive/Desktop/D206 Data Cleaning/churn_raw_data.csv')

# Dropping duplicates & checking for changes
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)

# Reviewing columns, nulls, and datatypes
print(df.info())

# Prepping columns: Dropping Unnamed: 0 column, setting index, renaming item1
# to item8 columns
df.drop(columns='Unnamed: 0', inplace=True)

df.set_index('CaseOrder', inplace=True)

survey_dict = {'item1': 'Response',
               'item2': 'Fix',
               'item3': 'Replacement',
               'item4': 'Reliability',
               'item5': 'Options',
               'item6': 'Respectful',
               'item7': 'Courteous',
               'item8': 'Listening'}
df.rename(columns=survey_dict, inplace=True)

# Heatmap to visualize location of null values.
fig, ax = pp.subplots(figsize=(15, 10))
sns.heatmap(df.isnull(), xticklabels=1, cbar=False, ax=ax)

# Dropping rows with > 75% missing data
nullThresh = df.shape[1] * 0.75
df.dropna(thresh=nullThresh, inplace=True)
print(df.shape)

# The following columns have null values that must be addressed:
# Children, Age, Income, Techie, Phone, TechSupport, Tenure, Bandwith_GB_Year
# Analyzing Children variable
maxChild = np.nanmax(df.Children)
minChild = np.nanmin(df.Children)
avgChild = np.nanmean(df.Children)
medChild = np.nanmedian(df.Children)
print("The maximum value in the Children column is ", maxChild)
print("The minimum value in the Children column is ", minChild)
print("The arithmetic mean value in the Children column is ", avgChild)
print("The median value in the Children column is ", medChild)
pp.hist(df.Children, edgecolor='black')
pp.show()

# Imputing using the median, converting to int64 & confirming no nulls remain
df.Children.fillna(medChild, inplace=True)
df['Children'] = df['Children'].astype('int64')
print(df.Children.dtypes)
pp.hist(df.Children, edgecolor='black')
pp.show()
print(df.Children.isnull().any())

# Analyzing the Age variable
maxAge = np.nanmax(df.Age)
minAge = np.nanmin(df.Age)
avgAge = np.nanmean(df.Age)
medAge = np.nanmedian(df.Age)
print("The maximum value in the Age column is ", maxAge)
print("The minimum value in the Age column is ", minAge)
print("The arithmetic mean value in the Age column is ", avgAge)
print("The median value in the Age column is ", medAge)
pp.hist(df.Age, edgecolor='black')
pp.show()

# Imputing using the median, converting to int64, & confirming no nulls remain
df.Age.fillna(medAge, inplace=True)
df['Age'] = df['Age'].astype('int64')
print(df.Age.dtypes)
pp.hist(df.Age, edgecolor='black')
pp.show()
print(df.Age.isnull().any())

# Analyzing the Income variable.
maxInc = np.nanmax(df.Income)
minInc = np.nanmin(df.Income)
avgInc = np.nanmean(df.Income)
medInc = np.nanmedian(df.Income)
print("The maximum value in the Income column is ", maxInc)
print("The minimum value in the Income column is ", minInc)
print("The arithmetic mean value in the Income column is ", avgInc)
print("The median value in the Income column is ", medInc)
pp.hist(df.Income, edgecolor='black')
pp.show()

# Histogram is heavily skewed right with visible outliers. Further analysis is
# required.

# Identifying & analyzing outliers
df['Income_z'] = stats.zscore(df['Income'], nan_policy='omit')
income_outliers = df.query('Income_z > 3 | Income_z < -3')
income_outliers_sort = income_outliers[[
    'Customer_id', 'Income', 'Income_z']].sort_values(['Income_z'],
                                                      ascending=False)
print("The percentage of outliers in the data set is",
      len(income_outliers) / df.shape[0] * 100)

# Removing outliers and recalculating values.
filtered_income = df.drop(df[abs(df['Income_z']) > 3].index)
filteredMaxInc = np.nanmax(filtered_income['Income'])
filteredMinInc = np.nanmin(filtered_income['Income'])
filteredAvgInc = np.nanmean(filtered_income['Income'])
filteredMedInc = np.nanmedian(filtered_income['Income'])
print("The filtered maximum value in the Income column is ", filteredMaxInc)
print("The filtered minimum value in the Income column is ", filteredMinInc)
print("The filtered arithmetic mean value in the Income column is ",
      filteredAvgInc)
print("The filtered median value in the Income column is ", filteredMedInc)
pp.hist(filtered_income['Income'], edgecolor='black')
pp.show()

# Imputing missing values, dropping Income_z column, and confirming changes
# have been made.
df.Income.fillna(filteredMedInc, inplace=True)
df.drop(columns='Income_z', inplace=True)

pp.hist(df.Income, edgecolor='black')
pp.show()
print(df.Age.isnull().any())
print(df.shape)

# Analyzing Techie column.
print(df.Techie.value_counts())
print(df.Techie.isnull().sum())

# Dropping column due to large amount of missing data (nearly 25%) & values are
# self-reported (therefore biased) and confirming it has been dropped.
print(df.shape)
df.drop(columns='Techie', inplace=True)
print(df.shape)

# Analyzing the Phone column using the Multiple column.
print("The Phone column contains", df.Phone.isnull().sum(), "missing values.")
print("The Multiple column contains",
      df.Multiple.isnull().sum(), "missing values.")

# Those with Multiple lines have phone service. Finding Customer_ids with
# multiple lines but phone is null & imputing with 'Yes'
phoneNull = df.loc[df['Phone'].isnull()]
phoneNullMultYes = phoneNull.loc[phoneNull['Multiple'] == 'Yes']
filteredPhoneIds = phoneNullMultYes['Customer_id']
df.loc[df.Customer_id.isin(filteredPhoneIds), 'Phone'] = 'Yes'
print("There are", df.Phone.isnull().sum(), "remaining missing values.")

# Checking mode
print(df.Phone.mode())

# Imputing remaining missing values with mode & confirming no nulls remain.
df.Phone.fillna('Yes', inplace=True)
print(df.Phone.value_counts())
print(df.Phone.isnull().any())

# Analyzing TechSupport column.
print(df.TechSupport.value_counts())
print(df.TechSupport.isnull().sum())

# Checking mode
print(df.TechSupport.mode())

# Imputing with mode & confirming no nulls remain.
df.TechSupport.fillna('No', inplace=True)
print(df.TechSupport.value_counts())
print(df.TechSupport.isnull().any())

# Analyzing Tenure column.
maxTenure = np.nanmax(df['Tenure'])
minTenure = np.nanmin(df['Tenure'])
avgTenure = np.nanmean(df['Tenure'])
medTenure = np.nanmedian(df['Tenure'])
print("The maximum value in the Tenure column is ", maxTenure)
print("The minimum value in the Tenure column is ", minTenure)
print("The arithmetic mean value in the Tenure column is ", avgTenure)
print("The median value in the Tenure column is ", medTenure)
pp.hist(df['Tenure'], edgecolor='black')
pp.show()

# Imputing with mode & confirming no nulls remain.
df.Tenure.fillna(medTenure, inplace=True)
pp.hist(df['Tenure'], edgecolor='black')
pp.show()
print(df.Tenure.isnull().any())

# Analyzing Bandwidth_GB_Year column.
maxBandwidth = np.nanmax(df['Bandwidth_GB_Year'])
minBandwidth = np.nanmin(df['Bandwidth_GB_Year'])
avgBandwidth = np.nanmean(df['Bandwidth_GB_Year'])
medBandwidth = np.nanmedian(df['Bandwidth_GB_Year'])
print("The maximum value in the Bandwidth_GB_Year column is ", maxBandwidth)
print("The minimum value in the Bandwidth_GB_Year column is ", minBandwidth)
print("The arithmetic mean value in the Bandwidth_GB_Year column is ",
      avgBandwidth)
print("The median value in the Bandwidth_GB_Year column is ", medBandwidth)
pp.hist(df['Bandwidth_GB_Year'], edgecolor='black')
pp.show()

# Imputing with median & confirming no nulls remain
df['Bandwidth_GB_Year'].fillna(medBandwidth, inplace=True)
pp.hist(df['Bandwidth_GB_Year'], edgecolor='black')
pp.show()
print(df.Bandwidth_GB_Year.isnull().any())

# Confirming all nulls have been replaced in the dataframe
print(df.isnull().any())

# All nulls have been imputed. Review remaining variables for errors and/or
# outliers.

# Analyzing City column.
print(len(df.City.unique()))

# This list accounts for > 60% of the values in the column. It is unreasonable
# to review this manually, so the column will be left as-is.

# Analyzing State column.
print(df.State.value_counts())
print(len(df.State.value_counts()))

# No apparent errors in state column.

# Analyzing County column.
print(len(df.County.unique()))

# This list accounts for > 16% of the values in the column. It is unreasonable
# to review this manually, so the column will be left as-is.

# Analyzing Zip column.
# The range for zip codes in the US is 00001 to 99950
print(min(df.Zip))
print(max(df.Zip))

# The minimum and maximum values are within the defined range for zip codes.

# Analyzing Lat and Lng columns.
# Latitude range: -90 to 90, Longitude range: -180 to 180
print(max(df.Lat))
print(min(df.Lat))
print(max(df.Lng))
print(min(df.Lng))

# These values are within the respective ranges.

# Analyzing Popluation column.
# Analyzing Bandwidth_GB_Year column.
maxPop = np.nanmax(df['Population'])
minPop = np.nanmin(df['Population'])
avgPop = np.nanmean(df['Population'])
medPop = np.nanmedian(df['Population'])
print("The maximum value in the Population column is ", maxPop)
print("The minimum value in the Population column is ", minPop)
print("The arithmetic mean value in the Population column is ", avgPop)
print("The median value in the Population column is ", medPop)
pp.hist(df.Population, edgecolor='black')
pp.show()

# Analyzing the 20 largest Population values.
cityPopSorted = df[['City', 'State', 'Population']
                   ].sort_values(by='Population', ascending=False)
print(cityPopSorted.head(20))

# These values correspond to highly populated areas and do not appear to be a
# recording error. While the zeroes in the population column seem unlikely, the
# values were pulled from census data and I do not feel properly equipped to
# replace them.

# Analyzing Timezone column.
print(df.Area.value_counts())
print(df.Timezone.value_counts())

# The outputs are as expected.

# Analyzing Job column.
print(len(df.Job.unique()))
print(df.Job.head(20))

# These values are self-reported and categorical. They may no longer be true,
# or may never have been true. With such a wide variance in entries, it would
# be challenging to draw any reasonable conclusions based on this variable.
# Therefore, it will be dropped.

# Dropping Job column & confirming it has been dropped.
print(df.shape)
df.drop(columns='Job', inplace=True)
print(df.shape)

# Analyzing Education, Employment, Marital, Gender, and Churn columns.
print(df.Education.value_counts())
print(df.Employment.value_counts())
print(df.Marital.value_counts())
print(df.Gender.value_counts())
print(df.Churn.value_counts())

# There are no apparent errors in these outputs.

# Analyzing Outage_sec_perweek
print(df.Outage_sec_perweek.max())
print(df.Outage_sec_perweek.min())
print(df.Outage_sec_perweek.mean())
print(df.Outage_sec_perweek.median())
pp.hist(df.Outage_sec_perweek, edgecolor='black')
pp.show()

# It appears that there are some negative values for time which cannot be true.
# There is a small grouping of values on the high end but the maximum value is
# still reasonable.

negOutage = df.query('Outage_sec_perweek < 0')
print(negOutage.shape)
print(negOutage.Outage_sec_perweek)

# There are only 11 entries that are negative, all of which are close to zero.
# I will replace these with zero as they are likely due to some recording error
# and such a minor change will not dramatically affect the data set.

# Replacing negative values in Outage_sec_perweek
negOutageIds = negOutage.Customer_id
df.loc[df.Customer_id.isin(negOutageIds), 'Outage_sec_perweek'] = '0'

# Analyzing Email, Contacts, and Yearly_equip_failure columns.
pp.hist(df.Email, bins=5, edgecolor='black')
pp.show()

pp.hist(df.Contacts, bins=5, edgecolor='black')
pp.show()

pp.hist(df.Yearly_equip_failure, bins=5, edgecolor='black')
pp.show()

# Email appears normally distributed. Contacts and Yearly_equip_failure are
# skewed right, but reasonable.

# Analyzing several categorical columns using value_counts
cols = ['Contract', 'Port_modem', 'Tablet', 'InternetService', 'Multiple',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'StreamingTV',
        'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']

for col in cols:
    print(df[col].value_counts())

# The outputs are as expected.

# Analyzing MonthlyCharge column.
pp.hist(df.MonthlyCharge, bins=5, edgecolor='black')

# There are no apparent errors.

# Analyzing Survey Response columns
cols2 = ['Response', 'Fix', 'Replacement', 'Reliability', 'Options',
         'Respectful', 'Courteous', 'Listening']

for col in cols2:
    print(df[col].value_counts())

# All outputs are as expected.

# Exporting cleaned data set to CSV
df.to_csv(
    r'C:/Users/sigsp/OneDrive/Desktop/D206 Data Cleaning/D206 PA/'
    'churn_data_cleaned.csv')

# Performing PCA on numeric columns
numericCols = df.select_dtypes(exclude=['object']).columns.tolist()

dfNumeric = df[numericCols]

numericCols_normalized = (dfNumeric - dfNumeric.mean()) / dfNumeric.std()

pca = PCA(n_components=dfNumeric.shape[1])

# Loop to name PC columns based on the number of numeric columns in the
# dataframe
PCcols = []
i = 1
while i < len(numericCols) + 1:
    col = 'PC' + str(i)
    PCcols.append(col)
    i += 1

pca.fit(numericCols_normalized)
numeric_pca = pd.DataFrame(pca.transform(
    numericCols_normalized), columns=numericCols)

cov_matrix = np.dot(numericCols_normalized.T,
                    numericCols_normalized) / dfNumeric.shape[0]
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector))
               for eigenvector in pca.components_]

pp.plot(eigenvalues, marker='x', linestyle='solid')
pp.hlines(1, xmin=0, xmax=len(numericCols) + 1, linestyles='dashed')
pp.xlabel('number of components')
pp.ylabel('eigenvalue')
pp.show()

# Determine the number of eigenvalues that are greater than 1.
pcCount = 0
for eigenvalue in eigenvalues:
    if eigenvalue > 1:
        pcCount += 1
print(pcCount)

# Printing loadings for only the first 8 PCs
loading = pd.DataFrame(pca.components_.T, columns=PCcols,
                       index=dfNumeric.columns)
print(loading[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']])
