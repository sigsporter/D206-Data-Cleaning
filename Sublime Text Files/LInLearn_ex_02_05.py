import pandas as pd
import numpy as np

df = pd.read_csv(
    "C:/Users/sigsp/OneDrive/Desktop/D206 Data Cleaning/Exercise Files/chapter2/02_05/tb.csv")

melted = df.melt(['country', 'year'], ['m04', 'm514', 'm014', 'm1524', 'm2534',
                                       'm3544', 'm4554', 'm5564', 'm65',
                                       'mu', 'f04', 'f514', 'f014', 'f1524',
                                       'f2534', 'f3544', 'f4554', 'f5564',
                                       'f65', 'fu'], 'sexage', 'cases')

melted['sex'] = melted['sexage'].str.slice(0, 1)
melted['age'] = melted['sexage'].str.slice(1)

melted['age'] = melted['age'].map({'04': '0-4', '514': '5-14', '1524': '15-24',
                                   '2534': '25-34', '3544': '35-44',
                                   '4554': '45-54', '5564': '55-64',
                                   '65': '65+', 'u': np.NaN})

final = melted.dropna(subset=['cases'])

final.sort_values(['country', 'year', 'age', 'sex'])
final = final[['country', 'year', 'age', 'sex', 'cases']]

final.to_csv('tb_final.csv', index=False)

check = pd.read_csv('tb_final.csv')

print(check.head(-5))
