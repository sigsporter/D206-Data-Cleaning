import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank_train = pd.read_csv("C:/Users/sigsp/OneDrive/Desktop/D206 Data Cleaning/Website Data Sets/bank_marketing_training")

bank_train['days_since_previous'] = bank_train['days_since_previous'].replace({999: np.NaN})


plt.hist(bank_train['days_since_previous'])
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.title('Days Since Previous')
plt.show()