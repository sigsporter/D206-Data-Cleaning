churnedCustomers = df.loc[df['Churn'] == 'Yes']
currentCustomers = df.loc[df['Churn'] == 'No']

xChurn = np.unique(churnedCustomers['State'])
yChurn = churnedCustomers['State'].value_counts().sort_index(ascending=True)

# pp.scatter(xAxis, yAxis)
# pp.show()

# txChurn = churnedCustomers.loc[churnedCustomers['State'] == 'TX']
# cols = ['Outage_sec_perweek','Contacts','Yearly_equip_failure','item4']
# print(txChurn[cols].sort_values('Yearly_equip_failure',ascending=False))


# txCurrent = currentCustomers.loc[currentCustomers['State'] == 'TX']

# xCurrent = np.unique(currentCustomers['State'])
# yCurrent = currentCustomers['State'].value_counts().sort_index(ascending=True)

# pp.scatter(xChurn, yChurn)
# pp.show()