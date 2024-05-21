import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load data
df = pd.read_csv('data.csv')


# Initialize label encoder
le = LabelEncoder()

# Perform label encoding on potential_issue, deck_risk, oe_constraint, ppap_risk, rev_stop, went_on_backorder
df['potential_issue'] = le.fit_transform(df['potential_issue'])
df['deck_risk'] = le.fit_transform(df['deck_risk'])
df['oe_constraint'] = le.fit_transform(df['oe_constraint'])
df['ppap_risk'] = le.fit_transform(df['ppap_risk'])
df['stop_auto_buy'] = le.fit_transform(df['stop_auto_buy'])
df['rev_stop'] = le.fit_transform(df['rev_stop'])
df['went_on_backorder'] = le.fit_transform(df['went_on_backorder'])



# Save the transformed data frame as cleaned_data.csv
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of cleaned dataframe
print(df.head())