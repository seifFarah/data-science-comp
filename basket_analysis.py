import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
import seaborn as sns
import matplotlib.pyplot as plt

# Change this to your file path
file_path = "C:\\Users\\seiff\\Desktop\\data-science-comp\\Order Data.xlsx"
df = pd.read_excel(file_path)

# Sample a portion of the data (e.g., 10%)
df_sample = df.sample(frac=0.1, random_state=1)


# Transform the data: create a matrix where each row is an order and each column is a product
basket = df_sample.groupby(['Order ID', 'Product Name'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Order ID')

# Make any value greater than zero to be equal to 1
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Ensure basket is boolean for apriori algorithm
basket = basket.astype(bool)

# Print basket data to check transformation
print("Basket Data:")
print(basket.head())
print(basket.shape)

# Generate frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(basket, min_support=0.0001, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets.head())
print(frequent_itemsets.shape)

# Generate association rules with a reasonable confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.005)
print("Association Rules:")
print(rules.head())
print(rules.shape)

# Sort rules by lift
rules = rules.sort_values('lift', ascending=False)

# Filter rules with lift > 1
filtered_rules = rules[rules['lift'] > 1]

# Display the filtered rules
top_10_rules = filtered_rules.sort_values('lift', ascending=False).head(10)

# Display the top 10 rules in a table format
print("Top 10 Strongest Relationships:")
print(top_10_rules[['antecedents', 'consequents']])

