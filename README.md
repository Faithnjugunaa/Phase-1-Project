import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Read the csv
df = pd.read_csv("/Aviation_Data.csv", low_memory=False)
df

Data exploration

df.shape

df.columns

#First 5 rows
df.head(5)

#Type of objects & missing values visibility
df.info()

Find missing values in a DataFrame

Data Cleaning & Missing Value Imputation

# Your code here
# check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print("Missing values:\n", missing)

# Drop irrelevant or consistently missing columns (example)
df.drop(columns=['investigation_type', 'publication_date'], inplace=True, errors='ignore')

To Standardize or Normalize all column names

df.info()

#Clean Column Names
# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace('.', '_').str.replace(' ', '_')

# Check again
print(df.columns.tolist())

df['event_date']

# Convert dates
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df[df['event_date'].notnull()]  # remove rows with invalid dates


df.head(3)

df['aircraft_damage']

df['engine_type']

df['make']

df.loc[:, 'aircraft_damage'] = df['aircraft_damage'].fillna('Unknown')
df.loc[:, 'engine_type'] = df['engine_type'].fillna('Unknown')
df.loc[:, 'make'] = df['make'].fillna('Unknown')


Check the actual column names

print(df.columns.tolist())

Risk Aggregation - Define risk based on fatalities, injuries, damage severity etc

# Aircraft accident count by manufacturer
top_makes = df['make'].value_counts().head(10)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x=top_makes.values, y=top_makes.index)
plt.title("Top 10 Aircraft Manufacturers by Number of Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Manufacturer")
plt.tight_layout()
plt.show()

to check aircraft make, model, and their corresponding risk score and injury counts.

# Filter valid dates and create a safe copy
df = df[df['event_date'].notnull()].copy()

# Fill missing injury values
df.loc[:, 'total_fatal_injuries'] = df['total_fatal_injuries'].fillna(0)
df.loc[:, 'total_serious_injuries'] = df['total_serious_injuries'].fillna(0)
df.loc[:, 'total_minor_injuries'] = df['total_minor_injuries'].fillna(0)

# Calculate risk score
df.loc[:, 'risk_score'] = (
    df['total_fatal_injuries'] * 3 +
    df['total_serious_injuries'] * 2 +
    df['total_minor_injuries'] * 1
)

# Confirm changes by printing first few rows
print(df[['make', 'model', 'risk_score', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries']].head())


 See Top Low-Risk Manufacturers (Summary Table)

# Top low-risk Manufacures(Summary Table)
# Group by aircraft manufacturer
risk_summary = df.groupby('make').agg({
    'risk_score': 'mean',
    'event_id': 'count'
}).rename(columns={'event_id': 'total_incidents'}).sort_values(by='risk_score')

# Show top 10 manufacturers with lowest average risk
print(risk_summary.head(10))


print("SHAPE OF DF:", df.shape)
print(df[['make', 'risk_score']].dropna().head(10))


import plotly.express as px

# Prepare summary DataFrame for scatter
risk_plot_df = risk_summary.reset_index()

fig = px.scatter(
    risk_plot_df,
    x='total_incidents',
    y='risk_score',
    hover_name='make',
    size='total_incidents',
    color='risk_score',
    color_continuous_scale='RdYlGn_r',
    title='Aircraft Manufacturers: Incidents vs. Risk Score'
)
fig.update_layout(xaxis_title='Total Incidents', yaxis_title='Average Risk Score')
fig.show()


Objective to find our the airline with less accident
Aircfraft by model, category, flight

which aircraft types are involved in the most accidents

accidents_by_type = df['model'].value_counts()
print(accidents_by_type.head(10))

Visualization

import matplotlib.pyplot as plt

top_models = accidents_by_type.head(10)

plt.figure(figsize=(10, 6))
top_models.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Accidents')
plt.ylabel('Aircraft Model')
plt.title('Top 10 Aircraft Models by Accident Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


print(df['investigation_type'].unique())


print(df.columns.tolist())

 Filter only accidents (exclude incidents)

accidents_df = df[df['investigation_type'] == 'Accident']  # ✅ correct
accidents_df

print(accidents_df['investigation_type'].value_counts())

# Step 5: Count number of accident records
num_accidents = accidents_df.shape[0]

print(f"Total number of accidents: {num_accidents}")

# Correct column names
accidents_by_make_model = accidents_df.groupby(['make', 'model']).size().reset_index(name='accident_count')
accidents_by_make_model

# Sort by number of accidents in descending order
accidents_by_make_model_sorted = accidents_by_make_model.sort_values(by='accident_count', ascending=False)
accidents_by_make_model_sorted

# By manufacturer (Make)
df['make'].value_counts()

# Or by specific aircraft model
df['model'].value_counts()

accidents_df = df[df['investigation_type'] == 'Accident']
accidents_df['make'].value_counts()

print(df.columns.tolist())

# Prepare data safely
top10 = accidents_by_make_model_sorted.head(10).copy()
bottom10 = accidents_by_make_model_sorted[accidents_by_make_model_sorted['accident_count'] > 0].tail(10).copy()

# Create aircraft label
top10['aircraft'] = top10['make'] + " " + top10['model']
bottom10['aircraft'] = bottom10['make'] + " " + bottom10['model']

# Set up subplots
fig, axes = plt.subplots(ncols=2, figsize=(18, 8))

# Top 10 plot
sns.barplot(data=top10, x='accident_count', y='aircraft', ax=axes[0])
axes[0].set_title("Top 10 Aircraft Make-Models by Accident Count")
axes[0].set_xlabel("accident_count")
axes[0].set_ylabel("")

# Bottom 10 plot
sns.barplot(data=bottom10, x='accident_count', y='aircraft', ax=axes[1])
axes[1].set_title("Bottom 10 Aircraft Make-Models by Accident Count")
axes[1].set_xlabel("accident_count")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

#common causes of accidents
print(df.columns.tolist())

# Check for common phases of flight where accidents occurred
top_causes = df['broad_phase_of_flight'].value_counts().head(10)

# Display the results
print("Top 10 Common Phases of Flight in Accidents:")
print(top_causes)


Saving the data to a clean version

df.to_csv("Refined_aviation_data.csv", index=False)
