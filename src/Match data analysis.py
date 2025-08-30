import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# load the 10 json files with LoL match data in, by starting these variable names
# with an underscore they wont show up in the Variable Explorer panel
_matches1 = pd.read_json("matches1.json", orient = "records", encoding = "latin")
_matches2 = pd.read_json("matches2.json", orient = "records", encoding = "latin")
_matches3 = pd.read_json("matches3.json", orient = "records", encoding = "latin")
_matches4 = pd.read_json("matches4.json", orient = "records", encoding = "latin")
_matches5 = pd.read_json("matches5.json", orient = "records", encoding = "latin")
_matches6 = pd.read_json("matches6.json", orient = "records", encoding = "latin")
_matches7 = pd.read_json("matches7.json", orient = "records", encoding = "latin")
_matches8 = pd.read_json("matches8.json", orient = "records", encoding = "latin")
_matches9 = pd.read_json("matches9.json", orient = "records", encoding = "latin")
_matches10 = pd.read_json("matches10.json", orient = "records", encoding = "latin")

# glue together all 10 of the above files to make one dataframe
matches = pd.concat([_matches1, _matches2, _matches3, _matches4, _matches5, _matches6, _matches7, _matches8, _matches9, _matches10], ignore_index = True)

# extract data about teams, players, and playerIDs to seperate dataframes
teamdata   = pd.json_normalize(data = matches["matches"], record_path = "teams", meta = ["gameId"])
playerdata = pd.json_normalize(data = matches["matches"], record_path = "participants", meta = ["gameId"])
playerIddata = pd.json_normalize(data = matches["matches"], record_path = "participantIdentities", meta = ["gameId"])

# get the matchdata in its own dataframe in a more readable format
matchdata = pd.read_json(matches["matches"].to_json(), orient = "index")
############################################################################### DATA LOADING FROM LAB 5 ABOVE THIS LINE
import json

# load all the item data, its a few steps!
# (some of these lines could be combined though)
_itemFile = open("item.json").readlines()[0]                               
_itemFileDict = json.loads(_itemFile)                                       
itemdata = pd.DataFrame.from_dict(_itemFileDict["data"], orient = "index")  

# copy index column to a new column called itemId, this is useful as we can
# then filter by this column if we need to
itemdata["itemId"] = itemdata.index

# convert our itemId column to numeric data (as opposed to strings or objects)
# this makes the next operations work smoothly as the playerdata also uses numeric
# data for its contents
itemdata["itemId"] = pd.to_numeric(itemdata["itemId"])                                  
###############################################################################
from collections import Counter

# Create a dictionary mapping item IDs to item names
id_to_name = itemdata.set_index('itemId')['name'].to_dict()

# Replace item IDs in stats.item0-6 columns with item names
columns_to_replace = [f'stats.item{i}' for i in range(7)]
playerdata[columns_to_replace] = playerdata[columns_to_replace].replace(id_to_name)  

# Combine stats.item0-6 columns into a single Series (a list of all items used)
combined_items = playerdata[columns_to_replace].stack()

# Use Counter to count how many times each item was used
item_counts = Counter(combined_items)


# get the matchdata in its own dataframe in a more readable format
matchdata = pd.read_json(matches["matches"].to_json(), orient = "index")
######################################################################## DATA LOADING FROM LAB 5 ABOVE THIS LINE

# load the champion file into champdata, then turn the "key" column into numbers as opposed to text
_champs = pd.read_json("champion.json", orient = "records", encoding = "Latin")
champdata = pd.json_normalize(_champs["data"])
champdata["key"] = pd.to_numeric(champdata["key"])

# function to return the championName from champdata when given a numeric id
def GetChampionNameFromId(champId):
    
    # the column on champdata with the ids in is called "key"
    # this filter should give us just the champdata row with
    # a "key" of champId
    filter = champdata["key"] == champId
    champ = champdata[filter]
    
    # return the "id" column from the one row we filtered above
    # the id column on champdata contains the champion name
    return champ.iloc[0]["id"]

# apply above function to playerdata to make an extra column called "championName"
playerdata["championName"] = playerdata["championId"].apply(GetChampionNameFromId)

# what are the 10 most chosen champions
top10chosen = playerdata.groupby("championName").size().nlargest(10)

# Group by championName and stats.win, then count occurrences
champWins = playerdata.groupby(["championName", "stats.win"]).size().unstack(fill_value=0)

# Rename columns for clarity
champWins.columns = ['Losses', 'Wins']

# Calculate total games played per champion
champWins['Total Games'] = champWins['Wins'] + champWins['Losses']

# Calculate win percentage
champWins['Win%'] = (champWins['Wins'] / champWins['Total Games']) * 100

# Optional: Sort by Win% to see the champions with the highest win rates
champWins = champWins.sort_values(by='Win%', ascending=False)


# Create a pie chart for the top 10 most chosen champions with win percentages
plt.figure(figsize=(12, 8))

# Data for pie chart
sizes = top10chosen.values  # Player counts
labels = top10chosen.index  # Champion names
win_percentages = champWins.loc[labels, 'Win%']  # Match win percentages for the top champions

# Create pie chart
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%\n({int(pct * sum(sizes) / 100)} players)",  # Display player count
    startangle=140,
    textprops={'fontsize': 10},
    pctdistance=0.85,
)

# Add a legend that shows win percentages
legend_labels = [f"{label} ({win_percentages.loc[label]:.1f}% Win)" for label in labels]
plt.legend(wedges, legend_labels, title="Champions (Win%)", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

# Add title
plt.title('Player Count and Win% Distribution for Top 10 Champions', fontsize=16)

# Display the pie chart
plt.show()


# Define a mapping of champion names to their primary lanes
lane_mapping = {
    "Jinx"    : "Bottom",
    "Kayn"    : "Jungle",
    "Twitch"  : "Bottom",
    "Yasuo"   : "Mid",
    "Janna"   : "Bottom",
    "Vayne"   : "Bottom",
    "Thresh"  : "Bottom",
    "Tristana": "Bottom",
    "Warwick" : "Jungle",
    "Lucian"  : "Bottom",
    # Add mappings for other champions in your top 10
}

# Filter the top 10 champions and map them to lanes
top10_champions = top10chosen.index.tolist()  # Get the names of the top 10 champions
lanes = [lane_mapping.get(champ, "Unknown") for champ in top10_champions]  # Map champions to lanes

# Count the number of champions in each lane
lane_counts = pd.Series(lanes).value_counts()

# Plot the bar graph
plt.figure(figsize=(10, 6))
lane_counts.reindex(["Top", "Jungle", "Mid", "Bottom",]).plot(kind="bar", color="skyblue", edgecolor="black")

# Add titles and labels
plt.title("Distribution of Top 10 Champions by Lane", fontsize=16)
plt.xlabel("Lane", fontsize=12)
plt.ylabel("Number of Champions", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


###################### SCATTER GRAPH #######################################


# Calculate the total pick rate for all champions
total_picks = playerdata.groupby("championName").size()
total_picks_percentage = (total_picks / total_picks.sum()) * 100

# Calculate the win rates for all champions
champion_wins = playerdata.groupby(["championName", "stats.win"]).size().unstack(fill_value=0)
champion_wins['Total Games'] = champion_wins.sum(axis=1)
champion_wins['Win%'] = (champion_wins[True] / champion_wins['Total Games']) * 100

# Merge pick rate and win rate into a single DataFrame
champion_stats = pd.DataFrame({
    'Pick Rate (%)': total_picks_percentage,
    'Win Rate (%)': champion_wins['Win%']
}).dropna()

# Filter for bottom lane champions in the top 10
bottom_lane_champions = [
    champ for champ in top10_champions if lane_mapping.get(champ) == "Bottom"
]
bottom_lane_stats = champion_stats.loc[bottom_lane_champions]

# Plot the correlation
plt.figure(figsize=(10, 6))

# Scatter plot for bottom lane champions in top 10
plt.scatter(
    bottom_lane_stats['Pick Rate (%)'], 
    bottom_lane_stats['Win Rate (%)'], 
    color='blue', 
    label='Bottom Lane Champions (Top 10)', 
    s=100
)

# Scatter plot for all champions
plt.scatter(
    champion_stats['Pick Rate (%)'], 
    champion_stats['Win Rate (%)'], 
    color='gray', 
    alpha=0.6, 
    label='All Champions'
)

# Add labels and title
plt.title("Correlation between Pick Rate and Win Rate for Bottom Lane Champions (Top 10)", fontsize=16)
plt.xlabel("Pick Rate (%)", fontsize=12)
plt.ylabel("Win Rate (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)

# champions in the top 10 bottom lane
for champ, row in bottom_lane_stats.iterrows():
    plt.text(row['Pick Rate (%)'], row['Win Rate (%)'], champ, fontsize=10)

plt.tight_layout()
plt.show()

# Calculate Pearson's r for Bottom Lane Champions
pearson_r_bottom = bottom_lane_stats['Pick Rate (%)'].corr(bottom_lane_stats['Win Rate (%)'], method='pearson')

# Print the result
print("Pearson's correlation coefficient (r) for Bottom Lane Champions:", pearson_r_bottom)





# Scatter plot for bottom lane champions in the top 10
plt.figure(figsize=(10, 6))
plt.scatter(
    bottom_lane_stats['Pick Rate (%)'], 
    bottom_lane_stats['Win Rate (%)'], 
    color='blue', 
    label='Bottom Lane Champions (Top 10)', 
    s=100
)

# Scatter plot for all champions
plt.scatter(
    champion_stats['Pick Rate (%)'], 
    champion_stats['Win Rate (%)'], 
    color='gray', 
    alpha=0.6, 
    label='All Champions'
)


######### LINE OF BEST FIT ######################################

# Calculate the line of best fit for the bottom lane champions
x = bottom_lane_stats['Pick Rate (%)']
y = bottom_lane_stats['Win Rate (%)']
slope, intercept = np.polyfit(x, y, 1)  # Linear regression: slope and intercept

# Create the regression line
regression_line = slope * x + intercept

# Plot the regression line
plt.plot(x, regression_line, color='red', label=f'Line of Best Fit (y = {slope:.2f}x + {intercept:.2f})')

# Add labels and title
plt.title("Correlation between Pick Rate and Win Rate for Bottom Lane Champions (Top 10)", fontsize=16)
plt.xlabel("Pick Rate (%)", fontsize=12)
plt.ylabel("Win Rate (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)

# Champions in the top 10 bottom lane
for champ, row in bottom_lane_stats.iterrows():
    plt.text(row['Pick Rate (%)'], row['Win Rate (%)'], champ, fontsize=10)

plt.tight_layout()
plt.show()

# Print the equation of the line
print(f"Equation of the line of best fit: y = {slope:.2f}x + {intercept:.2f}")

from scipy.stats import pearsonr

# Extract the pick rate and win rate for bottom lane champions
x = bottom_lane_stats['Pick Rate (%)']
y = bottom_lane_stats['Win Rate (%)']

# Calculate Pearson's correlation coefficient and the p-value
correlation_coefficient, p_value = pearsonr(x, y)

# Print the results
print(f"Pearson's correlation coefficient (r): {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.4e}")



# Extract Pick Rate and Win Rate for all champions
x_all = champion_stats['Pick Rate (%)']
y_all = champion_stats['Win Rate (%)']

# Calculate Pearson's correlation coefficient and the p-value
correlation_coefficient_all, p_value_all = pearsonr(x_all, y_all)

# Print the results
print(f"Pearson's correlation coefficient (r) for all champions: {correlation_coefficient_all:.4f}")
print(f"P-value: {p_value_all:.4e}")


#################### REGRESSION SUMMARY ####################

# Aggregate relevant statistics for bottom lane champions
bottom_lane_features = playerdata[playerdata['championName'].isin(bottom_lane_champions)].groupby('championName').agg({
    'stats.kills': 'mean',           # Average kills
    'stats.totalDamageDealtToChampions': 'mean',  # Average damage done
    'stats.timeCCingOthers': 'mean',  # Crowd control time
    'stats.goldEarned': 'mean',      # Average gold earned
    'stats.win': 'mean'              # Win rate (mean of 1s and 0s gives the win rate)
}).rename(columns={
    'stats.kills': 'Average Kills',
    'stats.totalDamageDealtToChampions': 'Average Damage',
    'stats.timeCCingOthers': 'Crowd Control',
    'stats.goldEarned': 'Average Gold',
    'stats.win': 'Win Rate'
})

# Normalize the Win Rate to percentage
bottom_lane_features['Win Rate (%)'] = bottom_lane_features['Win Rate'] * 100

# Display the prepared data
print(bottom_lane_features)


# Define predictors and the target
X = bottom_lane_features[['Average Kills', 'Average Damage', 'Crowd Control', 'Average Gold']]
y = bottom_lane_features['Win Rate (%)']

# Add a constant to the predictors (intercept term)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression summary
print(model.summary())



############## REGRESSION TABLE ######################################


# Define predictors and response variable
X = bottom_lane_features[['Average Kills', 'Average Damage', 'Crowd Control', 'Average Gold']]
y = bottom_lane_features['Win Rate (%)']

# Add constant for the intercept term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Extract regression summary data
results_summary = pd.DataFrame({
    "Coefficient": model.params,
    "Standard Error": model.bse,
    "T-Value": model.tvalues,
    "P-Value": model.pvalues,
})

# Add a significance column based on the 0.05 threshold
results_summary['Significance'] = results_summary['P-Value'] < 0.05

# Convert to Markdown-friendly table
markdown_table = results_summary.to_markdown(index=True)
print(markdown_table)


######################## ROLE SYNERGY HEAT MAP ##################################


# Step 1: Define lane mapping
lane_mapping = {
    "Jinx": "Bottom",
    "Tristana": "Bottom",
    "Lucian": "Bottom",
    "Vayne": "Bottom",
    "Twitch": "Bottom",
    "Janna": "Support",
    "Thresh": "Support",
    "Leona": "Support",
    "Nami": "Support",
    "Lulu": "Support",

}

# lane column based on championName
playerdata['lane'] = playerdata['championName'].map(lane_mapping).fillna("Unknown")

# Filter for ADC and Support players
adc_data = playerdata[playerdata['lane'] == "Bottom"]
support_data = playerdata[playerdata['lane'] == "Support"]

# Pair ADCs and Supports based on gameId and teamId
adc_support_pairs = adc_data.merge(
    support_data, 
    on=["gameId", "teamId"], 
    suffixes=('_ADC', '_Support')
)

# Count ADC-Support combinations
adc_support_combinations = (
    adc_support_pairs.groupby(['championName_ADC', 'championName_Support'])
    .size()
    .reset_index(name='Count')
    .sort_values(by='Count', ascending=False)
)

# Step 1: Pair ADC and Support champions based on gameId and teamId
adc_support_pairs = adc_data.merge(
    support_data, 
    on=["gameId", "teamId"], 
    suffixes=('_ADC', '_Support')
)

# Step 2: Group by ADC-Support combinations and calculate wins
adc_support_pairs['Win'] = adc_support_pairs['stats.win_ADC'] & adc_support_pairs['stats.win_Support']

# Count the number of wins for each ADC-Support pair
adc_support_combinations = (
    adc_support_pairs.groupby(['championName_ADC', 'championName_Support'])
    .agg(Win_Count=('Win', 'sum'), Total_Count=('Win', 'size'))
    .reset_index()
)

# Step 3: Calculate win rates for each ADC-Support pairing
adc_support_combinations['Win_Rate'] = (adc_support_combinations['Win_Count'] / adc_support_combinations['Total_Count']) * 100

# Step 4: Select the top 10 ADCs and Supports based on frequency
top_adcs = adc_support_combinations['championName_ADC'].value_counts().head(10).index
top_supports = adc_support_combinations['championName_Support'].value_counts().head(10).index

# Filter for the top 10 ADC and Support combinations
adc_support_combinations_filtered = adc_support_combinations[
    (adc_support_combinations['championName_ADC'].isin(top_adcs)) &
    (adc_support_combinations['championName_Support'].isin(top_supports))
]

# Step 5: Pivot the data for the heatmap
heatmap_data = adc_support_combinations_filtered.pivot(
    index='championName_ADC', 
    columns='championName_Support', 
    values='Win_Rate'
).fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt='.2f', 
    cmap='coolwarm',  # Changed to 'coolwarm' for better contrast
    cbar_kws={'label': 'Win Rate (%)'},
    linewidths=0.5,  # Adds borders between cells for clarity
    linecolor='white'  # White borders for better separation
)

# Add titles and labels
plt.title("Top 10 ADC-Support Champion Pairing Win Rates", fontsize=16)
plt.xlabel("Support Champion", fontsize=12)
plt.ylabel("ADC Champion", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show the heatmap
plt.show()

from scipy.stats import chi2_contingency

# Step 1: Prepare the contingency table
# Use `heatmap_data` for the top ADC-Support combinations
contingency_table = heatmap_data.copy()

# Normalize the table to focus on relative distribution of win rates
contingency_table_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

# Chi-Squared test
chi2, p, dof, expected = chi2_contingency(contingency_table_normalized)

# Print the results
print(f"Chi-Squared Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-Value: {p:.4e}")

# Step 3: Interpret the results
if p < 0.05:
    print("The results suggest a significant association between ADC and Support pairings (Reject H0).")
else:
    print("No significant association between ADC and Support pairings (Fail to reject H0).")



