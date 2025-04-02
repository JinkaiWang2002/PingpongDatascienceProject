import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression




## Data Import and Manipulation

info_style_numeric = pd.read_csv("ittf_player_info_style_numeric.csv")

# Clean the "Name" column by removing all non-alphabetic characters
info_style_numeric['Name'] = info_style_numeric['Name'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Transform the df into a dict, of which the value is a list containing the numeric values of four styles
styles_dict = info_style_numeric.set_index('Name')[['Playing_hand_numeric', 'Playing_style_numeric', 'Grip_numeric', 'Style_commonness']].apply(lambda row: row.tolist(), axis=1).to_dict()


ranking_not_aggregated = pd.read_csv("ittf_ranking_50.csv")

# Apply styles_dict to all cases of all columns of the ranking_50 df. 
# The result is a tuple of the player name and a list of style infos.
# For those players who are not in the styles_dict, the value is 99.
ranking_styles = ranking_not_aggregated.map(lambda x: (x, styles_dict.get(x, 99)))

# Rename column "Unnamed: 0" to "ranking"
ranking_styles = ranking_styles.rename(columns={"Unnamed: 0": "Ranking"})

# Replace tuples in the "ranking" column with their first element (to avoid adding 99 to this column)
ranking_styles['Ranking'] = ranking_styles['Ranking'].apply(lambda x: x[0] if isinstance(x, tuple) else x)

# Apply cleaning only to columns *other than* 'Ranking'
ranking_not_aggregated.loc[:, ranking_not_aggregated.columns != 'Ranking'] = (
    ranking_not_aggregated.loc[:, ranking_not_aggregated.columns != 'Ranking']
    .applymap(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)
)

# Display result
print(ranking_styles.head())




## First Round of Ratio Calculation and Visualization (Without Applying Varied Weights)

# The function "extract_and_count" is used to extract the numeric value of each of the four styles from the tuple in each cell
# before counting unique values in the df.
def extract_and_count(dataframe, list_index):    
    extracted_values = dataframe.drop(columns=["Ranking"]).map(
        lambda x: x[1][list_index] if isinstance(x[1], list) else x[1]
    )
    return extracted_values.apply(pd.Series.value_counts)

# Applying the function to extract different indices
playinghand_numeric_value = extract_and_count(ranking_styles, 0)
playingstyle_numeric_value = extract_and_count(ranking_styles, 1)
grip_numeric_value = extract_and_count(ranking_styles, 2)
stylecommonness_numeric_value = extract_and_count(ranking_styles, 3)

# Display the results
print("Playing Hand Numeric Value:\n", playinghand_numeric_value)
print("Playing Style Numeric Value:\n", playingstyle_numeric_value)
print("Grip Numeric Value:\n", grip_numeric_value)
print("Style Commonness Numeric Value:\n", stylecommonness_numeric_value)


# The function "calculate_and_visualize_ratios" is used to calculate the proportion of the common-style players in the whole
# before visualizing the results (a dict) in a line plot.

def calculate_and_visualize_ratios(dataframe, title):
    
    # Step 1: Compute Ratios
    ratios = {}
    for column in dataframe.columns:
        ratio = dataframe[column][0] / (dataframe[column][1] + dataframe[column][0])
        ratios[column] = ratio

    # Step 2: Convert dictionary to two separate lists
    x = list(ratios.keys())  # Categories (e.g., Years)
    y = list(ratios.values())  # Corresponding Ratios

    # Step 3: Create the Line Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')  # Adding markers to highlight data points

    # Add labels and title
    plt.xlabel('Categories')  # Generalized since it can be years, styles, etc.
    plt.ylabel(title)  # Dynamic y-label based on input title
    plt.title(title)

    # Ensure all categories are displayed on the x-axis
    plt.xticks(x, rotation=45)

    # Display the plot
    plt.show()
    
    
list_to_plot = [playinghand_numeric_value, playingstyle_numeric_value, grip_numeric_value, stylecommonness_numeric_value]
calculate_and_visualize_ratios(list_to_plot[0], "Playing Hand Ratios")
calculate_and_visualize_ratios(list_to_plot[1], "Playing Style Ratios")
calculate_and_visualize_ratios(list_to_plot[2], "Grip Ratios")
calculate_and_visualize_ratios(list_to_plot[3], "Style Commonness Ratios")




## Applying Varied Weights to Players of Different Rankings  

# Sample data (assuming ranking_styles is already defined)
data = ranking_styles.copy()  # Create a copy to modify without altering the original

# Define the weight function
def compute_weight(index, max_index):
    return 2 - (index / max_index)

# Calculate the maximum index
max_index = len(data) - 1

# Create a new DataFrame to store the weighted values
weighted_df = data.copy()

# Iterate over each year (column) in the DataFrame
for year in data.columns:
    # Iterate over each player's ranking (row index) and value
    for idx, value in enumerate(data[year]):
        # Apply weight only if the value is a tuple and contains a list
        if isinstance(value, tuple) and isinstance(value[1], list):
            weight = compute_weight(idx, max_index)
            # Multiply the entire list by the weight
            modified_tuple = (value[0], [element * weight for element in value[1]])
            weighted_df.at[idx, year] = modified_tuple  # Store the modified tuple

# Display DataFrame in Jupyter Notebook or VS Code (if using interactive mode)
from IPython.display import display  
display(weighted_df)  # Show DataFrame directly in the output

# Export the wieghted DataFrame to a CSV file
weighted_df.to_csv("styles_weighted_by_rankings.csv", index=False)


## Calculating the Weighted Ratios to See the Evolution of Styles

# Sample DataFrame (Assuming ranking_styles exists)
data = ranking_styles.copy()  # Create a copy to modify without altering the original

# Initialize dictionaries to store the sum and count for each component (0,1,2,3)
component_sums = {i: {} for i in range(4)}  # Store total sums for four components
component_counts = {i: {} for i in range(4)}  # Store valid row counts for each component

# Iterate over each column (year)
for column in data.columns:
    # Initialize total sum and valid count for each component
    total_sums = [0] * 4  # [Sum for list[0], Sum for list[1], Sum for list[2], Sum for list[3]]
    valid_counts = [0] * 4  # Count valid rows that contain a list as second element

    # Iterate over each cell in the column
    for value in data[column]:
        if isinstance(value, tuple) and len(value) > 1 and isinstance(value[1], list):
            for i in range(min(4, len(value[1]))):  # Ensure we do not exceed list length
                total_sums[i] += value[1][i]  # Sum up values for each list element
                valid_counts[i] += 1  # Count how many valid entries exist

    # Store the results for each component
    for i in range(4):
        component_sums[i][column] = total_sums[i] / valid_counts[i] if valid_counts[i] > 0 else 0

# Convert results to a single DataFrame
styles_evolution_weighted_df = pd.DataFrame({
    "Year": list(component_sums[0].keys()),
    "Playing_Hand_Ratio": list(component_sums[0].values()),
    "Playing_Style_Ratio": list(component_sums[1].values()),
    "Grip_Ratio": list(component_sums[2].values()),
    "Style_Commonness_Ratio": list(component_sums[3].values()),
}).set_index("Year")

# Remove the "Ranking" row from styles_evolution_weighted_df
styles_evolution_weighted_df = styles_evolution_weighted_df.drop("Ranking", errors="ignore")

# Esport the final DataFrame to a CSV file
styles_evolution_weighted_df.to_csv("styles_evolution_weighted.csv")

# Display the final DataFrame
styles_evolution_weighted_df.head()


# Generate a line plot for each component style ratio as well as the commonness ratio

# Define the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each style component
for column in styles_evolution_weighted_df.columns:
    ax.plot(styles_evolution_weighted_df.index, styles_evolution_weighted_df[column], label=column)
    
# Set the title and labels
ax.set_title("Weighted Evolution of Playing Styles")
ax.set_xlabel("Year")
ax.set_ylabel("Style Ratio")

# Set the legend
ax.legend()




## Influence of Each Dimensions to the Evolution of the Style Commonness (Multiple Linear Regression)

x1 = styles_evolution_weighted_df['Playing_Hand_Ratio'] 
x2 = styles_evolution_weighted_df['Playing_Style_Ratio']
x3 = styles_evolution_weighted_df['Grip_Ratio']
X = [x1, x2, x3]    
X_T = np.transpose(X)
y = styles_evolution_weighted_df['Style_Commonness_Ratio']

# Initialize and train the model
model = LinearRegression()
# Fit the model
model.fit(X_T, y)

# Display the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# Visualization

# Define the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set the width of the bars
bar_width = 0.4

# Plot the influence of each component
ax.bar(styles_evolution_weighted_df.columns[:3], model.coef_[:3], width=bar_width)

# Set the title and labels
ax.set_title("Influence of Style Components on Style Commonness")

# Display the plot
plt.show()
    

