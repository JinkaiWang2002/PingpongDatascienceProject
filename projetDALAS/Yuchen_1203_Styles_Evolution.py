import pandas as pd
import matplotlib.pyplot as plt



info_style_numeric = pd.read_csv("ittf_player_info_style_numeric.csv")

# Transform the df into a dict, of which the value is a list containing the numeric values of four styles
styles_dict = info_style_numeric.set_index('Name')[['Playing_hand_numeric', 'Playing_style_numeric', 'Grip_numeric', 'Style_commonness']].apply(lambda row: row.tolist(), axis=1).to_dict()



ranking_not_aggregated = pd.read_csv("ittf_ranking_50.csv")

# Apply styles_dict to all cases of all columns of the ranking_50 df. 
# The result is a tuple of the player name and a list of style infos.
# For those players who are not in the styles_dict, the value is 99.
ranking_styles = ranking_not_aggregated.map(lambda x: (x, styles_dict.get(x, 99)))

# Rename column "Unnamed: 0" to "ranking"
ranking_styles = ranking_styles.rename(columns={"Unnamed: 0": "Ranking"})

# Replace tuples in the "ranking" column with their first element
ranking_styles['Ranking'] = ranking_styles['Ranking'].apply(lambda x: x[0] if isinstance(x, tuple) else x)

# Display result
print(ranking_styles.head())



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
    

