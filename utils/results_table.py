import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('../results/results.csv')

# Calculate the mean values for accuracy, f1_score, and AUC
df['mean_accuracy'] = df['accuracy'].apply(lambda x: np.mean(eval(x)))
df['mean_f1_score'] = df['f1_score'].apply(lambda x: np.mean(eval(x)))
df['mean_AUC'] = df['AUC'].apply(lambda x: np.mean(eval(x)))

# Select the required columns
latex_df = df[['model', 'mean_accuracy', 'mean_f1_score', 'mean_AUC']]


# Define a function to map model names to features and model type
def map_features_model(model_name):
    if 'CountVectorizer' in model_name:
        feature = 'BoW'
    elif 'TfidfVectorizer' in model_name:
        feature = 'TF-IDF'
    else:
        feature = 'Unknown'

    if 'VP' in model_name:
        feature += '+VP'
    if 'WE1' in model_name:
        feature += '+WE1'
    if 'WE2' in model_name:
        feature += '+WE2'

    if 'RandomForestClassifier' in model_name:
        model = 'RF'
    elif 'CalibratedClassifierCV' in model_name:
        model = 'SVC'
    else:
        model = 'Unknown'

    return feature, model


# Apply the mapping function to create new columns
latex_df['Features'], latex_df['Model'] = zip(*latex_df['model'].apply(map_features_model))

# Sort the dataframe by Features and Model
latex_df = latex_df.sort_values(by=['Features', 'Model'])

# Generate the LaTeX table in the desired format
latex_table = "\\begin{tabular}{l l c c c}\n"
latex_table += "    \\hline\n"
latex_table += "    Features & Model & Accuracy & F1 score & AUC \\\\\n"
latex_table += "    \\hline\n"

current_feature = ""
for index, row in latex_df.iterrows():
    if row['Features'] != current_feature:
        if current_feature != "":
            latex_table += "    \\hline\n"
        current_feature = row['Features']
        latex_table += f"    \\multirow[t]{{2}}{{*}}{{{current_feature}}} & {row['Model']} & {row['mean_accuracy'] * 100:.2f}\\% & {row['mean_f1_score'] * 100:.2f}\\% & {row['mean_AUC'] * 100:.2f}\\% \\\\\n"
    else:
        latex_table += f"                                & {row['Model']} & {row['mean_accuracy'] * 100:.2f}\\% & {row['mean_f1_score'] * 100:.2f}\\% & {row['mean_AUC'] * 100:.2f}\\% \\\\\n"

latex_table += "    \\hline\n"
latex_table += "\\end{tabular}"

# Save the LaTeX table to a file
with open('../results/results_table_custom.tex', 'w') as f:
    f.write(latex_table)

print("The custom LaTeX table has been generated and saved to results_table_custom.tex.")