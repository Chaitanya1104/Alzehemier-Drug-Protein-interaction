import os
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns

def load_models(models_folder):
    models = {}
    for file in os.listdir(models_folder):
        if file.endswith(".json"):
            model_path = os.path.join(models_folder, file)
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model_name = file.replace("_best_model.json", "")
            models[model_name] = model
    return models

def read_data(csv_path):
    df = pd.read_csv(csv_path, sep=" ", names=["smile", "name"], engine='python')
    counter = 1
    for i, row in df.iterrows():
        if pd.isnull(row['name']):
            df.at[i, 'name'] = f'Ligand_{counter}'
            counter += 1
    return df

def calculate_fp(molecule, method='morgan2_c'):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2)

def to_fingerprint(df, fp_name, verbose=False):
    if verbose:
        print('> Constructing fingerprints from SMILES')
    df[fp_name] = df['smile'].map(lambda s: calculate_fp(Chem.MolFromSmiles(s), method=fp_name))
    return df[['smile', 'name', fp_name]]

def decode_fingerprints(df, fp_column='morgan2_c'):
    decoded_fps = []
    for fp in df[fp_column]:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        decoded_fps.append(arr)
    return np.array(decoded_fps)

def predictor(models_path, data_path):
    models = load_models(models_path)
    data_df = read_data(data_path)
    data_df.dropna(subset=['smile'], inplace=True)

    df_fingerprints = to_fingerprint(data_df, 'morgan2_c')
    df_fingerprints.to_csv('Prediction/fingerprints.csv', index=False)
    df_fps_array = decode_fingerprints(df_fingerprints)

    print(f'Fingerprints shape: {df_fps_array.shape}')

    model_names = list(models.keys())
    df_results = pd.DataFrame(columns=['Ligand'] + model_names)
    df_results['Ligand'] = data_df['name'] + ' [' + data_df['smile'] + ']'

    for model_name, model in models.items():
        pred_prob = model.predict_proba(df_fps_array)[:, 1]
        df_results[model_name] = pred_prob

    df_results.to_csv('Prediction/raw_results.csv', index=False)

    targets = sorted(set(col.split('_')[0] for col in df_results.columns if col != 'Ligand'))
    for target in targets:
        cols = [col for col in df_results.columns if col.startswith(target)]
        df_results[target] = df_results[cols].mean(axis=1)
        df_results.drop(columns=cols, inplace=True)

    df_results.to_csv('Prediction/results.csv', index=False)
    return df_results

def to_graphos(df_results, threshold=0.5):
    df_ligands = pd.DataFrame({'node': df_results['Ligand'], 'type': 'Ligand'})
    df_targets = pd.DataFrame({'node': df_results.columns[1:], 'type': 'Target'})
    df_nodes = pd.concat([df_ligands, df_targets])
    df_nodes.to_csv('Prediction/nodes.csv', index=False)

    df_melted = df_results.melt(id_vars='Ligand', var_name='target', value_name='probability')
    df_melted.to_csv('Prediction/ligand_target_probability.csv', index=False)

    df_melted_filtered = df_melted[df_melted['probability'] >= threshold]
    df_melted_filtered.to_csv('Prediction/ligand_target_probability_filtered.csv', index=False)

def plot_heatmap_filtered(df_results, threshold=0.7, save_path='Prediction/ligand_target_heatmap_filtered.png'):
    df_heatmap = df_results.set_index('Ligand')
    df_filtered = df_heatmap.where(df_heatmap >= threshold)
    df_filtered = df_filtered.dropna(axis=0, how='all')
    df_filtered = df_filtered.dropna(axis=1, how='all')

    if df_filtered.empty:
        print(f"No ligand-target pairs above threshold {threshold}. No heatmap generated.")
        return

    row_order = df_filtered.mean(axis=1).sort_values(ascending=False).index
    col_order = df_filtered.mean(axis=0).sort_values(ascending=False).index
    df_ordered = df_filtered.loc[row_order, col_order]

    plt.figure(figsize=(12, max(6, 0.4 * len(df_ordered))))
    sns.set(font_scale=0.9)
    sns.heatmap(
        df_ordered,
        cmap='Blues',
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor='lightgrey',
        cbar_kws={"label": "Probability"}
    )
    plt.title(f'Ligand-Target Heatmap (Probability ≥ {threshold})', fontsize=15, pad=12)
    plt.xlabel('Targets', fontsize=12)
    plt.ylabel('Ligands', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Filtered heatmap saved to {save_path}")

def rank_ligands_top_k(df_results, top_k=5, save_path='Prediction/ligand_top_k_rankings.csv'):
    df_rankings = df_results.set_index('Ligand')
    top_k_rows = []

    for target in df_rankings.columns:
        top_k_ligands = df_rankings[target].sort_values(ascending=False).head(top_k)
        for ligand, prob in top_k_ligands.items():
            top_k_rows.append({'Ligand': ligand, 'Target': target, 'Probability': prob})

    df_top_k = pd.DataFrame(top_k_rows)
    df_pivot = df_top_k.pivot(index='Ligand', columns='Target', values='Probability')
    df_pivot.to_csv(save_path)

    plot_heatmap_filtered(
        df_pivot.reset_index(),
        threshold=0.0,
        save_path='Prediction/ligand_top_k_heatmap.png'
    )

    print(f"Top-{top_k} ligand rankings and heatmap saved.")

# Run everything
if __name__ == "__main__":
    if not os.path.exists('Prediction'):
        os.makedirs('Prediction')

    models_path = './output/Trained_models'
    data_path = './molecules-to-screen.csv'

    df_results = predictor(models_path, data_path)
    to_graphos(df_results, 0.7)
    plot_heatmap_filtered(df_results, threshold=0.7)
    rank_ligands_top_k(df_results, top_k=5)
