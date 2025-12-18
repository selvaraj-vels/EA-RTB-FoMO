# File: visualize_results.py
# Purpose: Visualize EA-RTB results (FoMO, CTR, bid shading, auction metrics)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 1. Load results CSV
# -----------------------
results_file = "EA_RTB_results.csv"
df = pd.read_csv(results_file)

# -----------------------
# 2. FoMO Score Distribution
# -----------------------
plt.figure(figsize=(8,6))
sns.histplot(df['FoMO_score'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of FoMO Scores")
plt.xlabel("FoMO Score")
plt.ylabel("Number of Ads")
plt.savefig("fo_mo_distribution.png")
plt.close()

# -----------------------
# 3. CTR vs FoMO Score
# -----------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='FoMO_score', y='ctr', data=df, alpha=0.6)
plt.title("CTR vs FoMO Score")
plt.xlabel("FoMO Score")
plt.ylabel("CTR")
plt.savefig("ctr_vs_fomo.png")
plt.close()

# -----------------------
# 4. Bid Shading Analysis
# -----------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='shading_action', y='final_bid', data=df)
plt.title("Final Bid vs Shading Action")
plt.xlabel("Shading Action Index")
plt.ylabel("Final Bid")
plt.savefig("final_bid_vs_shading.png")
plt.close()

# -----------------------
# 5. Average Reward by Shading Action
# -----------------------
reward_summary = df.groupby('shading_action')['reward'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x='shading_action', y='reward', data=reward_summary, palette='viridis')
plt.title("Average Reward by Shading Action")
plt.xlabel("Shading Action Index")
plt.ylabel("Average Reward")
plt.savefig("reward_by_shading.png")
plt.close()

# -----------------------
# 6. Save summary table
# -----------------------
summary_table = df.groupby('shading_action').agg({
    'ctr':'mean',
    'final_bid':'mean',
    'reward':'mean',
    'FoMO_score':'mean'
}).reset_index()
summary_table.to_csv("EA_RTB_summary_table.csv", index=False)
print("Visualization completed. Figures and summary table saved.")
