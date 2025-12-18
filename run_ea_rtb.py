# File: run_ea_rtb.py
# Purpose: Full EA-RTB pipeline integration (FoMO + RL Bid Shading)

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models.fomo_model import FoMODataset, FoMOModel
from models.rl_bid_shading import DQN, step

# -----------------------
# 1. Load dataset
# -----------------------
csv_file = "dataset/metadata_preprocessed.csv"  # preprocessed CSV with 'text', 'image_path', 'url'
image_dir = "dataset/images_preprocessed/"

dataset = FoMODataset(csv_file, image_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# -----------------------
# 2. Load FoMO model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fomo_model = FoMOModel().to(device)
fomo_model.eval()  # evaluation mode

# -----------------------
# 3. Load RL model
# -----------------------
input_dim = 4
output_dim = 7
rl_model = DQN(input_dim, output_dim)
rl_model.eval()  # assume pretrained weights loaded if available

# -----------------------
# 4. Simulate auctions
# -----------------------
results = []

for batch in loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    images = batch['image'].to(device)
    
    # Compute FoMO scores
    with torch.no_grad():
        fomo_scores = fomo_model(input_ids, attention_mask, images).squeeze().cpu().numpy()
    
    # Example: simulate simplified win probability and CTR
    # Normally, these would come from predictive models trained on historical auction logs
    p_win = np.random.uniform(0.4, 0.7, size=len(fomo_scores))
    ctr = np.random.uniform(0.02, 0.05, size=len(fomo_scores))
    floor_price = np.random.uniform(1.0, 3.0, size=len(fomo_scores))
    bid_base = np.random.uniform(1.0, 3.0, size=len(fomo_scores))
    
    for i in range(len(fomo_scores)):
        state = np.array([floor_price[i], p_win[i], fomo_scores[i], ctr[i]], dtype=np.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # RL selects shading action
        action_idx = torch.argmax(rl_model(state_tensor)).item()
        
        # Apply shading and compute reward
        next_state, reward, done = step(state, action_idx, bid_base[i], fomo_scores[i], p_win[i], ctr[i], floor_price[i])
        
        results.append({
            'url': batch['url'][i],
            'FoMO_score': fomo_scores[i],
            'base_bid': bid_base[i],
            'shading_action': action_idx,
            'final_bid': bid_base[i]*(1 + [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1][action_idx]),
            'reward': reward,
            'ctr': ctr[i]
        })

# -----------------------
# 5. Save results
# -----------------------
results_df = pd.DataFrame(results)
results_df.to_csv("EA_RTB_results.csv", index=False)
print("EA-RTB simulation completed. Results saved to EA_RTB_results.csv")
