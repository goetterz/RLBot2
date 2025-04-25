import os
import torch

# Path to your checkpoint directory
checkpoint_dir = "data/checkpoints/rlgym-ppo-run"

# Find the most recent checkpoint folder
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    latest_checkpoint = max(os.listdir(checkpoint_dir), key=lambda d: int(d))
    checkpoint_folder = f"{checkpoint_dir}/{latest_checkpoint}"
    print(f"Found checkpoint folder: {checkpoint_folder}")
else:
    print("No checkpoint folders found!")
    exit(1)

# Inspect what files are in the checkpoint folder
files = os.listdir(checkpoint_folder)
print("Files in checkpoint folder:", files)

# Look specifically for the policy file
policy_file = None
for file in files:
    if "POLICY" in file and "OPTIMIZER" not in file:
        policy_file = file
        break

if policy_file:
    policy_path = f"{checkpoint_folder}/{policy_file}"
    print(f"Loading policy from {policy_path}")
    
    # Load the policy
    policy_state_dict = torch.load(policy_path, map_location=torch.device('cpu'))
    
    # Print information about the loaded file
    print(f"Policy loaded, type: {type(policy_state_dict)}")
    
    # Create the structure expected by the bot
    if isinstance(policy_state_dict, dict):
        # It's already a state dict
        output_dict = {"policy_state_dict": policy_state_dict}
    else:
        # It might be the full model, try to get its state dict
        try:
            output_dict = {"policy_state_dict": policy_state_dict.state_dict()}
            print("Extracted state_dict from the model")
        except AttributeError:
            print("Could not extract state_dict, using the loaded object directly")
            output_dict = {"policy_state_dict": policy_state_dict}
    
    # Save to model.pt
    model_path = f"{checkpoint_folder}/model.pt"
    torch.save(output_dict, model_path)
    print(f"Saved model to {model_path}")
else:
    print("Could not find a policy file in the checkpoint folder")