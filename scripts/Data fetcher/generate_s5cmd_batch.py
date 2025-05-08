import random
import os


# Set the random seed for reproducibility
random.seed(42)

def sample_and_split_commands(list_path, dataset_name, output_path, percentage=0.01, train_ratio=0.8):
    with open(list_path, 'r') as f:
        sequence_dirs = [line.strip().replace("DIR  ", "").strip() for line in f if line.strip()]
    
    # Calculate total sample size
    sample_size = max(1, int(len(sequence_dirs) * percentage))
    sampled_dirs = random.sample(sequence_dirs, sample_size)
    
    # Split the sampled directories into train and test sets
    split_idx = int(len(sampled_dirs) * train_ratio)
    train_dirs = sampled_dirs[:split_idx]
    test_dirs = sampled_dirs[split_idx:]
    
    # Write all commands to a single file
    with open(output_path, 'w') as out:
        # Write train commands
        for seq_id in train_dirs:
            cmd = f'cp "s3://argoverse/datasets/av2/sensor/{dataset_name}/{seq_id}*" "./data/train/{seq_id}"\n'
            out.write(cmd)
        
        # Write test commands
        for seq_id in test_dirs:
            cmd = f'cp "s3://argoverse/datasets/av2/sensor/{dataset_name}/{seq_id}*" "./data/test/{seq_id}"\n'
            out.write(cmd)
    
    return len(train_dirs), len(test_dirs)

if __name__ == "__main__":
    # Generate s5cmd batch commands with train/test split in a single file
    train_count, test_count = sample_and_split_commands(
        'train.txt', 
        'train', 
        'download_5percent.s5cmd', 
        percentage=0.02,
        train_ratio=0.8
    )
    
    print(f"Generated commands to download {train_count} training sequences and {test_count} test sequences")
    print(f"Total sequences: {train_count + test_count}")
    print("Run command with:")
    print("s5cmd --no-sign-request run download_5percent.s5cmd")
