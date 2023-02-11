import pandas as pd
from sklearn.model_selection import train_test_split


from data.stats import show_class_distribution

if __name__ == "__main__":
    # parameters
    metadata_path = "data/csvs/metadata.csv"
    save_dir = "data/csvs"
    seed = 137
    test_size = 0.25

    meta_df = pd.read_csv(metadata_path)
    train_df, test_df = train_test_split(meta_df, test_size=test_size, 
                                         stratify=meta_df.num_label, random_state=seed)
    show_class_distribution(train_df.txt_label, "train")
    show_class_distribution(test_df.txt_label, "test")
    
    train_df.to_csv(f"{save_dir}/train_1.csv", index=False)
    test_df.to_csv(f"{save_dir}/test_1.csv", index=False)
