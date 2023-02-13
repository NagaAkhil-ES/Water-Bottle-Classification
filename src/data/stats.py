import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def show_class_distribution(l_labels, title=None):
    l_labels = pd.Series(l_labels)
    dist_df = pd.DataFrame({"size": l_labels.value_counts(),
                            "ratio": l_labels.value_counts(normalize=True).round(2)})
    if title is not None:
        print(f"\n{title} distribution")
    print(dist_df)
    print(f"total size: {len(l_labels)} \n")

def get_class_weights(l_labels, method=1, f_show=False):
    """l_labels should be list of numeric labels
    
    recommended method is 1"""
    l_labels = pd.Series(l_labels)
    weight_df = pd.DataFrame()
    weight_df["size"] = l_labels.value_counts()
    weight_df["method_1"] = (weight_df["size"].max()/l_labels.value_counts()).round(2)
    weight_df["method_2"] = compute_class_weight(class_weight="balanced", classes=l_labels.unique(), y=l_labels).round(2)
    weight_df["method_3"] = (1/l_labels.value_counts()).round(2)
    weight_df.sort_index(inplace=True)
    if f_show:
        print("class weights\n", weight_df, "\n")
    return weight_df[f"method_{method}"].tolist()

# unit test block
if __name__ == "__main__":
    l_temp = [1,0,1,1,0,1,1,1,0,0,0,0,1,1,2,2,0,0,2,2,1]
    show_class_distribution(l_temp, "Temp list")
    print(get_class_weights(l_temp, method=1, f_show=True))