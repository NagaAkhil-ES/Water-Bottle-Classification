import pandas as pd

def show_class_distribution(l_labels, title=None):
    l_labels = pd.Series(l_labels)
    dist_df = pd.DataFrame({"size": l_labels.value_counts(),
                            "ratio": l_labels.value_counts(normalize=True).round(2)})
    if title is not None:
        print(f"\n{title} distribution")
    print(dist_df, "\n")

# unit test block
if __name__ == "__main__":
    l_temp = [1,0,1,1,0,1,1,1,0,0,0,0,1,1,2,2,0,0,2,2,1]
    show_class_distribution(l_temp, "Temp list")