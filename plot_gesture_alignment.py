import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import TOKEN_KEY


def create_gesture_word_alignment_density_plot(data: pd.DataFrame,
                                               offset_key: str,
                                               outfile: str,
                                               title: str = None,
                                               ):
    """Plot density plot with relative alignments of gesture apices to words (word onsets)."""
    # Drop NA entries from dataframe to remove entries without aligned gesture
    data = data.dropna()
    
    # Get unique words aligned with gestures
    words = set(data[TOKEN_KEY].to_list())
    
    # Convert offsets to milliseconds
    data[offset_key] = data[offset_key].apply(lambda seconds: seconds * 1008)

    # Plot the density distributions per word
    plt.figure(figsize=(8, 6))
    for word in words:
        sns.kdeplot(data[data[TOKEN_KEY] == word][offset_key], fill=True, label=f"Apex to <{word}> onset", alpha=0.6, warn_singular=False)
    # Plot total distribution
    sns.kdeplot(data[offset_key], fill=True, label=f"Apex to any demonstrative onset", alpha=0.6)
    # Add axis labels, plot title, legend
    plt.xlabel("Apex before onset < 0 < Apex after onset (ms)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    plt.legend(fontsize="9", loc="best")

    # Plot to outfile and close figure
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
