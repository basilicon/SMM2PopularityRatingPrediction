import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(16,8))

source_file = "./data/tabular_only.csv"
df = pd.read_csv(source_file)

gamestyle_counts = df['gamestyle'].value_counts()
gamestyle_counts.plot(kind='bar', ax=axes[0], figsize=(12,4), color='skyblue', alpha=0.8)
gamestyle_mapping = {
    0: "SMB1",
    1: "SMB3",
    2: "SMW",
    3: "NSMBU",
    4: "SM3DW"
}
axes[0].set_xticklabels([gamestyle_mapping[label] for label in gamestyle_counts.index])
axes[0].set_title("Gamestyle Distribution")
axes[0].set_xlabel("Gamestyle")

theme_counts = df['theme'].value_counts()
theme_counts.plot(kind='bar', ax=axes[1], figsize=(12,4), color='skyblue', alpha=0.8)
theme_mapping = {
    0: "Overworld",
    1: "Underground",
    2: "Castle",
    3: "Airship",
    4: "Underwater",
    5: "Ghost house",
    6: "Snow",
    7: "Desert",
    8: "Sky",
    9: "Forest"
}
axes[1].set_xticklabels([theme_mapping[label] for label in theme_counts.index])
axes[1].set_title("Theme Distribution")
axes[1].set_xlabel("Theme")

tag_counts = pd.concat([df['tag1'], df['tag2']], axis=0).value_counts()
tag_counts.plot(kind='bar', ax=axes[2], figsize=(12,4), color='skyblue', alpha=0.8)
tag_mapping = {
    0: "None",
    1: "Standard",
    2: "Puzzle solving",
    3: "Speedrun",
    4: "Autoscroll",
    5: "Auto mario",
    6: "Short and sweet",
    7: "Multiplayer versus",
    8: "Themed",
    9: "Music",
    10: "Art",
    11: "Technical",
    12: "Shooter",
    13: "Boss battle",
    14: "Single player",
    15: "Link"
}
axes[2].set_xticklabels([tag_mapping[label] for label in tag_counts.index])
axes[2].set_title("Tag Distribution")
axes[2].set_xlabel("Tag")

timer_bins = [0, 100, 200, 300, 400, 500]
timer_labels = ['0-90', '100-190', '200-290', '300-390', '400-500']
df['timer-range'] = pd.cut(df['timer'], bins=timer_bins, labels=timer_labels, right=False)
timer_counts = df['timer-range'].value_counts(sort=False)
timer_counts.plot(kind='bar', ax=axes[3], figsize=(12,4), color='skyblue', alpha=0.8)
axes[3].set_title("Timer Distribution")
axes[3].set_xlabel("Timer")

axes[4].hist(df['likes-norm'], bins=10, color='skyblue', alpha=0.8)
axes[4].set_yscale('log')
axes[4].set_title("Distribution of normalized likes")

plt.tight_layout()
plt.show()