import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Always load the CSV from the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "last_150_playoff_games_quarter_scores.csv")

# Load
df = pd.read_csv(CSV_PATH)

# Team -> conference map
AFC_TEAMS = {
    "BAL","BUF","CIN","CLE","DEN","HOU","IND","JAX","KC","LAC","LV","MIA","NE","NYJ","PIT","TEN"
}
NFC_TEAMS = {
    "ARI","ATL","CAR","CHI","DAL","DET","GB","LA","MIN","NO","NYG","PHI","SEA","SF","TB","WAS"
}

def conf(team: str):
    if team in AFC_TEAMS:
        return "AFC"
    if team in NFC_TEAMS:
        return "NFC"
    return None

df["away_conf"] = df["away_team"].apply(conf)
df["home_conf"] = df["home_team"].apply(conf)

def true_final_scores(row) -> tuple[int, int]:
    """
    Playoff rule:
    - If OT exists (Q5 cumulative totals present), use that (final score).
    - Otherwise use Q4 cumulative totals (end of regulation = final).
    This prevents treating end-of-Q4 ties as 'final' outcomes.
    """
    if (not pd.isna(row.get("away_cum_Q5"))) and (not pd.isna(row.get("home_cum_Q5"))):
        return int(row["away_cum_Q5"]), int(row["home_cum_Q5"])
    return int(row["away_cum_Q4"]), int(row["home_cum_Q4"])

# Build (NFC_last_digit, AFC_last_digit) pairs using your conference-aware logic
# with corrected OT handling (final score always trumps Q4 if OT happened).
pairs = []

for _, r in df.iterrows():
    away_final, home_final = true_final_scores(r)
    ac = r["away_conf"]
    hc = r["home_conf"]

    # NFC vs AFC (Super Bowl)
    if (ac in ("AFC", "NFC")) and (hc in ("AFC", "NFC")) and (ac != hc):
        if ac == "NFC":
            nfc_score, afc_score = away_final, home_final
        else:
            nfc_score, afc_score = home_final, away_final
        pairs.append((nfc_score % 10, afc_score % 10))

    else:
        # Same-conference games: add both directions (team vs opponent)
        pairs.append((away_final % 10, home_final % 10))
        pairs.append((home_final % 10, away_final % 10))

# 10x10 matrix: rows = NFC digit (0-9), cols = AFC digit (0-9)
mat = np.zeros((10, 10), dtype=int)
for n, a in pairs:
    mat[int(n), int(a)] += 1

prob = mat / mat.sum()

# Heatmap
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(prob, aspect="equal")

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))

ax.set_xlabel("AFC last digit (final score)")
ax.set_ylabel("NFC last digit (final score)")

ax.set_title(
    "NFL Squares Heatmap (Playoffs)\n"
    "NFC vs AFC Final Score Last Digits (Conference-aware, OT-corrected, last 150 games)"
)

for i in range(10):
    for j in range(10):
        if mat[i, j] > 0:
            ax.text(j, i, f"{prob[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
plt.tight_layout()
plt.show()

# Bar chart of top combos (horizontal)
counts = {}
for n, a in pairs:
    key = f"{n}-{a}"
    counts[key] = counts.get(key, 0) + 1

total = sum(counts.values())

combo_df = (
    pd.DataFrame([{"Combo": k, "Count": v, "Probability": v / total} for k, v in counts.items()])
    .sort_values("Probability", ascending=False)
    .head(15)
)

plt.figure(figsize=(8, 6))
plt.barh(combo_df["Combo"], combo_df["Probability"])
plt.gca().invert_yaxis()
plt.xlabel("Probability")
plt.title(
    "Most Common Final Score Last-Digit Combos (Playoffs)\n"
    "Conference-aware, OT-corrected, last 150 games"
)

for i, p in enumerate(combo_df["Probability"]):
    plt.text(p, i, f"{p * 100:.1f}%", va="center")

plt.tight_layout()
plt.show()

# Print top combos
print("\nTop NFC/AFC last-digit combos (OT-corrected):")
for _, row in combo_df.iterrows():
    print(f'{row["Combo"]}: {int(row["Count"])} ({row["Probability"]*100:.2f}%)')
