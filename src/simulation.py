from monte_carlo import simulate_games
import pandas as pd
import matplotlib.pyplot as plt
import os

N = 100000

os.makedirs("../results", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

res_p1 = simulate_games(N, start_player=1)
res_p2 = simulate_games(N, start_player=-1)

df = pd.DataFrame([
    {"Start Player": "Player 1", **res_p1},
    {"Start Player": "Player 2", **res_p2}
])

df.to_csv("../results/results.csv", index=False)

win_rates = [
    res_p1["start_player_wins"] / N,
    res_p2["start_player_wins"] / N
]

plt.figure()
plt.bar(["Player 1 starts", "Player 2 starts"], win_rates)
plt.ylabel("Winning Probability")
plt.title("Winning Probability of the Starting Player")
plt.savefig("../figures/win_rates.png")
plt.close()

print(df)

# ---------------------------------------
# FIGURE 2: Outcome distribution
# ---------------------------------------

labels = ["Start Player Wins", "Other Player Wins", "Draws"]

p1_values = [
    res_p1["start_player_wins"],
    res_p1["other_player_wins"],
    res_p1["draws"]
]

p2_values = [
    res_p2["start_player_wins"],
    res_p2["other_player_wins"],
    res_p2["draws"]
]

x = range(len(labels))

plt.figure()
plt.bar(x, p1_values, label="Player 1 starts")
plt.bar(x, p2_values, bottom=p1_values, label="Player 2 starts")
plt.xticks(x, labels)
plt.ylabel("Number of Games")
plt.title("Outcome Distribution of Connect Four Games")
plt.legend()
plt.savefig("../figures/outcome_distribution.png")
plt.close()