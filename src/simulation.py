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
