"""Generate a small dataset with the exact Altman schema, including a planted
fraud pattern (bursts of high-value online transactions on a compromised card)
so feature and leakage tests have something realistic to bite on."""
import numpy as np, pandas as pd, sys

rng = np.random.default_rng(7)
N_USERS, N_DAYS = 60, 400
rows = []
start = pd.Timestamp("2018-01-01")
MCCS = [5411, 5812, 5541, 4121, 7995, 5999, 4829]
STATES = ["CA","NY","TX","FL","WA","IL"]

for user in range(N_USERS):
    for card in range(rng.integers(1, 3)):
        t = start + pd.Timedelta(days=float(rng.uniform(0, 30)))
        home = rng.choice(STATES)
        base = rng.uniform(20, 120)
        compromised_at = (start + pd.Timedelta(days=float(rng.uniform(60, N_DAYS-10)))
                          if rng.random() < 0.25 else None)
        while t < start + pd.Timedelta(days=N_DAYS):
            t += pd.Timedelta(hours=float(rng.exponential(30)))
            if t >= start + pd.Timedelta(days=N_DAYS): break
            # legitimate traffic is a realistic mix of channels, otherwise
            # channel alone separates the classes and every model looks perfect
            chip = rng.choice(["Swipe Transaction", "Chip Transaction", "Online Transaction"],
                              p=[0.45, 0.35, 0.20])
            fraud, amt, state = 0, abs(rng.normal(base, base*0.4)), home
            if compromised_at is not None and compromised_at <= t < compromised_at + pd.Timedelta(hours=6):
                # burst: several large online charges, often out of state
                fraud, amt = 1, abs(rng.normal(base*7, base*2))
                state = rng.choice(STATES)
                chip = rng.choice(["Online Transaction", "Swipe Transaction"], p=[0.75, 0.25])
            elif rng.random() < 0.004:
                fraud, amt = 1, abs(rng.normal(base*5, base*2))
                chip = rng.choice(["Online Transaction", "Chip Transaction"], p=[0.7, 0.3])
            rows.append({
                "User": user, "Card": card, "Year": t.year, "Month": t.month, "Day": t.day,
                "Time": f"{t.hour:02d}:{t.minute:02d}", "Amount": f"${amt:.2f}",
                "Use Chip": chip, "Merchant Name": str(rng.integers(10**14, 10**15)),
                "Merchant City": "City" + str(rng.integers(1, 40)), "Merchant State": state,
                "Zip": str(rng.integers(10000, 99999)), "MCC": int(rng.choice(MCCS)),
                "Errors?": "" if rng.random() > .03 else "Bad PIN",
                "Is Fraud?": "Yes" if fraud else "No",
            })

df = pd.DataFrame(rows)
out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df):,} rows to {out} | fraud {df['Is Fraud?'].eq('Yes').mean():.4%}")
