"""
Generira sintetički skup podataka koji simulira strukturu
'eCommerce behavior data from multi category store' (Kaggle, mkechinov).

Izvorni Kaggle skup ima preko 100 milijuna redaka (events: view, cart, purchase)
i prevelik je za jednostavan studentski projekt, pa ovdje generiramo manji,
ali strukturno i statistički sličan skup (events log po korisniku/proizvodu),
s namjerno ubačenim nedostacima (null vrijednosti, duplikati, outlieri)
kako bi se mogao demonstrirati cijeli proces čišćenja podataka.

Kolone (isti format kao originalni Kaggle skup):
    event_time, event_type, product_id, category_code, brand, price, user_id, user_session
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_USERS = 4000
CATEGORIES = [
    "electronics.smartphone", "electronics.audio.headphone", "computers.notebook",
    "appliances.kitchen.refrigerators", "apparel.shoes", "apparel.shirt",
    "furniture.living_room.sofa", "construction.tools.light", "sport.bicycle",
    "kids.toys", "accessories.bag", "auto.accessories"
]
BRANDS = ["samsung", "apple", "xiaomi", "lg", "huawei", "bosch", "nike",
          "adidas", "ikea", "logitech", "sony", np.nan]

EVENT_TYPES = ["view", "cart", "purchase"]
EVENT_PROBS = [0.78, 0.14, 0.08]  # purchase je rjeđi event, kao u stvarnosti

start = pd.Timestamp("2023-01-01")
end = pd.Timestamp("2023-03-31")

rows = []
user_ids = rng.integers(100000, 999999, size=N_USERS)

# svakom korisniku dodijelimo "intenzitet" aktivnosti (neki su puno aktivniji)
user_activity = rng.gamma(shape=2.0, scale=3.0, size=N_USERS)

for i, uid in enumerate(user_ids):
    n_events = max(1, int(rng.poisson(lam=user_activity[i] * 3) + 1))
    n_sessions = max(1, n_events // rng.integers(2, 6))
    session_ids = [f"s{uid}_{j}" for j in range(n_sessions)]

    # korisnikova sklonost kupovini (neki kupuju puno, neki nikad)
    purchase_bias = rng.beta(1.5, 6)

    for _ in range(n_events):
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, int((end - start).total_seconds()))))
        cat = rng.choice(CATEGORIES)
        brand = rng.choice(BRANDS)
        base_price = {
            "electronics.smartphone": 450, "electronics.audio.headphone": 80,
            "computers.notebook": 700, "appliances.kitchen.refrigerators": 550,
            "apparel.shoes": 60, "apparel.shirt": 25,
            "furniture.living_room.sofa": 400, "construction.tools.light": 35,
            "sport.bicycle": 300, "kids.toys": 20, "accessories.bag": 45,
            "auto.accessories": 50,
        }[cat]
        price = max(1.0, rng.normal(base_price, base_price * 0.3))

        probs = np.array(EVENT_PROBS, dtype=float)
        probs[2] *= (1 + 4 * purchase_bias)  # pristrani korisnici kupuju češće
        probs = probs / probs.sum()
        etype = rng.choice(EVENT_TYPES, p=probs)

        rows.append({
            "event_time": ts,
            "event_type": etype,
            "product_id": rng.integers(1000000, 9999999),
            "category_code": cat,
            "brand": brand,
            "price": round(price, 2),
            "user_id": uid,
            "user_session": rng.choice(session_ids),
        })

df = pd.DataFrame(rows)

# --- namjerno ubacivanje "prljavih" podataka, kao u stvarnom skupu ---

# 1) nedostajuće vrijednosti u brand (već imamo NaN u listi) i category_code
mask_cat_null = rng.random(len(df)) < 0.03
df.loc[mask_cat_null, "category_code"] = np.nan

mask_price_null = rng.random(len(df)) < 0.01
df.loc[mask_price_null, "price"] = np.nan

# 2) duplicirani redovi (cijeli redovi ponovljeni)
dup_idx = rng.choice(df.index, size=int(len(df) * 0.02), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 3) outlieri u cijeni (npr. greška u unosu / pogrešan decimalni zarez)
outlier_idx = rng.choice(df.index, size=30, replace=False)
df.loc[outlier_idx, "price"] = df.loc[outlier_idx, "price"] * rng.integers(50, 200, size=30)

# 4) promiješaj redove (kao stvaran log)
df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)

df.to_csv("data/ecommerce_events_raw.csv", index=False)
print("Spremljeno:", df.shape)
print(df.head())
