# src/generate_mock_data.py
import pandas as pd
import random
from datetime import datetime, timedelta

boroughs = ["Camden", "Westminster", "Tower Hamlets", "Islington", "Hackney", "Kensington and Chelsea"]
property_types = ["Flat", "House", "Studio", "Maisonette"]
agents = ["Prime Estates", "London Homes", "City Lettings", "Metro Realty"]

def random_postcode():
    # simplified UK-like postcodes
    return f"{random.choice(['NW','E','W','SW','SE','N'])}{random.randint(1,9)} {random.randint(1,9)}{random.choice('ABCD')}"

rows = []
for i in range(1, 61):
    pid = f"P{i:03d}"
    borough = random.choice(boroughs)
    ptype = random.choice(property_types)
    bedrooms = random.choice([0,1,2,3,4])
    price = random.choice([250000, 325000, 475000, 550000, 750000, 895000])
    # add some "k" or £ signs randomly to simulate messy data
    price_str = random.choice([f"£{price}", f"{price//1000}k", str(price), f"£{price:,}"])
    rows.append({
        "property_id": pid,
        "address": f"{random.randint(1,200)} {random.choice(['High Street','Road','Lane','Gardens'])}",
        "borough": random.choice([borough.lower(), borough.upper(), borough.title()]),
        "postcode": random_postcode(),
        "property_type": random.choice([ptype, ptype.lower()]),
        "bedrooms": bedrooms,
        "price": price_str,
        "listing_date": (datetime.now() - timedelta(days=random.randint(0,120))).date().isoformat(),
        "agent_name": random.choice(agents)
    })

df = pd.DataFrame(rows)
df.to_csv("data/london_properties.csv", index=False)
print("Wrote data/london_properties.csv with", len(df), "rows")
