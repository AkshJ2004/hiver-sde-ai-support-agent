"""Quick exploration of the full TWCS dataset to understand AppleSupport scale."""
import pandas as pd
from collections import Counter

print("Loading full TWCS dataset...")
df = pd.read_csv("data/raw/twcs.csv")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nInbound vs outbound:")
print(df["inbound"].value_counts())

# Top brands by outbound tweet count
outbound = df[df["inbound"] == False]
brand_counts = outbound["author_id"].value_counts().head(20)
print(f"\nTop 20 brands by outbound (reply) count:")
for brand, count in brand_counts.items():
    print(f"  {brand}: {count:,}")

# AppleSupport specifically
apple_out = outbound[outbound["author_id"] == "AppleSupport"]
apple_in = df[(df["inbound"] == True)]
print(f"\nAppleSupport outbound tweets: {len(apple_out):,}")

# How many inbound tweets have AppleSupport replies?
apple_reply_ids = set()
for resp in apple_out["in_response_to_tweet_id"].dropna():
    apple_reply_ids.add(str(int(float(resp))))

inbound_with_apple_reply = df[df["tweet_id"].astype(str).isin(apple_reply_ids)]
print(f"Inbound tweets with AppleSupport replies: {len(inbound_with_apple_reply):,}")

# Sample a few to understand structure
print(f"\nSample AppleSupport conversations:")
for _, row in inbound_with_apple_reply.head(3).iterrows():
    print(f"  Customer: {str(row['text'])[:120]}")
    # Find the reply
    replies = apple_out[apple_out["in_response_to_tweet_id"] == row["tweet_id"]]
    if len(replies):
        print(f"  Apple:    {str(replies.iloc[0]['text'])[:120]}")
    print()
