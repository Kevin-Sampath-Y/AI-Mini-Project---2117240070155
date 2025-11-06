# ============================================================
# Sentiment Analysis Dataset Generator
# Creates sentiment_dataset.csv with 1000 reviews
# Distribution: 520 positive, 480 negative
# ============================================================

import pandas as pd
import random

# Positive reviews
positive_reviews = [
    "This product is amazing and works perfectly!",
    "Absolutely loved it, highly recommended!",
    "Excellent quality, exceeded my expectations!",
    "Best purchase I've made this year!",
    "Outstanding service and great product!",
    "Fantastic experience, will buy again!",
    "Incredible value for money!",
    "Very satisfied with this purchase!",
    "This is exactly what I needed!",
    "Superb quality and fast delivery!",
    "Amazing product, love it so much!",
    "Highly recommend to everyone!",
    "Perfect in every way possible!",
    "Great product, very happy with it!",
    "Exceeded all my expectations!",
    "Wonderful experience from start to finish!",
    "Top quality product, worth every penny!",
    "Brilliant purchase, very pleased!",
    "Exceptional quality and service!",
    "Love this, best decision ever!",
    "Five stars, absolutely fantastic!",
    "This is the best thing I've bought!",
    "Truly amazing, can't fault it!",
    "Outstanding product, works great!",
    "Extremely satisfied with this!",
    "Impressive quality and performance!",
    "Excellent product, highly satisfied!",
    "Perfect product, no complaints!",
    "Very good quality, great value!",
    "Amazing, better than expected!",
    "Great quality and reasonable price!",
    "Wonderful product, love using it!",
    "Fantastic quality, very impressed!",
    "Highly recommended, excellent buy!",
    "Superior quality, great purchase!",
    "Awesome product, totally worth it!",
    "Very pleased with this purchase!",
    "Excellent value and quality!",
    "Best quality I've seen!",
    "Absolutely perfect, love it!",
    "Great buy, very satisfied!",
    "Amazing experience overall!",
    "Top-notch quality and service!",
    "Perfect condition, works great!",
    "Brilliant product, highly recommend!",
    "Extremely happy with purchase!",
    "Outstanding value for money!",
    "Love everything about this!",
    "Fantastic product and service!",
    "Very good, meets all needs!",
    "Excellent purchase, no regrets!",
    "Amazing quality, very durable!",
    "Great product, fast shipping!",
    "Wonderful, exactly as described!",
    "Very impressed with quality!",
    "Perfect fit, works perfectly!",
    "Awesome quality, love it!",
    "Highly satisfied, great product!",
    "Brilliant quality and design!",
    "Excellent choice, very happy!",
    "Best product in its category!",
    "Amazing features, works well!",
    "Great experience, recommend it!",
    "Very good product, satisfied!",
    "Fantastic, worth the price!",
    "Love the quality and design!",
    "Perfect product for my needs!",
    "Outstanding, couldn't be better!",
    "Great value, highly recommend!",
    "Excellent quality and finish!",
    "Very pleased, works perfectly!",
    "Amazing purchase, no issues!",
    "Top quality, very reliable!",
    "Wonderful product, great buy!",
    "Impressive performance overall!",
    "Great product, fast delivery!",
    "Excellent, meets expectations!",
    "Very satisfied, good quality!",
    "Fantastic, exactly what wanted!",
    "Love it, perfect condition!",
    "Outstanding quality product!",
    "Great purchase, very happy!"
]

# Negative reviews
negative_reviews = [
    "Terrible quality, waste of money!",
    "Worst purchase ever, completely disappointed!",
    "Poor quality and bad customer service!",
    "Do not buy this, total waste!",
    "Awful product, broke immediately!",
    "Very disappointed, not as described!",
    "Horrible experience, would not recommend!",
    "Complete waste of time and money!",
    "Terrible, doesn't work at all!",
    "Very poor quality, not satisfied!",
    "Disappointing purchase, regret buying!",
    "Bad quality, fell apart quickly!",
    "Not worth the price, disappointing!",
    "Terrible product, avoid at all costs!",
    "Very unhappy with this purchase!",
    "Poor service and worse product!",
    "Useless product, complete failure!",
    "Worst quality I've ever seen!",
    "Don't waste your money on this!",
    "Extremely disappointed with quality!",
    "Awful, broke after one use!",
    "Very poor, not recommended!",
    "Terrible investment, big mistake!",
    "Bad experience, unsatisfied customer!",
    "Horrible quality, stopped working!",
    "Not good, returned immediately!",
    "Disappointing, not worth buying!",
    "Poor construction, cheap materials!",
    "Very bad quality control!",
    "Terrible, doesn't match description!",
    "Awful service and product!",
    "Complete disappointment, waste!",
    "Bad purchase, regret it!",
    "Not satisfied, poor quality!",
    "Horrible, malfunctioned quickly!",
    "Terrible value for money!",
    "Very unhappy, doesn't work!",
    "Poor design, impractical!",
    "Disappointing quality throughout!",
    "Bad product, many defects!",
    "Worst experience, avoid!",
    "Not as advertised, misleading!",
    "Terrible durability, broke fast!",
    "Very poor performance!",
    "Awful, unreliable product!",
    "Bad investment, not worth!",
    "Disappointing results overall!",
    "Poor functionality, useless!",
    "Horrible build quality!",
    "Not good at all!",
    "Terrible features, lacking!",
    "Very dissatisfied customer!",
    "Poor performance, sluggish!",
    "Bad design, uncomfortable!",
    "Awful quality materials!",
    "Not recommended, poor!",
    "Terrible finish, cheap!",
    "Very bad, malfunctions!",
    "Disappointing purchase!",
    "Poor value, overpriced!",
    "Horrible, defective unit!",
    "Bad quality assurance!",
    "Not worth money!",
    "Terrible workmanship!",
    "Very unsatisfactory!",
    "Poor standards!",
    "Awful experience!",
    "Bad all around!",
    "Disappointing quality!",
    "Terrible choice!",
    "Very poor!",
    "Not good!",
    "Bad buy!",
    "Awful!",
    "Poor!",
    "Terrible!",
    "Disappointing!",
    "Horrible!",
    "Unsatisfied!",
    "Regret purchase!"
]

# Generate dataset
dataset = []

# Add 520 positive reviews
for i in range(520):
    review = random.choice(positive_reviews)
    # Add some variation
    if random.random() > 0.3:
        variations = [
            review,
            review + " Really happy!",
            "Great! " + review,
            review + " Thumbs up!",
            review + " Five stars!",
            "Wow! " + review,
            review + " Absolutely wonderful!",
            review + " Will recommend!",
            "Impressive! " + review,
            review + " Perfect choice!"
        ]
        review = random.choice(variations)
    dataset.append({"review": review, "sentiment": "positive"})

# Add 480 negative reviews
for i in range(480):
    review = random.choice(negative_reviews)
    # Add some variation
    if random.random() > 0.3:
        variations = [
            review,
            review + " Not happy!",
            "Sad. " + review,
            review + " Big mistake!",
            review + " Zero stars!",
            "Unfortunately, " + review,
            review + " Very upset!",
            review + " Will not buy again!",
            "Sadly, " + review,
            review + " Refund requested!"
        ]
        review = random.choice(variations)
    dataset.append({"review": review, "sentiment": "negative"})

# Shuffle the dataset
random.shuffle(dataset)

# Create DataFrame
df = pd.DataFrame(dataset)

# Save to CSV
df.to_csv("sentiment_dataset.csv", index=False)

print("✅ Dataset created successfully!")
print(f"\nDataset Shape: {df.shape}")
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\n💾 File saved as 'sentiment_dataset.csv'")