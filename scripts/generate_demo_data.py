"""Generate the SYNTHETIC demo dataset data/demo_electronics_reviews.csv.

These reviews are invented for demonstration and testing. They are NOT real
customer feedback. The generator is deterministic (fixed random seed) and
deliberately includes patterns the dashboard should detect:

- English, Hinglish and Hindi (Devanagari) reviews
- a rise in Bluetooth/connectivity complaints for Nimbus ANC Headphones in September
- a burst of near-identical generic 5-star reviews for PulseFit Smartwatch on one day
- a couple of exact duplicate reviews

Run:  python scripts/generate_demo_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_electronics_reviews.csv"
START, END = date(2026, 4, 1), date(2026, 9, 30)

# product -> polarity -> aspect -> sentences
POOLS = {
    "Nimbus ANC Headphones": {
        "pos": {
            "sound": [
                "Sound quality is excellent with deep, clean bass.",
                "Noise cancellation blocks out the metro noise completely.",
                "Vocals are crisp and the soundstage feels wide.",
                "ANC works brilliantly on long flights.",
                "Sound quality ekdum mast hai.",
                "Bass bahut accha hai, gaane sunne mein maza aata hai.",
                "आवाज़ की गुणवत्ता बहुत अच्छी है।",
            ],
            "comfort": [
                "Very comfortable even after four hours of use.",
                "The ear cushions are soft and light on the head.",
                "Comfort bhi badhiya hai, ghanto tak pehen sakte ho.",
            ],
            "battery": [
                "Battery easily lasts the whole week of commuting.",
                "Battery lasts all day with ANC switched on.",
                "Battery backup kaafi accha hai.",
            ],
            "connectivity": [
                "Pairs instantly with my phone and laptop.",
                "Bluetooth connection is rock solid across the room.",
            ],
            "value": ["Great value for the price.", "Worth every rupee at the sale price."],
            "delivery": ["Delivery was quick and the box was well packed."],
        },
        "neg": {
            "connectivity": [
                "Bluetooth keeps disconnecting every few minutes.",
                "The connection drops when I move away from the phone.",
                "The left earcup randomly loses connection.",
                "Pairing fails again and again after the latest update.",
                "Audio cuts out during calls, very frustrating.",
                "Bluetooth baar baar disconnect ho jata hai.",
                "Connection itna weak hai ki dusre kamre mein jaate hi cut ho jata hai.",
            ],
            "battery": [
                "Battery drains within five hours.",
                "Battery life is much shorter than advertised.",
                "Battery bahut jaldi khatam ho jati hai.",
                "बैटरी बहुत जल्दी खत्म हो जाती है।",
            ],
            "quality": [
                "Left ear stopped working after three weeks.",
                "The headband cracked near the hinge.",
                "Build feels cheap and creaky.",
                "Ek mahine mein hi kharab ho gaya.",
            ],
            "comfort": [
                "Ear cups get hot and uncomfortable after an hour.",
                "Clamping force is too tight for long sessions.",
            ],
            "service": [
                "Customer support did not respond to my emails.",
                "My replacement request was refused without any reason.",
                "Customer care ne koi jawab nahi diya.",
            ],
            "delivery": ["Delivery was late by a week.", "Package arrived damaged and dented.", "Delivery late thi."],
            "value": ["Overpriced for what you get."],
        },
    },
    "PulseFit Smartwatch": {
        "pos": {
            "features": [
                "Sleep tracking is surprisingly accurate.",
                "Step counting matches my phone almost exactly.",
                "Notifications are clear and the app is simple to use.",
                "Heart rate readings look reliable during runs.",
                "Watch ka display bahut bright aur clear hai.",
            ],
            "battery": ["Battery lasts five days on a single charge.", "Charges fully in under an hour."],
            "comfort": ["The strap is soft and comfortable to sleep in."],
            "value": ["Excellent value compared to bigger brands."],
        },
        "neg": {
            "features": [
                "GPS takes forever to lock and loses track on runs.",
                "Heart rate readings jump around randomly.",
                "The app keeps crashing after sync.",
            ],
            "battery": ["Battery barely lasts a day with GPS on.", "Stopped charging after two months."],
            "comfort": ["The band caused skin irritation.", "Strap material feels cheap and sweaty."],
            "service": ["Customer service took three weeks to reply."],
            "delivery": ["Delivery was delayed twice."],
        },
    },
    "AeroBrew Coffee Maker": {
        "pos": {
            "performance": [
                "Brews fast and the coffee tastes rich.",
                "The warming plate keeps coffee hot for an hour.",
                "Coffee ka taste ekdum cafe jaisa hai.",
            ],
            "design": ["Looks great on the counter and is easy to clean."],
            "value": ["Good value for daily use."],
            "delivery": ["Shipping was fast and the packaging was sturdy."],
        },
        "neg": {
            "quality": [
                "The carafe cracked after two weeks.",
                "The lid leaks when pouring.",
                "Machine stopped working after a month.",
            ],
            "performance": ["Very noisy while brewing.", "Water tank is too small, I refill constantly."],
            "service": ["Customer service refused a return."],
            "delivery": ["Arrived late and the box was crushed."],
        },
    },
}

PRODUCT_WEIGHTS = {"Nimbus ANC Headphones": 0.5, "PulseFit Smartwatch": 0.3, "AeroBrew Coffee Maker": 0.2}
GENERIC_POSITIVE = ["Good product.", "Nice, happy with it.", "Value for money.", "Accha product hai."]

SUSPICIOUS_BURST = [
    "Best smartwatch ever!!! Must buy, highly recommended!!!",
    "Best smartwatch ever!! Must buy. Highly recommended!!!",
    "BEST SMARTWATCH EVER!!! MUST BUY!!! HIGHLY RECOMMENDED!!!",
    "Best smart watch ever!!! Must buy, highly recommended!!",
    "Best smartwatch ever, must buy!!! Highly recommended!!!",
    "Amazing!!! Best smartwatch ever!!! Must buy!!!",
]


def sentence_key(text: str) -> frozenset:
    """Order-independent key so "A. B." and "B. A." count as the same review."""
    parts = text.replace("However, ", ". ").replace("But ", ". ").lower().split(".")
    return frozenset(p.strip(" ,") for p in parts if p.strip(" ,"))


def negative_probability(product: str, day: date) -> float:
    """Base complaint rate, with a rise for the headphones in September."""
    if product == "Nimbus ANC Headphones" and day >= date(2026, 9, 1):
        return 0.6
    return 0.3


def pick_aspect(rng: random.Random, product: str, polarity: str, day: date) -> str:
    aspects = list(POOLS[product][polarity])
    weights = [1.0] * len(aspects)
    if product == "Nimbus ANC Headphones" and polarity == "neg" and day >= date(2026, 9, 1):
        weights = [6.0 if a == "connectivity" else 1.0 for a in aspects]
    return rng.choices(aspects, weights)[0]


def lower_first(sentence: str) -> str:
    """Lower-case the first letter unless the first word is an acronym like ANC or GPS."""
    first_word = sentence.split()[0]
    return sentence if first_word.isupper() else sentence[0].lower() + sentence[1:]


def make_review(rng: random.Random, product: str, day: date) -> tuple[str, int]:
    clause_count = rng.choices([1, 2, 3], [0.2, 0.5, 0.3])[0]
    p_neg = negative_probability(product, day)
    parts, polarities, used = [], [], set()
    for _ in range(clause_count):
        polarity = "neg" if rng.random() < p_neg else "pos"
        aspect = pick_aspect(rng, product, polarity, day)
        if aspect in used:
            continue
        used.add(aspect)
        parts.append(rng.choice(POOLS[product][polarity][aspect]))
        polarities.append(polarity)
    negatives = polarities.count("neg")
    if negatives == 0:
        rating = rng.choices([5, 4], [0.65, 0.35])[0]
    elif negatives == len(polarities):
        rating = rng.choices([1, 2], [0.6, 0.4])[0]
    else:
        rating = rng.choice([2, 3, 3, 4])
    text = parts[0]
    for previous, polarity, part in zip(polarities, polarities[1:], parts[1:]):
        # Add a contrast word when the review switches between praise and complaint.
        text += (" " + rng.choice(["But", "However,"]) + " " + lower_first(part)) if polarity != previous else " " + part
    return text, rating


def generate() -> list[dict]:
    rng = random.Random(42)
    rows, seen = [], set()
    day = START
    while day <= END:
        daily_reviews = rng.choice([0, 1, 1, 2, 2, 3]) + (1 if day >= date(2026, 9, 1) else 0)
        for _ in range(daily_reviews):
            product = rng.choices(list(PRODUCT_WEIGHTS), list(PRODUCT_WEIGHTS.values()))[0]
            if day >= date(2026, 9, 1) and rng.random() < 0.35:
                product = "Nimbus ANC Headphones"  # more headphone reviews arrive in September
            for _attempt in range(5):  # retry a few times to avoid exact repeats
                if rng.random() < 0.04:
                    text, rating = rng.choice(GENERIC_POSITIVE), rng.choice([4, 5])
                else:
                    text, rating = make_review(rng, product, day)
                key = sentence_key(text)
                if key not in seen:
                    break
            if key in seen:
                continue
            seen.add(key)
            rows.append({"product_name": product, "rating": rating, "review_text": text, "review_date": day})
        day += timedelta(days=1)

    # Burst of near-identical generic 5-star reviews posted on one day.
    for text in SUSPICIOUS_BURST:
        rows.append({"product_name": "PulseFit Smartwatch", "rating": 5, "review_text": text,
                     "review_date": date(2026, 8, 14)})
    # Two exact duplicates of existing reviews, posted later.
    for source_index, offset in ((10, 40), (55, 25)):
        source = rows[source_index]
        rows.append({**source, "review_date": source["review_date"] + timedelta(days=offset)})

    rows.sort(key=lambda r: r["review_date"])
    for i, row in enumerate(rows, start=1):
        row["review_id"] = f"D{i:03d}"
    return rows


def main() -> None:
    rows = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "product_name", "rating", "review_text", "review_date"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic reviews to {OUT_PATH}")


if __name__ == "__main__":
    main()
