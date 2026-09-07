import random
import pandas as pd

SEED = 105

random.seed(SEED)

CATEGORIES = [
    "Beauty",
    "Apparel",
    "Footwear",
    "Home",
    "Electronics"
]

STATUSES = [
    "Delivered",
    "Shipped",
    "Placed",
    "Returned",
    "Refunded"
]

CATEGORY_WEIGHTS = {
    "Beauty": 0.40,
    "Apparel": 0.20,
    "Footwear": 0.15,
    "Home": 0.15,
    "Electronics": 0.10
}

STATUS_WEIGHTS = {
    "Delivered": 0.55,
    "Shipped": 0.18,
    "Placed": 0.12,
    "Returned": 0.09,
    "Refunded": 0.06
}

PRICE_RANGES = {
    "Beauty": (199, 8000),
    "Apparel": (499, 6000),
    "Footwear": (799, 8000),
    "Home": (399, 12000),
    "Electronics": (999, 25000)
}

def generate_days():
    return random.randint(0, 30)


def generate_order(record_id):
    category = random.choices(
        CATEGORIES,
        weights=CATEGORY_WEIGHTS.values()
    )[0]

    status = random.choices(
        STATUSES,
        weights=STATUS_WEIGHTS.values()
    )[0]

    low_price, high_price = PRICE_RANGES[category]

    order_value = random.randint(low_price, high_price)

    return {
        "record_id": record_id,
        "category": category,
        "status": status,
        "order_value_inr": order_value,
        "days_since_created": generate_days(),
        "delayed_shipment": random.choices(
    [True, False],
    weights=[0.20, 0.80]
)[0]
    }

def validate_dataset(orders):
    required_categories = set(CATEGORIES)
    required_statuses = set(STATUSES)

    actual_categories = {
        order["category"] for order in orders
    }

    actual_statuses = {
        order["status"] for order in orders
    }

    delayed_count = sum(
        order["delayed_shipment"] for order in orders
    )

    delayed_percentage = (
        delayed_count / len(orders)
    ) * 100

    categories_valid = all(
        category in actual_categories
        for category in required_categories
    )

    statuses_valid = all(
        status in actual_statuses
        for status in required_statuses
    )

    delayed_valid = 10 <= delayed_percentage <= 30

    return (
        len(orders) == 50
        and categories_valid
        and statuses_valid
        and delayed_valid
    )

orders = []

for i in range(1, 51):
    order = generate_order(f"NYK{i:03d}")
    orders.append(order)
if validate_dataset(orders):
    print("\nDataset validation: PASSED")
else:
    print("\nDataset validation: FAILED")    

print(orders)

from collections import Counter

category_counts = Counter(order["category"] for order in orders)
status_counts = Counter(order["status"] for order in orders)

delayed_count = sum(
    order["delayed_shipment"] for order in orders
)

delayed_percentage = (delayed_count / len(orders)) * 100

print("\nCategory counts:")
print(category_counts)

print("\nStatus counts:")
print(status_counts)

print("\nDelayed shipments:")
print(f"{delayed_percentage:.2f}%")


df = pd.DataFrame(orders)

df.to_csv("data/orders.csv", index=False)

print("\nDataset saved to data/orders.csv")