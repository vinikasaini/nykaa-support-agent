from src.dataset import orders

ESCALATION_THRESHOLD = 0.50


def check_order_status(record_id: str) -> dict:

    for order in orders:

        if order["record_id"] == record_id:

            recency_signal = (
                order["days_since_created"] / 30
            )

            delayed_signal = (
                1 if order["delayed_shipment"] else 0
            )

            escalation_score = (
                0.6 * recency_signal
                + 0.4 * delayed_signal
            )

            escalation_score = round(
                escalation_score,
                3
            )

            return {
                "record_id": record_id,
                "status": order["status"],
                "order_value_inr": order["order_value_inr"],
                "escalation_score": escalation_score,
                "recommended_escalation": (
                    escalation_score >= ESCALATION_THRESHOLD
                )
            }

    return {
        "record_id": record_id,
        "status": "Not Found",
        "order_value_inr": None,
        "escalation_score": 0.0,
        "recommended_escalation": False
    }


if __name__ == "__main__":

    delayed_order = next(
        order for order in orders
        if order["delayed_shipment"] is True
    )

    non_delayed_order = next(
        order for order in orders
        if order["delayed_shipment"] is False
    )

    test_ids = [
        delayed_order["record_id"],
        non_delayed_order["record_id"],
        "NYK999"
    ]

    print("Order Status Tool Tests")
    print()

    for record_id in test_ids:

        result = check_order_status(record_id)

        print("Record ID:", result["record_id"])
        print("Status:", result["status"])
        print("Order Value:", result["order_value_inr"])
        print(
            "Escalation Score:",
            result["escalation_score"]
        )
        print(
            "Recommended Escalation:",
            result["recommended_escalation"]
        )
        print()