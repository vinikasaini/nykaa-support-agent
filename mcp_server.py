from fastmcp import FastMCP

from src.order_tools import check_order_status


mcp = FastMCP(
    "Nykaa Order Lookup MCP Server"
)


@mcp.tool
def lookup_order(record_id: str) -> dict:
    """
    Look up a Nykaa order by its record ID.

    Args:
        record_id: The unique Nykaa order ID, such as NYK001.

    Returns:
        A dictionary containing the order status, order value,
        escalation score, and recommended escalation decision.
    """

    result = check_order_status(record_id)

    return {
        "record_id": record_id,
        "status": result["status"],
        "order_value_inr": result["order_value_inr"],
        "escalation_score": result["escalation_score"],
        "recommended_escalation": result["recommended_escalation"]
    }


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001
    )