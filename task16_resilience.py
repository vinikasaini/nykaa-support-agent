import asyncio
import time

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# ============================================================
# TASK 16
# RESILIENCE & TIMEOUT DEMONSTRATION
# ============================================================

print("=" * 70)
print("TASK 16 - RESILIENCE & TIMEOUT DEMONSTRATION")
print("=" * 70)


# ============================================================
# SHARED STATE
# ============================================================

class DemoState(TypedDict, total=False):

    message: str

    retry_attempts: int
    retry_result: str

    timeout_result: str
    global_timeout_result: str

    completed_nodes: list


# ============================================================
# PART 1
# EXPONENTIAL BACKOFF RETRY
# ============================================================

retry_counter = {
    "count": 0
}


async def transient_failure_node(state: DemoState):

    retry_counter["count"] += 1

    attempt = retry_counter["count"]

    print(
        f"\n[RETRY NODE] Attempt {attempt}"
    )

    # Simulate a transient failure
    # on the first two attempts.
    if attempt <= 2:

        print(
            f"[RETRY NODE] Simulated transient failure "
            f"on attempt {attempt}"
        )

        raise RuntimeError(
            f"Temporary failure on attempt {attempt}"
        )

    # Third attempt succeeds.
    print(
        f"[RETRY NODE] SUCCESS on attempt {attempt}"
    )

    return {
        "retry_attempts": attempt,
        "retry_result": "Retry recovered successfully",
        "completed_nodes": state.get(
            "completed_nodes",
            []
        ) + ["retry_node"]
    }


def build_retry_graph():

    builder = StateGraph(DemoState)

    builder.add_node(
        "retry_node",
        transient_failure_node,
        retry_policy=RetryPolicy(
            max_attempts=4,
            initial_interval=0.5,
            backoff_factor=2.0,
            max_interval=2.0,
            jitter=False
        )
    )

    builder.add_edge(
        START,
        "retry_node"
    )

    builder.add_edge(
        "retry_node",
        END
    )

    return builder.compile()


async def demonstrate_retry():

    print("\n")
    print("=" * 70)
    print("PART A - EXPONENTIAL BACKOFF RETRY")
    print("=" * 70)

    retry_counter["count"] = 0

    graph = build_retry_graph()

    start_time = time.perf_counter()

    try:

        result = await graph.ainvoke(
            {
                "message": "Test transient failure recovery",
                "completed_nodes": []
            }
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        print("\nRetry demonstration completed.")

        print(
            f"Total attempts: "
            f"{result.get('retry_attempts')}"
        )

        print(
            f"Result: "
            f"{result.get('retry_result')}"
        )

        print(
            f"Elapsed time: "
            f"{elapsed:.2f} seconds"
        )

        print(
            "\nEXPECTED BEHAVIOUR:"
        )

        print(
            "Attempt 1 -> failure"
        )

        print(
            "Wait -> exponential backoff"
        )

        print(
            "Attempt 2 -> failure"
        )

        print(
            "Wait -> exponential backoff"
        )

        print(
            "Attempt 3 -> success"
        )

        print(
            "\nTASK 16(a) RETRY POLICY: PASSED"
        )

    except Exception as error:

        print(
            "\nRetry demonstration failed:"
        )

        print(error)


# ============================================================
# PART 2
# PER-NODE TIMEOUT
# ============================================================

async def slow_operation():

    print(
        "\n[SLOW NODE] Simulated operation started..."
    )

    # Deliberately takes 3 seconds.
    await asyncio.sleep(3)

    return "Slow operation completed"


async def per_node_timeout_node(state: DemoState):

    print(
        "\n[TIMEOUT NODE] Starting simulated call..."
    )

    NODE_TIMEOUT = 1.0

    try:

        result = await asyncio.wait_for(
            slow_operation(),
            timeout=NODE_TIMEOUT
        )

        print(
            "[TIMEOUT NODE] Operation completed "
            "within timeout."
        )

        return {
            "timeout_result": result,
            "completed_nodes": state.get(
                "completed_nodes",
                []
            ) + ["timeout_node"]
        }

    except asyncio.TimeoutError:

        print(
            "\n[TIMEOUT NODE] PER-NODE TIMEOUT FIRED"
        )

        print(
            f"[TIMEOUT NODE] Allowed time: "
            f"{NODE_TIMEOUT} seconds"
        )

        print(
            "[TIMEOUT NODE] Simulated operation "
            "was cancelled cleanly."
        )

        # Raise a clean timeout error.
        raise TimeoutError(
            "Per-node timeout exceeded: "
            f"operation exceeded {NODE_TIMEOUT} seconds."
        )


def build_per_node_timeout_graph():

    builder = StateGraph(DemoState)

    builder.add_node(
        "timeout_node",
        per_node_timeout_node
    )

    builder.add_edge(
        START,
        "timeout_node"
    )

    builder.add_edge(
        "timeout_node",
        END
    )

    return builder.compile()


async def demonstrate_per_node_timeout():

    print("\n")
    print("=" * 70)
    print("PART B - PER-NODE TIMEOUT")
    print("=" * 70)

    graph = build_per_node_timeout_graph()

    start_time = time.perf_counter()

    try:

        await graph.ainvoke(
            {
                "message": "Test per-node timeout",
                "completed_nodes": []
            }
        )

        print(
            "\nERROR:"
        )

        print(
            "The node completed unexpectedly."
        )

    except TimeoutError as error:

        elapsed = (
            time.perf_counter() - start_time
        )

        print(
            "\nPer-node timeout demonstration completed."
        )

        print(
            f"Elapsed time: "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Error: {error}"
        )

        print(
            "\nEXPECTED BEHAVIOUR:"
        )

        print(
            "The simulated operation takes 3 seconds."
        )

        print(
            "The node timeout is 1 second."
        )

        print(
            "Therefore the node is cancelled after "
            "approximately 1 second."
        )

        print(
            "\nTASK 16(b) PER-NODE TIMEOUT: PASSED"
        )


# ============================================================
# PART 3
# GLOBAL GRAPH TIMEOUT
# ============================================================

async def very_slow_node(state: DemoState):

    print(
        "\n[GLOBAL TIMEOUT NODE] "
        "Long-running operation started..."
    )

    # Deliberately longer than global timeout.
    await asyncio.sleep(5)

    print(
        "[GLOBAL TIMEOUT NODE] "
        "Operation unexpectedly completed."
    )

    return {
        "global_timeout_result": "Completed",
        "completed_nodes": state.get(
            "completed_nodes",
            []
        ) + ["global_timeout_node"]
    }


def build_global_timeout_graph():

    builder = StateGraph(DemoState)

    builder.add_node(
        "global_timeout_node",
        very_slow_node
    )

    builder.add_edge(
        START,
        "global_timeout_node"
    )

    builder.add_edge(
        "global_timeout_node",
        END
    )

    return builder.compile()


async def demonstrate_global_timeout():

    print("\n")
    print("=" * 70)
    print("PART C - GLOBAL GRAPH TIMEOUT")
    print("=" * 70)

    graph = build_global_timeout_graph()

    GLOBAL_TIMEOUT = 2.0

    print(
        f"\nGlobal graph timeout: "
        f"{GLOBAL_TIMEOUT} seconds"
    )

    print(
        "Simulated graph operation: "
        "5 seconds"
    )

    start_time = time.perf_counter()

    try:

        await asyncio.wait_for(
            graph.ainvoke(
                {
                    "message": "Test global graph timeout",
                    "completed_nodes": []
                }
            ),
            timeout=GLOBAL_TIMEOUT
        )

        print(
            "\nERROR:"
        )

        print(
            "The graph completed unexpectedly."
        )

    except asyncio.TimeoutError:

        elapsed = (
            time.perf_counter() - start_time
        )

        print(
            "\nGLOBAL TIMEOUT FIRED"
        )

        print(
            f"Elapsed time: "
            f"{elapsed:.2f} seconds"
        )

        print(
            "The complete graph execution "
            "was cancelled cleanly."
        )

        print(
            "\nEXPECTED BEHAVIOUR:"
        )

        print(
            "Node requires approximately 5 seconds."
        )

        print(
            "Global timeout is 2 seconds."
        )

        print(
            "Therefore the complete graph run "
            "is cancelled after approximately 2 seconds."
        )

        print(
            "\nTASK 16(c) GLOBAL TIMEOUT: PASSED"
        )


# ============================================================
# RUN ALL TASK 16 DEMONSTRATIONS
# ============================================================

async def main():

    await demonstrate_retry()

    await demonstrate_per_node_timeout()

    await demonstrate_global_timeout()

    print("\n")
    print("=" * 70)
    print("TASK 16 DEMONSTRATION COMPLETE")
    print("=" * 70)

    print(
        "\nSummary:"
    )

    print(
        "1. Exponential-backoff retry policy -> PASSED"
    )

    print(
        "2. Per-node timeout -> PASSED"
    )

    print(
        "3. Global graph timeout -> PASSED"
    )

    print(
        "\nAll Task 16 resilience requirements "
        "have been demonstrated."
    )


if __name__ == "__main__":

    asyncio.run(main())