from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_architecture_image():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background color
    fig.patch.set_facecolor("#0F172A")  # Deep slate navy

    # Title
    ax.text(
        50,
        94,
        "MEMORYOS: System Architecture & Data Flow",
        ha="center",
        va="center",
        color="#F8FAFC",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        50,
        90,
        "Convico AI 24-Hour Engineering Challenge | Persistent Multilingual Memory Backend",
        ha="center",
        va="center",
        color="#94A3B8",
        fontsize=10,
    )

    # Box 1: Client Layer
    box_client = patches.FancyBboxPatch(
        (10, 78), 80, 8, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5
    )
    ax.add_patch(box_client)
    ax.text(50, 82, "API Client / Frontend", ha="center", va="center", color="#38BDF8", fontsize=11, fontweight="bold")
    ax.text(
        50,
        79,
        "Authenticated Requests (Bearer JWT) | Character & Conversation Scoping",
        ha="center",
        va="center",
        color="#E2E8F0",
        fontsize=9,
    )

    # Down arrow
    ax.annotate(
        "",
        xy=(50, 68),
        xytext=(50, 77),
        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2),
    )

    # Box 2: FastAPI Application Gateway
    box_api = patches.FancyBboxPatch(
        (10, 58), 80, 9, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#818CF8", linewidth=1.5
    )
    ax.add_patch(box_api)
    ax.text(
        50,
        63,
        "FastAPI Application & Security Gateway",
        ha="center",
        va="center",
        color="#818CF8",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        50,
        60,
        "Pydantic v2 Validation | JWT Principal Extraction | Row-Level Tenancy Guard | Audit Middleware",
        ha="center",
        va="center",
        color="#E2E8F0",
        fontsize=8.5,
    )

    # Arrows to pipelines
    ax.annotate(
        "",
        xy=(28, 48),
        xytext=(35, 57),
        arrowprops=dict(arrowstyle="->", color="#34D399", lw=1.8),
    )
    ax.annotate(
        "",
        xy=(72, 48),
        xytext=(65, 57),
        arrowprops=dict(arrowstyle="->", color="#F472B6", lw=1.8),
    )

    # Box 3A: Memory Extraction & Lifecycle Pipeline
    box_mem = patches.FancyBboxPatch(
        (8, 28), 40, 19, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#34D399", linewidth=1.5
    )
    ax.add_patch(box_mem)
    ax.text(28, 44, "Memory Pipeline", ha="center", va="center", color="#34D399", fontsize=11, fontweight="bold")
    mem_details = (
        "- English & Hinglish Neural Extraction\n"
        "- Assertion vs. Question Filtering\n"
        "- Polarity & Preference Analysis\n"
        "- Temporal Lifecycle & Supersession\n"
        "- Atomic Multi-Fact Transactions"
    )
    ax.text(28, 35, mem_details, ha="center", va="center", color="#CBD5E1", fontsize=8, linespacing=1.4)

    # Box 3B: Hybrid RAG Retrieval Pipeline
    box_rag = patches.FancyBboxPatch(
        (52, 28), 40, 19, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#F472B6", linewidth=1.5
    )
    ax.add_patch(box_rag)
    ax.text(72, 44, "Hybrid RAG Pipeline", ha="center", va="center", color="#F472B6", fontsize=11, fontweight="bold")
    rag_details = (
        "- Lane A: Structured Exact Entity/Domain\n"
        "- Lane B: Lexical Full-Text & Synonym Search\n"
        "- Lane C: pgvector 384-dim Dense Cosine\n"
        "- Reciprocal Rank Fusion (RRF k=60)\n"
        "- Multi-Factor & Temporal Reranker"
    )
    ax.text(72, 35, rag_details, ha="center", va="center", color="#CBD5E1", fontsize=8, linespacing=1.4)

    # Down arrows to Storage
    ax.annotate(
        "",
        xy=(35, 17),
        xytext=(28, 27),
        arrowprops=dict(arrowstyle="->", color="#60A5FA", lw=1.8),
    )
    ax.annotate(
        "",
        xy=(65, 17),
        xytext=(72, 27),
        arrowprops=dict(arrowstyle="->", color="#F87171", lw=1.8),
    )

    # Box 4A: Authoritative Database (PostgreSQL + pgvector)
    box_pg = patches.FancyBboxPatch(
        (10, 6), 38, 10, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#60A5FA", linewidth=1.5
    )
    ax.add_patch(box_pg)
    ax.text(
        29,
        12,
        "PostgreSQL 16 + pgvector",
        ha="center",
        va="center",
        color="#60A5FA",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        29,
        9,
        "Authoritative Durable Record | Provenance\nTemporal History | HNSW Vector Index",
        ha="center",
        va="center",
        color="#E2E8F0",
        fontsize=7.5,
    )

    # Box 4B: Ephemeral Cache (Redis)
    box_redis = patches.FancyBboxPatch(
        (52, 6), 38, 10, boxstyle="round,pad=1", facecolor="#1E293B", edgecolor="#F87171", linewidth=1.5
    )
    ax.add_patch(box_redis)
    ax.text(71, 12, "Redis 7 Cache", ha="center", va="center", color="#F87171", fontsize=10.5, fontweight="bold")
    ax.text(
        71,
        9,
        "Sliding Context Window | Query Cache\nTTL Expiration | Mutation Invalidation",
        ha="center",
        va="center",
        color="#E2E8F0",
        fontsize=7.5,
    )

    # Save diagram
    paths = [
        Path("docs/architecture.png"),
        Path("CANDIDATE_NAME_CONVICO_AI_ASSIGNMENT/05_ARCHITECTURE/architecture.png"),
        Path("RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/05_ARCHITECTURE/architecture.png"),
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"Architecture diagram saved to {p}")

    plt.close()


if __name__ == "__main__":
    generate_architecture_image()

