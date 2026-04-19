from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_W, FIG_H = 16, 10


def add_box(ax, xy, w, h, title, subtitle_lines, edge, face, lw=2, title_size=11):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        edgecolor=edge,
        facecolor=face,
        linewidth=lw,
    )
    ax.add_patch(box)

    x, y = xy
    cx = x + w / 2
    ax.text(cx, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, fontweight="bold")

    for i, line in enumerate(subtitle_lines):
        ax.text(cx, y + h - 0.62 - (i * 0.28), line, ha="center", va="top", fontsize=9)


def add_arrow(ax, start, end, color="#34495E", style="-", lw=2, ms=16):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=style,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        8,
        9.75,
        "Intelligent Credit Risk Scoring and Agentic Lending Decision Support",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        8,
        9.38,
        "Current Architecture: Streamlit + Random Forest + LangGraph + FAISS RAG + Groq LLM",
        ha="center",
        va="top",
        fontsize=10,
        color="#4A4A4A",
    )

    # Core inference flow (shared by both pages)
    add_box(
        ax,
        (0.6, 7.9),
        6.2,
        1.2,
        "Input Layer (Streamlit)",
        [
            "app.py (Credit Risk Scoring) and pages/lending_advisor.py",
            "Borrower profile: demographics, financials, loan details, credit history",
        ],
        edge="#1F77B4",
        face="#DCEEFF",
    )

    add_box(
        ax,
        (0.6, 6.2),
        6.2,
        1.2,
        "Preprocessing",
        [
            "Label encoding for categorical features (saved encoder mapping)",
            "Feature ordering + standard scaling consistency",
        ],
        edge="#FF8C00",
        face="#FFE6CC",
    )

    add_box(
        ax,
        (0.6, 4.5),
        6.2,
        1.2,
        "ML Inference Engine",
        [
            "Random Forest (100 trees, max_depth=10)",
            "Outputs: risk class + repayment/default probabilities",
        ],
        edge="#2CA02C",
        face="#DCF5DC",
    )

    add_arrow(ax, (3.7, 7.9), (3.7, 7.4), color="#1F77B4")
    add_arrow(ax, (3.7, 6.2), (3.7, 5.7), color="#FF8C00")

    # Branch A: Milestone 1 direct UI output
    add_box(
        ax,
        (0.6, 2.4),
        6.2,
        1.5,
        "Branch A: Credit Risk Scoring Output",
        [
            "Low/High risk decision and probability metrics",
            "Sidebar visualizations (confusion matrix, ROC)",
        ],
        edge="#D62728",
        face="#FFE0E0",
    )
    add_arrow(ax, (2.6, 4.5), (2.6, 3.9), color="#D62728")

    # Branch B: Agentic workflow
    add_box(
        ax,
        (8.2, 7.9),
        7.0,
        1.2,
        "Agent Trigger (Lending Advisor Page)",
        [
            "Uses borrower input + ML outputs + GROQ_API_KEY",
            "Builds shared LangGraph state for multi-step reasoning",
        ],
        edge="#9467BD",
        face="#EEE0FF",
    )
    add_arrow(ax, (6.8, 5.1), (8.2, 8.45), color="#9467BD")

    # LangGraph nodes
    add_box(
        ax,
        (8.2, 6.2),
        2.1,
        1.2,
        "Node 1",
        ["Risk Analyzer", "Summarizes borrower risk"],
        edge="#7F7F7F",
        face="#F4F4F4",
        title_size=10,
    )
    add_box(
        ax,
        (10.65, 6.2),
        2.1,
        1.2,
        "Node 2",
        ["Regulation Retriever", "FAISS similarity search"],
        edge="#7F7F7F",
        face="#F4F4F4",
        title_size=10,
    )
    add_box(
        ax,
        (13.1, 6.2),
        2.1,
        1.2,
        "Node 3",
        ["Report Generator", "LLM structured output"],
        edge="#7F7F7F",
        face="#F4F4F4",
        title_size=10,
    )

    add_arrow(ax, (11.25, 7.9), (9.25, 7.4), color="#9467BD")
    add_arrow(ax, (10.3, 6.8), (10.65, 6.8), color="#7F7F7F")
    add_arrow(ax, (12.75, 6.8), (13.1, 6.8), color="#7F7F7F")

    # RAG knowledge base
    add_box(
        ax,
        (11.0, 4.5),
        4.2,
        1.2,
        "RAG Knowledge Base",
        [
            "data/regulations.txt -> embeddings (all-MiniLM-L6-v2)",
            "Stored and queried with FAISS index",
        ],
        edge="#17BECF",
        face="#DDF9FF",
    )
    add_arrow(ax, (12.75, 5.7), (11.7, 6.2), color="#17BECF")

    add_box(
        ax,
        (8.2, 2.4),
        7.0,
        1.5,
        "Branch B: Final Lending Assessment Report",
        [
            "4 sections: Profile/Risk, Lending Decision, Regulatory References, Disclaimer",
            "Displayed in Streamlit expanders + retrieved source snippets",
        ],
        edge="#8C564B",
        face="#FBE7E3",
    )
    add_arrow(ax, (14.15, 6.2), (14.15, 3.9), color="#8C564B")

    # Offline training pipeline note
    add_box(
        ax,
        (0.6, 0.7),
        6.2,
        1.2,
        "Offline Training and Evaluation",
        [
            "train_model.py + compare_models.py generate metrics and visualizations",
            "Saved parameters in models/model_params.py used during app inference",
        ],
        edge="#4A4A4A",
        face="#EFEFEF",
    )
    add_arrow(ax, (3.7, 1.9), (3.7, 4.5), color="#4A4A4A", style="--", lw=1.6, ms=13)

    ax.text(
        8,
        0.18,
        "Deployment: Streamlit Cloud | Repository: github.com/officialravleensingh/credit-risk-scoring",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    plt.tight_layout()

    output_path = Path(__file__).with_name("project_architecture_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved architecture diagram: {output_path}")


if __name__ == "__main__":
    main()
