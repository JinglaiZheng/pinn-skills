#!/usr/bin/env python3
"""Generate a star history chart from data/stars.json."""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "stars.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "star_chart.png")


def main():
    if not os.path.exists(DATA_FILE):
        print(f"No data file at {DATA_FILE}")
        return

    with open(DATA_FILE) as f:
        data = json.load(f)

    if len(data) < 2:
        print(f"Only {len(data)} entries — need at least 2 to plot")
        return

    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
    stars = [d["stars"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(dates, 0, stars, alpha=0.15, color="#0969da")
    ax.plot(dates, stars, color="#0969da", linewidth=2.5, marker="o", markersize=4)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stars", fontsize=12)
    ax.set_title("pinn-skill — Star History", fontsize=14, fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(0, max(stars) * 1.15)

    plt.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {OUTPUT_FILE} ({len(data)} data points, {stars[-1]} stars)")


if __name__ == "__main__":
    main()
