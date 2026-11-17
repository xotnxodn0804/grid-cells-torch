
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NORMAL_DIR = Path("results/oracle_500m_normal")
ZERO_DIR = Path("results/oracle_500m_zero")
OUT_DIR = Path("results/oracle_500m_compare")

# 이전 그림에 사용한 동일한 에피소드
EPISODES = [5, 25, 78, 74]


def first_return(folder, ep):
    path = folder / f"trajectory_ep{ep:03d}.csv"
    df = pd.read_csv(path)

    # 실제 목표 도달 이벤트 찾기
    hits = np.flatnonzero(df["goal_hit"].to_numpy() > 0)

    if len(hits) < 2:
        return None

    first_hit = hits[0]
    second_hit = hits[1]

    # 첫 목표 도달 이후 순간이동한 위치부터
    # 두 번째 목표 도달을 유발한 행동 직전까지
    trajectory = df.iloc[first_hit + 1:second_hit + 1].copy()

    latency = int(
        df.iloc[second_hit]["frame"]
        - df.iloc[first_hit]["frame"]
    )

    gx = float(df.iloc[0]["goal_x_m_debug"])
    gy = float(df.iloc[0]["goal_y_m_debug"])

    return trajectory, gx, gy, latency, len(hits)


def draw(ax, result, color, mode, episode):
    if result is None:
        ax.text(
            0.5, 0.5, "No first revisit",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_title(f"Episode {episode} | {mode}")
    else:
        traj, gx, gy, latency, total_hits = result

        x = traj["x_m"].to_numpy()
        y = traj["y_m"].to_numpy()

        # 첫 재방문 경로
        ax.plot(x, y, color=color, linewidth=2.0, alpha=0.85)

        # 순간이동 이후 출발 위치
        ax.scatter(
            x[0], y[0],
            color="black", marker="s",
            s=45, zorder=5
        )

        # 기록된 마지막 위치 (보상 발생 행동 직전)
        ax.scatter(
            x[-1], y[-1],
            color=color, marker="o",
            edgecolor="black", linewidth=0.5,
            s=45, zorder=6
        )

        # 목표 위치 (평가용 디버그 좌표)
        ax.scatter(
            gx, gy,
            color="#D9534F",
            marker="*", s=180, zorder=7
        )

        ax.set_title(
            f"Episode {episode} | {mode}\n"
            f"First return: {latency / 60:.1f} s "
            f"({latency} frames) | Total hits: {total_hits}",
            fontsize=10
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def main():
    fig, axes = plt.subplots(
        len(EPISODES), 2,
        figsize=(10, 4 * len(EPISODES)),
        squeeze=False
    )

    for row, ep in enumerate(EPISODES):
        normal = first_return(NORMAL_DIR, ep)
        zero = first_return(ZERO_DIR, ep)

        draw(axes[row, 0], normal, "#2474B5", "Normal", ep)
        draw(axes[row, 1], zero, "#DD8935", "Zero", ep)

    fig.suptitle(
        "Oracle Place Agent (500M)\n"
        "First Goal Revisit: Normal vs Zero Goal Memory",
        fontsize=14
    )

    fig.text(
        0.5, 0.01,
        "Square = start after teleport | "
        "Circle = last recorded position | "
        "Star = goal",
        ha="center", fontsize=10
    )

    fig.tight_layout(rect=[0, 0.035, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    png = OUT_DIR / "first_return_normal_vs_zero.png"
    pdf = OUT_DIR / "first_return_normal_vs_zero.pdf"

    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


if __name__ == "__main__":
    main()
