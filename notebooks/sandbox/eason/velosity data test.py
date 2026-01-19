import channels_13
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ---- Config
    base_dir = r"E:\R\open-data-master\data"
    match_id = "3788741"

    # ---- Load StatsBomb open-data (events + 360)
    events, threesixty, _ = channels_13.load_match_json(match_id, base_dir=base_dir)
    combined = channels_13.merge_freeze_frames(events, threesixty)
    df = channels_13.events_to_df(combined)

    # ---- Estimate event-to-event velocity 
    df_vel = channels_13.estimate_actor_velocity_vector(df, smooth_window=3)

    # ---- Basic diagnostics
    s_raw = df_vel["speed"]
    non_na = int(s_raw.notna().sum())
    total = int(len(s_raw))

    print(f"Total events: {total}")
    print(f"Non-NA speed values: {non_na}")

    # If speed is all NaN, stop early with a clear explanation
    if non_na == 0:
        print(
            "\nNo speed values were computed (all NaN).\n"
            "This is expected if your events DataFrame does not have valid time and/or location pairs\n"
            "to compute displacement to a 'next event' for the same player.\n"
            "Common causes:\n"
            "- minute/second missing or not numeric\n"
            "- location or next location missing\n"
            "- grouping keys (match_id/player_id) not consistent\n"
        )
        return

    # Work with cleaned speed series
    s = s_raw.dropna().astype(float)

    print("\nSpeed summary (m/s):")
    print(s.describe(percentiles=[0.9, 0.95, 0.99, 0.999]))

    # ---- Count "high speed" events
    thresholds = [8, 10, 12, 15]
    for th in thresholds:
        count = int((s > th).sum())
        print(f"Speed > {th:>4} m/s : {count:>6} events")

    # ---- Inspect top fastest events
    # dt should be computed as (t_next - t). If t_next is NaN, dt is NaN.
    if "t" in df_vel.columns and "t_next" in df_vel.columns:
        df_vel["dt"] = df_vel["t_next"] - df_vel["t"]

    top_fast = df_vel.sort_values("speed", ascending=False).head(20)
    cols_to_show = [
        "match_id", "event_id", "player_id", "minute", "second",
        "location", "loc_next", "t", "t_next", "dt", "vx", "vy", "speed"
    ]
    cols_to_show = [c for c in cols_to_show if c in top_fast.columns]
    print("\nTop 20 fastest rows:")
    print(top_fast[cols_to_show])

    # ---- Plot histogram with a robust x-range so bars are visible
    p99 = float(np.quantile(s, 0.99))
    xmax = max(0.5, p99 * 1.2)
    xmax = min(xmax, 25.0)  # keep plot readable

    plt.figure(figsize=(7, 4))
    plt.hist(s, bins=80, range=(0, xmax))
    for th in [8, 10, 12]:
        plt.axvline(th, linestyle="--", label=f"{th} m/s")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("Distribution of event-to-event speed estimates")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

