from lerobot.datasets.lerobot_dataset import LeRobotDataset
from collections import Counter
import numpy as np


def analyze_metadata(ds: LeRobotDataset):
    if ds is None:
        print("Dataset is None, skip metadata analysis.")
        return
    meta = ds.meta
    info = meta.info
    print("=== Info ===")
    for k in [
        "codebase_version",
        "robot_type",
        "fps",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "total_chunks",
    ]:
        if k in info:
            print(k + ":", info[k])
    print("splits:", info.get("splits"))
    print("data_path:", info.get("data_path"))
    print("video_path:", info.get("video_path"))
    print("features:", sorted(list(info.get("features", {}).keys())))
    episodes = meta.episodes
    print("=== Episodes ===")
    num_eps = len(episodes)
    print("num_episodes (from meta.episodes):", num_eps)
    lengths = []
    tasks_counter = Counter()
    for ep in episodes.values():
        length = ep.get("length")
        if isinstance(length, (int, float)):
            lengths.append(length)
        ep_tasks = ep.get("tasks")
        if isinstance(ep_tasks, list):
            for t in ep_tasks:
                tasks_counter[str(t)] += 1
        elif ep_tasks is not None:
            tasks_counter[str(ep_tasks)] += 1
    if lengths:
        arr = np.array(lengths, dtype=np.float64)
        print(
            "episode_length min/mean/max:",
            float(arr.min()),
            float(arr.mean()),
            float(arr.max()),
        )
    print("episode tasks distribution (by text):")
    for t, c in tasks_counter.most_common(10):
        print(" ", repr(t), "->", c)
    tasks = meta.tasks
    print("=== Tasks Table ===")
    print("num_tasks (from meta.tasks):", len(tasks))
    for idx, t in list(tasks.items())[:10]:
        print(" task_index", idx, ":", repr(t))
    episodes_stats = meta.episodes_stats
    print("=== Episodes Stats ===")
    print("episodes_stats entries:", len(episodes_stats))
    if episodes_stats:
        first = next(iter(episodes_stats.values()))
        print("stats available for features:", sorted(list(first.keys())))
        if "reward_bins" in first:
            counts_list = []
            for s in episodes_stats.values():
                rb = s.get("reward_bins")
                if rb is None:
                    continue
                count = rb.get("count")
                if count is None:
                    continue
                counts_list.append(count)
            if counts_list:
                try:
                    arr = np.stack(counts_list, axis=0)
                    total_counts = arr.sum(axis=0)
                    nonzero = total_counts > 0
                    print("reward_bins num_bins:", int(total_counts.shape[0]))
                    print("reward_bins nonzero_bins:", int(nonzero.sum()))
                    if total_counts.shape[0] <= 20:
                        print(
                            "reward_bins counts:",
                            [int(x) for x in total_counts],
                        )
                    else:
                        topk = 10
                        indices = np.argsort(-total_counts)[:topk]
                        pairs = [
                            (int(i), int(total_counts[i])) for i in indices
                        ]
                        print(
                            "reward_bins top",
                            topk,
                            "bins (index, count):",
                            pairs,
                        )
                except Exception as e:
                    print("Failed to aggregate reward_bins stats:", e)


def analyze_dataset(ds: LeRobotDataset, max_episodes: int = 5, max_frames_per_ep: int = 5):
    if ds is None:
        print("Dataset is None, skip analysis.")
        return
    print("=== Dataset Basic Info ===")
    try:
        meta = ds.meta
        info = getattr(meta, "info", None)
        if info is not None:
            print("codebase_version:", info.get("codebase_version"))
            print("features keys:", sorted(list(info.get("features", {}).keys())))
            print("fps:", info.get("fps"))
        print("num_episodes:", meta.num_episodes)
        print("num_frames:", meta.num_frames)
    except Exception as e:
        print("Failed to read meta info:", e)
    print("=== Sample Frames ===")
    try:
        length = len(ds)
        print("total dataset length (frames):", length)
        indices = list(range(min(length, max_episodes)))
        for idx in indices:
            sample = ds[idx]
            print(f"[frame {idx}] keys:", list(sample.keys()))
            obs = sample.get("observation", {})
            action = sample.get("action", None)
            if isinstance(obs, dict):
                for k, v in obs.items():
                    try:
                        if hasattr(v, "shape"):
                            print("  obs", k, "shape:", v.shape)
                        else:
                            print("  obs", k, "type:", type(v))
                    except Exception:
                        print("  obs", k, "type:", type(v))
            if action is not None:
                try:
                    if hasattr(action, "shape"):
                        print("  action shape:", action.shape)
                    else:
                        print("  action type:", type(action))
                except Exception:
                    print("  action type:", type(action))
    except Exception as e:
        print("Failed to sample frames:", e)
    print("=== Episode Length Statistics ===")
    try:
        lengths = []
        for ep_idx in range(min(getattr(ds.meta, "num_episodes", 0), max_episodes)):
            ep = ds.get_episode(ep_idx)
            l = len(ep["index"]) if "index" in ep else len(ep)
            lengths.append(l)
            print(f"episode {ep_idx} length:", l)
        if lengths:
            arr = np.array(lengths)
            print("episode length min/mean/max:", arr.min(), arr.mean(), arr.max())
    except Exception as e:
        print("Failed to compute episode stats:", e)


if __name__ == "__main__":
    rl_dataset = "/data/model_checkpoint/clear_brain/notebooks_check/aloha/rl_data"
    rl_dataset_id = "lerobot/realworld_aloha_success_bhc"
    try:
        ds = LeRobotDataset(repo_id=rl_dataset_id, root=rl_dataset)
        print("Loaded LeRobot dataset:", rl_dataset_id)
    except Exception as e:
        print("Failed to load dataset:", e)
        ds = None
    analyze_metadata(ds)
