import gzip
import os
import sys
import json
import argparse
import multiprocessing as mp
from typing import List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

CONFIG_PATH = "config/vln_r2r.yaml"
BASE = "/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA"
HM3D_SCENE_DATASET_CONFIG = os.path.join(BASE, "scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json")
DATA_SOURCES = [
    ("envdrop", os.path.join(BASE, "trajectory_data/EnvDrop"), os.path.join(BASE, "datasets/envdrop/envdrop.json.gz")),
    ("scalevln", os.path.join(BASE, "trajectory_data/ScaleVLN"), os.path.join(BASE, "datasets/scalevln/scalevln_subset_150k.json.gz")),
]
GOAL_RADIUS = 0.25
# id, video, actions — not SFT annotations.json
HABITAT_ANNOTATIONS_FILE = "annotations_bak.json"

# RECOLLECT_VIDEO_DIRS = [
#     "B6ByNegPMKs_envdrop_096694",
#     "B6ByNegPMKs_envdrop_096695",
#     "B6ByNegPMKs_envdrop_096696",
# ]


def _video_stem(video_field: str) -> str:
    v = str(video_field).strip().replace("\\", "/")
    if v.startswith("images/"):
        return v[len("images/") :]
    return v


def _rgb_dir(data_root: str, video_id: str) -> str:
    return os.path.join(data_root, "images", _video_stem(video_id), "rgb")


def _pose_dir(data_root: str, video_id: str) -> str:
    return os.path.join(data_root, "images", _video_stem(video_id))


def _poses_path(data_root: str, video_id: str) -> str:
    return os.path.join(_pose_dir(data_root, video_id), "poses.npy")


def _intrinsics_path(data_root: str, video_id: str) -> str:
    return os.path.join(_pose_dir(data_root, video_id), "intrinsics.npy")


def _quat_to_rotmat(q) -> np.ndarray:
    if hasattr(q, "w") and hasattr(q, "x") and hasattr(q, "y") and hasattr(q, "z"):
        w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    elif hasattr(q, "real") and hasattr(q, "imag"):
        imag = np.asarray(q.imag, dtype=np.float64).reshape(-1)
        w, x, y, z = float(q.real), float(imag[0]), float(imag[1]), float(imag[2])
    else:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 4:
            raise ValueError(f"Unsupported quaternion format: shape={arr.shape}")
        # Habitat commonly uses [x, y, z, w] for stored coefficients.
        x, y, z, w = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _state_to_pose44(state) -> np.ndarray:
    t = np.asarray(state.position, dtype=np.float64).reshape(3)
    r = _quat_to_rotmat(state.rotation)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose


def _get_rgb_pose44(env) -> np.ndarray:
    agent_state = env.sim.get_agent_state()
    sensor_states = getattr(agent_state, "sensor_states", {}) or {}
    rgb_state = None
    for k, v in sensor_states.items():
        if "rgb" in str(k).lower():
            rgb_state = v
            break
    if rgb_state is None and sensor_states:
        rgb_state = next(iter(sensor_states.values()))
    if rgb_state is None:
        rgb_state = agent_state
    return _state_to_pose44(rgb_state)


def _build_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = np.deg2rad(float(hfov_deg))
    fx = (float(width) / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="Collect trajectory camera poses/intrinsics (resume + multi-GPU).")
    p.add_argument("--dataset", choices=["envdrop", "scalevln", "all"], default="all")
    p.add_argument("--num_workers", type=int, default=1, help="Processes; one GPU each.")
    p.add_argument("--gpu_ids", type=str, default=None, help="e.g. 0,1,2,3 (default 0..num_workers-1).")
    p.add_argument(
        "--recollect",
        action="store_true",
        help="Only collect episodes listed in RECOLLECT_VIDEO_DIRS (see top of file).",
    )
    return p.parse_args()


def _allowed_videos_from_args(args: argparse.Namespace) -> Optional[Set[str]]:
    if args.recollect:
        return set(RECOLLECT_VIDEO_DIRS)
    return None


def is_episode_done(data_root, annotation):
    poses_path = _poses_path(data_root, annotation["video"])
    intr_path = _intrinsics_path(data_root, annotation["video"])
    if not (os.path.isfile(poses_path) and os.path.isfile(intr_path)):
        return False
    try:
        poses = np.load(poses_path, allow_pickle=False)
        intr = np.load(intr_path, allow_pickle=False)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            return False
        if intr.shape != (4,):
            return False
        n_expected = len(annotation["actions"])
        return poses.shape[0] == n_expected
    except Exception:
        return False


def _get_todo_episode_ids(data_root: str, annotations: list) -> Set[int]:
    id_to_annot = {a["id"]: a for a in annotations}
    todo = set()
    for eid, ann in id_to_annot.items():
        if not is_episode_done(data_root, ann):
            todo.add(eid)
    return todo


def _dataset_episode_order(data_path: str) -> List[int]:
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
    if not isinstance(episodes, list):
        return []
    return [ep["episode_id"] for ep in episodes]


def _worker_process_episodes(args: Tuple[int, str, str, str, List[int], dict]):
    gpu_id, name, data_root, data_path, episode_ids, id_to_annot = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import habitat
    from vln.habitat_extensions import measures  
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    overrides = [f"habitat.dataset.data_path={data_path}"]
    if name == "scalevln":
        overrides.append(f"habitat.simulator.scene_dataset={HM3D_SCENE_DATASET_CONFIG}")
    cfg = get_config(CONFIG_PATH, overrides=overrides)
    if name == "scalevln":
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        for ep in dataset.episodes:
            ep.scene_dataset_config = HM3D_SCENE_DATASET_CONFIG
        env = habitat.Env(config=cfg, dataset=dataset)
    else:
        env = habitat.Env(config=cfg)
    rgb_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    intrinsics = _build_intrinsics(rgb_cfg.width, rgb_cfg.height, rgb_cfg.hfov)

    ep_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    for eid in tqdm(sorted(set(episode_ids)), desc=f"{name} gpu{gpu_id}", position=gpu_id):
        episode = ep_by_id.get(eid)
        if episode is None:
            continue
        annotation = id_to_annot[eid]
        env.current_episode = episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
        observation = env.reset()
        reference_actions = annotation["actions"][1:] + [0]
        poses = []
        while not env.episode_over:
            poses.append(_get_rgb_pose44(env))
            action = reference_actions.pop(0)
            observation = env.step(action)
        pose_dir = _pose_dir(data_root, annotation["video"])
        os.makedirs(pose_dir, exist_ok=True)
        np.save(_poses_path(data_root, annotation["video"]), np.stack(poses, axis=0))
        np.save(_intrinsics_path(data_root, annotation["video"]), intrinsics)
    env.close()


def _run_single_worker(
    name: str,
    data_root: str,
    data_path: str,
    gpu_id: int,
    allowed_videos: Optional[Set[str]] = None,
    force_recollect: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import habitat
    from vln.habitat_extensions import measures 
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    overrides = [f"habitat.dataset.data_path={data_path}"]
    if name == "scalevln":
        overrides.append(f"habitat.simulator.scene_dataset={HM3D_SCENE_DATASET_CONFIG}")
    cfg = get_config(CONFIG_PATH, overrides=overrides)
    if name == "scalevln":
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        for ep in dataset.episodes:
            ep.scene_dataset_config = HM3D_SCENE_DATASET_CONFIG
        env = habitat.Env(config=cfg, dataset=dataset)
    else:
        env = habitat.Env(config=cfg)
    rgb_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    intrinsics = _build_intrinsics(rgb_cfg.width, rgb_cfg.height, rgb_cfg.hfov)
    annotations = json.load(open(os.path.join(data_root, HABITAT_ANNOTATIONS_FILE), "r"))
    if allowed_videos is not None:
        annotations = [a for a in annotations if _video_stem(a.get("video", "")) in allowed_videos]
    id_to_annot = {a["id"]: a for a in annotations}
    if force_recollect:
        todo_list = sorted(id_to_annot.keys())
    else:
        todo_list = sorted(eid for eid, ann in id_to_annot.items() if not is_episode_done(data_root, ann))
    ep_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    for eid in tqdm(todo_list, desc=name):
        episode = ep_by_id.get(eid)
        if episode is None:
            continue
        annotation = id_to_annot[eid]
        env.current_episode = episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
        observation = env.reset()
        reference_actions = annotation["actions"][1:] + [0]
        poses = []
        while not env.episode_over:
            poses.append(_get_rgb_pose44(env))
            action = reference_actions.pop(0)
            observation = env.step(action)
        pose_dir = _pose_dir(data_root, annotation["video"])
        os.makedirs(pose_dir, exist_ok=True)
        np.save(_poses_path(data_root, annotation["video"]), np.stack(poses, axis=0))
        np.save(_intrinsics_path(data_root, annotation["video"]), intrinsics)
    env.close()


def main():
    args = parse_args()
    allowed_videos = _allowed_videos_from_args(args)
    gpu_ids = args.gpu_ids
    if gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_workers))

    sources = [
        (name, data_root, data_path)
        for name, data_root, data_path in DATA_SOURCES
        if args.dataset == "all" or args.dataset == name
    ]

    for name, data_root, data_path in sources:
        annotations = json.load(open(os.path.join(data_root, HABITAT_ANNOTATIONS_FILE), "r"))
        if allowed_videos is not None:
            annotations = [a for a in annotations if _video_stem(a.get("video", "")) in allowed_videos]
            if not annotations:
                print(f"[{name}] No annotations match filter, skip.")
                continue
            print(f"[{name}] Filter: {len(allowed_videos)} video dirs, {len(annotations)} annotations.")
        id_to_annot = {a["id"]: a for a in annotations}
        if args.recollect:
            todo_ids = set(id_to_annot.keys())
        else:
            todo_ids = _get_todo_episode_ids(data_root, annotations)
        if not todo_ids:
            print(f"[{name}] All episodes already done, skip.")
            continue

        n_workers = min(args.num_workers, len(gpu_ids), len(todo_ids))
        if n_workers <= 1:
            gpu_id = gpu_ids[0] if gpu_ids else 0
            _run_single_worker(name, data_root, data_path, gpu_id, allowed_videos, force_recollect=args.recollect)
            continue

        dataset_order = _dataset_episode_order(data_path)
        todo_list = [eid for eid in dataset_order if eid in todo_ids]
        if len(todo_list) < len(todo_ids):
            extra = todo_ids - set(todo_list)
            todo_list = todo_list + list(extra)
        chunk_size = (len(todo_list) + n_workers - 1) // n_workers
        chunks = [todo_list[i : i + chunk_size] for i in range(0, len(todo_list), chunk_size)]
        chunks = chunks[:n_workers]
        procs = []
        for i, chunk in enumerate(chunks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            p = mp.Process(
                target=_worker_process_episodes,
                args=((gpu_id, name, data_root, data_path, chunk, id_to_annot),),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
