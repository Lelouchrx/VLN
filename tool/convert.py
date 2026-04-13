import json
import os
from pathlib import Path
import base64
import argparse
import random
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _empty_image_paths(img_files):
    """返回 0 字节或无法 stat 的图片路径（视为空/无效）。"""
    bad = []
    for p in img_files:
        try:
            sz = p.stat().st_size
        except OSError:
            bad.append(p)
            continue
        if sz == 0:
            bad.append(p)
    return bad


def convert(data_path, out_path, batch_size=4, workers=None, seed=42, random_sample=False):
    action_map = {
        1: "Move forward 25 cm",
        2: "Turn left 15 degrees",
        3: "Turn right 15 degrees",
        0: "Stop"
    }

    conjunctions = [
        "you can see ",
        "in front of you is ",
        "there is ",
        "you can spot ",
        "you are toward the ",
        "ahead of you is ",
        "in your sight is ",
        "you notice ",
        "directly ahead lies ",
        "before you stands ",
        "nearby you see ",
        "in the distance is ",
        "lying ahead is ",
        "situated ahead is "
    ]

    dataset_config = {
        # 'R2R': 'annotations_v1-3.json',
        # 'RxR': 'annotations.json',
        'ScaleVLN': 'annotations_bak.json',
        'EnvDrop': 'annotations_bak.json'
    }

    if workers is None:
        workers = min(32, max(1, (os.cpu_count() or 1) * 2))
    workers = max(1, int(workers))

    for dataset, annot_file in tqdm(dataset_config.items(), desc="Datasets"):
        annot_path = Path(data_path) / dataset / annot_file
        img_base = Path(data_path) / dataset / 'images'

        if not annot_path.exists() and dataset == "RxR":
            alt_path = Path(data_path) / dataset / "annotations_bak.json"
            if alt_path.exists():
                annot_path = alt_path

        if not annot_path.exists() or not img_base.exists():
            print(f"Skipping {dataset}: missing files")
            continue

        img_dir = img_base / 'images' if (img_base / 'images').exists() else img_base
        print(f"Processing {dataset}...")
        results = []

        if annot_path.suffix == '.gz':
            with gzip.open(annot_path, 'rt') as f:
                episodes = json.load(f)
        else:
            with open(annot_path, 'r') as f:
                episodes = json.load(f)

        sub_dirs = [sub for sub in img_dir.iterdir() if sub.is_dir()]
        sub_dir_by_name = {sub.name: sub for sub in sub_dirs}
        scene_to_dir = {}
        scene_lock = threading.Lock()

        def stable_int_from_id(x) -> int:
            s = str(x)
            h = 0
            for ch in s:
                h = (h * 131 + ord(ch)) & 0x7fffffff
            return h

        def resolve_ep_dir(scene):
            with scene_lock:
                if scene in scene_to_dir:
                    return scene_to_dir[scene]
            ep_dir = sub_dir_by_name.get(scene)
            if ep_dir is None:
                for sub in sub_dirs:
                    if scene in sub.name:
                        ep_dir = sub
                        break
            with scene_lock:
                scene_to_dir[scene] = ep_dir
            return ep_dir

        def process_episode(ep):
            ep_id = ep['id']
            video = ep['video']
            scene = video.split('/')[-1]

            ep_dir = resolve_ep_dir(scene)

            if ep_dir is None or not (ep_dir / 'rgb').exists():
                return None

            img_files = sorted((ep_dir / 'rgb').glob('*.jpg'))
            actions = ep['actions']
            instr = ep['instructions'][0] if ep['instructions'] else ""

            if actions and actions[0] == -1:
                actions = actions[1:]

            if len(actions) < 2:
                return None

            actions = actions + [0]

            if len(actions) != len(img_files):
                return None

            if not img_files:
                return None

            rng = random.Random(seed + stable_int_from_id(ep_id))

            step_images = [img_files[j] for j in range(0, len(img_files), batch_size)]
            n_turns = len(step_images)
            if n_turns <= 0:
                return None

            step_action_texts = []
            for s in range(n_turns):
                start = s * batch_size
                end = min((s + 1) * batch_size, len(actions))
                act_batch = actions[start:end]
                if len(act_batch) < batch_size:
                    act_batch = act_batch + [0] * (batch_size - len(act_batch))

                act_texts = [
                    action_map[act] if act in action_map else str(act) for act in act_batch
                ]
                step_action_texts.append(", ".join(act_texts))

            system_msg = {
                "from": "system",
                "value": "You are an autonomous navigation assistant. Your task is to follow the given instruction in the environment. At each step, output the next four actions to take, exactly four, separated by commas. Use only the following actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop."
            }

            def build_conv(end_turn: int, sample_tag: str, start_turn: int = 0):
                conv = [system_msg]
                conv.append({"from": "user", "value": f"Your instruction is '{instr}' <image>"})
                for s in range(start_turn, end_turn + 1):
                    conv.append({"from": "assistant", "value": step_action_texts[s]})
                    if s < end_turn:
                        visual_prompt = rng.choice(conjunctions)
                        conv.append({"from": "user", "value": f"{visual_prompt}<image>"})
                return {
                    "episode_id": ep_id,
                    "dataset": dataset,
                    "sample_tag": sample_tag,
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "messages": conv,
                    "images": [str(p) for p in step_images[start_turn : end_turn + 1]],
                }

            full_end_turn = n_turns - 1
            if random_sample:
                max_len = min(20, n_turns)
                seg_len = rng.randint(1, max_len)
                st = rng.randint(0, n_turns - seg_len)
                en = st + seg_len - 1
                out = []
                if st != 0 or en != full_end_turn:
                    out.append(build_conv(en, "random", start_turn=st))
                out.append(build_conv(full_end_turn, "full", start_turn=0))
                return out
            return [build_conv(full_end_turn, "full", start_turn=0)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_episode, ep) for ep in episodes]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset} episodes"):
                item = future.result()
                if item:
                    results.extend(item)

        n_ok = sum(1 for f in futures if f.result() is not None)
        print(f"{dataset}: episodes {len(episodes)} | 成功 {n_ok} | 跳过 {len(episodes) - n_ok}")

        out_file = Path(data_path) / dataset / out_path
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"{dataset}: Converted {len(results)} training samples -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data")
    parser.add_argument("--out_path", type=str, default="annotations_random.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_sample", action="store_true")
    args = parser.parse_args()
    convert(
        args.data_path,
        args.out_path,
        args.batch_size,
        args.workers,
        args.seed,
        args.random_sample,
    )