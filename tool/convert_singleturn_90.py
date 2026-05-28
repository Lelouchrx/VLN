import argparse
import copy
import json
import os
import random
import re

from tqdm import tqdm


SYSTEM_PROMPT = "You are a visual language navigation model."

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations{} and an image of the current observation{}. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

IDM_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given an image of current view{} and an image of the goal view{}. "
    "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

FFS_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You are given a history of observations{}, followed by the current observation{} and the executed action: [{}]. "
    "Then you are given four candidate next observations in order. "
    "Please select the correct option letter from A/B/C/D."
)

FORWARD_DISTANCE = 25
TURN_ANGLE = 15
MAX_HISTORY = 8
MAX_ACTIONS_PER_SAMPLE = 3
COMBINE_PROB = 0.7
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def action_id_to_str(action_id):
    action_id = int(action_id)
    if action_id == -1 or action_id == 0:
        return "stop"
    if action_id == 1:
        return f"forward {FORWARD_DISTANCE} cm"
    if action_id == 2:
        return f"turn left {TURN_ANGLE} degree"
    if action_id == 3:
        return f"turn right {TURN_ANGLE} degree"
    raise ValueError(f"Invalid action id: {action_id}")


def to_action_text(action):
    if isinstance(action, str):
        text = action.strip()
        return "stop" if text == "-1" else text
    return action_id_to_str(action)


def try_combine(running_value, next_action):
    idx = running_value.rfind(", ")
    head = running_value[: idx + 2] if idx != -1 else ""
    tail = running_value[idx + 2 :] if idx != -1 else running_value
    m1 = int(re.search(r"-?\d+", tail).group())
    m2 = int(re.search(r"-?\d+", next_action).group())
    if "forward" in tail:
        cap, unit, kind = 3 * FORWARD_DISTANCE, "cm", "forward"
    elif "turn left" in tail:
        cap, unit, kind = 3 * TURN_ANGLE, "degree", "turn left"
    elif "turn right" in tail:
        cap, unit, kind = 3 * TURN_ANGLE, "degree", "turn right"
    else:
        return None
    if m1 + m2 > cap:
        return None
    return f"{head}{kind} {m1 + m2} {unit}"


def uniform_sample_with_ends(data, n):
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def list_episode_images(image_root, video_id):
    folder = os.path.join(image_root, str(video_id))
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]) if re.findall(r"\d+", x) else x)
    return [os.path.join(folder, f) for f in files]


def normalize_pair(images):
    return [images[0], images[0]] if len(images) == 1 else images


def sample_history_plus_current(images):
    if len(images) <= 1:
        return images
    history = images[:-1]
    if len(history) > MAX_HISTORY:
        history = uniform_sample_with_ends(history, MAX_HISTORY)
    return history + [images[-1]]


def build_vln_prompt(instruction, image_count):
    history_count = max(0, image_count - 1)
    return VLN_PROMPT_TEMPLATE.format("<image>" * history_count, "<image>", instruction)


def finalize_vln_sample(sample, instruction):
    user = sample["messages"][0]
    images = sample_history_plus_current(user["image"])
    images = normalize_pair(images)
    user["image"] = images
    user["value"] = build_vln_prompt(instruction, len(images))
    sample["images"] = list(images)
    del user["image"]
    return sample


def finalize_idm_sample(sample):
    user = sample["messages"][0]
    images = list(user["image"])
    user["value"] = IDM_PROMPT_TEMPLATE.format("<image>", "<image>")
    sample["images"] = images
    del user["image"]
    return sample


def prep_actions_and_images(episode, image_root, drop_stop):
    raw_actions = episode.get("actions") or []
    if not raw_actions:
        return None
    action_texts = [to_action_text(a) for a in raw_actions]
    while action_texts and action_texts[0] == "stop":
        action_texts.pop(0)
    if drop_stop:
        action_texts = [a for a in action_texts if a != "stop"]
    elif action_texts and action_texts[-1] != "stop":
        action_texts.append("stop")
    if not action_texts:
        return None

    video_id = episode.get("video_id", episode.get("episode_id"))
    images = list_episode_images(image_root, video_id)
    if not images:
        return None
    target_len = len(action_texts) + 1
    if len(images) < target_len:
        images = images + [images[-1]] * (target_len - len(images))
    else:
        images = images[:target_len]
    return action_texts, images


def get_instructions(episode, only_first):
    instructions = episode.get("instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    instructions = [s for s in instructions if isinstance(s, str) and s.strip()]
    if only_first:
        instructions = instructions[:1]
    return instructions


def convert_episode_vln(episode, image_root, out_samples):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=False)
    if prepared is None:
        return
    action_texts, images = prepared
    instructions = get_instructions(episode, only_first=False)
    if not instructions:
        return
    episode_id = str(episode.get("episode_id", episode.get("video_id", "")))

    for instruction in instructions:
        sample = {
            "system": SYSTEM_PROMPT,
            "messages": [
                {"from": "user", "value": "", "image": [images[0]]},
                {"from": "assistant", "value": action_texts[0]},
            ],
            "episode_id": episode_id,
            "task type": "vln",
        }
        last_action = action_texts[0]
        pending = []
        for i in range(1, len(action_texts)):
            cur = action_texts[i]
            running = sample["messages"][1]["value"]
            combined = try_combine(running, cur) if cur == last_action else None
            if combined is not None and random.random() <= COMBINE_PROB:
                sample["messages"][1]["value"] = combined
                pending.append(images[i])
            elif running.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
                sample["messages"][1]["value"] = running + ", " + cur
                pending.append(images[i])
            else:
                out_samples.append(finalize_vln_sample(copy.deepcopy(sample), instruction))
                sample["messages"][0]["image"].extend(pending)
                sample["messages"][0]["image"].append(images[i])
                sample["messages"][1]["value"] = cur
                pending = []
            last_action = cur
        out_samples.append(finalize_vln_sample(copy.deepcopy(sample), instruction))


def convert_episode_idm(episode, image_root, out_samples):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=True)
    if prepared is None:
        return
    action_texts, images = prepared
    if len(images) < 2:
        return
    instructions = get_instructions(episode, only_first=True)
    if not instructions:
        return
    episode_id = str(episode.get("episode_id", episode.get("video_id", "")))

    sample = {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"from": "user", "value": "", "image": [images[0], images[1]]},
            {"from": "assistant", "value": action_texts[0]},
        ],
        "episode_id": episode_id,
        "task type": "idm",
    }
    last_action = action_texts[0]
    for i in range(1, len(action_texts)):
        cur = action_texts[i]
        running = sample["messages"][1]["value"]
        next_goal = images[i + 1] if i + 1 < len(images) else images[-1]
        combined = try_combine(running, cur) if cur == last_action else None
        if combined is not None and random.random() <= COMBINE_PROB:
            sample["messages"][1]["value"] = combined
            sample["messages"][0]["image"][-1] = next_goal
        elif running.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            sample["messages"][1]["value"] = running + ", " + cur
            sample["messages"][0]["image"][-1] = next_goal
        else:
            out_samples.append(finalize_idm_sample(copy.deepcopy(sample)))
            sample["messages"][0]["image"] = [images[i], next_goal]
            sample["messages"][1]["value"] = cur
        last_action = cur
    if sample["messages"][1]["value"] != "stop":
        out_samples.append(finalize_idm_sample(copy.deepcopy(sample)))


def build_scene_frame_pool(episodes, image_root):
    pool = {}
    for episode in tqdm(episodes, desc="ffs scene pool"):
        video_id = str(episode.get("video_id", episode.get("episode_id", "")))
        if not video_id:
            continue
        scene_prefix = video_id.split("_")[0] if "_" in video_id else video_id
        frames = list_episode_images(image_root, video_id)
        if not frames:
            continue
        pool.setdefault(scene_prefix, {})[video_id] = frames
    return pool


def compose_ffs_action(action_texts, start_idx):
    merged = action_texts[start_idx]
    last_action = action_texts[start_idx]
    end_idx = start_idx
    for i in range(start_idx + 1, len(action_texts)):
        cur = action_texts[i]
        combined = try_combine(merged, last_action) if cur == last_action else None
        if combined is not None and random.random() <= COMBINE_PROB:
            merged = combined
            end_idx = i
        elif merged.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            merged += ", " + cur
            end_idx = i
        else:
            break
        last_action = cur
    return merged, end_idx


def make_ffs_sample(episode, episode_image_list, action_texts, start_idx, end_idx, action_text, scene_pool):
    if end_idx + 1 >= len(episode_image_list):
        return None
    current_img = episode_image_list[start_idx]
    true_next_img = episode_image_list[end_idx + 1]

    history_imgs = episode_image_list[:start_idx]
    if not history_imgs:
        history_imgs = [current_img]
    if len(history_imgs) > MAX_HISTORY:
        history_imgs = uniform_sample_with_ends(history_imgs, MAX_HISTORY)

    before_window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    neg1_candidates = [x for x in before_window if x != true_next_img]
    if not neg1_candidates:
        return None
    neg1 = random.choice(neg1_candidates)
    exclude = {true_next_img, neg1}

    if action_texts[end_idx] == "stop":
        window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    else:
        window = episode_image_list[end_idx + 2 : min(len(episode_image_list), end_idx + 6)]
        if not window:
            window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    neg2_candidates = [x for x in window if x not in exclude]
    if not neg2_candidates:
        return None
    neg2 = random.choice(neg2_candidates)
    exclude.add(neg2)

    video_id = str(episode.get("video_id", episode.get("episode_id", "")))
    scene_prefix = video_id.split("_")[0] if "_" in video_id else video_id
    cross_candidates = []
    siblings = scene_pool.get(scene_prefix, {})
    for other_traj, frames in siblings.items():
        if other_traj == video_id:
            continue
        cross_candidates.extend(f for f in frames if f not in exclude)
    if not cross_candidates:
        for prefix, trajs in scene_pool.items():
            if prefix == scene_prefix:
                continue
            for other_traj, frames in trajs.items():
                cross_candidates.extend(f for f in frames if f not in exclude)
            if cross_candidates:
                break
    if not cross_candidates:
        return None
    neg3 = random.choice(cross_candidates)

    options = [("neg", neg1), ("neg", neg2), ("neg", neg3), ("true", true_next_img)]
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    correct = None
    option_imgs = []
    for idx, (kind, path) in enumerate(options):
        option_imgs.append(path)
        if kind == "true":
            correct = labels[idx]

    history_tokens = "<image>" * len(history_imgs)
    prompt = FFS_PROMPT_TEMPLATE.format(history_tokens, "<image>", action_text)
    prompt += " Option A: <image>, Option B: <image>, Option C: <image>, Option D: <image>."

    images = history_imgs + [current_img] + option_imgs
    return {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": correct},
        ],
        "episode_id": str(episode.get("episode_id", episode.get("video_id", ""))),
        "task type": "ffs",
        "images": images,
    }


def convert_episode_ffs(episode, image_root, scene_pool, out_samples):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=False)
    if prepared is None:
        return
    action_texts, images = prepared
    i = 0
    while i < len(action_texts):
        merged, end_idx = compose_ffs_action(action_texts, i)
        sample = make_ffs_sample(episode, images, action_texts, i, end_idx, merged, scene_pool)
        if sample is not None:
            out_samples.append(sample)
        i = end_idx + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path",
        default="/wuji-vefps-D/wuji-il/caiwr/data/trajectory_data/R2R_90/r2r.jsonl",
    )
    parser.add_argument(
        "--image_root",
        default="/wuji-vefps-D/wuji-il/caiwr/data/trajectory_data/R2R_90/images",
    )
    parser.add_argument(
        "--output_path",
        default="/wuji-vefps-D/wuji-il/caiwr/data/trajectory_data/R2R_90/annotations_vln_llamafactory.jsonl",
    )
    parser.add_argument(
        "--task_type",
        nargs="+",
        choices=["vln", "idm", "ffs"],
        default=["vln"],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.annotation_path, "r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    scene_pool = None
    if "ffs" in args.task_type:
        scene_pool = build_scene_frame_pool(episodes, args.image_root)

    samples = []
    for task_type in args.task_type:
        if task_type == "vln":
            for episode in tqdm(episodes, desc="vln"):
                convert_episode_vln(episode, args.image_root, samples)
        elif task_type == "idm":
            for episode in tqdm(episodes, desc="idm"):
                convert_episode_idm(episode, args.image_root, samples)
        elif task_type == "ffs":
            for episode in tqdm(episodes, desc="ffs"):
                convert_episode_ffs(episode, args.image_root, scene_pool, samples)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
