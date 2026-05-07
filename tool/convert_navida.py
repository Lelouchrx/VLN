import json
import os
import random
import re
import copy
from tqdm import tqdm
import argparse


forward_distance = 25
turn_angle = 15


def action_id_to_str(action_id):
    if action_id == 0:
        return "stop"
    elif action_id == 1:
        return f"forward {forward_distance} cm"
    elif action_id == 2:
        return f"turn left {turn_angle} degree"
    elif action_id == 3:
        return f"turn right {turn_angle} degree"
    else:
        raise ValueError(f"Invalid action ID: {action_id}")


def uniform_sample_with_ends(data, n):
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def to_action_text(action):
    if isinstance(action, str):
        text = action.strip()
        if text == "-1":
            return "stop"
        return text
    action = int(action)
    if action == -1:
        action = 0
    return action_id_to_str(action)


def combine(action1, action2):
    idx = action1.rfind(', ')
    subaction0 = action1[:idx + 1] + ' ' if action1[:idx + 1] != '' else action1[:idx + 1]
    subaction1 = action1[idx + 1:]
    match1 = re.search(r'-?\d+', subaction1)
    match1 = int(match1.group())
    match2 = re.search(r'-?\d+', action2)
    match2 = int(match2.group())
    if "forward" in subaction1:
        if match1 + match2 <= 3 * forward_distance:
            return f"{subaction0}forward {match1 + match2} cm"
        else:
            return None
    elif "turn left" in subaction1:
        if match1 + match2 <= 3 * turn_angle:
            return f"{subaction0}turn left {match1 + match2} degree"
        else:
            return None
    elif "turn right" in subaction1:
        if match1 + match2 <= 3 * turn_angle:
            return f"{subaction0}turn right {match1 + match2} degree"
        else:
            return None
    else:
        raise ValueError(f"Invalid action: {action1}")


def get_json_items(annotation_path):
    with open(annotation_path, 'r', encoding='utf-8') as f:
        if annotation_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ["data", "annotations", "items"]:
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Unsupported annotation format: {annotation_path}")


def get_first(item, keys, default=None):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def get_episode_images(episode_item, image_root):
    direct_images = get_first(episode_item, ["images", "image", "image_paths", "rgb_paths"], None)
    if isinstance(direct_images, list) and len(direct_images) > 0:
        return direct_images

    video_id = get_first(episode_item, ["video_id", "video", "trajectory_id", "traj_id"], None)
    if video_id is None and "episode_id" in episode_item:
        video_id = str(episode_item["episode_id"])
    if video_id is None:
        raise ValueError(f"Cannot infer image folder from item keys: {list(episode_item.keys())}")

    video_id = str(video_id)
    if os.path.isabs(video_id):
        episode_image_path = video_id
    else:
        if video_id.startswith("images/"):
            video_id = video_id[len("images/"):]
        episode_image_path = os.path.join(image_root, video_id)
    rgb_path = os.path.join(episode_image_path, "rgb")
    if os.path.isdir(rgb_path):
        episode_image_path = rgb_path
    if not os.path.isdir(episode_image_path):
        raise FileNotFoundError(f"Image directory not found: {episode_image_path}")

    episode_image_list = [
        image for image in os.listdir(episode_image_path)
        if image.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    episode_image_list = sorted(
        episode_image_list,
        key=lambda x: int(re.findall(r"\d+", x)[-1]) if re.findall(r"\d+", x) else x
    )
    return [episode_image_path + '/' + image for image in episode_image_list]


def build_scene_frame_pool(annotation, image_root):
    scene_pool = {}
    for episode_item in tqdm(annotation, desc="build scene pools"):
        trajectory_id = get_first(episode_item, ["video_id", "video", "trajectory_id", "traj_id"], None)
        if trajectory_id is None:
            trajectory_id = get_first(episode_item, ['episode_id', 'path_id', 'id'], '')
        trajectory_id = str(trajectory_id)
        scene_prefix = trajectory_id.split("_")[0] if "_" in trajectory_id else trajectory_id
        try:
            frames = get_episode_images(episode_item, image_root)
        except Exception:
            continue
        if len(frames) == 0:
            continue
        if scene_prefix not in scene_pool:
            scene_pool[scene_prefix] = {}
        scene_pool[scene_prefix][trajectory_id] = frames
    return scene_pool


def compose_ffs_action(action_texts, start_idx):
    merged_action = action_texts[start_idx]
    last_action = action_texts[start_idx]
    end_idx = start_idx
    for i in range(start_idx + 1, len(action_texts)):
        prob = random.random()
        if prob <= 0.7 and action_texts[i] == last_action and combine(merged_action, last_action) is not None:
            merged_action = combine(merged_action, last_action)
            end_idx = i
        else:
            count = merged_action.count(',')
            if count < 2:
                merged_action += ', ' + action_texts[i]
                end_idx = i
            else:
                break
        last_action = action_texts[i]
    return merged_action, end_idx


def make_ffs_sample(episode_item, episode_image_list, action_texts, action_start_idx, action_end_idx, action_text, scene_pool, system_prompt, prompt_template):
    total_actions = len(action_texts)
    if action_start_idx >= total_actions:
        return None
    if action_end_idx + 1 >= len(episode_image_list):
        return None

    current_img = episode_image_list[action_start_idx]
    true_next_img = episode_image_list[action_end_idx + 1]

    history_imgs = episode_image_list[:action_start_idx]
    if len(history_imgs) == 0:
        history_imgs = [current_img]
    if len(history_imgs) > 8:
        history_imgs = uniform_sample_with_ends(history_imgs, 8)

    # 1) Neighbor hard negative before the true next frame.
    before_window_start = max(0, action_end_idx - 4)
    neg1_candidates = episode_image_list[before_window_start:action_end_idx + 1]
    neg1_filtered = [x for x in neg1_candidates if x != true_next_img]
    neg1 = random.choice(neg1_filtered) if len(neg1_filtered) > 0 else None
    if neg1 is None:
        return None

    # 2) Temporal negative in next 4 frames; for stop action, sample from previous 4 frames.
    exclude_set = {true_next_img, neg1}
    if action_texts[action_end_idx] == "stop":
        neg2_candidates = episode_image_list[max(0, action_end_idx - 4):action_end_idx + 1]
    else:
        neg2_candidates = episode_image_list[action_end_idx + 2:min(len(episode_image_list), action_end_idx + 6)]
        if len(neg2_candidates) == 0:
            neg2_candidates = episode_image_list[max(0, action_end_idx - 4):action_end_idx + 1]
    neg2_filtered = [x for x in neg2_candidates if x not in exclude_set]
    neg2 = random.choice(neg2_filtered) if len(neg2_filtered) > 0 else None
    if neg2 is None:
        return None

    # 3) Cross-trajectory negative from the same scene prefix.
    trajectory_id = get_first(episode_item, ["video_id", "video", "trajectory_id", "traj_id"], None)
    if trajectory_id is None:
        trajectory_id = get_first(episode_item, ['episode_id', 'path_id', 'id'], '')
    trajectory_id = str(trajectory_id)
    scene_prefix = trajectory_id.split("_")[0] if "_" in trajectory_id else trajectory_id
    neg3 = None
    if scene_prefix in scene_pool:
        cross_traj_candidates = []
        for other_traj, frames in scene_pool[scene_prefix].items():
            if other_traj == trajectory_id:
                continue
            for frame in frames:
                if frame not in exclude_set and frame != neg2:
                    cross_traj_candidates.append(frame)
        if len(cross_traj_candidates) > 0:
            neg3 = random.choice(cross_traj_candidates)
    if neg3 is None:
        return None

    option_items = [
        ("neg", neg1),
        ("neg", neg2),
        ("neg", neg3),
        ("true", true_next_img),
    ]
    random.shuffle(option_items)
    labels = ["A", "B", "C", "D"]
    correct_label = None
    option_images = []
    for idx, (kind, img_path) in enumerate(option_items):
        option_images.append(img_path)
        if kind == "true":
            correct_label = labels[idx]
    if correct_label is None:
        return None

    history_tokens = "<image>" * len(history_imgs)
    current_token = "<image>"
    prompt = prompt_template.format(history_tokens, current_token, action_text)
    prompt += " Option A: <image>, Option B: <image>, Option C: <image>, Option D: <image>."
    sample = {
        "system": system_prompt,
        "messages": [
            {
                "from": "user",
                "value": prompt,
                "image": history_imgs + [current_img] + option_images,
            },
            {"from": "assistant", "value": correct_label},
        ],
        "episode_id": str(get_first(episode_item, ['episode_id', 'path_id', 'id'], '')),
        "task type": "ffs",
    }
    return attach_llamafactory_mm(sample)


def sample_vln_images(image_list):
    if len(image_list) <= 1:
        return image_list
    history = image_list[:-1]
    if len(history) > 8:
        history = uniform_sample_with_ends(history, 8)
    return history + [image_list[-1]]


def build_vln_prompt(prompt_template, instruction, image_list):
    history_count = max(0, len(image_list) - 1)
    history_tokens = "<image>" * history_count
    return prompt_template.format(history_tokens, "<image>", instruction)


def normalize_vln_images(image_list):
    if len(image_list) == 1:
        return [image_list[0], image_list[0]]
    return image_list


def attach_llamafactory_mm(example):
    user = example["messages"][0]
    imgs = user.get("image", [])
    if not isinstance(imgs, list):
        imgs = [imgs] if imgs else []
    example["images"] = imgs[:]
    if "<image>" not in user["value"]:
        user["value"] = "<image>" * len(imgs) + user["value"]
    if "image" in user:
        del user["image"]
    return example


def process_single_type(annotation, image_root, system_prompt, prompt_template, task_type, scene_pool=None):
    data2save = []
    for episode_item in tqdm(annotation):
        episode_id = get_first(episode_item, ['episode_id', 'path_id', 'id'], '')
        actions = get_first(episode_item, ['actions', 'action', 'action_ids', 'nav_actions'], [])
        if not isinstance(actions, list) or len(actions) == 0:
            continue

        all_action_texts = [to_action_text(a) for a in actions]
        if len(all_action_texts) == 0:
            continue

        action_texts = all_action_texts[:]
        while len(action_texts) > 0 and action_texts[0] == "stop":
            action_texts = action_texts[1:]
        if task_type == "idm":
            action_texts = [a for a in action_texts if a != "stop"]
        elif len(action_texts) > 0 and action_texts[-1] != "stop":
            action_texts.append("stop")
        if len(action_texts) == 0:
            continue

        instruction_value = get_first(episode_item, ['instruction', 'instructions'], "")
        if isinstance(instruction_value, list):
            instruction_list = instruction_value
        else:
            instruction_list = [instruction_value]
        instruction_list = [x for x in instruction_list if isinstance(x, str) and x.strip()]
        if len(instruction_list) == 0:
            continue
        if task_type == "idm":
            instruction_list = instruction_list[:1]

        episode_image_list = get_episode_images(episode_item, image_root)
        if len(episode_image_list) == 0:
            continue
        if len(episode_image_list) < len(action_texts) + 1:
            episode_image_list += [episode_image_list[-1]] * (len(action_texts) + 1 - len(episode_image_list))
        elif len(episode_image_list) > len(action_texts) + 1:
            episode_image_list = episode_image_list[:len(action_texts) + 1]

        for instruction in instruction_list:
            tmp_data = {
                "system": system_prompt,
                "messages": [],
                "episode_id": str(episode_id),
                "task type": task_type,
            }
            if task_type == "vln":
                formated_instruction = build_vln_prompt(prompt_template, instruction, [episode_image_list[0]])
                tmp_data["messages"].append({
                    "from": "user",
                    "value": formated_instruction,
                    "image": normalize_vln_images([episode_image_list[0]])
                })
                tmp_data["messages"].append({"from": "assistant", "value": action_texts[0]})
                last_action = action_texts[0]
                pending_rgb_list = []
                for i in range(1, len(action_texts)):
                    prob = random.random()
                    if prob <= 0.7 and action_texts[i] == last_action and combine(tmp_data["messages"][-1]['value'], last_action) is not None:
                        tmp_data["messages"][-1]['value'] = combine(tmp_data["messages"][-1]['value'], last_action)
                        pending_rgb_list.append(episode_image_list[i])
                    else:
                        count = tmp_data["messages"][-1]['value'].count(',')
                        if count < 2:
                            tmp_data["messages"][-1]['value'] += ', ' + action_texts[i]
                            pending_rgb_list.append(episode_image_list[i])
                        else:
                            out_item = copy.deepcopy(tmp_data)
                            out_item["messages"][0]["image"] = sample_vln_images(out_item["messages"][0]["image"])
                            out_item["messages"][0]["image"] = normalize_vln_images(out_item["messages"][0]["image"])
                            out_item["messages"][0]["value"] = build_vln_prompt(
                                prompt_template, instruction, out_item["messages"][0]["image"]
                            )
                            data2save.append(attach_llamafactory_mm(out_item))
                            while len(pending_rgb_list) != 0:
                                item = pending_rgb_list.pop(0)
                                tmp_data["messages"][0]['image'].append(item)
                            tmp_data["messages"][0]['image'].append(episode_image_list[i])
                            tmp_data["messages"][1]['value'] = action_texts[i]
                    last_action = action_texts[i]
                out_item = copy.deepcopy(tmp_data)
                out_item["messages"][0]["image"] = sample_vln_images(out_item["messages"][0]["image"])
                out_item["messages"][0]["image"] = normalize_vln_images(out_item["messages"][0]["image"])
                out_item["messages"][0]["value"] = build_vln_prompt(
                    prompt_template, instruction, out_item["messages"][0]["image"]
                )
                data2save.append(attach_llamafactory_mm(out_item))
            elif task_type == "idm":
                formated_instruction = prompt_template.format("<image>", "<image>")
                tmp_data["messages"].append({"from": "user", "value": formated_instruction, "image": [episode_image_list[0], episode_image_list[1]]})
                tmp_data["messages"].append({"from": "assistant", "value": action_texts[0]})
                last_action = action_texts[0]
                for i in range(1, len(action_texts)):
                    prob = random.random()
                    if prob <= 0.7 and action_texts[i] == last_action and combine(tmp_data["messages"][-1]['value'], last_action) is not None:
                        tmp_data["messages"][-1]['value'] = combine(tmp_data["messages"][-1]['value'], last_action)
                        tmp_data["messages"][-2]['image'][-1] = episode_image_list[i + 1]
                    else:
                        count = tmp_data["messages"][-1]['value'].count(',')
                        if count < 2:
                            tmp_data["messages"][-1]['value'] += ', ' + action_texts[i]
                            tmp_data["messages"][-2]['image'][-1] = episode_image_list[i + 1]
                        else:
                            data2save.append(attach_llamafactory_mm(copy.deepcopy(tmp_data)))
                            tmp_data["messages"][0]['image'] = [episode_image_list[i], episode_image_list[i + 1]]
                            tmp_data["messages"][1]['value'] = action_texts[i]
                    last_action = action_texts[i]
                if tmp_data["messages"][1]['value'] != 'stop':
                    data2save.append(attach_llamafactory_mm(copy.deepcopy(tmp_data)))
            elif task_type == "ffs":
                if scene_pool is None:
                    continue
                i = 0
                while i < len(action_texts):
                    composed_action, end_idx = compose_ffs_action(action_texts, i)
                    out_item = make_ffs_sample(
                        episode_item=episode_item,
                        episode_image_list=episode_image_list,
                        action_texts=action_texts,
                        action_start_idx=i,
                        action_end_idx=end_idx,
                        action_text=composed_action,
                        scene_pool=scene_pool,
                        system_prompt=system_prompt,
                        prompt_template=prompt_template,
                    )
                    if out_item is not None:
                        data2save.append(out_item)
                    i = end_idx + 1
    return data2save


def main(annotation_path, image_root, task_type_list, output_path):
    system_prompt = "You are a visual language navigation model."
    vln_prompt_template = "Imagine you are a robot programmed for navigation tasks. " \
        "You have been given a video of historical observations{} and an image of the current observation{}. " \
        "Your assigned task is: '{}'. Analyze this series of images to decide your next move, " \
        "which could involve turning left or right by a specific degree or moving forward a certain distance."

    idm_prompt_template = "Imagine you are a robot programmed for navigation tasks. " \
        "You have been given an image of current view{} and an image of the goal view{}. " \
        "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, " \
        "which could involve turning left " \
        "or right by a specific degree or moving forward a certain distance."

    ffs_prompt_template = "Imagine you are a robot programmed for navigation tasks. " \
        "You are given a history of observations{}, followed by the current observation{} and the executed action: [{}]. " \
        "Then you are given four candidate next observations in order. " \
        "Please select the correct option letter from A/B/C/D."

    data2save = []
    annotation = get_json_items(annotation_path)
    scene_pool = None
    if "ffs" in task_type_list:
        scene_pool = build_scene_frame_pool(annotation, image_root)

    for task_type in task_type_list:
        if task_type == "vln":
            prompt_template = vln_prompt_template
        elif task_type == "idm":
            prompt_template = idm_prompt_template
        elif task_type == "ffs":
            prompt_template = ffs_prompt_template
        else:
            continue
        data2save.extend(process_single_type(annotation, image_root, system_prompt, prompt_template, task_type, scene_pool=scene_pool))

    print(f"total number of samples = {len(data2save)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data2save:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(len(data2save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        nargs="+",
        default=["vln", "idm", "ffs"],
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data/R2R/annotations_v1-3.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="data/images/r2r",
    )
    parser.add_argument("--output_path", type=str, default='/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data/R2R/annotations_vlnidsffs.json')
    args = parser.parse_args()

    main(args.annotation_path, args.image_root, args.task_type, args.output_path)