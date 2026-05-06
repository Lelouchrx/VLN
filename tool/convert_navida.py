import json
import os
import random
import re
import copy
from tqdm import tqdm
import argparse


forward_distance = 25
turn_angle = 15


def noise_action(action_id):
    p = random.random()
    if p < 0.7 and action_id != 0:
        return random.randint(1, 3)
    else:
        return action_id


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


def image_sort_key(x):
    matched = re.findall(r"\d+", x)
    if matched:
        return int(matched[-1])
    return x


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
    episode_image_list = sorted(episode_image_list, key=image_sort_key)
    return [episode_image_path + '/' + image for image in episode_image_list]


def sample_vln_images(image_list):
    if len(image_list) <= 1:
        return image_list
    history = image_list[:-1]
    if len(history) > 8:
        history = uniform_sample_with_ends(history, 8)
    return history + [image_list[-1]]


def attach_llamafactory_mm(example):
    user = example["messages"][0]
    imgs = user.get("image", [])
    if not isinstance(imgs, list):
        imgs = [imgs] if imgs else []
    example["images"] = imgs[:]
    user["value"] = "<image>" * len(imgs) + user["value"]
    if "image" in user:
        del user["image"]
    return example


def process_single_type(annotation, image_root, system_prompt, prompt_template, task_type):
    data2save = []
    for episode_item in tqdm(annotation):
        episode_id = get_first(episode_item, ['episode_id', 'path_id', 'id'], '')
        actions = get_first(episode_item, ['actions', 'action', 'action_ids', 'nav_actions'], [])
        if not isinstance(actions, list) or len(actions) == 0:
            continue

        action_texts = [to_action_text(a) for a in actions]
        if len(action_texts) > 0 and action_texts[-1] == "stop":
            action_texts = action_texts[:-1]
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
                "action_history": [],
                "episode_id": str(episode_id),
                "task type": task_type,
            }
            if task_type == "vln":
                formated_instruction = prompt_template.format(instruction)
                tmp_data["messages"].append({"from": "user", "value": formated_instruction, "image": [episode_image_list[0]]})
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
                            data2save.append(attach_llamafactory_mm(out_item))
                            tmp_data["action_history"].append(tmp_data["messages"][1]['value'])
                            while len(pending_rgb_list) != 0:
                                item = pending_rgb_list.pop(0)
                                tmp_data["messages"][0]['image'].append(item)
                            tmp_data["messages"][0]['image'].append(episode_image_list[i])
                            tmp_data["messages"][1]['value'] = action_texts[i]
                    last_action = action_texts[i]
                out_item = copy.deepcopy(tmp_data)
                out_item["messages"][0]["image"] = sample_vln_images(out_item["messages"][0]["image"])
                data2save.append(attach_llamafactory_mm(out_item))
            elif task_type == "idm":
                formated_instruction = prompt_template
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
                            tmp_data["action_history"].append(tmp_data["messages"][1]['value'])
                            tmp_data["messages"][0]['image'] = [episode_image_list[i], episode_image_list[i + 1]]
                            tmp_data["messages"][1]['value'] = action_texts[i]
                    last_action = action_texts[i]
                if tmp_data["messages"][1]['value'] != 'stop':
                    data2save.append(attach_llamafactory_mm(copy.deepcopy(tmp_data)))
    return data2save


def main(annotation_path, image_root, task_type_list, output_path):
    system_prompt = "You are a helpful assistant."
    vln_prompt_template = "Imagine you are a robot programmed for navigation tasks. " \
        "You have been given a video of historical observations and an image of the current observation. " \
        "Your assigned task is: '{}'. Analyze this series of images to decide your next move, " \
        "which could involve turning left or right by a specific degree or moving forward a certain distance."

    idm_prompt_template = "Imagine you are a robot programmed for navigation tasks. " \
        "You have been given an image of current view and an image of the goal view. " \
        "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, " \
        "which could involve turning left " \
        "or right by a specific degree or moving forward a certain distance."

    data2save = []
    annotation = get_json_items(annotation_path)

    for task_type in task_type_list:
        if task_type == "vln":
            prompt_template = vln_prompt_template
        elif task_type == "idm":
            prompt_template = idm_prompt_template
        else:
            continue
        data2save.extend(process_single_type(annotation, image_root, system_prompt, prompt_template, task_type))

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
        default=["vln", "idm"],
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/media/mldadmin/home/s125mdg38_06/StreamVLN/data/trajectory_data/R2R/annotations_v1-3.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="data/images/r2r",
    )
    parser.add_argument("--output_path", type=str, default='data/navida_train_data.jsonl')
    args = parser.parse_args()

    main(args.annotation_path, args.image_root, args.task_type, args.output_path)