import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
from tqdm import tqdm
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
import argparse, habitat
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image, ImageFont, ImageDraw
import multiprocessing as mp
import time, math
from openai import OpenAI
import base64
from io import BytesIO


SYSTEM_PROMPT = "You are a visual language navigation model."

FIRST_TURN_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given an image of the current observation<image>. "
    "Your assigned task is: '{}'. "
    "Analyze this image to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
)

FOLLOWUP_TURN_TEXT = "<image>"

NUM_ACTIONS_PER_TURN = 3
NUM_ACTIONS_EXECUTE = 3

# SFT used LLaMA-Factory's qwen3_vl ReasoningTemplate, which prepends
# "<think>\n\n</think>\n\n" to every assistant response (with loss). At
# inference the model therefore emits an empty think block on turn 0; from
# turn 1 onward it stops because the recorded history has none. Strip it
# unconditionally so behaviour is identical across turns.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def encode_image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def seed_all():
    np.random.seed(41)
    random.seed(41)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_done_result_keys(result_path_or_file):
    if result_path_or_file.endswith(".json"):
        result_file = result_path_or_file
    else:
        result_file = os.path.join(result_path_or_file, "result.json")
    done = set()
    if not os.path.exists(result_file):
        return done
    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scene_id" in item and "episode_id" in item and "episode_instruction" in item:
                done.add((item["scene_id"], str(item["episode_id"]), item["episode_instruction"]))
    return done


def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path, num_generations,
                   forward_distance, turn_angle, max_action_history, resolution_ratio,
                   enable_early_stop, skip_result_file, save_sharegpt, save_video) -> None:

    env = Env(config.habitat, dataset)

    agent = MultiTurn_Agent(
        api_key,
        base_url,
        result_path,
        forward_distance,
        turn_angle,
        max_action_history,
        resolution_ratio,
        num_generations,
        require_map=save_video,
        save_sharegpt=save_sharegpt,
    )

    num_episodes = len(env.episodes)
    done_result_keys = load_done_result_keys(result_path)
    if skip_result_file:
        done_result_keys.update(load_done_result_keys(skip_result_file))

    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    for _ in range(num_episodes):
        episode_start_time = time.time()

        obs = env.reset()
        iter_step = 0
        agent.reset()

        t_dict = {
            "t_episode": 0,
        }

        continuse_rotation_count = 0
        last_dtg = 999
        scene_id = env.current_episode.scene_id.split('/')[-2]
        episode_id = env.current_episode.episode_id
        episode_instruction = obs["instruction"]["text"]
        if (scene_id, str(episode_id), episode_instruction) in done_result_keys:
            t_dict["t_episode"] = time.time() - episode_start_time
            t_dict["skipped"] = 1
            result_queue.put(t_dict)
            continue
        while not env.episode_over:

            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id)

            if enable_early_stop and (continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS):
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)

        info = env.get_metrics()

        result = {
            "scene_id": scene_id,
            "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
            "trial_id": 0,
            "trial_total": 1,
            "success": info["success"],
            "spl": info["spl"],
            "os": info["oracle_success"],
            "ne": info["distance_to_goal"],
            "steps": iter_step,
            "episode_instruction": episode_instruction
        }
        if save_sharegpt:
            agent.dump_sharegpt(
                scene_id=scene_id,
                episode_id=episode_id,
                steps=iter_step,
                trial_id=0,
            )
        with open(os.path.join(result_path, "result.json"), "a") as f:
            f.write(json.dumps(result) + "\n")
        done_result_keys.add((scene_id, str(episode_id), episode_instruction))

        t_dict["t_episode"] = time.time() - episode_start_time
        t_dict["success"] = float(result["success"])
        t_dict["spl"] = result["spl"]
        t_dict["os"] = result["os"]
        t_dict["ne"] = result["ne"]
        t_dict["steps"] = result["steps"]
        result_queue.put(t_dict)


class MultiTurn_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance,
                 turn_angle, max_action_history, resolution_ratio, num_generations=1,
                 require_map=False, save_sharegpt=False):

        print("Initialize MultiTurn Agent")

        self.result_path = result_path
        self.require_map = require_map
        self.save_sharegpt = save_sharegpt
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        os.makedirs(self.result_path, exist_ok=True)
        if self.require_map:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = os.environ.get("OPENAI_MODEL") or self.client.models.list().data[0].id

        self.temperature = 0.3
        self.top_p = 0.95
        self.top_k = 20
        self.max_tokens = 512

        # atomic action ids: 0-stop, 1-forward, 2-left, 3-right
        self.idx2action_text = {
            0: "stop",
            1: "forward {} cm",
            2: "turn left {} degree",
            3: "turn right {} degree",
        }

        self.history_rgb_tensor = None
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.sharegpt_conversations = []
        self.is_first_round = True

        self.reset()

    def predict_inference(self):
        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k,
                # Inject empty <think></think> into the prompt suffix so the
                # model doesn't waste tokens (or risk truncation) emitting it.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        output_text = outputs.choices[0].message.content
        output_text = _THINK_BLOCK_RE.sub("", output_text, count=1)
        output_text = output_text.strip()
        return output_text

    def extract_multi_result(self, output):
        """Parse one assistant turn into a list of (atomic_id, magnitude) compound actions."""
        output_match = re.search(r'<answer>(.*?)</answer>', output, flags=re.DOTALL)
        output = output_match.group(1).strip() if output_match else output.strip()
        sub_actions = [s for s in output.split(',') if s.strip()]
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            if action_index is None:
                continue
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        """Map a single sub-action string to (id, magnitude). id: 0-stop, 1-forward, 2-left, 3-right."""
        output = output.lower().strip()
        if "stop" in output:
            return 0, None
        if "forward" in output:
            match = re.search(r'-?\d+', output)
            numeric = float(match.group()) if match else float(self.forward_distance)
            return 1, numeric
        if "left" in output:
            match = re.search(r'-?\d+', output)
            numeric = float(match.group()) if match else float(self.turn_angle)
            return 2, numeric
        if "right" in output:
            match = re.search(r'-?\d+', output)
            numeric = float(match.group()) if match else float(self.turn_angle)
            return 3, numeric
        return None, None

    def expand_to_atomic(self, compound_actions):
        """Expand 3 compound actions into atomic 25cm / 15deg steps for habitat."""
        atomic = []
        for action_index, numeric in compound_actions:
            if action_index == 0:
                atomic.append(0)
                break
            elif action_index == 1:
                steps = max(1, min(3, round(numeric / self.forward_distance)))
                atomic.extend([1] * steps)
            elif action_index == 2:
                steps = max(1, min(3, round(numeric / self.turn_angle)))
                atomic.extend([2] * steps)
            elif action_index == 3:
                steps = max(1, min(3, round(numeric / self.turn_angle)))
                atomic.extend([3] * steps)
        return atomic

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self, action_id):
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")

    def reset(self):
        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(self.result_path, "video", "{}.gif".format(self.episode_id))
                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })
        self.sharegpt_conversations = []
        self.is_first_round = True

    def dump_sharegpt(self, scene_id, episode_id, steps, trial_id=0):
        output_id = f"{scene_id}_{episode_id}_trial{trial_id}"
        output_dir = os.path.join(self.result_path, "sharegpt", output_id)
        os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "id": output_id,
            "system": SYSTEM_PROMPT,
            "conversations": self.sharegpt_conversations,
            "metrics": {
                "scene_id": scene_id,
                "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                "trial_id": trial_id,
                "steps": steps,
            },
        }

        with open(os.path.join(output_dir, "sharegpt.json"), "w") as f:
            json.dump(output_data, f, indent=2)

    def _build_first_turn_content(self, instruction, image_b64):
        """First user turn: instruction + embedded <image>. Send the literal <image> token as the training prompt did, then attach the actual image_url so the model sees it."""
        text = FIRST_TURN_TEMPLATE.format(instruction)
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]

    def _build_followup_content(self, image_b64):
        """Follow-up turns: just the <image> token + the new image."""
        return [
            {"type": "text", "text": FOLLOWUP_TURN_TEXT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]

    def act(self, observations, info, episode_id):
        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        self.rgb_list.append(rgb_)
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        # Drain pending atomic actions before issuing a new model call.
        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        image_b64 = encode_image_base64(self.rgb_list[-1])
        if self.is_first_round:
            content = self._build_first_turn_content(observations["instruction"]["text"], image_b64)
            user_text_for_sharegpt = FIRST_TURN_TEMPLATE.format(observations["instruction"]["text"])
        else:
            content = self._build_followup_content(image_b64)
            user_text_for_sharegpt = FOLLOWUP_TURN_TEXT

        self.conversations.append({
            "role": "user",
            "content": content,
        })

        navigation = self.predict_inference()

        compound = self.extract_multi_result(navigation)
        # Model outputs 3 compound actions; execute all 3 and record all 3 in
        # history (matches the SFT distribution where every assistant turn has 3).
        compound = compound[:NUM_ACTIONS_PER_TURN]
        if len(compound) == 0:
            print('random select an action')
            fallback_id = random.randint(1, 3)
            mag = self.forward_distance if fallback_id == 1 else self.turn_angle
            compound = [[fallback_id, mag]]

        executed = compound[:NUM_ACTIONS_EXECUTE]
        atomic_seq = self.expand_to_atomic(executed)
        if len(atomic_seq) == 0:
            atomic_seq = [0]

        # Re-render the assistant text in the canonical training format,
        # logging only the compound actions we actually execute.
        assistant_parts = []
        for action_index, numeric in executed:
            if action_index == 0:
                assistant_parts.append("stop")
            elif action_index == 1:
                assistant_parts.append(self.idx2action_text[1].format(int(numeric)))
            elif action_index == 2:
                assistant_parts.append(self.idx2action_text[2].format(int(numeric)))
            elif action_index == 3:
                assistant_parts.append(self.idx2action_text[3].format(int(numeric)))
        assistant_text = ", ".join(assistant_parts)

        self.conversations.append({
            "role": "assistant",
            "content": assistant_text,
        })

        if self.save_sharegpt:
            self.sharegpt_conversations.append({
                "from": "human",
                "value": user_text_for_sharegpt,
            })
            self.sharegpt_conversations.append({
                "from": "gpt",
                "value": assistant_text,
            })
        self.is_first_round = False

        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], assistant_text)
            self.topdown_map_list.append(img)

        self.pending_action_list.extend(atomic_seq)
        if len(self.pending_action_list) == 0:
            self.pending_action_list.append(0)

        return {"action": self.pending_action_list.pop(0)}


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config", type=str, required=True, help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num", type=int, required=True, help="chunks of evluation")
    parser.add_argument("--resolution-ratio", type=float, help="image resize ratio", default=1.0)
    parser.add_argument("--result-path", type=str, required=True, help="location to save results")
    parser.add_argument("--forward-distance", type=int, help="distance that one forward action takes", default=25)
    parser.add_argument("--turn-angle", type=int, help="angle that one turn action takes", default=15)
    parser.add_argument("--max-action-history", type=int, help="the maximum num of action history", default=10)
    parser.add_argument("--num-generations", type=int, help="whether use video or multi image", default=1)
    parser.add_argument("--enable-early-stop", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable heuristic early stop based on repeated no-progress rotation or max steps.")
    parser.add_argument("--skip-result-file", type=str, default=None,
                        help="Optional extra result json/jsonl(or dir) to skip already-finished episodes.")
    parser.add_argument("--save-sharegpt", action="store_true", default=False,
                        help="Save ShareGPT json under result-path/sharegpt/")
    parser.add_argument("--save-video", action="store_true", default=False,
                        help="Save per-episode top-down GIF under result-path/video/")
    args = parser.parse_args()
    if args.skip_result_file is None:
        default_skip_file = os.path.join(args.result_path, "result.json")
        if os.path.exists(default_skip_file):
            args.skip_result_file = default_skip_file

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)

    num_episodes = len(dataset.episodes)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (result_queue, api_key, base_url, config, dataset_splits[i], args.result_path,
                       args.num_generations, args.forward_distance, args.turn_angle,
                       args.max_action_history, args.resolution_ratio, args.enable_early_stop,
                       args.skip_result_file, args.save_sharegpt, args.save_video)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)
    for p in processes:
        p.join()

    result_file = os.path.join(args.result_path, "result.json")
    n_run, s_suc, s_spl, s_os, s_ne, s_step = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not all(k in item for k in ("scene_id", "episode_id", "episode_instruction", "success", "spl", "os", "ne", "steps")):
                    continue
                n_run += 1
                s_suc += item["success"]
                s_spl += item["spl"]
                s_os += item["os"]
                s_ne += item["ne"]
                s_step += item["steps"]
    if n_run:
        summary = {
            "sucs_all": s_suc / n_run,
            "spls_all": s_spl / n_run,
            "oss_all": s_os / n_run,
            "ones_all": s_ne / n_run,
            "avg_step": s_step / n_run,
        }
    else:
        summary = {k: None for k in ("sucs_all", "spls_all", "oss_all", "ones_all", "avg_step")}
    print(json.dumps(summary, ensure_ascii=False))
    with open(result_file, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()