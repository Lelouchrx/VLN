"""
eval_vllm_navida_nocombine.py

与 annotations_vln_nocombine.jsonl 训练格式对齐的评测脚本：
  - 帧累积策略：每步都追加到 rgb_list（与 eval_vllm_navida.py 完全相同）
  - VLM 输入：uniform_sample_with_ends(rgb_list[:-1], 8) 历史帧 + rgb_list[-1] 当前帧
  - VLM 输出：单个原子动作（forward 25 cm / turn left 15 degrees / stop）
               模型不输出大动作，无需拆分，直接执行
"""

import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import tqdm
import os
import re
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
from PIL import Image
import multiprocessing as mp
import time
from openai import OpenAI
import base64
from io import BytesIO


SYSTEM_PROMPT = "You are a helpful assistant."


def encode_image_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def seed_all():
    np.random.seed(41)
    random.seed(41)


def load_done_result_keys(result_path: str) -> set:
    result_file = os.path.join(result_path, "result.json")
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
                trial_id = int(item.get("trial_id", 0))
                trial_total = int(item.get("trial_total", 1))
                done.add((
                    item["scene_id"],
                    str(item["episode_id"]),
                    item["episode_instruction"],
                    trial_id,
                    trial_total,
                ))
    return done


def evaluate_agent(
    result_queue, api_key, base_url, config, dataset, result_path,
    num_generations, forward_distance, turn_angle, max_action_history,
    resolution_ratio, save_video, pass_k
) -> None:

    original_episodes = list(dataset.episodes)

    agent = NaVIDA_NoCombine_Agent(
        api_key=api_key,
        base_url=base_url,
        result_path=result_path,
        forward_distance=forward_distance,
        turn_angle=turn_angle,
        max_action_history=max_action_history,
        resolution_ratio=resolution_ratio,
        num_generations=num_generations,
        save_video=save_video,
    )

    num_episodes = len(original_episodes)
    done_result_keys = load_done_result_keys(result_path)

    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    for trial_id in range(pass_k):
        dataset.episodes = list(original_episodes)
        env = Env(config.habitat, dataset)
        for _ in range(num_episodes):
            episode_start_time = time.time()

            obs = env.reset()
            iter_step = 0
            agent.reset()

            t_dict = {"t_episode": 0}

            continuse_rotation_count = 0
            last_dtg = 999
            scene_id = env.current_episode.scene_id.split("/")[-2]
            episode_id = env.current_episode.episode_id
            episode_instruction = obs["instruction"]["text"]

            if (scene_id, str(episode_id), episode_instruction, trial_id, pass_k) in done_result_keys:
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

                if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                    action = {"action": 0}

                iter_step += 1
                obs = env.step(action)

            info = env.get_metrics()
            result = {
                "scene_id": scene_id,
                "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                "trial_id": trial_id,
                "trial_total": pass_k,
                "success": info["success"],
                "spl": info["spl"],
                "os": info["oracle_success"],
                "ne": info["distance_to_goal"],
                "steps": iter_step,
                "episode_instruction": episode_instruction,
            }
            with open(os.path.join(result_path, "result.json"), "a") as f:
                f.write(json.dumps(result) + "\n")
            done_result_keys.add((scene_id, str(episode_id), episode_instruction, trial_id, pass_k))

            t_dict["t_episode"] = time.time() - episode_start_time
            result_queue.put(t_dict)
        env.close()


class NaVIDA_NoCombine_Agent(Agent):
    """
    与 annotations_vln_nocombine.jsonl 对齐的评测 Agent。

    帧累积策略（与 eval_vllm_navida.py 完全相同）：
      - 每一步（无论是否调用 VLM）都将当前 rgb 追加到 rgb_list
      - 历史帧：uniform_sample_with_ends(rgb_list[:-1], 8)
      - 当前帧：rgb_list[-1]

    动作策略（与 nocombine 训练格式对齐）：
      - 每次 VLM 调用输出单个原子动作：
          "forward 25 cm"    → action 1（前进一步）
          "turn left 15 degrees"  → action 2（左转一步）
          "turn right 15 degrees" → action 3（右转一步）
          "stop"             → action 0
      - 不做大动作拆分，直接返回对应 Habitat 动作 id
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        result_path: str,
        forward_distance: int,
        turn_angle: int,
        max_action_history: int,
        resolution_ratio: float,
        num_generations: int = 1,
        save_video: bool = False,
    ):
        print("Initialize NaVIDA_NoCombine_Agent")

        self.result_path = result_path
        self.save_video = save_video
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations

        os.makedirs(self.result_path, exist_ok=True)
        if self.save_video:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.client.models.list().data[0].id

        self.temperature = 0.3
        self.top_p = 0.95
        self.max_tokens = 512

        # 与 eval_vllm_navida.py 保持相同 prompt 模板
        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        )

        self.episode_id = None
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]

        self.reset()

    # ------------------------------------------------------------------
    # 工具方法（与 eval_vllm_navida.py 完全相同）
    # ------------------------------------------------------------------

    def uniform_sample_with_ends(self, data: list, n: int) -> list:
        if len(data) <= n:
            return data
        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def predict_inference(self) -> str:
        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs.choices[0].message.content.strip()

    def extract_result(self, output: str):
        """解析单个原子动作字符串，返回 (action_id, numeric)。
        action_id: 0-stop, 1-forward, 2-turn left, 3-turn right
        """
        m = re.search(r"<answer>(.*?)</answer>", output)
        output = m.group(1).strip() if m else output.strip()
        output = output.lower()

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            m = re.search(r"-?\d+", output)
            return 1, float(m.group()) if m else float(self.forward_distance)
        elif "left" in output:
            m = re.search(r"-?\d+", output)
            return 2, float(m.group()) if m else float(self.turn_angle)
        elif "right" in output:
            m = re.search(r"-?\d+", output)
            return 3, float(m.group()) if m else float(self.turn_angle)
        return None, None

    def action_id_to_str(self, action_id: int) -> str:
        return {0: "stop", 1: "forward", 2: "turn left", 3: "turn right"}.get(action_id, "unknown")

    def addtext(self, image, instruction: str, navigation: str):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2
        y_line = textY
        x = 10
        line = ""
        for word in instruction.split(" "):
            test_line = (line + " " + word).lstrip()
            ts, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if ts[0] > w - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line
        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line += textsize[1] + 10
        cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self):
        if self.save_video and self.topdown_map_list:
            gif_path = os.path.join(self.result_path, "video", f"{self.episode_id}.gif")
            imageio.mimsave(gif_path, self.topdown_map_list)

        self.topdown_map_list = []
        self.rgb_list = []

        self.conversations = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]

    def act(self, observations, info, episode_id) -> dict:
        self.episode_id = episode_id

        # ----------------------------------------------------------------
        # 1. 帧累积：每步都追加到 rgb_list（与 eval_vllm_navida.py 相同）
        # ----------------------------------------------------------------
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype("uint8")).convert("RGB")
        self.rgb_list.append(rgb_)
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.save_video:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        # ----------------------------------------------------------------
        # 2. 构建 VLM 输入（与 eval_vllm_navida.py 完全相同的帧采样逻辑）
        # ----------------------------------------------------------------
        self.conversations = self.conversations[:1]  # 只保留 system prompt
        content = []

        content.append({
            "type": "text",
            "text": "Imagine you are a robot programmed for navigation tasks. "
                    "You have been given a video of historical observations",
        })
        if len(self.rgb_list) > 1:
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"},
                }
                for item in self.uniform_sample_with_ends(self.rgb_list[:-1], 8)
            ])
        else:
            # 第一步尚无历史，用当前帧充当历史占位（与原版相同）
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"},
            })
        content.append({"type": "text", "text": "and an image of the current observation"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"},
        })
        suffix = self.promt_template.format(
            observations["instruction"]["text"]
        ).split("current observation")[1]
        content.append({"type": "text", "text": suffix})

        self.conversations.append({"role": "user", "content": content})

        # ----------------------------------------------------------------
        # 3. VLM 推理，获取单个原子动作
        # ----------------------------------------------------------------
        navigation = self.predict_inference()

        if self.save_video:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)

        # ----------------------------------------------------------------
        # 4. 解析原子动作，直接返回（不拆分，不积累 pending list）
        # ----------------------------------------------------------------
        action_index, _ = self.extract_result(navigation)

        if action_index is None:
            print("Warning: failed to parse VLM output, random fallback")
            action_index = random.randint(1, 3)

        return {"action": action_index}


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------

def main():
    seed_all()
    parser = argparse.ArgumentParser(
        description="Evaluate VLM agent aligned with nocombine (atomic action) training format."
    )
    parser.add_argument("--exp-config", type=str, required=True,
                        help="Path to Habitat config yaml")
    parser.add_argument("--split-num", type=int, required=True,
                        help="Number of parallel evaluation workers")
    parser.add_argument("--resolution-ratio", type=float, default=1.0,
                        help="Image resolution downscale ratio")
    parser.add_argument("--result-path", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--forward-distance", type=int, default=25,
                        help="Forward step distance in cm (default: 25)")
    parser.add_argument("--turn-angle", type=int, default=15,
                        help="Turn step angle in degrees (default: 15)")
    parser.add_argument("--max-action-history", type=int, default=10,
                        help="Max frames to keep in rgb_list history")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--pass-k", type=int, default=1,
                        help="Number of trials per episode for pass@k")
    parser.add_argument("--save_vedio", action="store_true",
                        help="Save per-episode top-down GIFs under result-path/video/")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None, \
        "Must set OPENAI_API_KEY and OPENAI_API_BASE environment variables"

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)
    num_episodes = len(dataset.episodes)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (
            result_queue, api_key, base_url, config, dataset_splits[i],
            args.result_path, args.num_generations, args.forward_distance,
            args.turn_angle, args.max_action_history, args.resolution_ratio,
            args.save_vedio, args.pass_k,
        )
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes * args.pass_k, desc="Evaluating") as pbar:
        for _ in range(num_episodes * args.pass_k):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)

    for p in processes:
        p.join()

    # ------------------------------------------------------------------
    # 汇总结果
    # ------------------------------------------------------------------
    result_file = os.path.join(args.result_path, "result.json")
    n_run, s_suc, s_spl, s_os, s_ne, s_step = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    traj_results = {}

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
                if not all(k in item for k in (
                    "scene_id", "episode_id", "episode_instruction",
                    "success", "spl", "os", "ne", "steps",
                )):
                    continue
                n_run += 1
                success = float(item["success"])
                spl = float(item["spl"])
                os_val = float(item["os"])
                ne = float(item["ne"])
                s_suc += success
                s_spl += spl
                s_os += os_val
                s_ne += ne
                s_step += float(item["steps"])
                traj_key = (item["scene_id"], str(item["episode_id"]), item["episode_instruction"])
                if traj_key not in traj_results:
                    traj_results[traj_key] = {
                        "pass_success": success,
                        "pass_spl": spl,
                        "pass_os": os_val,
                        "pass_ne": ne,
                    }
                else:
                    traj_results[traj_key]["pass_success"] = max(traj_results[traj_key]["pass_success"], success)
                    traj_results[traj_key]["pass_spl"] = max(traj_results[traj_key]["pass_spl"], spl)
                    traj_results[traj_key]["pass_os"] = max(traj_results[traj_key]["pass_os"], os_val)
                    traj_results[traj_key]["pass_ne"] = min(traj_results[traj_key]["pass_ne"], ne)

    n_traj = len(traj_results)
    pass_k_value = int(args.pass_k)
    pass_at_k_key = f"pass@{pass_k_value}"

    if n_run and n_traj:
        pass_suc = sum(v["pass_success"] for v in traj_results.values()) / n_traj
        if pass_k_value > 1:
            summary = {
                "sucs_all": s_suc / n_run,
                "spls_all": sum(v["pass_spl"] for v in traj_results.values()) / n_traj,
                "oss_all": sum(v["pass_os"] for v in traj_results.values()) / n_traj,
                "ones_all": sum(v["pass_ne"] for v in traj_results.values()) / n_traj,
                "pass_k": pass_k_value,
                pass_at_k_key: pass_suc,
                f"avg{pass_k_value}": s_suc / n_run,
            }
        else:
            summary = {
                "sucs_all": s_suc / n_run,
                "spls_all": s_spl / n_run,
                "oss_all": s_os / n_run,
                "ones_all": s_ne / n_run,
                "avg_step": s_step / n_run,
                "pass_k": pass_k_value,
                pass_at_k_key: pass_suc,
            }
    else:
        summary = {
            "sucs_all": None, "spls_all": None, "oss_all": None, "ones_all": None,
            "pass_k": pass_k_value, pass_at_k_key: None, "avg_step": None,
        }

    print(json.dumps(summary, ensure_ascii=False))
    with open(result_file, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
