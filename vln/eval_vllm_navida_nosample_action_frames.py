"""
eval_jsonl_navida.py

评测脚本，匹配 jsonl 训练格式的帧累积策略：
  - 每次 VLM 输出的 3 个大动作只取前 2 个执行
  - 每个大动作拆分为若干微步骤：
      forward X cm  → round(X / forward_distance) 步, 最多 3 步
      turn   Y deg  → round(Y / turn_angle)        步, 最多 3 步
  - 只在大动作的 **最后一个微步骤完成后** 捕获帧，将该帧加入关键帧历史
  - 下一轮 VLM 调用时使用 [历史关键帧(采样到8帧)] + [当前关键帧] 作为输入，
    与 jsonl 训练数据格式保持一致
"""

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
from PIL import Image
import multiprocessing as mp
import time
from openai import OpenAI
import base64
from io import BytesIO


# 与训练 jsonl 保持一致的系统提示
SYSTEM_PROMPT = "You are a visual language navigation model."


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

    agent = NaVIDA_JsonL_Agent(
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


class NaVIDA_JsonL_Agent(Agent):
    """
    与 jsonl 训练格式对齐的导航智能体。

    帧累积策略（与训练数据生成逻辑一致）：
      key_frame_list[:-1]  → 历史帧（VLM 输入中的 "historical observations"）
      key_frame_list[-1]   → 当前帧（VLM 输入中的 "current observation"）

      何时向 key_frame_list 追加帧：
        1. 首次调用 act()：将初始观测作为起始关键帧
        2. 每当一个大动作的最后一个微步骤执行完毕后：
           将该时刻的观测追加为新的关键帧

    动作选取：
      - VLM 输出 3 个大动作，只取前 2 个执行（与 eval_vllm_navida.py 保持一致）
      - 每个大动作拆分为微步骤时，记录哪一步是该大动作的最后一步
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
        print("Initialize NaVIDA_JsonL_Agent")

        self.result_path = result_path
        self.save_video = save_video
        self.forward_distance = forward_distance   # 一个 forward 微步骤的距离，默认 25 cm
        self.turn_angle = turn_angle               # 一个 turn 微步骤的角度，默认 15°
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

        # 与训练 jsonl 中用户消息格式完全一致
        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        )

        self.episode_id = None
        self.reset()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def uniform_sample_with_ends(self, data: list, n: int) -> list:
        """均匀采样，保留首尾元素，用于限制历史帧数量。"""
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

    def extract_multi_result(self, output: str) -> list:
        """将 VLM 输出解析为大动作列表，格式: [[action_id, numeric], ...]"""
        sub_actions = output.split(", ")
        return [list(self.extract_result(s)) for s in sub_actions]

    def extract_result(self, output: str):
        """解析单个大动作字符串，返回 (action_id, numeric)。
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

    def decompose_macro_action(self, action_index, numeric) -> list:
        """
        将一个大动作拆分为微步骤列表：[(micro_action_id, is_last_in_macro), ...]

        例如：
          forward 75 cm (forward_distance=25) → 3 步 → [(1,F),(1,F),(1,T)]
          turn left 30° (turn_angle=15)        → 2 步 → [(2,F),(2,T)]
          stop                                  → [(0,T)]
        """
        if action_index == 0:
            return [(0, True)]

        if action_index == 1:  # forward
            n = min(3, max(1, round(numeric / self.forward_distance))) if numeric else 1
            return [(1, i == n - 1) for i in range(n)]

        if action_index == 2:  # turn left
            n = min(3, max(1, round(numeric / self.turn_angle))) if numeric else 1
            return [(2, i == n - 1) for i in range(n)]

        if action_index == 3:  # turn right
            n = min(3, max(1, round(numeric / self.turn_angle))) if numeric else 1
            return [(3, i == n - 1) for i in range(n)]

        return [(1, True)]  # fallback: 前进一步

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
        """每个 episode 开始前调用，清空所有状态。"""
        if self.save_video and hasattr(self, "topdown_map_list") and self.topdown_map_list:
            gif_path = os.path.join(self.result_path, "video", f"{self.episode_id}.gif")
            imageio.mimsave(gif_path, self.topdown_map_list)

        self.topdown_map_list = []

        # 关键帧列表：只存储每个大动作最后微步骤完成时刻的帧
        # key_frame_list[-1]  = 当前帧 (current observation)
        # key_frame_list[:-1] = 历史帧 (historical observations)
        self.key_frame_list: list = []

        # 待执行的微步骤队列：每个元素为 (action_id, is_last_in_macro)
        self.pending_micro_steps: list = []

        # 上一个被弹出执行的微步骤是否为某大动作的最后一步
        self.last_was_macro_end: bool = False

        self.conversations = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]

    def act(self, observations, info, episode_id) -> dict:
        self.episode_id = episode_id

        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype("uint8")).convert("RGB")

        if self.save_video:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        # ----------------------------------------------------------------
        # 关键帧捕获逻辑（与训练 jsonl 数据生成方式对齐）：
        #   情况1：key_frame_list 为空 → 首次调用，将初始观测设为起始关键帧
        #   情况2：上一步是某大动作的最后一个微步骤 → 此时观测即为该大动作
        #          执行完毕时刻的帧，追加到 key_frame_list
        # ----------------------------------------------------------------
        if not self.key_frame_list:
            # 初始帧：尚未执行任何动作时的观测
            self.key_frame_list.append(rgb_)
        elif self.last_was_macro_end:
            # 大动作完成时刻的帧 → 追加为新关键帧
            self.key_frame_list.append(rgb_)
            if len(self.key_frame_list) > self.max_action_history:
                self.key_frame_list = self.key_frame_list[1:]

        # ----------------------------------------------------------------
        # 若仍有待执行的微步骤，直接弹出执行，不调用 VLM
        # ----------------------------------------------------------------
        if self.pending_micro_steps:
            micro_action, is_macro_end = self.pending_micro_steps.pop(0)
            self.last_was_macro_end = is_macro_end

            if self.save_video:
                label = self.action_id_to_str(micro_action)
                if is_macro_end:
                    label += " [macro-end★]"
                img = self.addtext(output_im, observations["instruction"]["text"],
                                   f"Pending: {label}")
                self.topdown_map_list.append(img)

            return {"action": micro_action}

        # ----------------------------------------------------------------
        # 待执行队列已空 → 调用 VLM 获取下一批大动作
        # 输入格式与训练 jsonl 完全一致：
        #   "historical observations" → key_frame_list[:-1] (均匀采样至8帧)
        #   "current observation"     → key_frame_list[-1]
        # ----------------------------------------------------------------
        self.conversations = self.conversations[:1]  # 只保留 system prompt
        content = []

        history_frames = self.key_frame_list[:-1]
        current_frame = self.key_frame_list[-1]

        content.append({
            "type": "text",
            "text": "Imagine you are a robot programmed for navigation tasks. "
                    "You have been given a video of historical observations",
        })

        if history_frames:
            sampled_history = self.uniform_sample_with_ends(history_frames, 8)
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(f)}"},
                }
                for f in sampled_history
            ])
        else:
            # 尚无历史帧时，将当前帧同时充当历史，与 eval_vllm_navida.py 保持一致
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(current_frame)}"},
            })

        content.append({"type": "text", "text": "and an image of the current observation"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(current_frame)}"},
        })

        # prompt 中 "current observation" 之后的部分，包含任务指令
        suffix = self.prompt_template.format(
            observations["instruction"]["text"]
        ).split("current observation")[1]
        content.append({"type": "text", "text": suffix})

        self.conversations.append({"role": "user", "content": content})

        navigation = self.predict_inference()

        if self.save_video:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)

        # ----------------------------------------------------------------
        # 解析 VLM 输出：取前 2 个大动作（与训练数据对齐），拆分为微步骤
        # ----------------------------------------------------------------
        macro_actions = self.extract_multi_result(navigation)
        macro_actions = macro_actions[:2]  # 只取前 2 个大动作

        for action_index, numeric in macro_actions:
            if action_index is None:
                continue
            micro_steps = self.decompose_macro_action(action_index, numeric)
            self.pending_micro_steps.extend(micro_steps)

        # 若解析失败，随机选一个动作作为兜底
        if not self.pending_micro_steps:
            print("Warning: failed to parse VLM output, random fallback")
            rand_action = random.randint(1, 3)
            self.pending_micro_steps.append((rand_action, True))

        micro_action, is_macro_end = self.pending_micro_steps.pop(0)
        self.last_was_macro_end = is_macro_end
        return {"action": micro_action}


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------

def main():
    seed_all()
    parser = argparse.ArgumentParser(
        description="Evaluate VLM navigation agent using jsonl training-format frame accumulation."
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
                        help="One forward micro-step distance in cm (default: 25)")
    parser.add_argument("--turn-angle", type=int, default=15,
                        help="One turn micro-step angle in degrees (default: 15)")
    parser.add_argument("--max-action-history", type=int, default=10,
                        help="Max number of key frames to keep in history")
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
    # 汇总结果（与 eval_vllm_navida.py 相同）
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
            "pass_k": pass_k_value, pass_at_k_key: None,
        }
        if pass_k_value > 1:
            summary["avg_step"] = None
        else:
            summary["avg_step"] = None

    print(json.dumps(summary, ensure_ascii=False))
    with open(result_file, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
