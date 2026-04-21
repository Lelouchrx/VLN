import os
import sys
import torch
import json
import argparse
import transformers
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dist import *
import torch.distributed as dist
from vln.qwenvl.Evaluator import VLNEvaluator

import os
import re
import itertools
import random
import numpy as np
import torch
import tqdm
import copy
import threading
import json
import random
import habitat
import time
import gzip
from PIL import Image
from omegaconf import OmegaConf
from typing import List, Dict
from PIL import Image
import io

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, append_text_underneath_image


from utils.dist import *
from habitat_extensions.maps import image_resize


import base64
from vln.qwen3vln_eval import VLN_Inference_VLLM
from vln.qwenvl.Evaluator import NAV_SYSTEM_PROMPT
DATASET = "rxr"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = "./generated_data"

DEFAULT_EPISODE_LENGTH = 60
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85

class DAggerCollector:
    def __init__(self, args, rank, world_size):
        self.device = torch.device("cuda")
        self.args = args
        self.rank = rank
        self.world_size = world_size
        logging.getLogger("habitat").setLevel(logging.ERROR)
        logging.getLogger("habitat_sim").setLevel(logging.ERROR)

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)
        with read_write(self.config):
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(self.args.local_rank)
        print(OmegaConf.to_yaml(self.config))

        gt_path = self.args.dagger_gt_annotations_path
        gt_open = gzip.open if str(gt_path).endswith('.gz') else open
        with gt_open(gt_path, 'rt', encoding='utf-8') as f:
            gt_data = json.load(f)
        gt_map = {}
        if isinstance(gt_data, dict):
            # Support both {"episode_id": annotation, ...} and wrapped list formats.
            candidate_list = None
            for key in ("annotations", "episodes", "data"):
                if isinstance(gt_data.get(key), list):
                    candidate_list = gt_data[key]
                    break
            if candidate_list is None:
                for epid, item in gt_data.items():
                    if isinstance(item, dict):
                        gt_map[str(epid)] = item
            else:
                for item in candidate_list:
                    if not isinstance(item, dict):
                        continue
                    epid = item.get("episode_id", item.get("id"))
                    if epid is None:
                        continue
                    gt_map[str(epid)] = item
        else:
            for item in gt_data:
                if not isinstance(item, dict):
                    continue
                epid = item.get("episode_id", item.get("id"))
                if epid is None:
                    continue
                gt_map[str(epid)] = item
        self.gt_annotations = gt_map
        
        with read_write(self.config):
            measurements = {}
            if self.args.dagger_save_video:
                measurements.update(
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
            if measurements:
                self.config.habitat.task.measurements.update(measurements)

        self.dagger_config = OmegaConf.create({
            "p": self.args.dagger_p,
            "update_size": self.args.dagger_update_size,
            "commit_freq": self.args.dagger_commit_freq,
        })
        print(self.dagger_config)
        self.llm_executor = ThreadPoolExecutor(
            max_workers=max(1, int(max(self.args.vllm_max_workers, self.args.parallel_envs)))
        )
        self._llm_cond = threading.Condition()
        self._llm_pending = {}
        self._llm_done = {}
        self._llm_next_req_id = 0
        self._llm_shutdown = False
        self._llm_scheduler_thread = threading.Thread(target=self._llm_scheduler_loop, daemon=True)
        self._llm_scheduler_thread.start()
        self.visual_prompts = [
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
            "situated ahead is ",
        ]

    def _llm_scheduler_loop(self):
        while True:
            with self._llm_cond:
                while (not self._llm_shutdown) and (not self._llm_pending):
                    self._llm_cond.wait()
                if self._llm_shutdown and (not self._llm_pending):
                    return
                futures = list(self._llm_pending.keys())
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            with self._llm_cond:
                for fut in done:
                    req_id = self._llm_pending.pop(fut, None)
                    if req_id is None:
                        continue
                    try:
                        self._llm_done[req_id] = (True, fut.result())
                    except Exception as e:
                        self._llm_done[req_id] = (False, e)
                self._llm_cond.notify_all()

    def _request_llm(self, evaluator, messages, llm_state, max_tokens):
        if hasattr(evaluator, "_run_single_vllm"):
            fut = self.llm_executor.submit(
                evaluator._run_single_vllm,
                messages,
                llm_state,
            )
        else:
            fut = self.llm_executor.submit(
                evaluator.model.call_model,
                messages,
                max_new_tokens=max_tokens,
            )
        with self._llm_cond:
            req_id = self._llm_next_req_id
            self._llm_next_req_id += 1
            self._llm_pending[fut] = req_id
            self._llm_cond.notify_all()
            while req_id not in self._llm_done:
                self._llm_cond.wait()
            ok, payload = self._llm_done.pop(req_id)
        if ok:
            return payload
        raise payload

    def config_env(self, scene=None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        return habitat.Env(config=self.config)

    def generate(self, env: habitat.Env, evaluator = None, save_video: bool = True, force_expert: bool = False) -> Dict:
        """
        Generate a trajectory for the given episode.
        """
        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it
        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        instructions = episode.instruction.instruction_text
        ref_path = episode.reference_path

        observation = env.reset()
        annotation_actions = []
        rgb_data_list, rgb_list = [], []
        step_id = 0
        actions = [-1]
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, 'videos'), exist_ok=True)

        mem_ids = []
        vis_frames = []
        left_expert_actions_num = 0
        from_expert = True if force_expert else False
        force_episode_end = False
        model_success = True
        action_seq, action_mask = [], []
        time_ids = []
        metrics = None
        accumulated_error = 0 

        ref_actions_len = len(self.gt_annotations[str(episode_id)]["actions"])
        messages = [{"role": "system", "content": NAV_SYSTEM_PROMPT}]
        messages_started = False

        trace_steps = []
        instr_str = instructions if isinstance(instructions, str) else (instructions[0] if instructions else "")
        while not env.episode_over:
            step_trace_llm = None
            generated_this_round = False
            generated_branch = None
            generated_actions = None
            time_ids.append(step_id)
            rgb = observation["rgb"]

            if self.args.collect_images:
                rgb_path = os.path.join(
                    self.output_path,
                    "images",
                    f"{scene_id}_{self.dataset}_{episode_id:06d}",
                    "rgb",
                    f"{step_id:03d}.jpg",
                )
                rgb_data_list.append((rgb, rgb_path))

            if evaluator is not None:
                image = Image.fromarray(rgb).convert("RGB")
                rgb_list.append(image)

                if len(action_seq) == 0 and left_expert_actions_num == 0:
                    from_expert = True if force_expert else random.random() < beta

                if len(action_seq) == 0:
                    if left_expert_actions_num > 0:
                        action = agent.get_next_action(ref_path[next_waypoint_id])
                        action_seq = [action]
                        left_expert_actions_num -= 1
                        generated_this_round = True
                        generated_branch = "expert"
                        generated_actions = list(action_seq)
                    else:
                        if from_expert:
                            action = agent.get_next_action(ref_path[next_waypoint_id])
                            action_seq = [action]
                            left_expert_actions_num = 0
                            generated_this_round = True
                            generated_branch = "expert"
                            generated_actions = list(action_seq)
                        else:
                            if not messages_started:
                                user_text = f"Your instruction is '{instructions}' "
                                messages_started = True
                            else:
                                user_text = random.choice(self.visual_prompts)
                            img = Image.fromarray(rgb).convert('RGB')
                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            messages.append({'role': 'user', 'content': user_text, 'image_url': f"data:image/png;base64,{b64}"})

                            max_tokens = getattr(self.args, "max_new_tokens", 32)
                            llm_state = {"last_info": metrics or {}}
                            llm_outputs = self._request_llm(
                                evaluator=evaluator,
                                messages=messages,
                                llm_state=llm_state,
                                max_tokens=max_tokens,
                            )
                            llm_text = llm_outputs[0] if isinstance(llm_outputs, list) and len(llm_outputs) > 0 else str(llm_outputs)
                            if self.args.dagger_save_model_trace:
                                step_trace_llm = llm_text
                            action_seq = evaluator.parse_actions(llm_text)
                            if len(action_seq) == 0:
                                action_seq = evaluator.parse_actions(llm_text.upper().replace(" ", ""))
                            if len(action_seq) == 0:
                                digit_actions = re.findall(r"[0-3]", llm_text)
                                action_seq = [int(a) for a in digit_actions]
                            if len(action_seq) == 0:
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": "You should output only a comma-separated action list using exactly these actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop.",
                                    }
                                )
                                action_seq = [0]
                            chat_action_seq = []
                            for a in action_seq:
                                if a == 0:
                                    fallback_action = agent.get_next_action(ref_path[next_waypoint_id])
                                    chat_action_seq.append(fallback_action if fallback_action != 0 else 1)
                                else:
                                    chat_action_seq.append(a)

                            for idx in range(0, len(chat_action_seq), 4):
                                chunk = chat_action_seq[idx:idx + 4]
                                if idx > 0:
                                    chunk_user_text = random.choice(self.visual_prompts)
                                    chunk_img = Image.fromarray(rgb).convert('RGB')
                                    chunk_buf = io.BytesIO()
                                    chunk_img.save(chunk_buf, format='PNG')
                                    chunk_b64 = base64.b64encode(chunk_buf.getvalue()).decode("utf-8")
                                    messages.append({'role': 'user', 'content': chunk_user_text, 'image_url': f"data:image/png;base64,{chunk_b64}"})
                                assistant_text = ", ".join(evaluator.idx2action_text.get(a, "Stop") for a in chunk)
                                messages.append({'role': 'assistant', 'content': assistant_text})
                            generated_this_round = True
                            generated_branch = "model"
                            generated_actions = list(chat_action_seq)

            else:
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_seq = [action]
                generated_this_round = True
                generated_branch = "expert"
                generated_actions = list(action_seq)

            action_source = "expert" if from_expert else "model"

            if len(action_seq) == 0:
                action_seq = [0]

            if self.args.dagger_save_model_trace:
                full_seq_before_pop = list(action_seq)
            else:
                full_seq_before_pop = None
            action = action_seq.pop(0)
            expert_action = agent.get_next_action(ref_path[next_waypoint_id])
            if action != expert_action:
                accumulated_error += 1

            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break

            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)

            accumulated_error_ratio_denom = max(1, int(ref_actions_len / (len(ref_path) - 1)))
            accumulated_error_ratio = accumulated_error / accumulated_error_ratio_denom
            error_not_toleranted = (
                (from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0)
                or (accumulated_error_ratio > 0.8)
                or accumulated_error > 12
            )
            acc_err_before_reset = None

            if wp_id_available and error_not_toleranted:
                model_success = False
                acc_err_before_reset = int(accumulated_error)
                accumulated_error = 0
                force_expert = True
                action_source = "expert"

                action = agent.get_next_action(ref_path[next_waypoint_id])
                if action == 0 and not force_episode_end:
                    action = 1
                action_seq = []

                if evaluator is not None:
                    user_text = random.choice(self.visual_prompts)
                    img = Image.fromarray(rgb).convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    messages.append({'role': 'user', 'content': user_text, 'image_url': f"data:image/png;base64,{b64}"})
                    expert_text = evaluator.idx2action_text.get(action, "Move forward 25 cm")
                    messages.append({'role': 'assistant', 'content': expert_text})
                generated_this_round = True
                generated_branch = "expert"
                generated_actions = [int(action)]
            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])
                if action == 0 and not force_episode_end:
                    action = 1

            if self.args.dagger_save_model_trace and generated_this_round:
                pre_m = env.get_metrics()
                trace_entry = {
                    "step_id": step_id,
                    "decision_branch": generated_branch,
                    "distance_to_goal": float(pre_m["distance_to_goal"]),
                    "instruction": instr_str,
                    "actions": [int(x) for x in generated_actions] if generated_actions is not None else None,
                    "model_raw": step_trace_llm if generated_branch == "model" else None,
                    "expert_action": int(expert_action),
                    "next_waypoint_id": int(next_waypoint_id),
                    "ref_path_len": int(len(ref_path)),
                    "from_expert": bool(from_expert),
                    "force_episode_end": bool(force_episode_end),
                    "accumulated_error": int(accumulated_error),
                    "accumulated_error_ratio_denom": int(accumulated_error_ratio_denom),
                    "accumulated_error_ratio": float(accumulated_error_ratio),
                }
                if acc_err_before_reset is not None:
                    trace_entry["accumulated_error_before_reset"] = acc_err_before_reset
                trace_steps.append(trace_entry)

            observation = env.step(action)
            metrics = env.get_metrics()

            if save_video:
                metrics = env.get_metrics()
                if metrics['top_down_map'] is not None:
                    resized_rgb = np.array(image_resize(img=observation['rgb'],
                                                        size=(int(observation['rgb'].shape[0] * 1.6), int(observation['rgb'].shape[1] * 1.6)),
                                                        channels_last=True))
                    frame = observations_to_image({'rgb': resized_rgb}, metrics)
                    frame = append_text_underneath_image(frame, episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")
                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:
                break
            actions.append(action)
            step_id += 1

        if self.args.collect_images:
            assert len(rgb_data_list) == len(actions), f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {(actions)}"

        annotation_actions.append({
            "id": episode_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "video": f"images/{scene_id}_{self.dataset}_{episode_id:06d}",
            "instructions": [instructions] if isinstance(instructions, str) else instructions,
            "actions": actions,
        })

        episode_save = metrics["distance_to_goal"] < MIDGOAL_RADIUS and (((not model_success) and (metrics["pl"] < RELATIVE_PATH_LENGTH_THRESHOLD)) or (metrics["pl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD))
        if episode_save and self.args.collect_images:
            os.makedirs(os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb"), exist_ok=True)

            for rgb, rgb_path in rgb_data_list:
                Image.fromarray(rgb).convert("RGB").save(rgb_path)

        if save_video:
            if episode_save:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'save_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()
            else:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'notsave_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()

        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_episode_end,
            }
        )

        if self.args.dagger_save_model_trace and trace_steps:
            trace_dir = os.path.join(self.output_path, "model_traces")
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(
                trace_dir,
                f"{scene_id}_{episode_id:06d}_{trajectory_id}_r{self.rank}.json",
            )
            with open(trace_file, "w", encoding="utf-8") as tf:
                json.dump(
                    {
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "instruction": instr_str,
                        "steps": trace_steps,
                    },
                    tf,
                    ensure_ascii=False,
                    indent=2,
                )

        episode_dict = dict(
            anno_actions=annotation_actions,
            metrics=metrics,
        )

        return episode_dict

    def update_dataset(self, evaluator, dataset=None):
        """Update dataset with the collected data."""

        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type, config=self.config.habitat.dataset
        )
        scene_episode_dict = {}
        episode_uuids = []
        start = time.time()
        for episode in dataset.episodes:
            episode_uuid = (episode.scene_id, episode.episode_id, episode.trajectory_id)
            episode_uuids.append(episode_uuid)
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sampled_episodes_uuids = episode_uuids
        sampled_episodes_by_scene = {}
        for scene_id in sorted(scene_episode_dict.keys()):
            sampled_episodes_traj_ids = [
                (episode_uuid[1], episode_uuid[2])
                for episode_uuid in sampled_episodes_uuids
                if episode_uuid[0] == scene_id
            ]
            sampled_episodes_by_scene[scene_id] = [
                ep for ep in scene_episode_dict[scene_id]
                if (ep.episode_id, ep.trajectory_id) in sampled_episodes_traj_ids
            ]

        assigned_episodes = []
        for scene_id in sorted(scene_episode_dict.keys()):
            episodes = sampled_episodes_by_scene[scene_id]
            assigned_episodes.extend(episodes[self.rank::self.world_size])

        # Resume behavior: skip episodes that already exist in result.json.
        processed_keys = set()
        result_path = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    scene = item.get("scene")
                    episode_id = item.get("episode_id")
                    trajectory_id = item.get("trajectory_id")
                    if scene is None or episode_id is None or trajectory_id is None:
                        continue
                    processed_keys.add((str(scene), str(episode_id), str(trajectory_id)))

        if processed_keys:
            original_total = len(assigned_episodes)
            assigned_episodes = [
                ep
                for ep in assigned_episodes
                if (
                    ep.scene_id.split('/')[-2],
                    str(ep.episode_id),
                    str(ep.trajectory_id),
                ) not in processed_keys
            ]
            skipped_total = original_total - len(assigned_episodes)
        thread_envs = {}
        thread_envs_lock = threading.Lock()

        def _get_thread_env():
            tid = threading.get_ident()
            with thread_envs_lock:
                if tid not in thread_envs:
                    thread_envs[tid] = self.config_env()
                return thread_envs[tid]

        def _collect_episode(ep):
            env_local = _get_thread_env()
            env_local.current_episode = ep
            env_local.current_episode.goals[0].radius = MIDGOAL_RADIUS
            episode_dagger = self.generate(
                env=env_local,
                evaluator=evaluator,
                save_video=self.args.dagger_save_video,
                force_expert=self.args.force_expert,
            )
            return ep, episode_dagger

        num_collect_episodes = 0
        annotations_actions = []
        assigned_total = len(assigned_episodes)
        max_workers = max(1, int(getattr(self.args, 'parallel_envs', 1)))

        def _dedup_records(records, key_name, fallback_keys):
            seen = set()
            out = []
            for item in records:
                key = item.get(key_name)
                if key is None:
                    key = tuple(item.get(k) for k in fallback_keys)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            return out

        with tqdm.tqdm(
            total=assigned_total,
            dynamic_ncols=True,
            desc=f"rank{self.rank}",
            disable=(self.rank != 0),
        ) as pbar, torch.no_grad(), ThreadPoolExecutor(max_workers=max_workers) as env_executor:
            it = iter(assigned_episodes)
            future_to_ep = {}
            processed_episodes = 0

            def _submit_available():
                while len(future_to_ep) < max_workers:
                    try:
                        ep = next(it)
                    except StopIteration:
                        break
                    fut = env_executor.submit(_collect_episode, ep)
                    future_to_ep[fut] = ep

            _submit_available()

            stop_collect = False
            while future_to_ep and (not stop_collect):
                done, _ = wait(list(future_to_ep.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    ep = future_to_ep.pop(fut)
                    try:
                        episode, episode_dagger = fut.result()
                    except ZeroDivisionError:
                        processed_episodes += 1
                        pbar.update()
                        pbar.set_postfix({
                            "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                            "processed": f"{processed_episodes}/{assigned_total}",
                        })
                        if processed_episodes >= self.dagger_config.update_size:
                            stop_collect = True
                        continue
                    scan = episode.scene_id.split('/')[-2]

                    with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                        rel_pl = episode_dagger['metrics'].get('pl', None)
                        result = {
                            'scene': scan,
                            'episode_id': episode.episode_id,
                            'trajectory_id': episode.trajectory_id,
                            'save': episode_dagger['metrics']['save'],
                            'model_success': episode_dagger['metrics']['model_success'],
                            'success': episode_dagger['metrics']['success'],
                            'distance_to_goal': episode_dagger['metrics'].get('distance_to_goal', None),
                            'relative_pl': rel_pl,
                            'step_id': episode_dagger['metrics']['step_id'],
                            'ref_actions': episode_dagger['metrics']['ref_actions_len'],
                            'accumulated_error': episode_dagger['metrics']['accumulated_error'],
                            'force_episode_end': episode_dagger['metrics']['force_episode_end'],
                        }
                        f.write(json.dumps(result) + "\n")
                    if not episode_dagger['metrics']['save']:
                        processed_episodes += 1
                        pbar.update()
                        pbar.set_postfix({
                            "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                            "processed": f"{processed_episodes}/{assigned_total}",
                        })
                        if processed_episodes >= self.dagger_config.update_size:
                            stop_collect = True
                        continue

                    for k, v in episode_dagger.items():
                        if isinstance(v, torch.Tensor):
                            episode_dagger[k] = v.numpy()

                    print(
                        f"model_success = {episode_dagger['metrics']['model_success']}, "
                        f"scene {scan} id {episode.episode_id} trajectory {episode.trajectory_id}"
                    )

                    annotations_actions.extend(episode_dagger['anno_actions'])
                    processed_episodes += 1
                    pbar.update()
                    num_collect_episodes += 1
                    pbar.set_postfix({
                        "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                        "processed": f"{processed_episodes}/{assigned_total}",
                    })

                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        tgt_actions_path = os.path.join(self.output_path, f'{self.args.annotations_actions}_{self.rank}.json')
                        if os.path.exists(tgt_actions_path):
                            merged_actions = json.load(open(tgt_actions_path))
                        else:
                            merged_actions = []
                        with open(tgt_actions_path, 'w') as json_file:
                            merged_actions.extend(annotations_actions)
                            merged_actions = _dedup_records(merged_actions, "video", ("episode_id", "trajectory_id"))
                            json_data = json.dumps(merged_actions, indent=4)
                            json_file.write(json_data)

                    if processed_episodes >= self.dagger_config.update_size:
                        stop_collect = True

                if stop_collect:
                    for fut in future_to_ep:
                        fut.cancel()
                    break

                _submit_available()

            tgt_actions_path = os.path.join(self.output_path, f'{self.args.annotations_actions}_{self.rank}.json')
            if os.path.exists(tgt_actions_path):
                merged_actions = json.load(open(tgt_actions_path))
            else:
                merged_actions = []
            with open(tgt_actions_path, 'w') as json_file:
                merged_actions.extend(annotations_actions)
                merged_actions = _dedup_records(merged_actions, "video", ("episode_id", "trajectory_id"))
                json_data = json.dumps(merged_actions, indent=4)
                json_file.write(json_data)

            print(f"save total episodes {num_collect_episodes} time cost {time.time() - start}")

            def _close_thread():
                tid = threading.get_ident()
                with thread_envs_lock:
                    env_local = thread_envs.pop(tid, None)
                if env_local is not None:
                    try:
                        env_local.close()
                    except Exception:
                        pass

            close_futs = [
                env_executor.submit(_close_thread)
                for _ in range(max_workers)
            ]
            for cf in close_futs:
                try:
                    cf.result()
                except Exception:
                    pass

        for env_local in list(thread_envs.values()):
            try:
                env_local.close()
            except Exception:
                pass

        dist.barrier()
        if get_rank() == 0:
            tgt_actions_path = os.path.join(self.output_path, f'{self.args.annotations_actions}.json')
            merged_actions = []
            sub_tgt_actions_list = [
                os.path.join(self.output_path, f)
                for f in os.listdir(self.output_path)
                if f.startswith(f'{self.args.annotations_actions}_') and f.endswith('.json')
            ]
            for sub_tgt_actions_path in sub_tgt_actions_list:
                if os.path.exists(sub_tgt_actions_path):
                    merged_actions.extend(json.load(open(sub_tgt_actions_path)))
            merged_actions = sorted(merged_actions, key=lambda x: x['id'])
            merged_actions = _dedup_records(merged_actions, "video", ("episode_id", "trajectory_id"))
            with open(tgt_actions_path, 'w') as json_file:
                json_data = json.dumps(merged_actions, indent=4)
                json_file.write(json_data)

        with self._llm_cond:
            self._llm_shutdown = True
            self._llm_cond.notify_all()
        self._llm_scheduler_thread.join()
        self.llm_executor.shutdown(wait=True, cancel_futures=False)

def parse_args():

    parser = argparse.ArgumentParser()

    # VLLM
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model_name", type=str, default="qwen3vl")
    parser.add_argument("--vllm_max_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="data/dagger")

    # Habitat
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_dagger.yaml')
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")

    # DAgger
    parser.add_argument("--dagger_p",type=float, default=0.9)
    parser.add_argument("--dagger_update_size", type=int, default=1)
    parser.add_argument("--dagger_commit_freq",type=int, default=1)
    parser.add_argument("--dagger_dataset", type=str, default=DATASET)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_output_path",type=str, default="data/dagger")
    parser.add_argument("--dagger_data_path", type=str, default="data/datasets/vln_datasets/{split}.json.gz")
    parser.add_argument("--dagger_gt_annotations_path", type=str, default="data/datasets/vln_datasets/annotations.json")
    parser.add_argument(
        "--annotations_actions",
        type=str,
        default="annotations_actions",
        help="action annotation filename prefix, output as <prefix>.json",
    )
    parser.add_argument(
        "--dagger_save_video",
        action="store_true",
        default=False,
        help="whether to save video during dagger collection",
    )
    parser.add_argument(
        "--collect_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to save per-step RGB images under dagger_output_path/images",
    )
    parser.add_argument(
        "--dagger-save-model-trace",
        action="store_true",
        default=False,
        dest="dagger_save_model_trace",
        help="Write per-episode JSON traces (model vs expert, actions) under dagger_output_path/model_traces/",
    )

    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank")
    parser.add_argument("--gpu", default=0, type=int, help="gpu")
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    return parser

if __name__ == "__main__":

    global local_rank

    parser = parse_args()
    args = parser.parse_args()
    args.save_video = args.dagger_save_video

    init_distributed_mode(args)
    local_rank = args.local_rank

    model = VLN_Inference_VLLM(
        base_url=args.vllm_base_url,
        model_name=args.vllm_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    rank = get_rank()
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    collector = DAggerCollector(args=args, rank=rank, world_size=world_size)
    collector.update_dataset(evaluator=evaluator)
