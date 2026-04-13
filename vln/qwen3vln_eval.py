import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import asyncio
import tqdm
import torch
import json
import random
import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import defaultdict

from typing import Any
from omegaconf import OmegaConf
from collections import OrderedDict

import habitat
from habitat import Env
from vln.habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from transformers import Qwen3VLForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from peft import PeftConfig, PeftModel
from openai import OpenAI

from vln.data import (
    build_user_image_message,
    dump_sharegpt_trial,
    dump_vllm_error_sharegpt,
    parse_subset_episode_string,
    parse_subset_triplet_string,
    subset_episodes_to_triplets,
    to_local_chat_and_images,
    to_vllm_chat_messages,
)
from vln.utils.dist import *

CURATED_SUBSET_EPISODES = (
    # ("QUCTc6BB5sX", 994),
    # ("2azQ1b91cZZ", 81),
    # ("EU6Fwq7SyZv", 1812),
    ("QUCTc6BB5sX", 149),
    ("QUCTc6BB5sX", 148),
)


def _build_worker_config(config_path: str, split: str, save_video: bool, use_collision_prompt: bool = False):
    # Build habitat config with optional top-down map and collision metrics
    config = get_habitat_config(config_path)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        measurements = {}
        if save_video:
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
                    )
                }
            )
        if save_video or use_collision_prompt:
            measurements.update({"collisions": CollisionsMeasurementConfig()})
        if measurements:
            config.habitat.task.measurements.update(measurements)
    return config

def _env_worker_loop(worker_id, config_path, split, save_video, use_collision_prompt, cmd_q, res_q):
    # Multiprocessing worker for habitat env reset/step
    env = None
    try:
        worker_config = _build_worker_config(config_path, split, save_video, use_collision_prompt)
        env = Env(config=worker_config)
        episode_map = {}
        for ep in env.episodes:
            ep_instruction = ep.instruction.instruction_text if 'objectnav' not in config_path else ep.object_category
            scene_id = ep.scene_id.split('/')[-2]
            ep_key = (scene_id, str(ep.episode_id), ep_instruction)
            episode_map[ep_key] = ep

        while True:
            cmd, payload = cmd_q.get()
            if cmd == "close":
                break
            if cmd == "start_episode":
                key = payload["key"]
                episode_obj = episode_map[key]
                env.current_episode = episode_obj
                observations = env.reset()
                info = env.get_metrics() if (save_video or use_collision_prompt) else None
                res_q.put({"type": "started", "worker_id": worker_id, "observations": observations, "info": info})
                continue
            if cmd == "step":
                action = payload["action"]
                observations = env.step(action)
                done = env.episode_over
                info = env.get_metrics() if (save_video or use_collision_prompt or done) else None
                res_q.put(
                    {
                        "type": "step",
                        "worker_id": worker_id,
                        "observations": observations,
                        "done": done,
                        "info": info,
                    }
                )
                continue
    finally:
        if env is not None:
            env.close()

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.fail_on_llm_error = getattr(args, "fail_on_llm_error", True)
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            measurements = {}
            if self.save_video:
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
                        )
                    }
                )
            if self.save_video or getattr(self.args, 'use_collision_prompt', False):
                measurements.update({"collisions": CollisionsMeasurementConfig()})
            if measurements:
                self.config.habitat.task.measurements.update(measurements)

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self.processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        self.actions2idx = OrderedDict({
            "MOVE FORWARD 25 CM": [1],
            "TURN LEFT 15 DEGREES": [2],
            "TURN RIGHT 15 DEGREES": [3],
            "STOP": [0],
        })
        self.idx2action_text = {
            0: "Stop",
            1: "Move forward 25 cm",
            2: "Turn left 15 degrees",
            3: "Turn right 15 degrees",
        }
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
        self.num_frames = args.num_frames if hasattr(args, 'num_frames') else 1
        # self.num_future_steps = args.num_future_steps if hasattr(args, "num_future_steps") else 4
        self.max_new_tokens = args.max_new_tokens if hasattr(args, "max_new_tokens") else 32
        self.pass_k = max(1, int(args.pass_k)) if hasattr(args, "pass_k") else 4
        if hasattr(args, "use_vllm") and args.use_vllm:
            self.parallel_envs = max(1, int(args.parallel_envs))
        else:
            self.parallel_envs = 1
        self.trace_mode = args.trace_mode if hasattr(args, "trace_mode") else "fail"
        self.trace_dir = os.path.join(self.output_path, "traces")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.subset_triplets = getattr(args, "subset_triplets_parsed", None)
        self.subset_force_rerun = bool(getattr(args, "subset_force_rerun", False))
        self.save_sharegpt = bool(getattr(args, "save_sharegpt", False))

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env


    def eval_action(self, idx) -> None:
        env_for_episodes = self.config_env()
        scene_episode_dict = {}
        for episode in env_for_episodes.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        env_for_episodes.close()

        sucs, spls, oss, ones = [], [], [], []
        done_res = set()
        instruction_trial_success = defaultdict(dict)
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    if "scene_id" not in res:
                        continue
                    key = (res["scene_id"], str(res["episode_id"]), res["episode_instruction"])
                    trial_id = int(res.get("trial_id", 0))
                    done_res.add((key[0], key[1], key[2], trial_id))
                    if "success" in res:
                        instruction_trial_success[key][trial_id] = float(res["success"])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])

        def build_slot_state(worker_id, episode_obj, scene_name, trial_id, observations, init_info):
            # Initialize per-trial state dict with system prompt and metadata
            episode_instruction = episode_obj.instruction.instruction_text if 'objectnav' not in self.config_path else episode_obj.object_category
            episode_id = episode_obj.episode_id
            print(
                f"Episode start | scene={scene_name} episode_id={episode_id} trial={trial_id} | {episode_instruction}",
                flush=True,
            )
            state = {
                "worker_id": worker_id,
                "episode_obj": episode_obj,
                "scene_id": scene_name,
                "episode_id": episode_id,
                "episode_instruction": episode_instruction,
                "observations": observations,
                "system_message": "You are an autonomous navigation assistant. Your task is to follow the given instruction in the environment. At each step, output the next four actions to take, exactly four, separated by commas. Use only the following actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop.",
                "vis_frames": [],
                "step_id": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an autonomous navigation assistant. Your task is to follow the given instruction in the environment. At each step, output the next four actions to take, exactly four, separated by commas. Use only the following actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop."
                    }
                ],
                "messages_started": False,
                "action_seq": [],
                "trace_records": [],
                "last_info": init_info,
                "trial_id": int(trial_id),
                "trial_total": int(self.pass_k),
                "base_key": (scene_name, str(episode_id), episode_instruction),
            }
            if self.save_video:
                os.makedirs(
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_name}_{episode_id}_trial{trial_id}'),
                    exist_ok=True,
                )
            return state

        def run_single_vllm(req_messages, state):
            last_user = None
            for m in reversed(req_messages):
                if m.get("role") == "user":
                    last_user = m
                    break
            lc = last_user.get("content") if last_user else ""
            is_stuck = isinstance(lc, str) and "STUCK" in lc
            info = state.get("last_info") or {}
            coll = info.get("collisions")
            coll_flag = None
            if isinstance(coll, dict):
                coll_flag = bool(coll.get("is_collision"))
            dump_meta = {
                "output_path": self.output_path,
                "scene_id": state["scene_id"],
                "episode_id": state["episode_id"],
                "trial_id": state["trial_id"],
                "step_id": state["step_id"],
                "worker_id": state["worker_id"],
                "is_stuck_user_prompt": is_stuck,
                "collision_is_collision": coll_flag,
                "episode_instruction": state.get("episode_instruction", ""),
            }
            out = self.model.call_model(
                req_messages,
                max_new_tokens=self.max_new_tokens,
                dump_meta=dump_meta,
            )
            if isinstance(out, list):
                return out[0] if len(out) > 0 else ""
            return str(out)

        def commit_llm_output(state, llm_outputs, action_seq, parse_mode):
            # Record LLM output, actions and trace for current step
            print(llm_outputs, flush=True)
            state["action_seq"] = action_seq
            assistant_text = self.actions_to_text(action_seq)
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )
            info = state["last_info"] or {}
            state["trace_records"].append(
                {
                    "step_id": int(state["step_id"]),
                    "event": "generation",
                    "instruction": state["episode_instruction"],
                    "llm_outputs": llm_outputs,
                    "assistant_text": assistant_text,
                    "parse_mode": parse_mode,
                    "predicted_actions": list(action_seq),
                    "distance_to_goal": float(info["distance_to_goal"]) if "distance_to_goal" in info else None,
                }
            )
            print('env_id:', state["worker_id"], 'step_id:', state["step_id"], 'actions:', state["action_seq"], flush=True)

        def finalize_slot(state, metrics):
            # Save video/ShareGPT, record metrics and write result.json
            if self.save_sharegpt:
                dump_sharegpt_trial(state, self.output_path, self.epoch)
            if self.save_video:
                images_to_video(
                    state["vis_frames"],
                    os.path.join(self.output_path, f'vis_{self.epoch}'),
                    f'{state["scene_id"]}_{state["episode_id"]}_trial{state["trial_id"]}',
                    fps=6,
                    quality=9
                )
            state["vis_frames"].clear()
            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            ones.append(metrics['distance_to_goal'])
            print(
                f"Finished | scene={state['scene_id']} episode_id={state['episode_id']} trial={state['trial_id']} | "
                f"success={metrics['success']} spl={metrics['spl']} os={metrics['oracle_success']} "
                f"ne={metrics['distance_to_goal']} steps={state['step_id']}",
                flush=True,
            )
            instruction_trial_success[state["base_key"]][state["trial_id"]] = float(metrics["success"])
            result = {
                "scene_id": state["scene_id"],
                "episode_id": state["episode_id"],
                "trial_id": state["trial_id"],
                "trial_total": state["trial_total"],
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": state["step_id"],
                "episode_instruction": state["episode_instruction"]
            }
            should_dump_trace = (self.trace_mode == "all") or (self.trace_mode == "fail" and metrics["success"] < 1.0)
            if should_dump_trace:
                trace_payload = {
                    "scene_id": state["scene_id"],
                    "episode_id": state["episode_id"],
                    "episode_instruction": state["episode_instruction"],
                    "success": float(metrics["success"]),
                    "spl": float(metrics["spl"]),
                    "os": float(metrics["oracle_success"]),
                    "ne": float(metrics["distance_to_goal"]),
                    "steps": int(state["step_id"]),
                    "trace": state["trace_records"],
                }
                trace_path = os.path.join(
                    self.trace_dir,
                    f"{state['scene_id']}_{state['episode_id']}_trial{state['trial_id']}.json"
                )
                with open(trace_path, "w") as tf:
                    json.dump(trace_payload, tf, ensure_ascii=False)

            with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

        os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
        ctx = mp.get_context("spawn")
        workers = []
        for worker_id in range(self.parallel_envs):
            cmd_q = ctx.Queue()
            res_q = ctx.Queue()
            proc = ctx.Process(
                target=_env_worker_loop,
                args=(worker_id, self.config_path, self.split, self.save_video, getattr(self.args, 'use_collision_prompt', False), cmd_q, res_q),
                daemon=True,
            )
            proc.start()
            workers.append({"id": worker_id, "cmd_q": cmd_q, "res_q": res_q, "proc": proc})
        llm_executor = ThreadPoolExecutor(max_workers=max(1, self.parallel_envs))
        io_executor = ThreadPoolExecutor(max_workers=max(1, self.parallel_envs * 2))

        try:
            scenes_wanted = None
            if self.subset_triplets is not None:
                scenes_wanted = {t[0] for t in self.subset_triplets}
            for scene in sorted(scene_episode_dict.keys()):
                episodes = scene_episode_dict[scene]
                scene_id = scene.split('/')[-2]
                if scenes_wanted is not None and scene_id not in scenes_wanted:
                    continue
                print(f"scene_id={scene_id}", flush=True)
                assigned = episodes[idx::self.env_num]
                if self.subset_triplets is not None:
                    need_ep = {t[1] for t in self.subset_triplets if t[0] == scene_id}
                    assigned = [ep for ep in assigned if str(ep.episode_id) in need_ep]
                process_bar = tqdm.tqdm(range(len(assigned)), desc=f"scene {scene_id}")

                pending = []
                completed_trials = defaultdict(set)
                for ep in assigned:
                    ep_instruction = ep.instruction.instruction_text if 'objectnav' not in self.config_path else ep.object_category
                    base_key = (scene_id, str(ep.episode_id), ep_instruction)
                    for trial_id in range(self.pass_k):
                        if self.subset_triplets is not None:
                            if (scene_id, str(ep.episode_id), trial_id) not in self.subset_triplets:
                                continue
                        trial_key = (base_key[0], base_key[1], base_key[2], trial_id)
                        if (not self.subset_force_rerun) and trial_key in done_res:
                            completed_trials[base_key].add(trial_id)
                            continue
                        pending.append({"episode": ep, "trial_id": trial_id})
                    if len(completed_trials[base_key]) >= self.pass_k:
                        process_bar.update(1)

                if len(pending) == 0:
                    continue

                states_by_worker = {}
                llm_future_to_worker = {}
                env_future_meta = {}

                def submit_llm(worker_id):
                    state = states_by_worker.get(worker_id)
                    if state is None:
                        return
                    if state.pop("collision", False):
                        stuck_text = (
                            "STATUS: the robot is STUCK; it collided with an obstacle during the last action block. "
                            "Think about how to get unstuck and continue toward that goal. Do NOT output your reasoning. "
                            "Must output exactly four actions to take next, separated by commas. Use only the following actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop."
                        )
                        state["messages"].append({"role": "user", "content": stuck_text})
                        fut = llm_executor.submit(run_single_vllm, state["messages"], state)
                        llm_future_to_worker[fut] = worker_id
                        return

                    if not state["messages_started"]:
                        user_text = f"Your instruction is '{state['episode_instruction']}' "
                        state["messages_started"] = True
                    else:
                        user_text = random.choice(self.visual_prompts)

                    state["messages"].append(build_user_image_message(user_text, state["observations"]["rgb"]))

                    fut = llm_executor.submit(run_single_vllm, state["messages"], state)
                    llm_future_to_worker[fut] = worker_id

                def submit_step(worker_id):
                    state = states_by_worker.get(worker_id)
                    if state is None:
                        return
                    if len(state["action_seq"]) == 0:
                        return
                    if self.save_video:
                        info = state["last_info"] or {}
                        if info.get('top_down_map') is not None:
                            frame = observations_to_image({'rgb': state["observations"]['rgb']}, info)
                            state["vis_frames"].append(frame)
                    else:
                        state["last_info"] = None
                    action = state["action_seq"].pop(0)
                    if state["step_id"] >= 400:
                        action = 0
                    state["trace_records"].append(
                        {
                            "step_id": int(state["step_id"]),
                            "event": "execute",
                            "action": int(action),
                            "remaining_cached_actions": int(len(state["action_seq"])),
                        }
                    )
                    worker = workers[worker_id]
                    worker["cmd_q"].put(("step", {"action": int(action)}))
                    fut = io_executor.submit(worker["res_q"].get)
                    env_future_meta[fut] = ("step", worker_id, None)

                def dispatch_start(worker_id, ep_task):
                    ep_obj = ep_task["episode"]
                    trial_id = int(ep_task["trial_id"])
                    ep_instruction = ep_obj.instruction.instruction_text if 'objectnav' not in self.config_path else ep_obj.object_category
                    ep_key = (scene_id, str(ep_obj.episode_id), ep_instruction)
                    worker = workers[worker_id]
                    worker["cmd_q"].put(("start_episode", {"key": ep_key}))
                    fut = io_executor.submit(worker["res_q"].get)
                    env_future_meta[fut] = ("start", worker_id, ep_obj, trial_id)

                for worker in workers:
                    if len(pending) == 0:
                        break
                    ep_task = pending.pop(0)
                    dispatch_start(worker["id"], ep_task)

                while len(pending) > 0 or len(states_by_worker) > 0 or len(llm_future_to_worker) > 0 or len(env_future_meta) > 0:
                    wait_pool = list(llm_future_to_worker.keys()) + list(env_future_meta.keys())
                    if len(wait_pool) == 0:
                        break
                    done, _ = wait(wait_pool, return_when=FIRST_COMPLETED)
                    for fut in done:
                        if fut in llm_future_to_worker:
                            worker_id = llm_future_to_worker.pop(fut)
                            state = states_by_worker.get(worker_id)
                            if state is None:
                                continue
                            try:
                                llm_outputs = fut.result()
                            except Exception as e:
                                err_s = str(e)
                                if (
                                    "VLLM returned empty text" in err_s
                                    or "LLM returned empty output" in err_s
                                ):
                                    states_by_worker.pop(worker_id, None)
                                    ep_task = {
                                        "episode": state["episode_obj"],
                                        "trial_id": state["trial_id"],
                                    }
                                    print(
                                        f"[LLM empty] restarting episode scene={state['scene_id']} "
                                        f"episode_id={state['episode_id']} trial={state['trial_id']} "
                                        f"after step={state['step_id']}: {err_s}",
                                        flush=True,
                                    )
                                    dispatch_start(worker_id, ep_task)
                                    continue
                                raise RuntimeError(
                                    f"LLM request failed at episode={state['episode_id']} step={state['step_id']}: {e}"
                                ) from e
                            if self.fail_on_llm_error and (not isinstance(llm_outputs, str) or not llm_outputs.strip()):
                                states_by_worker.pop(worker_id, None)
                                ep_task = {
                                    "episode": state["episode_obj"],
                                    "trial_id": state["trial_id"],
                                }
                                print(
                                    f"[LLM empty] restarting episode scene={state['scene_id']} "
                                    f"episode_id={state['episode_id']} trial={state['trial_id']} "
                                    f"after step={state['step_id']} (empty string)",
                                    flush=True,
                                )
                                dispatch_start(worker_id, ep_task)
                                continue
                            parse_mode = "keyword"
                            action_seq = self.parse_actions(llm_outputs)
                            if len(action_seq) == 0:
                                parse_mode = "compact_keyword"
                                action_seq = self.parse_actions(llm_outputs.upper().replace(" ", ""))
                            if len(action_seq) == 0:
                                parse_mode = "digit"
                                digit_actions = re.findall(r"[0-3]", llm_outputs)
                                action_seq = [int(a) for a in digit_actions]
                            if len(action_seq) == 0:
                                state["messages"].append(
                                    {
                                        "role": "user",
                                        "content": "You should output only a comma-separated action list using exactly these actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop.",
                                    }
                                )
                                parse_mode = "fallback_stop"
                                action_seq = [0]
                            commit_llm_output(state, llm_outputs, action_seq, parse_mode)
                            submit_step(worker_id)
                            continue

                        if fut in env_future_meta:
                            event_info = env_future_meta.pop(fut)
                            event_type = event_info[0]
                            worker_id = event_info[1]
                            try:
                                env_result = fut.result()
                            except Exception:
                                env_result = {}

                            if event_type == "start":
                                ep_obj = event_info[2]
                                trial_id = event_info[3]
                                state = build_slot_state(
                                    worker_id,
                                    ep_obj,
                                    scene_id,
                                    trial_id,
                                    env_result.get("observations"),
                                    env_result.get("info"),
                                )
                                states_by_worker[worker_id] = state
                                submit_llm(worker_id)
                                continue

                            state = states_by_worker.get(worker_id)
                            if state is None:
                                continue
                            state["observations"] = env_result.get("observations")
                            state["step_id"] += 1
                            state["last_info"] = env_result.get("info")

                            if env_result.get("done", False):
                                metrics = state.get("last_info") or {}
                                states_by_worker.pop(worker_id, None)
                                finalize_slot(state, metrics)
                                completed_trials[state["base_key"]].add(state["trial_id"])
                                if len(completed_trials[state["base_key"]]) >= self.pass_k:
                                    process_bar.update(1)
                                if len(pending) > 0:
                                    next_ep_task = pending.pop(0)
                                    dispatch_start(worker_id, next_ep_task)
                            else:
                                if state["step_id"] >= 400:
                                    state["action_seq"] = []
                                    worker = workers[worker_id]
                                    worker["cmd_q"].put(("step", {"action": 0}))
                                    fut = io_executor.submit(worker["res_q"].get)
                                    env_future_meta[fut] = ("step", worker_id, None)
                                elif len(state["action_seq"]) > 0:
                                    submit_step(worker_id)
                                elif (
                                    getattr(self.args, "use_collision_prompt", False)
                                    and state.get("last_info")
                                    and state["last_info"].get("collisions", {}).get("is_collision", False)
                                ):
                                    state["trace_records"].append(
                                        {
                                            "step_id": int(state["step_id"]),
                                            "event": "collision",
                                            "instruction": state["episode_instruction"],
                                        }
                                    )
                                    state["collision"] = True
                                    submit_llm(worker_id)
                                else:
                                    submit_llm(worker_id)
        finally:
            llm_executor.shutdown(wait=True, cancel_futures=False)
            io_executor.shutdown(wait=True, cancel_futures=False)
            for worker in workers:
                try:
                    worker["cmd_q"].put(("close", {}))
                except Exception:
                    pass
            for worker in workers:
                worker["proc"].join(timeout=5)
                if worker["proc"].is_alive():
                    worker["proc"].terminate()

        pass_hits = []
        for _, trial_success_dict in instruction_trial_success.items():
            if len(trial_success_dict) < self.pass_k:
                continue
            ordered_trial_ids = sorted(trial_success_dict.keys())[: self.pass_k]
            hit = 1.0 if any(float(trial_success_dict[t]) >= 1.0 for t in ordered_trial_ids) else 0.0
            pass_hits.append(hit)

        return (
            torch.tensor(sucs, dtype=torch.float32, device=self.device),
            torch.tensor(spls, dtype=torch.float32, device=self.device),
            torch.tensor(oss, dtype=torch.float32, device=self.device),
            torch.tensor(ones, dtype=torch.float32, device=self.device),
            torch.tensor(len(sucs), device=self.device),
            torch.tensor(pass_hits, dtype=torch.float32, device=self.device),
            torch.tensor(len(pass_hits), device=self.device),
        )

    def parse_actions(self, output):
        # Extract action indices from LLM text using regex
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns, flags=re.IGNORECASE)
        matches = regex.findall(output)
        actions = [self.actions2idx[match.upper()] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def actions_to_text(self, actions):
        # Convert action list to comma-separated text
        return ", ".join(self.idx2action_text.get(action, "Stop") for action in actions)
    
class VLN_Inference:
    def __init__(self, model_path, device="cuda", num_future_steps=4, processor_path=None):
        # Load Qwen3-VL model (with optional PEFT) and processor
        self.device = device if isinstance(device, str) else str(device)
        # self.num_future_steps = num_future_steps
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            base_config = AutoConfig.from_pretrained(base_model_path)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=base_config,
                low_cpu_mem_usage=True,
                device_map={"": self.device},
            )
            self.model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload().eval()
        else:
            config = AutoConfig.from_pretrained(model_path)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                device_map={"": self.device},
            ).eval()
        self.processor = self._load_processor(model_path, processor_path)

    @staticmethod
    def _load_processor(model_path: str, processor_path: str = None):
        # Find and load preprocessor (fallback to parent dirs if missing)
        if processor_path:
            return AutoProcessor.from_pretrained(processor_path)

        candidates = [model_path]
        parent_1 = os.path.dirname(model_path)
        parent_2 = os.path.dirname(parent_1) if parent_1 else ""
        if parent_1 and parent_1 not in candidates:
            candidates.append(parent_1)
        if parent_2 and parent_2 not in candidates:
            candidates.append(parent_2)

        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "preprocessor_config.json")):
                return AutoProcessor.from_pretrained(candidate)

        raise OSError(
            f"Can't load image processor from {model_path}. "
            "Please provide --processor_path that contains preprocessor_config.json."
        )

    def call_model(self, messages, max_new_tokens=32, **kwargs):
        chat_messages, image_inputs = to_local_chat_and_images(messages)
        text = self.processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True
        )
        
        inputs = inputs.to(self.device)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers


class VLN_Inference_VLLM:
    def __init__(
        self, 
        base_url="http://localhost:8000/v1", 
        model_name="qwen3vl",
        temperature=0.0,
        top_p=0.95,
        top_k=20):
        # vLLM client wrapper using OpenAI compatible API
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model_name = model_name
        self.processor = None
        self.tokenizer = None
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def call_model(self, messages, max_new_tokens=32, dump_meta=None):
        req_messages = to_vllm_chat_messages(messages)
        create_kwargs = {
            "model": self.model_name,
            "messages": req_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {"top_k": self.top_k},
        }
        if max_new_tokens is not None and max_new_tokens > 0:
            create_kwargs["max_tokens"] = max_new_tokens
        resp = self.client.chat.completions.create(**create_kwargs)
        if not resp.choices:
            if dump_meta is not None:
                dm = dict(dump_meta)
                dm["error_tag"] = "no_choices"
                dump_vllm_error_sharegpt(messages, req_messages, resp, dm)
            raise RuntimeError(f"VLLM returned no choices: {resp}")
        text = resp.choices[0].message.content
        if not text:
            if dump_meta is not None:
                dm = dict(dump_meta)
                dm["error_tag"] = "empty_text"
                dump_vllm_error_sharegpt(messages, req_messages, resp, dm)
            raise RuntimeError(f"VLLM returned empty text: {resp}")
        return [text]

    async def call_model_async(self, messages, max_new_tokens=32):
        return await asyncio.to_thread(self.call_model, messages, max_new_tokens)
   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--use_collision_prompt", action="store_true", default=False,
                        help="Prompt the model when it collides with a obstacle.")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to sample")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--trace_mode", type=str, default="fail")
    parser.add_argument("--processor_path", type=str, default="",
                        help="Optional processor path containing preprocessor_config.json")
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--fail_on_llm_error", action=argparse.BooleanOptionalAction, default=True,
                        help="Raise error immediately when vLLM request/output is invalid instead of fallback-to-stop")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model_name", type=str, default="qwen3vl")
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='top_p')
    parser.add_argument('--top_k', type=int, default=20,
                        help='top_k')
    parser.add_argument('--pass_k', type=int, default=1,
                        help='Number of sampled trajectories per instruction for pass@k')
    parser.add_argument(
        "--replay_curated_subset",
        action="store_true",
        help="Replay 5 fixed (scene, episode) pairs, each with pass_k trials (same as full pass@k)",
    )
    parser.add_argument(
        "--subset_episodes",
        type=str,
        default="",
        help="Subset as scene:episode_id;scene:episode_id;... (runs pass_k trials per episode)",
    )
    parser.add_argument(
        "--subset_triplets",
        type=str,
        default="",
        help="Exact trials: scene:episode_id:trial_id;... (overrides subset_episodes if set)",
    )
    parser.add_argument(
        "--subset_force_rerun",
        action="store_true",
        help="Ignore existing result.json lines for selected subset trials",
    )
    parser.add_argument(
        "--save_sharegpt",
        action="store_true",
        help="Write ShareGPT-style sharegpt.json plus per-turn images under output_path/sharegpt/",
    )
    args = parser.parse_args()
    pk = max(1, int(args.pass_k))
    if args.subset_triplets.strip():
        args.subset_triplets_parsed = parse_subset_triplet_string(args.subset_triplets)
    elif args.subset_episodes.strip():
        args.subset_triplets_parsed = subset_episodes_to_triplets(
            parse_subset_episode_string(args.subset_episodes), pk
        )
    elif args.replay_curated_subset:
        args.subset_triplets_parsed = subset_episodes_to_triplets(CURATED_SUBSET_EPISODES, pk)
    else:
        args.subset_triplets_parsed = None
    init_distributed_mode(args)
    local_rank = args.local_rank
    device = f"cuda:{local_rank}"

    if args.use_vllm:
        model = VLN_Inference_VLLM(
            base_url=args.vllm_base_url,
            model_name=args.vllm_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        model = VLN_Inference(
            args.model_path,
            device=device,
            num_future_steps=args.num_future_steps,
            processor_path=(args.processor_path or None),
        )
        model.model.requires_grad_(False)
    evaluate(model, args)


def evaluate(model, args):
    # Aggregate metrics across GPUs and write final result
    world_size = get_world_size()
    if getattr(args, "subset_triplets_parsed", None) is not None and world_size != 1:
        raise RuntimeError("Subset replay requires world_size==1 (single process).")
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num, pass_hits, pass_num = evaluator.eval_action(get_rank())
    if is_dist_avail_and_initialized() and world_size > 1:
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
        pass_num_all = [torch.zeros_like(pass_num) for _ in range(world_size)]
        dist.all_gather(ep_num_all, ep_num)
        dist.all_gather(pass_num_all, pass_num)
        ep_sizes = [int(x.item()) for x in ep_num_all]
        pass_sizes = [int(x.item()) for x in pass_num_all]
        sucs_all = [torch.zeros(ep_sizes[i], dtype=sucs.dtype, device=sucs.device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_sizes[i], dtype=spls.dtype, device=spls.device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_sizes[i], dtype=oss.dtype, device=oss.device) for i in range(world_size)]
        ones_all = [torch.zeros(ep_sizes[i], dtype=ones.dtype, device=ones.device) for i in range(world_size)]
        pass_hits_all = [torch.zeros(pass_sizes[i], dtype=pass_hits.dtype, device=pass_hits.device) for i in range(world_size)]
        dist.barrier()
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(ones_all, ones)
        dist.all_gather(pass_hits_all, pass_hits)
        dist.barrier()
        sucs_all = torch.cat(sucs_all, dim=0)
        spls_all = torch.cat(spls_all, dim=0)
        oss_all = torch.cat(oss_all, dim=0)
        ones_all = torch.cat(ones_all, dim=0)
        pass_hits_all = torch.cat(pass_hits_all, dim=0)
    else:
        sucs_all = sucs
        spls_all = spls
        oss_all = oss
        ones_all = ones
        pass_hits_all = pass_hits
    spls_all = torch.nan_to_num(spls_all, nan=0.0)
    finite_mask = torch.isfinite(ones_all)
    ones_finite = ones_all[finite_mask]
    ones_avg = ones_finite.mean() if ones_finite.numel() > 0 else torch.tensor(0.0, device=ones_all.device)
    pass_at_k = pass_hits_all.mean() if pass_hits_all.numel() > 0 else torch.tensor(0.0, device=ones_all.device)
    if sucs_all.numel() > 0:
        spls_mean = (sum(spls_all) / len(spls_all)).item()
        oss_mean = (sum(oss_all) / len(oss_all)).item()
    else:
        spls_mean = 0.0
        oss_mean = 0.0
    result_all = {
        "sucs_all": sucs_all.mean().item() if sucs_all.numel() > 0 else 0.0,
        "spls_all": spls_mean,
        "oss_all": oss_mean,
        "ones_all": ones_avg.item(),
        "length": ones_avg.item(),
        "pass_k": int(evaluator.pass_k),
        f"pass@{int(evaluator.pass_k)}": pass_at_k.item(),
    }

    print(result_all, flush=True)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
