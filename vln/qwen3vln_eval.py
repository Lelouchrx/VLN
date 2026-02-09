import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from vln.habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLProcessor
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from vln.utils.dist import *

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
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
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

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        # self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        # self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        # self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        # camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        # self._camera_fov = camera_fov_rad
        # self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        # self.image_processor = model.get_vision_tower().image_processor
        self.processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        # prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        # answer = ""
        # self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        # self.actions2idx = OrderedDict({
        #     'STOP': [0],
        #     "↑": [1],
        #     "←": [2],
        #     "→": [3]
        # })
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1], 
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })

        self.num_frames = args.num_frames if hasattr(args, 'num_frames') else 1
        
    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env


    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            # episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)

                vis_frames = []
                step_id = 0

                # 创建调试目录
                debug_dir = os.path.join(self.output_path, f'debug_{self.epoch}', f'{scene_id}_{episode_id}')
                os.makedirs(debug_dir, exist_ok=True)

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)

                rgb_list = []
                action_seq = []
                while not env.episode_over:
                    # self.model.eval()
                    rgb = observations["rgb"]

                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)
                    
                    info = env.get_metrics()
                    
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    # import ipdb; ipdb.set_trace()
                    # print(f"action_seq = {action_seq}", flush=True)
                    if len(action_seq) == 0:
                        
                        history_len = len(rgb_list) - 1
                        if history_len <= self.num_frames:
                            history_images = rgb_list[:history_len]
                            images = history_images + [rgb_list[-1]]
                        else:
                            indices = np.linspace(0, history_len, self.num_frames + 1, dtype=int)
                            images = [rgb_list[i] for i in indices]

                        llm_outputs = self.model.call_model(images, episode_instruction, step_id)[0]
                        
                        # outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000)
                        # output_ids = outputs.sequences
                        # llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        # action_seq = self.parse_actions(llm_outputs)
                        if llm_outputs in self.actions2idx:
                            action_seq = list(self.actions2idx[llm_outputs])
                        else:
                            action_seq = [0]

                        print('actions', action_seq, flush=True)
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                    action = action_seq.pop(0)
                    if step_id >= 400:
                        action = 0
                    
                    observations = env.step(action)

                    # try:
                    #     info = env.get_metrics()
                    #     if info['top_down_map'] is not None:
                    #         tdm = info["top_down_map"]["map"]
                    #         fog = info["top_down_map"]["fog_of_war_mask"]
                    #         agent_pos = info["top_down_map"]["agent_map_coord"]
                    #         agent_angle = info["top_down_map"]["agent_angle"]

                    #         tdm_colored = maps.colorize_topdown_map(tdm, fog)

                    #         maps.draw_agent(
                    #             tdm_colored,
                    #             agent_pos,
                    #             agent_angle,
                    #             agent_radius_px=6
                    #         )

                    #         debug_image_path = os.path.join(debug_dir, f'step_{step_id:03d}_action_{action}.png')
                    #         Image.fromarray(tdm_colored).save(debug_image_path)
                    # except Exception as e:
                    #     print(f"Debug map save failed: {e}")

                    step_id += 1
                        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
    
class VLN_Inference:
    def __init__(self, model_path, device="cuda"):
        config = AutoConfig.from_pretrained(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="eager",
                dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def call_model(self, observations, task, step_id):
        messages = [
            {
                "role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
            }
        ]
        
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        
        visual = observations
        if isinstance(visual, Image.Image): 
            messages.append({
                "role": "user", 
                "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]
            })
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
            image_content = []
            image_count = 0
            for v in visual:
                # if add_frame_index:
                    # image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                image_content.append({"type": "image", "image": v})
                image_count += 1
            messages.append({
                "role": "user", 
                "content": image_content + [{"type": "text", "text": context}]
            })
        else:
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": context}]
            })

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs = []
        if isinstance(visual, Image.Image):
            image_inputs = [visual]
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            image_inputs = list(visual)
        
        inputs = self.processor(
            text=text,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True
        )
        
        inputs = inputs.to(self.device)
    
        # TODO: Set generation parameters
        # if "max_new_tokens" not in gen_kwargs:
        #     gen_kwargs["max_new_tokens"] = 24
        # if "temperature" not in gen_kwargs:
        #     gen_kwargs["temperature"] = 0
        # if "top_p" not in gen_kwargs:
        #     gen_kwargs["top_p"] = None
        # if "num_beams" not in gen_kwargs:
        #     gen_kwargs["num_beams"] = 1
        
        with torch.no_grad():
            # outputs = self.model.generate(
            #     **inputs,
            #     do_sample=True if gen_kwargs["temperature"] > 0 else False,
            #     temperature=gen_kwargs["temperature"],
            #     top_p=gen_kwargs["top_p"],
            #     num_beams=gen_kwargs["num_beams"],
            #     max_new_tokens=gen_kwargs["max_new_tokens"],
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=self.tokenizer.pad_token_id,
            # )
            outputs = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=10000)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers
   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
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
    
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    model = VLN_Inference(args.model_path, device=args.device)
    
    model.model.requires_grad_(False)
    model.model.to(local_rank)
    evaluate(model, args)


def evaluate(model, args):
    # model.eval()
    
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
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
