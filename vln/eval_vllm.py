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


SYSTEM_PROMPT = "You are an autonomous navigation assistant. Your task is to follow the given instruction in the environment. At each step, output the next four actions to take, exactly four, separated by commas. Use only the following actions: Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop."
SAVE_PATH = "/home/cs22-hongly/VLN/results/qwen3vln_8b_r2rrxrenv"
VISUAL_PROMPTS = [
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
                    enable_early_stop, use_collision_prompt, skip_result_file) -> None:
 
    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(
        api_key, 
        base_url, 
        result_path, 
        forward_distance, 
        turn_angle, 
        max_action_history, 
        resolution_ratio, 
        num_generations,
        use_collision_prompt=use_collision_prompt)

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
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id)

            if enable_early_stop and (continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS):
                action = {"action": 0}

            
            iter_step+=1
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

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, num_generations = 1,
                    require_map=True, use_collision_prompt=False):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.use_collision_prompt = use_collision_prompt
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        
        self.temperature = 0.3
        self.top_p = 0.95
        self.top_k = 20
        self.max_tokens = 512
        self.actions2idx = {
            "MOVE FORWARD 25 CM": [1],
            "TURN LEFT 15 DEGREES": [2],
            "TURN RIGHT 15 DEGREES": [3],
            "STOP": [0],
        }
        self.idx2action_text = {
            0: "Stop",
            1: "Move forward 25 cm",
            2: "Turn left 15 degrees",
            3: "Turn right 15 degrees",
        }

        # self.promt_template_first = "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
        #     "which could involve turning left or right by a specific degree or moving forward a certain distance."
        # self.promt_template = "Analyze this series of images to decide your next move, "\
        #     "which could involve turning left or right by a specific degree or moving forward a certain distance."
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        self.sharegpt_conversations = []
        self.is_first_round = True

        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def predict_inference(self):

        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={"top_k": self.top_k},
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns, flags=re.IGNORECASE)
        matches = regex.findall(output)
        actions = [self.actions2idx[match.upper()] for match in matches]
        return [a for group in actions for a in group]

    def actions_to_text(self, actions):
        return ", ".join(self.idx2action_text.get(action, "Stop") for action in actions)

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

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
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
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
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
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
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        # rgb_ = rgb_.resize((308,252))
        self.rgb_list.append(rgb_)
        # do not cut down rgb list while using uniform sampling
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        # for observation1+observation2 action style
        # self.conversations = self.conversations[:1]
        content = []

        if self.use_collision_prompt and info.get("collisions", {}).get("is_collision", False):
            user_text = (
                "STATUS: the robot is STUCK; it collided with an obstacle during the last action block. "
                "Think about how to get unstuck and continue toward that goal. Do NOT output your reasoning. "
                "Must output exactly four actions to take next, separated by commas. Use only the following actions: "
                "Move forward 25 cm, Turn left 15 degrees, Turn right 15 degrees, Stop."
            )
        elif self.is_first_round:
            user_text = f"Your instruction is '{observations['instruction']['text']}' "
        else:
            user_text = random.choice(VISUAL_PROMPTS)
        content.append({"type": "text", "text": user_text})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        # if self.is_first_round:
        #     prompt_tail = self.promt_template_first.format(observations["instruction"]["text"])
        # else:
        #     prompt_tail = self.promt_template
        # content.append({"type": "text", "text": prompt_tail})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        user_text_parts = []
        for c in content:
            if c.get("type") == "text":
                user_text_parts.append(c.get("text", ""))
        user_text = " ".join([part.strip() for part in user_text_parts if part.strip()])

        navigation = self.predict_inference()
        action_seq = self.parse_actions(navigation)
        if len(action_seq) == 0:
            action_seq = self.parse_actions(navigation.upper().replace(" ", ""))
        if len(action_seq) == 0:
            digit_actions = re.findall(r"[0-3]", navigation)
            action_seq = [int(a) for a in digit_actions]
        if len(action_seq) == 0:
            action_seq = [0]
        assistant_text = self.actions_to_text(action_seq)

        self.conversations.append({
            "role": "assistant",
            "content": assistant_text,
        })

        self.sharegpt_conversations.append({
            "from": "human",
            "value": user_text
        })
        self.sharegpt_conversations.append({
            "from": "gpt",
            "value": assistant_text
        })
        self.is_first_round = False

        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], assistant_text)
            self.topdown_map_list.append(img)

        self.pending_action_list.extend(action_seq)
        if len(self.pending_action_list) == 0:
            self.pending_action_list.append(0)

        return {"action": self.pending_action_list.pop(0)}


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    parser.add_argument("--enable-early-stop", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable heuristic early stop based on repeated no-progress rotation or max steps.")
    parser.add_argument("--use-collision-prompt", action="store_true", default=False,
                        help="Prompt the model with a stuck message when collision is detected.")
    parser.add_argument("--skip-result-file", type=str, default=None,
                        help="Optional extra result json/jsonl(or dir) to skip already-finished episodes.")
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
        # self.config.habitat.task.measurements.success.success_distance=3.0
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
                args.use_collision_prompt, args.skip_result_file)
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