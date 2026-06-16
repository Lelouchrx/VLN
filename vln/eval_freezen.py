"""
eval_freezen.py — standalone gN + fN ("freezen") history-sampling experiment.

STRATEGY (env-configurable):
  SAMPLE_STRATEGY=freezen   FREEZE_N=<g>   FREEZE_PERIOD=<f>
  - When history length L <= 8: send all (uniform-8), no freezing.
  - When L > 8 each inference round builds a fresh uniform-8 over the whole history,
    then:
        front N  = full[:N]   -> FROZEN; only re-sampled ("re-anchored") every
                                 FREEZE_PERIOD rounds. Because these frames (and
                                 their JPEG+base64 bytes) stay byte-identical between
                                 re-anchors, the prompt prefix [system + intro + front-N]
                                 is identical across those rounds, so vLLM's prefix
                                 cache (and multimodal cache) SKIP re-encoding /
                                 re-prefilling them.
        back  (8-N) = full[N:] -> re-sampled EVERY round (tracks the recent trajectory).
    -> "gN + fN" = freeze g frames, re-anchor every f rounds.

  This generalizes v3_front4uni (which is g=4, f=4).

SPEED INSTRUMENTATION (see the long comment on `infer_time_s` below and the
vLLM /metrics helper at the bottom):
  - result.json per episode carries `infer_time_s` = cumulative CLIENT-side wall time
    of the chat.completions calls for that episode. This INCLUDES server queue wait.
  - For TRUE inference time excluding queue, read vLLM /metrics
    (request_prefill_time / request_decode_time / request_inference_time), which are
    SERVER-side and exclude queue. Use snapshot_vllm_metrics() around a run.

Env:
  OPENAI_API_KEY, OPENAI_API_BASE   (vLLM OpenAI endpoint, prefix caching ON)
  SAMPLE_STRATEGY=freezen|uniform   FREEZE_N (default 4)   FREEZE_PERIOD (default 3)
"""
import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
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
    CollisionsMeasurementConfig, FogOfWarConfig, TopDownMapMeasurementConfig,
)
from PIL import Image
import multiprocessing as mp
import time, math
from openai import OpenAI
import base64
from io import BytesIO
import urllib.request

SYSTEM_PROMPT = "You are a helpful assistant."


def encode_image_base64(image):
    # Cache JPEG+base64 ON the PIL image object so frozen frames yield byte-identical
    # bytes across rounds -> vLLM prefix/MM cache actually hit. This is REQUIRED for the
    # freezing speed benefit; without it every round re-encodes and the prefix differs.
    cached = getattr(image, "_b64_cache", None)
    if cached is not None:
        return cached
    buf = BytesIO()
    image.save(buf, format="JPEG")
    enc = base64.b64encode(buf.getvalue()).decode("utf-8")
    try:
        image._b64_cache = enc
    except (AttributeError, TypeError):
        pass
    return enc


def seed_all():
    np.random.seed(41); random.seed(41)


def snapshot_vllm_metrics(base_url):
    """Server-side cumulative counters. Returns a dict; take a snapshot before and
    after a run and diff. Two kinds of entries:
      - timing histograms  -> {name: (sum, count)}; queue is EXCLUDED from
        prefill/decode/inference. delta_sum/delta_count = mean per-request time.
      - cache/req counters  -> {name: value} (float). Diff across the run gives
        queries/hits over THIS run; hits/queries = hit rate. prefix_cache_* and
        mm_cache_* are counted in tokens/items processed by vLLM's caches.
    Returns {} on failure (then speed/cache fields are simply omitted from summary)."""
    url = base_url.rstrip("/").replace("/v1", "") + "/metrics"
    out = {}
    try:
        txt = urllib.request.urlopen(url, timeout=5).read().decode()
        for key in ("request_queue_time_seconds", "request_prefill_time_seconds",
                    "request_decode_time_seconds", "request_inference_time_seconds",
                    "time_to_first_token_seconds"):
            s = re.search(rf"^vllm:{key}_sum\S* ([0-9.eE+-]+)", txt, re.M)
            c = re.search(rf"^vllm:{key}_count\S* ([0-9.eE+-]+)", txt, re.M)
            if s and c:
                out[key] = (float(s.group(1)), float(c.group(1)))
        # Counters (summed across label series, e.g. multiple engine ids). Metric
        # names vary by vLLM version, so try a few aliases and keep the first found.
        counter_aliases = {
            "prefix_cache_queries": ("gpu_prefix_cache_queries", "prefix_cache_queries"),
            "prefix_cache_hits":    ("gpu_prefix_cache_hits", "prefix_cache_hits"),
            "mm_cache_queries":     ("mm_cache_queries",),
            "mm_cache_hits":        ("mm_cache_hits",),
        }
        for out_key, names in counter_aliases.items():
            for name in names:
                vals = re.findall(rf"^vllm:{name}(?:_total)?(?:\{{[^}}]*\}})? ([0-9.eE+-]+)",
                                  txt, re.M)
                if vals:
                    out[out_key] = sum(float(v) for v in vals)
                    break
    except Exception:
        pass
    return out


def load_done_result_keys(result_path):
    done = set()
    f = os.path.join(result_path, "result.json")
    if not os.path.exists(f):
        return done
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            it = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "scene_id" in it and "episode_id" in it and "episode_instruction" in it:
            done.add((it["scene_id"], str(it["episode_id"]), it["episode_instruction"],
                      int(it.get("trial_id", 0)), int(it.get("trial_total", 1))))
    return done


def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path,
                   forward_distance, turn_angle, max_action_history, resolution_ratio,
                   pass_k, max_episodes) -> None:
    env = Env(config.habitat, dataset)
    agent = NaVIDA_Agent(api_key, base_url, result_path, forward_distance, turn_angle,
                         max_action_history, resolution_ratio)
    done = load_done_result_keys(result_path)
    EARLY_STOP_ROTATION, EARLY_STOP_STEPS = 25, 400

    eps = list(env.episodes)
    if max_episodes and max_episodes > 0:        # fix N episodes (per worker) for speed tests
        eps = eps[:max_episodes]
    for target_ep in eps:
        for trial_id in range(pass_k):
            t0 = time.time()
            env.current_episode = target_ep
            obs = env.reset(); iter_step = 0; agent.reset()
            rot_count, last_dtg = 0, 999
            scene_id = env.current_episode.scene_id.split('/')[-2]
            episode_id = env.current_episode.episode_id
            instr = obs["instruction"]["text"]
            if (scene_id, str(episode_id), instr, trial_id, pass_k) in done:
                result_queue.put({"t_episode": time.time() - t0, "skipped": 1}); continue
            while not env.episode_over:
                info = env.get_metrics()
                if info["distance_to_goal"] != last_dtg:
                    last_dtg = info["distance_to_goal"]; rot_count = 0
                else:
                    rot_count += 1
                action = agent.act(obs, info, env.current_episode.episode_id)
                if rot_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                    action = {"action": 0}
                iter_step += 1
                obs = env.step(action)
            info = env.get_metrics()
            result = {
                "scene_id": scene_id,
                "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                "trial_id": trial_id, "trial_total": pass_k,
                "success": info["success"], "spl": info["spl"], "os": info["oracle_success"],
                "ne": info["distance_to_goal"], "steps": iter_step,
                "n_rounds": agent._stat_rounds,
                # CLIENT-side cumulative chat.completions wall time for this episode.
                # NOTE: includes server queue wait -> NOT pure inference. For pure
                # inference (excl. queue) use vLLM /metrics request_inference_time.
                "infer_time_s": round(agent._stat_infer_time, 3),
                "episode_instruction": instr,
            }
            if agent.profile_rt:
                result["rt_cold"] = agent._rt["cold"]
                result["rt_warm"] = agent._rt["warm"]
            with open(os.path.join(result_path, "result.json"), "a") as f:
                f.write(json.dumps(result) + "\n")
            done.add((scene_id, str(episode_id), instr, trial_id, pass_k))
            d = {"t_episode": time.time() - t0, "success": info["success"], "spl": info["spl"]}
            result_queue.put(d)


class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, turn_angle,
                 max_action_history, resolution_ratio):
        self.result_path = result_path
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        os.makedirs(result_path, exist_ok=True)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=600)
        self.model = self.client.models.list().data[0].id
        self.temperature, self.top_p, self.max_tokens = 0.3, 0.95, 256
        self.base_url = base_url
        # round-type speed split: per-round /metrics snapshot, COLD(re-anchor) vs WARM(frozen).
        # Server-side delta is only attributable at NO concurrency (run with --split-num 1).
        self.profile_rt = os.environ.get("ROUND_TYPE_PROFILE", "0") == "1"

        # ---- gN + fN config ----
        self.sample_strategy = os.environ.get("SAMPLE_STRATEGY", "freezen")
        self.freeze_n = int(os.environ.get("FREEZE_N", "4"))        # g: frozen front count
        self.freeze_period = int(os.environ.get("FREEZE_PERIOD", "3"))  # f: re-anchor period

        self.promt_template = ("Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward a certain distance.")
        self.reset()

    def reset(self):
        self.pending_action_list = []
        self.rgb_list = []
        self._fz_front = None        # frozen front-N anchors
        self.step_count = 0
        self._stat_rounds = 0
        self._stat_infer_time = 0.0  # cumulative chat.completions wall time (client-side)
        self._cur_round_cold = None  # set each round in _select_history (freezen only)
        # round-type speed buckets (server-side /metrics sum/count deltas, per episode)
        z = lambda: {"prefill_s": 0.0, "prefill_c": 0.0, "decode_s": 0.0, "decode_c": 0.0,
                     "infer_s": 0.0, "infer_c": 0.0, "n": 0}
        self._rt = {"cold": z(), "warm": z()}
        self.conversations = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

    def uniform_sample_with_ends(self, data, n):
        if len(data) <= n:
            return data
        idx = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in idx]

    def _select_history(self):
        """Returns the history frames to send (current frame is sent separately)."""
        hist = self.rgb_list[:-1]
        if not hist:
            return []
        if self.sample_strategy == "freezen":
            N = max(1, min(8, self.freeze_n)); K = max(1, self.freeze_period)
            full = self.uniform_sample_with_ends(hist, 8)   # fresh uniform-8 each round
            if self._fz_front is None or (self.step_count % K) == 1:
                self._fz_front = full[:N]                    # (re-)anchor the frozen front
                self._cur_round_cold = True                  # COLD round: front-N re-prefilled
            else:
                self._cur_round_cold = False                 # WARM round: front-N reused from cache
            return self._fz_front + full[N:]                 # frozen front + fresh back
        # default: plain uniform-8 (no freezing -> no warm rounds; leave round-type unset)
        self._cur_round_cold = None
        return self.uniform_sample_with_ends(hist, 8)

    def predict_inference(self):
        prof = self.profile_rt and self._cur_round_cold is not None
        m0 = snapshot_vllm_metrics(self.base_url) if prof else None
        t = time.time()
        out = self.client.chat.completions.create(
            messages=self.conversations, model=self.model,
            max_completion_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        self._stat_infer_time += (time.time() - t)   # client-side wall time (incl. queue)
        if prof:
            time.sleep(0.02)                          # let /metrics settle (valid only w/o concurrency)
            m1 = snapshot_vllm_metrics(self.base_url)
            b = self._rt["cold" if self._cur_round_cold else "warm"]
            for key, nm in (("request_prefill_time_seconds", "prefill"),
                            ("request_decode_time_seconds", "decode"),
                            ("request_inference_time_seconds", "infer")):
                if m0 and m1 and key in m0 and key in m1:
                    b[nm + "_s"] += m1[key][0] - m0[key][0]
                    b[nm + "_c"] += m1[key][1] - m0[key][1]
            b["n"] += 1
        txt = re.sub(r'<think>.*?</think>', '', out.choices[0].message.content or "", flags=re.DOTALL)
        return txt.strip()

    def extract_multi_result(self, output):
        return [self.extract_result(s) for s in output.split(', ')]

    def extract_result(self, output):
        m = re.search(r'<answer>(.*?)</answer>', output)
        output = (m.group(1).strip() if m else output.strip()).lower()
        if "stop" in output:
            return 0, None
        if "forward" in output:
            mm = re.search(r'-?\d+', output); return 1, (float(mm.group()) if mm else self.forward_distance)
        if "left" in output:
            mm = re.search(r'-?\d+', output); return 2, (float(mm.group()) if mm else self.turn_angle)
        if "right" in output:
            mm = re.search(r'-?\d+', output); return 3, (float(mm.group()) if mm else self.turn_angle)
        return None, None

    def act(self, observations, info, episode_id):
        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        self.rgb_list.append(Image.fromarray(rgb.astype('uint8')).convert('RGB'))
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.pending_action_list:
            return {"action": self.pending_action_list.pop(0)}

        self.conversations = self.conversations[:1]
        self.step_count += 1
        content = [{"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'}]
        if len(self.rgb_list) > 1:
            for im in self._select_history():
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(im)}"}})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": self.promt_template.format(observations["instruction"]["text"]).split('current observation')[1]})
        self.conversations.append({"role": "user", "content": content})
        self._stat_rounds += 1

        navigation = self.predict_inference()
        result = self.extract_multi_result(navigation)[:2]   # execute first 2 sub-actions
        for ai, num in result:
            if ai == 0:
                self.pending_action_list.append(0)
            elif ai == 1:
                for _ in range(min(3, round(num / self.forward_distance))): self.pending_action_list.append(1)
            elif ai == 2:
                for _ in range(min(3, round(num / self.turn_angle))): self.pending_action_list.append(2)
            elif ai == 3:
                for _ in range(min(3, round(num / self.turn_angle))): self.pending_action_list.append(3)
            if ai is None or not self.pending_action_list:
                self.pending_action_list.append(random.randint(1, 3))
        return {"action": self.pending_action_list.pop(0)}


def main():
    seed_all()
    p = argparse.ArgumentParser()
    p.add_argument("--exp-config", required=True)
    p.add_argument("--split-num", type=int, required=True)
    p.add_argument("--resolution-ratio", type=float, default=1.0)
    p.add_argument("--result-path", required=True)
    p.add_argument("--forward-distance", type=int, default=25)
    p.add_argument("--turn-angle", type=int, default=15)
    p.add_argument("--max-action-history", type=int, default=200)
    p.add_argument("--pass-k", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=int(os.environ.get("MAX_EPISODES", "0")),
                   help="cap episodes PER WORKER (0 = all). For fixed-N speed tests, e.g. 100.")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert base_url is not None, "set OPENAI_API_BASE to the vLLM endpoint"

    config = get_config(args.exp_config)
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)
    num_episodes = len(dataset.episodes)
    if args.max_episodes and args.max_episodes > 0:   # per-worker cap -> effective total for the bar
        eff_episodes = sum(min(args.max_episodes, len(s.episodes)) for s in splits)
    else:
        eff_episodes = num_episodes

    # SPEED: snapshot server-side timing BEFORE the run (pure inference, excl. queue).
    m0 = snapshot_vllm_metrics(base_url)

    manager = mp.Manager(); q = manager.Queue(); procs = []
    for i in range(args.split_num):
        pr = mp.Process(target=evaluate_agent, args=(
            q, api_key, base_url, config, splits[i], args.result_path,
            args.forward_distance, args.turn_angle, args.max_action_history,
            args.resolution_ratio, args.pass_k, args.max_episodes), daemon=True)
        pr.start(); procs.append(pr)
    with tqdm(total=eff_episodes * args.pass_k, desc=f"freezen g{os.environ.get('FREEZE_N','4')}f{os.environ.get('FREEZE_PERIOD','3')}") as bar:
        done = 0
        while done < eff_episodes * args.pass_k:
            try:
                r = q.get(timeout=10); done += 1; bar.update(1); bar.set_postfix(**r)
            except Exception:
                if not any(pr.is_alive() for pr in procs):
                    break
    for pr in procs:
        pr.join()
    m1 = snapshot_vllm_metrics(base_url)

    # ---- aggregate SR + speed ----
    rf = os.path.join(args.result_path, "result.json")
    n_run = s_suc = tot_infer = tot_steps = 0
    traj = {}
    zc = lambda: {"prefill_s": 0.0, "prefill_c": 0.0, "decode_s": 0.0, "decode_c": 0.0,
                  "infer_s": 0.0, "infer_c": 0.0, "n": 0}
    rt_agg = {"cold": zc(), "warm": zc()}
    if os.path.exists(rf):
        for line in open(rf):
            line = line.strip()
            if not line or 'sucs_all' in line:
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in it for k in ("scene_id", "episode_id", "episode_instruction", "success", "steps")):
                continue
            n_run += 1; s_suc += float(it["success"])
            tot_infer += float(it.get("infer_time_s", 0)); tot_steps += int(it.get("steps", 0))
            k = (it["scene_id"], str(it["episode_id"]), it["episode_instruction"])
            traj.setdefault(k, []).append(float(it["success"]))
            for rt in ("cold", "warm"):              # accumulate round-type speed buckets
                b = it.get("rt_" + rt)
                if b:
                    for kk in rt_agg[rt]:
                        rt_agg[rt][kk] += b.get(kk, 0)
    pass_k = int(args.pass_k)
    summary = {
        "freeze_n": int(os.environ.get("FREEZE_N", "4")), "freeze_period": int(os.environ.get("FREEZE_PERIOD", "3")),
        "n_run": n_run,
        "avg_sr": (s_suc / n_run) if n_run else None,
        f"pass@{pass_k}": (sum(1 for v in traj.values() if max(v) >= 1) / len(traj)) if traj else None,
        # client-side mean (incl. queue):
        "mean_infer_s_ep": round(tot_infer / n_run, 3) if n_run else None,
        "mean_infer_per_step_s": round(tot_infer / tot_steps, 4) if tot_steps else None,
    }
    # server-side PURE inference (excl. queue), from /metrics delta over this run:
    if m0 and m1:
        for key, label in [("request_prefill_time_seconds", "prefill"),
                           ("request_decode_time_seconds", "decode"),
                           ("request_inference_time_seconds", "inference"),  # = prefill+decode, excl. queue
                           ("request_queue_time_seconds", "queue"),
                           ("time_to_first_token_seconds", "ttft")]:
            if key in m0 and key in m1:
                ds = m1[key][0] - m0[key][0]; dc = m1[key][1] - m0[key][1]
                summary[f"vllm_{label}_per_req_s"] = round(ds / dc, 4) if dc > 0 else None
        # reqs served over this run (count delta of the inference histogram):
        ik = "request_inference_time_seconds"
        if ik in m0 and ik in m1:
            summary["reqs"] = int(m1[ik][1] - m0[ik][1])
        # cache hit rates over this run = (hits delta) / (queries delta):
        for label, q, h in [("prefix_hit", "prefix_cache_queries", "prefix_cache_hits"),
                            ("mm_hit", "mm_cache_queries", "mm_cache_hits")]:
            if q in m0 and q in m1 and h in m0 and h in m1:
                dq = m1[q] - m0[q]; dh = m1[h] - m0[h]
                summary[label] = round(dh / dq, 4) if dq > 0 else None
    # round-type split: COLD (re-anchor) vs WARM (frozen) server-side per-round means
    if rt_agg["cold"]["n"] or rt_agg["warm"]["n"]:
        for rt in ("cold", "warm"):
            b = rt_agg[rt]
            summary[f"{rt}_rounds"] = b["n"]
            for nm in ("prefill", "decode", "infer"):
                summary[f"{rt}_{nm}_s"] = round(b[nm + "_s"] / b[nm + "_c"], 4) if b[nm + "_c"] > 0 else None
        cp, cc = rt_agg["cold"]["prefill_s"], rt_agg["cold"]["prefill_c"]
        wp, wc = rt_agg["warm"]["prefill_s"], rt_agg["warm"]["prefill_c"]
        if cc > 0 and wc > 0 and (wp / wc) > 0:
            summary["cold_warm_prefill_ratio"] = round((cp / cc) / (wp / wc), 3)
    print(json.dumps(summary, ensure_ascii=False))
    with open(rf, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
