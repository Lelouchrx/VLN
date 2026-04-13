import base64
import io
import json
import os
import time

from PIL import Image


def _save_data_url_image(data_url: str, save_path: str):
    # Decode and save base64 data URL as PNG
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(save_path)


def dump_sharegpt_trial(state, output_path: str, epoch: int):
    # Save trial as ShareGPT format with images and metrics
    key = f"{state['scene_id']}_{state['episode_id']}_trial{state['trial_id']}"
    root = os.path.join(output_path, "sharegpt", f"vis_{epoch}", key)
    os.makedirs(root, exist_ok=True)
    img_idx = 0
    conversations = []
    system_prompt = None

    for m in state["messages"]:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system_prompt = content if isinstance(content, str) else str(content)
            continue
        if role == "user":
            parts = []
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(str(content))
            image_url = m.get("image_url")
            if isinstance(image_url, str) and image_url.startswith("data:image"):
                fn = f"image_{img_idx:04d}.png"
                _save_data_url_image(image_url, os.path.join(root, fn))
                parts.append(f"<image>{fn}</image>")
                img_idx += 1
            conversations.append({"from": "human", "value": "\n".join(parts)})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content if isinstance(content, str) else str(content)})

    payload = {
        "id": key,
        "system": system_prompt,
        "conversations": conversations,
        "metrics": {
            "scene_id": state["scene_id"],
            "episode_id": state["episode_id"],
            "trial_id": state["trial_id"],
            "steps": state["step_id"],
        },
    }
    out_json = os.path.join(root, "sharegpt.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ShareGPT] wrote {out_json}", flush=True)


def dump_vllm_error_sharegpt(messages, vllm_messages, resp, meta: dict):
    ts = time.time_ns()
    key = f"{meta['scene_id']}_ep{meta['episode_id']}_trial{meta['trial_id']}_step{meta['step_id']}_{ts}"
    root = os.path.join(meta["output_path"], "error", key)
    os.makedirs(root, exist_ok=True)
    img_idx = 0
    conversations = []
    system_prompt = None
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system_prompt = content if isinstance(content, str) else str(content)
            continue
        if role == "user":
            parts = []
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(str(content))
            image_url = m.get("image_url")
            if isinstance(image_url, str) and image_url.startswith("data:image"):
                fn = f"image_{img_idx:04d}.png"
                _save_data_url_image(image_url, os.path.join(root, fn))
                parts.append(f"<image>{fn}</image>")
                img_idx += 1
            conversations.append({"from": "human", "value": "\n".join(parts)})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content if isinstance(content, str) else str(content)})
    resp_dump = None
    try:
        if hasattr(resp, "model_dump"):
            resp_dump = resp.model_dump()
        elif hasattr(resp, "dict"):
            resp_dump = resp.dict()
        else:
            resp_dump = str(resp)
    except Exception:
        resp_dump = str(resp)
    payload = {
        "id": key,
        "system": system_prompt,
        "conversations": conversations,
        "vllm_messages": vllm_messages,
        "openai_response": resp_dump,
        "error": meta.get("error_tag", "empty_vllm_text"),
        "meta": {
            "scene_id": meta["scene_id"],
            "episode_id": meta["episode_id"],
            "trial_id": meta["trial_id"],
            "step_id": meta["step_id"],
            "worker_id": meta["worker_id"],
            "is_stuck_user_prompt": meta.get("is_stuck_user_prompt"),
            "collision_is_collision": meta.get("collision_is_collision"),
            "episode_instruction": meta.get("episode_instruction"),
        },
    }
    out_json = os.path.join(root, "sharegpt.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

