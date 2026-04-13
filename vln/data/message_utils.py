import base64
import io
from typing import List, Tuple

from PIL import Image


def _pil_to_data_url(img: Image.Image) -> str:
    # Convert PIL image to base64 data URL
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _data_url_to_pil(data_url: str) -> Image.Image:
    # Decode base64 data URL back to PIL Image
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid data url")
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _strip_image_tag(text: str) -> str:
    # Remove <image> prefix from text for local processor
    if not isinstance(text, str):
        return str(text)
    if text.startswith("<image>\n"):
        return text[len("<image>\n") :]
    if text.startswith("<image>"):
        return text[len("<image>") :].lstrip()
    return text


def build_user_image_message(text: str, rgb):
    # Build <image> prefixed message with base64 image URL
    img = Image.fromarray(rgb).convert("RGB")
    clean_text = (text or "").strip()
    return {
        "role": "user",
        "content": f"<image>\n{clean_text}",
        "image_url": _pil_to_data_url(img),
    }


def to_local_chat_and_images(messages) -> Tuple[List[dict], List[Image.Image]]:
    # Convert messages for local Qwen3VL processor (extract images)
    chat_msgs = []
    images = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        image_url = m.get("image_url")
        if image_url and role == "user":
            img = _data_url_to_pil(image_url)
            images.append(img)
            chat_msgs.append(
                {
                    "role": role,
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": _strip_image_tag(content)},
                    ],
                }
            )
        else:
            chat_msgs.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return chat_msgs, images


def to_vllm_chat_messages(messages):
    # Convert messages to OpenAI-compatible format for vLLM (base64 image_url)
    chat_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        image_url = m.get("image_url")
        if image_url and role == "user":
            chat_msgs.append(
                {
                    "role": role,
                    "content": [
                        {"type": "text", "text": content if isinstance(content, str) else str(content)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )
        else:
            chat_msgs.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return chat_msgs

