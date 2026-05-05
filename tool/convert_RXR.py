import json
import gzip

RXR_EPISODES_PATH = "path/to/rxr/val_unseen/val_unseen_guide.json.gz"
NEW_RXR_EPISODES_PATH "path/to/rxr/val_unseen/val_unseen_guide_en.json.gz"

episodes_modified = {}
with gzip.open(RXR_EPISODES_PATH, 'rb') as f:
    episodes = json.loads(f.read().decode('utf-8'))

# Extract episodes with "en" in the language field
for ep in episodes["episodes"]:
    if "en" in ep["instruction"]["language"]:
        ep.update(instruction=dict(
                instruction_text=ep["instruction"]["instruction_text"],
                instruction_tokens=[],
            ),
        )
        episodes_modified["episodes"] = episodes_modified.get("episodes", []) + [ep]

# Append empty vocabulary to match the expected R2R format
episodes_modified["instruction_vocab"] = {
        "word_list": [],
        "word2idx_dict": {},
        "stoi": {},
        "itos": [],
        "num_vocab": 0,
        "UNK_INDEX": 1,
        "PAD_INDEX": 0
    }

# Save the modified episodes to a new gzip file
with gzip.open(NEW_RXR_EPISODES_PATH, 'wt', encoding='utf-8') as f:
    json.dump(episodes_modified, f)
