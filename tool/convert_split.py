import argparse
import json
from pathlib import Path


def split_episode_by_rounds(episode):
    """Split one episode into cumulative samples by assistant turns."""
    messages = episode.get("messages", [])
    images = episode.get("images", [])

    if len(messages) < 3:
        return []

    # Keep header fixed: system + first user instruction.
    base_messages = messages[:2]
    split_samples = []
    assistant_count = 0

    for idx, msg in enumerate(messages):
        if msg.get("from") != "assistant":
            continue

        assistant_count += 1
        if assistant_count > len(images):
            break

        split_samples.append(
            {
                "episode_id": episode.get("episode_id"),
                "dataset": episode.get("dataset"),
                "messages": base_messages + messages[2 : idx + 1],
                "images": images[:assistant_count],
            }
        )

    return split_samples


def split_annotations(input_path, output_path):
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as f:
        episodes = json.load(f)

    split_results = []
    for episode in episodes:
        split_results.extend(split_episode_by_rounds(episode))

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(split_results, f, indent=2, ensure_ascii=False)

    print(f"Input episodes: {len(episodes)}")
    print(f"Split samples: {len(split_results)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data/R2R/annotations.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data/R2R/annotations_split.json",
    )
    args = parser.parse_args()

    split_annotations(args.input, args.output)
