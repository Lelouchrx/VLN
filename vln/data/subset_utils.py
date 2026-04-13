def subset_episodes_to_triplets(episodes, pass_k: int):
    # Convert episodes to (scene, episode, trial) triplets
    triplets = set()
    for scene_id, ep_id in episodes:
        ep_str = str(int(ep_id))
        for trial_id in range(max(1, int(pass_k))):
            triplets.add((scene_id, ep_str, trial_id))
    return triplets


def parse_subset_episode_string(spec: str):
    # Parse "scene:ep;scene:ep" string into list of pairs
    pairs = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 2:
            raise ValueError(f"Bad subset episode {part!r}, expected scene:episode_id")
        pairs.append((bits[0].strip(), str(int(bits[1]))))
    return pairs


def parse_subset_triplet_string(spec: str):
    # Parse "scene:ep:trial;..." string into set of triplets
    triplets = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(f"Bad subset triplet {part!r}, expected scene:episode_id:trial_id")
        triplets.append((bits[0].strip(), str(int(bits[1])), int(bits[2])))
    return set(triplets)

