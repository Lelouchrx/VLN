import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}
# Trajectory / VLN datasets (paths relative to project root, e.g. StreamVLN)
TRAJECTORY_R2R = {
    "annotation_path": "data/trajectory_data/R2R/annotations_v1-3.json",
    "data_path": "data/trajectory_data/R2R",
}
TRAJECTORY_RXR = {
    "annotation_path": "data/trajectory_data/RxR/annotations.json",
    "data_path": "data/trajectory_data/RxR",
}
TRAJECTORY_ENVDROP = {
    "annotation_path": "data/trajectory_data/EnvDrop/annotations.json",
    "data_path": "data/trajectory_data/EnvDrop",
}
DAGGER_R2R = {
    "annotation_path": "data/dagger_data/R2R/annotations.json",
    "data_path": "data/dagger_data/R2R",
}
DAGGER_RXR = {
    "annotation_path": "data/dagger_data/RxR/annotations.json",
    "data_path": "data/dagger_data/RxR",
}
DAGGER_ENVDROP = {
    "annotation_path": "data/dagger_data/EnvDrop/annotations.json",
    "data_path": "data/dagger_data/EnvDrop",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "data/trajectory_data/R2R": TRAJECTORY_R2R,
    "data/trajectory_data/RxR": TRAJECTORY_RXR,
    "data/trajectory_data/EnvDrop": TRAJECTORY_ENVDROP,
    "data/dagger_data/R2R": DAGGER_R2R,
    "data/dagger_data/RxR": DAGGER_RXR,
    "data/dagger_data/EnvDrop": DAGGER_ENVDROP,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
