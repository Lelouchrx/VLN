# VLN

## 0. Conda 环境
- `vln / vllm`：`python==3.10`
- `llamafactory`：`python==3.12`

## 1. llamafactory
```bash
pip install -e ".[torch,metrics]" --no-build-isolation -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

## 2. vllm
```bash
pip install vllm==0.16.0
bash VLN/scripts/vllm_qwenvln.sh
```

## 3. 仿真/交互环境（vln）
```bash
pip install -r requirements.txt
```

## 4.Habitat安装
```bash
bash tool/habitat.sh
```

## 5. VLN-CE Episodes

Download the VLN-CE episodes and extract them into the `data/datasets/` directory:

- [r2r](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) (Rename `R2R_VLNCE_v1-3_preprocessed/` -> `r2r/`)
- [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (Rename `RxR_VLNCE_v0/` -> `rxr/`)
- [scalevln](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/ScaleVLN/scalevln_subset_150k.json.gz) (Follow the StreamVLN to convert a subset of the ScaleVLN dataset into the VLN-CE format.)

## 6. 工具
```bash
# 评测
 python vln/qwen3vln_eval.py   --use_vllm   --vllm_base_url http://10.176.62.171:8003/v1   --vllm_model_name qwen3vl   --habitat_config_path config/vln_r2r.yaml   --eval_split val_unseen   --output_path ./results/qwen3vln_eval_2b_all  --temperature 0.3 --use_collision_prompt --parallel_envs 16 --save_sharegpt

# 轨迹处理
conda activate vln
python tool/trajectory_making.py

# 下载 MP3D
conda activate vln
python tool/download_mp.py

# 下载 ScanNetv2
conda activate vln
python tool/download_scannetv2.py
```

## 7. dagger运行
```bash
bash scripts/habitat.sh
```
