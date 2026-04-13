# VLN

## 0. Conda 环境
- `vln / vllm`：`python==3.10`
- `llamafactory`：`python==3.12`

## 2. llamafactory
```bash
pip install -e ".[torch,metrics]" --no-build-isolation -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

## 3. vllm
```bash
pip install vllm==0.16.0
bash VLN/scripts/vllm_qwenvln.sh
```

## 4. 仿真/交互环境（vln）
```bash
pip install -r requirements.txt
```

## 5.Habitat安装
```bash
bash tool/habitat.sh
```

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
