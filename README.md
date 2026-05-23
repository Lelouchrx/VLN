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
conda activate vllm
bash scripts/vllm_qwenvln.sh
```

在 `scripts/vllm_qwenvln.sh` 中修改 `BASE_MODEL`、`ADAPTER`、`CUDA_VISIBLE_DEVICES` 与端口后启动服务。评测前在 `vln` 环境中设置（端口需与脚本一致）：

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://127.0.0.1:8001/v1
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

## 7. NaVIDA 评测（vLLM + Habitat）

通过 OpenAI 兼容 API 调用本地 vLLM，在 VLN-CE（R2R / RxR）上跑 NaVIDA 式多图导航评测。

| 脚本 | 说明 |
|------|------|
| `vln/eval_vllm.py` | **原版**：每条 episode 只评测 1 次，`trial_id=0`、`trial_total=1` |
| `vln/eval_vllm_navida.py` | **pass@k**：同一条轨迹跑 `k` 次，汇总 `pass@k` / `avg@k` |

**流程**：先 `bash scripts/vllm_qwenvln.sh` 启动 vLLM，再在仓库根目录修改 `scripts/eval_vllm.sh` 中的 `CONFIG_PATH`、`SAVE_PATH`、`OPENAI_API_BASE` 后执行：

```bash
conda activate vln
bash scripts/eval_vllm.sh
```

仓库内 `scripts/eval_vllm.sh` 默认调用 `eval_vllm_navida.py` 且 `--pass-k 4`；若要用原版单次评测，将其中 Python 入口改为 `eval_vllm.py` 并去掉 `--pass-k`。

**`eval_vllm.py` 命令示例（原版，单次）**

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://127.0.0.1:8001/v1

python vln/eval_vllm.py \
  --exp-config config/vln_r2r.yaml \
  --split-num 16 \
  --result-path ./results/navida_r2r_single \
  --forward-distance 25 \
  --turn-angle 15 \
  --max-action-history 200 \
  --num-generations 1
```

**`eval_vllm_navida.py` 命令示例（pass@k）**

```bash
python vln/eval_vllm_navida.py \
  --exp-config config/vln_r2r.yaml \
  --split-num 16 \
  --pass-k 4 \
  --result-path ./results/navida_r2r_pass4 \
  --forward-distance 25 \
  --turn-angle 15 \
  --max-action-history 200 \
  --num-generations 1
```

结果写入 `--result-path/result.json`（JSONL）：每行一条 episode（navida 含 `trial_id` / `trial_total`），**最后一行**为汇总指标。中断后重跑会自动跳过已完成条目。

## 8. dagger运行
```bash
bash scripts/habitat.sh
```
