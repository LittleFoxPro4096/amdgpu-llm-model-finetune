## 环境说明
操作系统: Windows 10
显卡: AMD Radeon RX 7900 XT
模型: Qwen/Qwen3.5-9B
环境: Miniforge3

## 创建环境
conda create -n amdgpu-torch python=3.12 -y
conda activate amdgpu-torch

## 安装依赖
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl
curl -L -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl
pip install --no-cache-dir rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl rocm-7.2.0.dev0.tar.gz
pip install --no-cache-dir torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl torchaudio-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl
pip install transformers accelerate datasets peft trl
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

## 生成训练集
python generate_jsonl.py

## 模型微调
python model_finetune.py

## 将模型转成GGUF格式
git clone https://github.com/ggml-org/llama.cpp.git
python llama.cpp\convert_hf_to_gguf.py "<path>\merged" --outfile "<path>\qwen3.5-it-9b-q8_0.gguf" --outtype q8_0
