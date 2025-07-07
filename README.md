## Installation

Step 1: Create a new conda environment:

```
conda create -n prune_llm python=3.9
conda activate prune_llm
```

Step 2: Install relevant packages

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```

## MRP
```
model=PATH_model
model_arch=llama
for method in prune_wanda_outlier_srp
do
CUDA_VISIBLE_DEVICES=3,4,5 python -u main.py \
    --model_name_or_path /data/llama-13b-hf/ \
    --Lamda 0.08 \
    --Hyper_m 5 \
    --model /data/llama-13b-hf/ \
    --prune_method ${method} \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save save_test/ \
    --is_block 

done
```
