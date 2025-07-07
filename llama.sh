
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