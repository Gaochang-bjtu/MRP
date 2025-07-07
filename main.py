import argparse
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
# from importlib.metadata import version
from collections import defaultdict
from lib.prune_all import prune_wanda_outlier_structure_special,prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier, prune_wanda,prune_magnitude,prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
import sys
import time
print('# of gpus: ', torch.cuda.device_count())
from wanda import prune_wanda_outlier_srp, prune_mag_outlier_srp, prune_sparsegpt_outlier_srp
import logging

import json
import math

import random
from itertools import chain
from pathlib import Path

import datasets

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


#logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
def compupte_rate(args, model, genome):
    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    total_num = 0
    total_prune_num = 0
    index = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            total_num += W.numel()
            total_prune_num += W.numel()*genome[i]
            index = index + 1
    return total_prune_num / total_num
def get_llm(model, cache_dir="llm_weights"):
    if 'baichuan' in model:
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            #load_in_8bit = True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )        

    model.seqlen = 2048
    return model
def main():


    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--prune_method1", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')


########################################### for train
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--is_block",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--compute",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )   

    #### saving parameters #####
    
    parser.add_argument(
        "--method",
        type=str,
        default=None,

    )   
    


    #### data parameters #####
    
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )
    
    
     
    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=3, )
    
    parser.add_argument(
    "--outlier_by_activation", action="store_true", help="outlier_by_activation")  
    
    
    parser.add_argument(
    "--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")  
    parser.add_argument(
        '--initial_prune_ratio', 
        type=float,
        default=0.7, )   
        
    parser.add_argument(
        '--max_prune_ratio', 
        type=float,
        default=0.75, )    
    parser.add_argument(
        '--min_prune_ratio_step', 
        type=float,
        default=0, )    
    parser.add_argument(
        '--initial_prune_ratio_step', 
        type=float,
        default=0, )  
    args = parser.parse_args()
    model_name = args.model.split("/")[-2]
    if args.is_block:
        print("pruning block")
    else:
        print("pruning layer")
    print("model_name:", model_name)


    log_file_path = os.path.join("result" + '.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    #logging.info(f'sparsity_ratio: {args.sparsity_ratio}')
    print ("args.nsamples",args.nsamples)
    # Setting seeds for reproducibility
    #np.random.seed(args.seed)
    #torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "structured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model_name_or_path, args.cache_dir)
    #print ("model is =================================================================================")
    #print (model.__class__.__name__)
    
    if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    elif "llama3" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    elif "llama" in args.model or 'vicuna' in args.model:
    
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    elif 'baichuan' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)



    device = torch.device("cuda:0")

    layer_num = 0
    
    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer_num += 1
    

    if "30b" in args.model or "mixtral" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)



    print ("target sparsity", args.sparsity_ratio)   







    print("pruning starts")


    ############################ baseline   ############################
    if args.prune_method == "wanda":
        sparsity_ratio = [args.sparsity_ratio] * layer_num
        prune_wanda(args, model, sparsity_ratio, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "magnitude":
        sparsity_ratio = [args.sparsity_ratio] * layer_num
        prune_magnitude(args, model, sparsity_ratio, tokenizer, device, prune_n=prune_n, prune_m=prune_m)




    elif args.prune_method == "sparsegpt":
        sparsity_ratio = [args.sparsity_ratio] * layer_num
        prune_sparsegpt(args, model, tokenizer, device, sparsity_ratio, prune_n=prune_n, prune_m=prune_m)

    ############################ owl   ############################
    elif args.prune_method == "wanda_owl":

        prune_wanda_outlier(args, model, tokenizer,  device, prune_n=prune_n, prune_m=prune_m)
        #prune_wanda_outlier(args, model, tokenizer,  device, prune_n=prune_n, prune_m=prune_m)


    ############################ owl   ############################
    elif args.prune_method == "magnitude_owl":

        prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)




    elif args.prune_method == "sparsegpt_owl":
        #model = get_llm(args.model_name_or_path, args.cache_dir)
        prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    elif args.prune_method == "wanda_owl_structure":


        prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
        
    elif args.prune_method == "wanda_owl_structure_special":
        prune_wanda_outlier_structure_special(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_owl_srp":
        logging.info(f'initial_pruning_ratio: {args.initial_prune_ratio}')
        logging.info(f'max_pruning_ratio: {args.max_prune_ratio}')
        sparsity_ratio,_ = check_sparsity(model, args)
        th = None
        prune_ratio_before = [0] * layer_num
        it = 0
        while(sparsity_ratio < args.sparsity_ratio):
            th, prune_ratio_before, outlier_ratio = prune_wanda_outlier_srp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, it = it, prune_ratio_before = prune_ratio_before, th = th, num_layer = layer_num)
            sparsity_ratio,ratio_list = check_sparsity(model, args)
            prune_ratio_before = ratio_list
            logging.info(f'it: {it}')
            logging.info(f'sparsity Ratio List: {ratio_list}')
            rounded_numbers = [f"{num:.3f}" for num in prune_ratio_before]
            logging.info(f'prune_ratio_before: {rounded_numbers}')
            rounded_numbers = [f"{num:.3f}" for num in outlier_ratio]
            logging.info(f'outlier_ratio: {rounded_numbers}')
            it = it + 1
            if it==2:
                break  
    elif args.prune_method == "mag_owl_srp":
        sparsity_ratio,_ = check_sparsity(model, args)
        th = None
        prune_ratio_before = [0] * layer_num
        it = 0
        while(sparsity_ratio < args.sparsity_ratio):
            th, prune_ratio_before, outlier_ratio = prune_mag_outlier_srp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, it = it, prune_ratio_before = prune_ratio_before, th = th)
            sparsity_ratio,ratio_list = check_sparsity(model, args)
            logging.info(f'it: {it}')
            logging.info(f'sparsity Ratio List: {ratio_list}')
            rounded_numbers = [f"{num:.3f}" for num in prune_ratio_before]
            logging.info(f'prune_ratio_before: {rounded_numbers}')
            rounded_numbers = [f"{num:.3f}" for num in outlier_ratio]
            logging.info(f'outlier_ratio: {rounded_numbers}')
            it = it + 1  
  
    elif args.prune_method == "sparsegpt_owl_srp":
        sparsity_ratio,_ = check_sparsity(model, args)
        th = None
        layer_num = 32
        prune_ratio_before = [0] * layer_num
        it = 0
        while(sparsity_ratio < args.sparsity_ratio):
            th, prune_ratio_before, outlier_ratio = prune_sparsegpt_outlier_srp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, it = it, prune_ratio_before = prune_ratio_before, th = th, num_layer = layer_num)
            sparsity_ratio,ratio_list = check_sparsity(model, args)
            logging.info(f'it: {it}')
            logging.info(f'sparsity Ratio List: {ratio_list}')
            rounded_numbers = [f"{num:.3f}" for num in prune_ratio_before]
            logging.info(f'prune_ratio_before: {rounded_numbers}')
            rounded_numbers = [f"{num:.3f}" for num in outlier_ratio]
            logging.info(f'outlier_ratio: {rounded_numbers}')
            it = it + 1    
    ################################################################
    print("*"*30)
    sparsity_ratio, sparsity_ratio_list = check_sparsity(model, args)
    print(sparsity_ratio_list)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")
    logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    logging.info(f"ppl on wikitext {ppl}")
    sys.stdout.flush()

    print(f"final ppl on wikitext {ppl}")



    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")








if __name__ == '__main__':
    main()
