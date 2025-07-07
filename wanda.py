import time 
import heapq 
import torch 
import torch.nn as nn 

from lib.layerwrapper import WrappedGPT
from lib.sparsegpt import SparseGPT 
from lib.data import get_loaders 
import numpy as np
from pdb import set_trace as st 
from collections import defaultdict

def get_prune_ratio(epoch, initial_prune_ratio, min_prune_ratio, decay_rate):
    prune_ratio = initial_prune_ratio * (decay_rate ** epoch)
    return max(prune_ratio, min_prune_ratio)
def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids 




def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model, args):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        #print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity_mask(mask):


    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()



    print(f" density {float(count)/total_params:.6f}")



def check_outlier(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def check_outlier_mean(mask, threshold, max_shred = 0, prune_ratio = 0):


    W = mask
    count = 0 
    total_params = 0
    if max_shred == 0:
        max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += (W.numel() * (1 - prune_ratio))



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio, max_shred


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_wanda_outlier_srp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, it = 0, prune_ratio_before = [0] * 32, th = None, num_layer = 32):
    ##### calucalte outlier ratio
    
    
 # 初始剪枝比例
    min_prune_ratio_step = 0.05     # 最小剪枝比例
    decay_rate = 0.95 
    min_prune_ratio = 0.05 
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    all_layer_th = []
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    #print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            #activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
        if args.is_block:
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            if th == None:
                if args.is_block:
                    out_ratio_layer, middle=check_outlier_mean(layer_wmetric,out_ratio, 0, prune_ratio_before[i])
                    all_layer_th.append(middle)
                else:
                    out_ratio_layer_list = []
                    for gg in range(len(layer_wmetric)):
                        out_ratio_layer, middle=check_outlier_mean(layer_wmetric[gg],out_ratio, 0, prune_ratio_before[i * 7 + gg])
                        all_layer_th.append(middle)    
                        out_ratio_layer_list.append(out_ratio_layer)                    
            else:
                if args.is_block:
                    out_ratio_layer,_=check_outlier_mean(layer_wmetric,out_ratio, th[i], prune_ratio_before[i])
                else:
                    out_ratio_layer_list = []
                    for gg in range(len(layer_wmetric)):
                        out_ratio_layer, _=check_outlier_mean(layer_wmetric[gg],out_ratio, th[i * 7 + gg], prune_ratio_before[i * 7 + gg])  
                        out_ratio_layer_list.append(out_ratio_layer)
            if args.is_block:                         
                print ("layer outlier ratio",out_ratio,out_ratio_layer)
            else:
                print ("layer outlier ratio",out_ratio,out_ratio_layer_list)                

        if args.is_block:
            all_layer_ratio.append(out_ratio_layer)
        else:
            all_layer_ratio = all_layer_ratio + out_ratio_layer_list
        
        



    
    

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    if it !=0:
        return all_layer_th, prune_ratio_before, all_layer_ratio
    candidates = [(i, prune_ratio_before[i], all_layer_ratio[i]) for i in range(len(prune_ratio_before)) if prune_ratio_before[i] < args.max_prune_ratio]
    if candidates:
        if args.is_block:
            prune_index = min(candidates, key=lambda x: x[2])[0]  # 根据敏感度找到要剪枝的层    
        else:
            candidates_sorted = sorted(candidates, key=lambda x: x[2])
            prune_index = [x[0] for x in candidates_sorted[:7]]

    #prune_index = np.argmin(all_layer_ratio)
    if it == 0 :
        if args.is_block:
            prune_ratio_before = [args.initial_prune_ratio] * num_layer
            prune_ratio_before = [0.42181523, 0.34738219, 0.34189838, 0.33253796, 0.33605978, 0.33838634,0.33387491, 0.32802942, 0.31753393, 0.31626629, 0.30894848, 0.30690375,0.30279631 ,0.29804103, 0.2970049 , 0.29586346, 0.28864478, 0.28548061,0.28363522 ,0.27993641 ,0.27659233 ,0.2752918,  0.27517053, 0.27378828,0.27215041 ,0.27070406, 0.26855903 ,0.26560769, 0.26342765, 0.26181523,0.26324618 ,0.27260742]
            prune_ratio_before = [1 - i for i in prune_ratio_before]
            #prune_ratio_before = [0.6999581852043733, 0.6999581852043733, 0.6999581852043733, 0.6999581852043733, 0.700086413987691, 0.700086413987691]*num_layer
        else:
            prune_ratio_before = [args.initial_prune_ratio] * 32 * 7
            
        #prune_ratio_before = [0.5514649603977676, 0.5505559495657929, 0.5513409732787077, 0.5508677679645129, 0.5510467371664757, 0.5510954612542776, 0.5512374751823993, 0.5513946848467361, 0.5509260666271871, 0.5507841567362636, 0.5511089545636138, 0.5510174648820861, 0.5513150270320167, 0.5507920036631182, 0.5511170946862087, 0.55080674226619, 0.550864448232099, 0.6007143950659382, 0.6014058924903555, 0.6519028971017885, 0.7020628401070587, 0.7515409043997773, 0.7511080529078964, 0.7630437740609666, 0.7846673634426653, 0.7691818048146145, 0.7600899656942068, 0.7734663151512461, 0.7536998055197975, 0.7709770738585922, 0.7651792447429058, 0.7963117110827738, 0.77987497187843, 0.7706337384941164, 0.7569439123484714, 0.8016332988896646, 0.8291350750883749, 0.8162789935908041, 0.8019918394482826, 0.776312769740081]
        #prune_ratio_before = [0.32196618, 0.32935266, 0.35500197, 0.40332755, 0.35558179, 0.34443323, 0.36262086, 0.35437842, 0.34058214, 0.33704857, 0.33604622, 0.33339052,0.33132592, 0.32116841, 0.31515644, 0.29931418, 0.28756899, 0.28025891,0.27373835, 0.26758393, 0.26216196, 0.26244496, 0.25927708, 0.2589161,0.25522576, 0.25855091, 0.25173976, 0.24901508, 0.24751691, 0.24332755,0.24364583, 0.25833288]
        #prune_ratio_before = [1 - value for value in prune_ratio_before]    
    else:
        if args.is_block:
            prune_ratio_before[prune_index] = prune_ratio_before[prune_index] + get_prune_ratio(it, args.initial_prune_ratio_step, args.min_prune_ratio_step, decay_rate)
        else:
            for gg in range(7):
                prune_ratio_before[prune_index[gg]] = prune_ratio_before[prune_index[gg]] + get_prune_ratio(it, args.initial_prune_ratio_step, args.min_prune_ratio_step, decay_rate)
    

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        if args.is_block:

            if it != 0:
                break
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        #inps, outs = outs, inps
        for h in handles:
            h.remove()
            
        if args.is_block: 
            layer_index = i
        else:
            layer_index = i * 6
        for name in subset:
            

            print(f"pruning layer {layer_index} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            #activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio= prune_ratio_before[layer_index]
            if args.is_block!= True:
                layer_index = layer_index + 1
                if layer_index > max(prune_index) and it!=0:
                        if th != None:
                            all_layer_th = th
                        return all_layer_th, prune_ratio_before, all_layer_ratio
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if args.sparsity_type == "structured":
                W_norm = torch.norm(W_metric.cuda(), dim = 1, p =1)
                thresh = torch.sort(W_norm)[0][int(W_metric.shape[0]*layer_sparsity_ratio)].cpu()
                W_mask = (W_norm<=thresh)

            elif prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    #W_mask = (W_metric<0.18)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    if th != None:
        all_layer_th = th
    return all_layer_th, prune_ratio_before, all_layer_ratio


def prune_mag_outlier_srp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, it = 0, prune_ratio_before = [0] * 32, th = None):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    
    
    initial_prune_ratio = 0.15  # 初始剪枝比例
    min_prune_ratio = 0.05     # 最小剪枝比例
    decay_rate = 0.95 
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    all_layer_th = []
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    #print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            #activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                


        




        if args.is_block:
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            if th == None:
                if args.is_block:
                    out_ratio_layer, middle=check_outlier_mean(layer_wmetric,out_ratio, 0, prune_ratio_before[i])
                    all_layer_th.append(middle)
                else:
                    out_ratio_layer_list = []
                    for gg in range(len(layer_wmetric)):
                        out_ratio_layer, middle=check_outlier_mean(layer_wmetric[gg],out_ratio, 0, prune_ratio_before[i * 7 + gg])
                        all_layer_th.append(middle)    
                        out_ratio_layer_list.append(out_ratio_layer)                    
            else:
                if args.is_block:
                    out_ratio_layer,_=check_outlier_mean(layer_wmetric,out_ratio, th[i], prune_ratio_before[i])
                else:
                    out_ratio_layer_list = []
                    for gg in range(len(layer_wmetric)):
                        out_ratio_layer, _=check_outlier_mean(layer_wmetric[gg],out_ratio, th[i * 7 + gg], prune_ratio_before[i * 7 + gg])  
                        out_ratio_layer_list.append(out_ratio_layer)
            if args.is_block:                         
                print ("layer outlier ratio",out_ratio,out_ratio_layer)
            else:
                print ("layer outlier ratio",out_ratio,out_ratio_layer_list)                

        if args.is_block:
            all_layer_ratio.append(out_ratio_layer)
        else:
            all_layer_ratio = all_layer_ratio + out_ratio_layer_list
        
        



    
    

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    candidates = [(i, prune_ratio_before[i], all_layer_ratio[i]) for i in range(len(prune_ratio_before)) if prune_ratio_before[i] < args.max_prune_ratio]
    if candidates:
        if args.is_block:
            prune_index = min(candidates, key=lambda x: x[2])[0]  # 根据敏感度找到要剪枝的层    
        else:
            candidates_sorted = sorted(candidates, key=lambda x: x[2])
            prune_index = [x[0] for x in candidates_sorted[:7]]

    #prune_index = np.argmin(all_layer_ratio)
    if it == 0 :
        if args.is_block:
            prune_ratio_before = [args.initial_prune_ratio] * 32
            #prune_ratio_before = [0.7] *32
            #prune_ratio_before = [0.67803382, 0.6706473399999999, 0.64499803, 0.59667245, 0.64441821, 0.6555667700000001, 0.63737914, 0.64562158, 0.65941786, 0.66295143, 0.6639537799999999, 0.66660948, 0.66867408, 0.6788315899999999, 0.68484356, 0.70068582, 0.71243101,0.71974109, 0.72626165, 0.73241607, 0.73783804, 0.7375550399999999, 0.7407229200000001, 0.7410839, 0.7447742399999999, 0.7414490899999999, 0.74826024, 0.7509849200000001, 0.75248309, 0.7566724499999999, 0.75635417, 0.74166712]

        else:
            prune_ratio_before = [args.initial_prune_ratio] * 32 * 7
            
        #prune_ratio_before = [0.5514649603977676, 0.5505559495657929, 0.5513409732787077, 0.5508677679645129, 0.5510467371664757, 0.5510954612542776, 0.5512374751823993, 0.5513946848467361, 0.5509260666271871, 0.5507841567362636, 0.5511089545636138, 0.5510174648820861, 0.5513150270320167, 0.5507920036631182, 0.5511170946862087, 0.55080674226619, 0.550864448232099, 0.6007143950659382, 0.6014058924903555, 0.6519028971017885, 0.7020628401070587, 0.7515409043997773, 0.7511080529078964, 0.7630437740609666, 0.7846673634426653, 0.7691818048146145, 0.7600899656942068, 0.7734663151512461, 0.7536998055197975, 0.7709770738585922, 0.7651792447429058, 0.7963117110827738, 0.77987497187843, 0.7706337384941164, 0.7569439123484714, 0.8016332988896646, 0.8291350750883749, 0.8162789935908041, 0.8019918394482826, 0.776312769740081]
        #prune_ratio_before = [0.32196618, 0.32935266, 0.35500197, 0.40332755, 0.35558179, 0.34443323, 0.36262086, 0.35437842, 0.34058214, 0.33704857, 0.33604622, 0.33339052,0.33132592, 0.32116841, 0.31515644, 0.29931418, 0.28756899, 0.28025891,0.27373835, 0.26758393, 0.26216196, 0.26244496, 0.25927708, 0.2589161,0.25522576, 0.25855091, 0.25173976, 0.24901508, 0.24751691, 0.24332755,0.24364583, 0.25833288]
        #prune_ratio_before = [1 - value for value in prune_ratio_before]    
    else:
        if args.is_block:
            prune_ratio_before[prune_index] = prune_ratio_before[prune_index] + get_prune_ratio(it, args.initial_prune_ratio_step, args.min_prune_ratio_step, decay_rate)
        else:
            for gg in range(7):
                prune_ratio_before[prune_index[gg]] = prune_ratio_before[prune_index[gg]] + get_prune_ratio(it, args.initial_prune_ratio_step, args.min_prune_ratio_step, decay_rate)
    

    
    
    ############## prune


    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    #print (layers)
    
    for i in range(len(layers)):
        if args.is_block:
            if i != prune_index and it !=0:
                continue          
        layer = layers[i]
        subset = find_layers(layer)
        bias = 0
        for name in subset:
            if args.is_block:
                layer_sparsity_ratio= prune_ratio_before[i]
            else:
                layer_sparsity_ratio= prune_ratio_before[i * 7 + bias]
                bias = bias + 1
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if args.sparsity_type == 'unstructured':
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                else:
                    W_norm = torch.norm(W_metric.cuda(), dim = 1, p =1)
                    #print(W_norm.shape, W.shape)
                    thresh = torch.sort(W_norm)[0][int(W.shape[0]*layer_sparsity_ratio)].cpu()
                    thresh = 0.0190
                    W_mask = (W_norm<=thresh)

            W[W_mask] = 0
    if th != None:
        all_layer_th = th
    return all_layer_th, prune_ratio_before, all_layer_ratio


@torch.no_grad()
def prune_sparsegpt_outlier_srp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, it = 0, prune_ratio_before = [0] * 32, th = None, num_layer = 40):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    

    
    initial_prune_ratio = 0.2  # 初始剪枝比例
    min_prune_ratio = 0.05     # 最小剪枝比例
    decay_rate = 0.95 
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    all_layer_th = []
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    #print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if "OPT" in model.__class__.__name__:
            if f"model.decoder.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.decoder.layers.{i}"]
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)          
        else:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            #activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                


        





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            if th == None:
                out_ratio_layer, middle=check_outlier_mean(layer_wmetric,out_ratio, 0, prune_ratio_before[i])
                all_layer_th.append(middle)
            else:
                out_ratio_layer,_=check_outlier_mean(layer_wmetric,out_ratio, th[i], prune_ratio_before[i])
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        



    
    

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    candidates = [(i, prune_ratio_before[i], all_layer_ratio[i]) for i in range(len(prune_ratio_before)) if prune_ratio_before[i] < args.max_prune_ratio]
    if candidates:
        prune_index = min(candidates, key=lambda x: x[2])[0]  # 根据敏感度找到要剪枝的层    
    #prune_index = np.argmin(all_layer_ratio)
    if it == 0 :
        #prune_ratio_before = [args.initial_prune_ratio] * num_layer
        prune_ratio_before = [0.7] * num_layer
        #prune_ratio_before = [0.67803382, 0.6706473399999999, 0.64499803, 0.59667245, 0.64441821, 0.6555667700000001, 0.63737914, 0.64562158, 0.65941786, 0.66295143, 0.6639537799999999, 0.66660948, 0.66867408, 0.6788315899999999, 0.68484356, 0.70068582, 0.71243101,0.71974109, 0.72626165, 0.73241607, 0.73783804, 0.7375550399999999, 0.7407229200000001, 0.7410839, 0.7447742399999999, 0.7414490899999999, 0.74826024, 0.7509849200000001, 0.75248309, 0.7566724499999999, 0.75635417, 0.74166712]
        #prune_ratio_before = [0.32196618, 0.32935266, 0.35500197, 0.40332755, 0.35558179, 0.34443323, 0.36262086, 0.35437842, 0.34058214, 0.33704857, 0.33604622, 0.33339052,0.33132592, 0.32116841, 0.31515644, 0.29931418, 0.28756899, 0.28025891,0.27373835, 0.26758393, 0.26216196, 0.26244496, 0.25927708, 0.2589161,0.25522576, 0.25855091, 0.25173976, 0.24901508, 0.24751691, 0.24332755,0.24364583, 0.25833288]
        #prune_ratio_before = [1 - value for value in prune_ratio_before]    
    else:

        prune_ratio_before[prune_index] = prune_ratio_before[prune_index] + get_prune_ratio(it, args.initial_prune_ratio_step, args.min_prune_ratio_step, decay_rate)
    
    


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    print('Starting ...')


    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)





    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    if "OPT" in model.__class__.__name__:
        if "model.decoder.embed_tokens" in model.hf_device_map:
            dev = model.hf_device_map["model.decoder.embed_tokens"]        
    else:
        if "model.embed_tokens" in model.hf_device_map:
            dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            #cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    #position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        if i > prune_index and it !=0:
            break


        layer_sparsity_ratio= prune_ratio_before[i]
        
        
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01


        layer = layers[i]
        if "OPT" in model.__class__.__name__:
            if f"model.decoder.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.decoder.layers.{i}"]
                print(f"layer {i} device {dev}")
                #inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
        else:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                print(f"layer {i} device {dev}")
                #inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps








    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    if th != None:
        all_layer_th = th
    return all_layer_th, prune_ratio_before, all_layer_ratio