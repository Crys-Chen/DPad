# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

# Copyright 2025 Xinhua Chen, Duke CEI Center
# 
# This file has been modified by Xinhua Chen, Duke CEI Center. Changes include:
# 1. Integrated Diffusion Scratchpad (DPad) for efficient inference.



import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from sampler import Sampler, GaussianSampler, UniformSampler
torch.set_printoptions(threshold=np.inf)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_indices, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_indices.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_indices.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_indices, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_indices, x0, x)
    confidence = torch.where(mask_indices, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_indices.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def suffix_dropout(x, sampler: Sampler, block_end):
    q_indices = torch.arange(block_end, device=x.device).unsqueeze(0).expand(x.shape[0],-1)
    suffix_indices = sampler.sample(torch.arange(block_end, x.shape[1], device=x.device)).unsqueeze(0).expand(x.shape[0],-1)
    
    q_indices = torch.cat([q_indices, suffix_indices], dim=-1)
    k_indices = q_indices.clone()

    assert q_indices.max() < x.shape[1]
    return q_indices, k_indices

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
             dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    if dropout == 'gaussian':
        sampler = GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
    elif dropout == 'uniform':
        sampler = UniformSampler(length=gen_length, number=preserved_tokens, window=window)
    else:
        raise ValueError(f"dropout {dropout} not recognized")
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block+1) * block_length
        block_mask_indices = (x[:, block_start: block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]

        i = 0
        while True:
            nfe += 1
            logits = model(x_pruned, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=(i==0)).logits
            mask_indices = (x_pruned == mask_id)
            mask_indices[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
     
            x0, transfer_index = get_transfer_index(logits, 
                                                    temperature, 
                                                    remasking, 
                                                    mask_indices, 
                                                    x_pruned, 
                                                    num_transfer_tokens[:, i] if threshold is None else None, 
                                                    threshold=threshold)                    
          
            x_pruned[transfer_index] = x0[transfer_index]
  
            i += 1
            if (x_pruned[:, block_start: block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break

    return x, nfe


@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
             dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    if dropout == 'gaussian':
        sampler = GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
    elif dropout == 'uniform':
        sampler = UniformSampler(length=gen_length, number=preserved_tokens, window=window)
    else:
        raise ValueError(f"dropout {dropout} not recognized")
            
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_indices = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]

        output = model(x_pruned, use_cache=True, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=True)
        past_key_values = output.past_key_values
        logits = output.logits
        mask_indices = (x_pruned == mask_id)
        mask_indices[:, block_end:] = 0

        i = 0
        x0, transfer_index = get_transfer_index(logits, 
                                                temperature, 
                                                remasking, 
                                                mask_indices, 
                                                x_pruned, 
                                                num_transfer_tokens[:, i] if threshold is None else None, 
                                                threshold=threshold)
        
        x_pruned[transfer_index] = x0[transfer_index]

        q_indices = q_indices[:,block_start:]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1

        while True:
            if (x_pruned[:, block_start:block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break
            nfe += 1
    
            logits = model(x_pruned[:, block_start:], past_key_values=past_key_values, use_cache=True, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=(i==1)).logits

            mask_indices = (x_pruned[:, block_start:] == mask_id)
            mask_indices[:, block_length:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned[:, block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)

            x_pruned[:, block_start:][transfer_index] = x0[transfer_index]

            i += 1
    
    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
            dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  

    if dropout == 'gaussian':
        sampler = GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
    elif dropout == 'uniform':
        sampler = UniformSampler(length=gen_length, number=preserved_tokens, window=window)
    else:
        raise ValueError(f"dropout {dropout} not recognized")
        
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_indices = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]

        # cache init and update
        output = model(x_pruned, use_cache=True, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=True)
        past_key_values = output.past_key_values
        logits = output.logits
        mask_indices = (x_pruned == mask_id)
        mask_indices[:, block_end:] = 0

        i = 0
        x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned, num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)
        x_pruned[transfer_index] = x0[transfer_index]

        q_indices = q_indices[:,block_start:block_end]

        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x_pruned, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        
        while True:
            if (x_pruned[:, block_start:block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break

            nfe += 1
   
            logits = model(x_pruned[:, block_start: block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=(i==1)).logits
            mask_indices = (x_pruned[:, block_start: block_end] == mask_id)
 
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned[:, block_start: block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)
            x_pruned[:, block_start: block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe
