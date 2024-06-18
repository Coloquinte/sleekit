import argparse
import os
import time

import torch
import torch.nn as nn

from sleekit import Sleekit
from datautils import *
import tqdm


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def is_bloom(model_name):
    return model_name.startswith("bigscience/bloom")


def is_opt(model_name):
    return model_name.startswith("facebook/opt")


def get_model(model, force_fp32):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    dtype = torch.float32 if force_fp32 else "auto"

    if is_opt(model):
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype=dtype)
        model.seqlen = model.config.max_position_embeddings
    elif is_bloom(model):
        from transformers import BloomForCausalLM

        model = BloomForCausalLM.from_pretrained(model, torch_dtype=dtype)
        model.seqlen = 2048
    else:
        raise ValueError(f"Unknown model type {model}")
    return model


def extract_layer_statistics(layers, args, inps, **kwargs):
    outs = torch.zeros_like(inps)
    for i in tqdm.tqdm(range(len(layers))):
        layer = layers[i].to(args.device)

        subset = find_layers(layer)
        sleekit = {}
        for name in subset:
            sleekit[name] = Sleekit(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                sleekit[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in tqdm.tqdm(range(args.nsamples), leave=False):
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        for h in handles:
            h.remove()

        for name in subset:
            sleekit[name].export(
                os.path.join(args.path, f"{i}.{name}"), npy_format=args.numpy
            )
            sleekit[name].free()

        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        del sleekit
        torch.cuda.empty_cache()


class Catcher(nn.Module):
    def __init__(self, module, inps, cache):
        super().__init__()
        self.module = module
        self.cache = cache
        self.cnt = 0
        self.inps = inps

    def forward(self, inp, **kwargs):
        self.inps[self.cnt] = inp
        self.cnt += 1
        for k, v in kwargs.items():
            self.cache[k] = v
        raise ValueError


def extract_inputs_opt(model, layers, dataloader, args):
    dev = args.device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {}
    layers[0] = Catcher(layers[0], inps, cache)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()
    return inps, cache


def extract_inputs_bloom(model, layers, dataloader, args):
    dev = args.device
    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = (
        model.transformer.word_embeddings_layernorm.to(dev)
    )
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {}
    layers[0] = Catcher(layers[0], inps, cache)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = (
        model.transformer.word_embeddings_layernorm.cpu()
    )
    torch.cuda.empty_cache()
    return inps, cache


@torch.no_grad()
def extract_inputs(model, dataloader, args):
    if is_opt(args.model):
        layers = model.model.decoder.layers
        inps, cache = extract_inputs_opt(model, layers, dataloader, args)
    elif is_bloom(args.model):
        layers = model.transformer.h
        inps, cache = extract_inputs_bloom(model, layers, dataloader, args)
    else:
        raise ValueError(f"Unsupported model {args.model}")
    return layers, inps, cache


@torch.no_grad()
def extract_statistics(model, dataloader, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers, inps, cache = extract_inputs(model, dataloader, args)
    extract_layer_statistics(layers, args, inps, **cache)

    model.config.use_cache = use_cache
