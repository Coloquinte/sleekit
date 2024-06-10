import argparse
import os
import time

import torch
import torch.nn as nn

from sleekit import Sleekit
from datautils import *
from modelutils import *
import tqdm


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


def extract_layer_statistics(layers, dev, inps, **kwargs):
    outs = torch.zeros_like(inps)
    for i in tqdm.tqdm(range(len(layers))):
        layer = layers[i].to(dev)

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
            sleekit[name].export(os.path.join(args.path, f"{i}.{name}"))
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


def extract_inputs_opt(model, layers, dataloader, dev):
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


def extract_inputs_bloom(model, layers, dataloader, dev):
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
def extract_statistics(model, dataloader, args):
    dev = args.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if is_opt(args.model):
        layers = model.model.decoder.layers
        inps, cache = extract_inputs_opt(model, layers, dataloader, dev)
    elif is_bloom(args.model):
        layers = model.transformer.h
        inps, cache = extract_inputs_bloom(model, layers, dataloader, dev)
    else:
        raise ValueError(f"Unsupported model {args.model}")

    extract_layer_statistics(layers, dev, inps, **cache)

    model.config.use_cache = use_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="Model to load; pass `facebook/opt-X` or `bigscience/bloom-X`.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Destination directory for statistics."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.device_count() > 0 else "cpu",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        help="Device to use.",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force float32 datatype for faster CPU inference",
    )

    args = parser.parse_args()

    model = get_model(args.model, args.force_fp32)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    extract_statistics(model, dataloader, args)
