import argparse
import pprint
import numpy as np
import torch
import time

from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit

model_sizes = [
    (32, 32, 128), #llama-7b
    (40, 40, 128), #llama-13b
    (80, 64, 128)  #llama-70b   
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 5
num_bench_steps = 100

def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated()
    
    end_time = time.perf_counter()
    
    
    return (end_time - start_time) * 1000 / num_bench_steps, memory_usage

def quantized_kv_cache_decode(
    n_layers, num_heads, head_dim, 
    batch_size, dtype, seq_len, 
    hadamard_dtype=torch.float16):
    device = torch.device("cuda:0")
    cache = MultiLayerPagedKVCache4Bit(
        batch_size=batch_size,
        page_size=seq_len, 
        max_seq_len=seq_len, 
        device=device, 
        n_layers=n_layers, # Ignornig n_layers as it does not affect speed
        num_heads=num_heads,
        head_dim=head_dim,
        disable_quant=dtype == torch.float16,
        hadamard_dtype=hadamard_dtype,
    )
    query_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    key_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    value_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    def _fake_prefill_and_decode():
        cache._needs_init = [False] * len(cache._needs_init)
        cache.length = seq_len - 1
        forward_func = cache.update(key_states, value_states, layer_idx=0, cache_kwargs={})
        attn_out = forward_func(query_states)

    times = []
    for i in range(10):
        times.append(module_benchmark(_fake_prefill_and_decode))
    return zip(*times)


def qattention_benchmark(args):
    
    for n_layers, num_heads, head_dim in model_sizes:
        time_fp16, memory_fp16 = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.batch_size,
            dtype=torch.float16,
            seq_len=args.seq_len,
            hadamard_dtype=None
        )
        
        time_int4, memory_int4 = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.batch_size,
            dtype="int4",
            seq_len=args.seq_len,
            hadamard_dtype=None
        )
        time_int4_hadfp16, _ = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.batch_size,
            dtype="int4",
            seq_len=args.seq_len,
            hadamard_dtype=torch.float16
        )
        time_int4_hadfp32, _ = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.batch_size,
            dtype="int4",
            seq_len=args.seq_len,
            hadamard_dtype=torch.float32
        )

        print(f"Int4 time: {np.mean(time_int4):.3f} +- {1.96 * np.std(time_int4):.3f}ms")

        print(f"Int4 (+FP16had) time: {np.mean(time_int4_hadfp16):.3f} +- {1.96 * np.std(time_int4_hadfp16):.3f}ms")

        print(f"Int4 (+FP32had) time: {np.mean(time_int4_hadfp32):.3f} +- {1.96 * np.std(time_int4_hadfp32):.3f}ms")
        
        print(f"FP16 time: {np.mean(time_fp16):.3f} +- {1.96 * np.std(time_fp16):.3f}ms")
        
        print(f"Speedup: {np.mean(time_fp16) / np.mean(time_int4_hadfp16):.3f}x")

        print(f"Int4 memory: {np.mean(memory_int4):.3f} +- {1.96 * np.std(memory_int4):.3f}ms")
        print(f"FP16 memory: {np.mean(memory_fp16):.3f} +- {1.96 * np.std(memory_fp16):.3f}ms")
        print(f"Memory Saving: {np.mean(memory_fp16) / np.mean(memory_int4):.3f}x")
        
        # table-style output
        print(f'{n_layers}x{num_heads}x{head_dim} & {args.batch_size} & {np.mean(time_fp16):.3f} & {np.mean(time_int4):.3f} & {np.mean(time_int4_hadfp32):.3f} & {np.mean(time_int4_hadfp16):.3f}\\\\')
        print('--------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    qattention_benchmark(args)
