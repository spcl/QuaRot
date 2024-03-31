import argparse
import transformers
import torch
import shutil
import json

from e2e.quantized_llama import modeling_llama
from e2e.checkpoint_utils import data_utils, gptq_utils, rotation_utils
from quarot.functional import pack_i4

def main(args):
    model = transformers.LlamaForCausalLM.from_pretrained(args.pretraiend_path_or_name)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.seqlen = 2048
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model)
    if not args.w_rtn:
        trainloader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.pretraiend_path_or_name,
            seqlen=model.seqlen, eval_mode=False
        )
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)
    else:
        quantizers = gptq_utils.rtn_fwrd(model, device, args)


    old_dict = model.state_dict()
    key_maps = {
        "mlp.down_proj": "mlp.down_proj.2",
        "self_attn.o_proj": "self_attn.o_proj.1"
    }
    bad_key_names = {
        "post_attention_layernorm.weight",
        "input_layernorm.weight"
    }
    def _get_new_key(key):
        new_key = key
        for old_name, new_name in key_maps.items():
            new_key = new_key.replace(old_name, new_name)
        return new_key
    
    def _keep_key(key):
        return all(bad_name not in key for bad_name in bad_key_names)

    new_dict = {_get_new_key(key): value for key, value in old_dict.items() if _keep_key(key)}
    for key, value in quantizers.items():
        new_key = _get_new_key(key)
        weight_scales = value.scale
        new_dict[f"{new_key}.weight_scales"] = weight_scales
        weight_matrix = new_dict[f"{new_key}.weight"]
        int_rounded_weight = (weight_matrix/weight_scales).round()
        new_dict[f"{new_key}.weight"] = pack_i4(int_rounded_weight.to(torch.int8))

    config = modeling_llama.QuarotLlamaConfig.from_pretrained(
        args.pretraiend_path_or_name,
        attn_implementation="flash_attention_2"
    )
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        new_model = modeling_llama.QuarotLlamaForCausalLM(config=config)

    result = new_model.load_state_dict(new_dict, strict=False)
    assert all("had_rem_dim" in key for key in result.missing_keys), result
    assert len(result.unexpected_keys) == 0, result

    new_model = new_model.cpu()

    new_model.save_pretrained(args.save_path)
    with open(f"{args.save_path}/config.json") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "quarot.LlamaConfig",
        "AutoModelForCausalLM": "quarot.QuarotLlamaForCausalLM"
    }
    config["model_type"] =  "llama_quarot"
    with open(f"{args.save_path}/config.json", "w") as f:
        json.dump(config, f)
    
    shutil.copy("e2e/quantized_llama/modeling_llama.py", f"{args.save_path}/quarot.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    supported_models = [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-70b-hf',
    ]

    supported_datasets = ['wikitext2', 'ptb', 'c4']

    # General Arguments
    parser.add_argument('--pretraiend_path_or_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    

    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')

    args = parser.parse_args()

    args.w_bits = 4
    main(args)
