# MODEL=Qwen/Qwen2.5-0.5B  # Will fail because the head number is 14.
# MODEL=Qwen/Qwen2.5-1.5B
MODEL=Qwen/Qwen2.5-7B
# MODEL=Qwen/Qwen2.5-32B
# MODEL='NousResearch/Llama-3.2-1B'
# MODEL='NousResearch/Meta-Llama-3.1-8B'
# MODEL='NousResearch/Meta-Llama-3.1-70B'
# MODEL=mistralai/Mistral-7B-v0.3

python main.py \
    --rotate \
    --w_bits 4 --w_clip \
    --a_bits 8 \
    --percdamp 0.1 \
    --w_asym \
    --model $MODEL
    # --w_groupsize 128 --a_groupsize 128 \
    # --w_rtn \
