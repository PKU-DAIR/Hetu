import argparse
def cal_patch_embed_flops(patch_size, seq_len, temporal_patch_size, in_channels, out_channels):
    kernel_size = patch_size
    in_channels = in_channels * temporal_patch_size
    return 4 * kernel_size * kernel_size * in_channels * out_channels * seq_len


def cal_patch_merger_flops(batch_size, seq_len, vision_hidden_size, llm_hidden_size):
    linear1_flops = 2 * batch_size * seq_len * vision_hidden_size * vision_hidden_size
    linear2_flops = 2 * batch_size * seq_len * vision_hidden_size * llm_hidden_size
    return linear1_flops + linear2_flops

def cal_attn_flops(batch_size, seq_len, hidden_size):
    return 8 * batch_size * seq_len * hidden_size * hidden_size + 4 * batch_size * seq_len * seq_len * hidden_size

def cal_mlp_flops(batch_size, seq_len, hidden_size, mlp_dim):
    return 4 * batch_size * seq_len * hidden_size * mlp_dim 


def cal_vocab_layer_flops(batch_size, seq_len, hidden_size, vocab_size):
    return 2 * batch_size * seq_len * hidden_size * vocab_size


def cal_vision_flops(batch_size, seq_len, num_layers, vision_hidden_size, vision_mlp_dim, llm_hidden_size, patch_size, temporal_patch_size, in_channels):
    patch_embed_flops = cal_patch_embed_flops(patch_size, seq_len, temporal_patch_size, in_channels, vision_hidden_size)
    attn_flops = cal_attn_flops(batch_size, seq_len, vision_hidden_size)
    mlp_flops = cal_mlp_flops(batch_size, seq_len, vision_hidden_size, vision_mlp_dim)
    patch_merger_flops = cal_patch_merger_flops(batch_size, seq_len, vision_hidden_size, llm_hidden_size)
    fw_flops = (attn_flops + mlp_flops) * num_layers + patch_merger_flops + patch_embed_flops
    return 3 * fw_flops


def cal_llama_flops(batch_size, seq_len, num_layers, hidden_size, llm_mlp_dim, vocab_size):
    attn_flops = cal_attn_flops(batch_size, seq_len, hidden_size)
    mlp_flops = cal_mlp_flops(batch_size, seq_len, hidden_size, llm_mlp_dim)
    vocab_layer_flops = cal_vocab_layer_flops(batch_size, seq_len, hidden_size, vocab_size)
    fw_flops = (attn_flops + mlp_flops) * num_layers + vocab_layer_flops
    return 3 * fw_flops


def cal_patch_embed_memory(patch_size, temporal_patch_size, in_channels, out_channels):
    kernel_size = patch_size
    in_channels = in_channels * temporal_patch_size
    return out_channels * kernel_size * kernel_size * in_channels

def cal_patch_merger_memory(vision_hidden_size, llm_hidden_size):
    return 2 * vision_hidden_size * vision_hidden_size + 2 * vision_hidden_size * llm_hidden_size

def cal_attn_memory(hidden_size):
    return 4 * hidden_size * hidden_size 

def cal_mlp_memory(hidden_size, mlp_dim):
    return 2 * hidden_size * mlp_dim 

def cal_layernorm_memory(hidden_size):
    return 2 * hidden_size 

def cal_vocab_layer_memory(hidden_size, vocab_size):
    return hidden_size * vocab_size 

def cal_vision_memory(num_layers, vision_hidden_size, vision_mlp_dim, llm_hidden_size, patch_size, temporal_patch_size, in_channels):
    patch_embed_memory = cal_patch_embed_memory(patch_size, temporal_patch_size, in_channels, vision_hidden_size)
    patch_merger_memory = cal_patch_merger_memory(vision_hidden_size, llm_hidden_size)
    attn_memory = cal_attn_memory(vision_hidden_size)
    mlp_memory = cal_mlp_memory(vision_hidden_size, vision_mlp_dim)
    layernorm_memory = cal_layernorm_memory(vision_hidden_size)
    return patch_embed_memory + patch_merger_memory + (attn_memory + mlp_memory + 2 * layernorm_memory) * num_layers

def cal_llama_memory(num_layers, hidden_size, llm_mlp_dim, vocab_size):
    attn_memory = cal_attn_memory(hidden_size)
    mlp_memory = cal_mlp_memory(hidden_size, llm_mlp_dim)
    vocab_layer_memory = cal_vocab_layer_memory(hidden_size, vocab_size)
    layernorm_memory = cal_layernorm_memory(hidden_size)
    return (attn_memory + mlp_memory + 2 * layernorm_memory) * num_layers + vocab_layer_memory

def cal_mixed_training_memory(vision_num_layers, llm_num_layers, vision_hidden_size, vision_mlp_dim, llm_hidden_size, llm_mlp_dim, patch_size, temporal_patch_size, in_channels, vocab_size):
    vision_memory = cal_vision_memory(vision_num_layers, vision_hidden_size, vision_mlp_dim, llm_hidden_size, patch_size, temporal_patch_size, in_channels)
    llm_memory = cal_llama_memory(llm_num_layers, llm_hidden_size, llm_mlp_dim, vocab_size)
    return 20 * (vision_memory + llm_memory)


def cal_vision_activation_memory(batch_size, seq_len, num_layers, vision_hidden_size, vision_mlp_dim):
    return num_layers * (16 * batch_size * seq_len * vision_hidden_size + 2 * batch_size * seq_len * vision_mlp_dim) + 10 * batch_size * seq_len * vision_hidden_size

def cal_llama_activation_memory(batch_size, seq_len, num_layers, hidden_size, llm_mlp_dim):
    return num_layers * (16 * batch_size * seq_len * hidden_size + 2 * batch_size * seq_len * llm_mlp_dim) + 6 * batch_size * seq_len * hidden_size


def cal_activation_memory(batch_size, vision_seq_len, llm_seq_len, vision_num_layers, llm_num_layers, vision_hidden_size, vision_mlp_dim, llm_hidden_size, llm_mlp_dim):
    vision_activation_memory = cal_vision_activation_memory(batch_size, vision_seq_len, vision_num_layers, vision_hidden_size, vision_mlp_dim)
    llm_activation_memory = cal_llama_activation_memory(batch_size, llm_seq_len, llm_num_layers, llm_hidden_size, llm_mlp_dim)
    return vision_activation_memory + llm_activation_memory


def analyze_FLOPs_memory(vision_seq_len_list, llm_seq_len_list, vision_config, llm_config):
    vision_num_layers = vision_config.num_hidden_layers
    vision_hidden_size = vision_config.embed_dim
    vision_mlp_dim = vision_config.mlp_dim
    llm_hidden_size = llm_config.hidden_size
    llm_mlp_dim = llm_config.ffn_hidden_size
    llm_num_layers = llm_config.num_hidden_layers
    llm_vocab_size = llm_config.vocab_size
    patch_size = vision_config.patch_size
    temporal_patch_size = vision_config.temporal_patch_size
    in_channels = vision_config.in_channels
    total_vision_flops = 0
    total_llm_flops = 0
    for vision_seq_len in vision_seq_len_list:
        # 计算FLOPs
        vision_flops = cal_vision_flops(
            1, 
            vision_seq_len, 
            vision_num_layers, 
            vision_hidden_size, 
            vision_mlp_dim,
            llm_hidden_size,
            patch_size,
        temporal_patch_size,
            in_channels
        )
        total_vision_flops += vision_flops
    for llm_seq_len in llm_seq_len_list:
        llm_flops = cal_llama_flops(
            1, 
            llm_seq_len, 
            llm_num_layers, 
            llm_hidden_size, 
            llm_mlp_dim,
            llm_vocab_size
        )
        total_llm_flops += llm_flops
    total_flops = total_vision_flops + total_llm_flops
    
    # 计算模型参数内存
    vision_model_memory = cal_vision_memory(
        vision_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim,
        llm_hidden_size,
        patch_size,
        temporal_patch_size,
        in_channels
    )
    llm_model_memory = cal_llama_memory(
        llm_num_layers, 
        llm_hidden_size, 
        llm_mlp_dim,
        llm_vocab_size
    )
    total_model_memory = vision_model_memory + llm_model_memory
    
    # 计算混合精度训练内存
    mixed_training_memory = cal_mixed_training_memory(
        vision_num_layers, 
        llm_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim,
        llm_hidden_size, 
        llm_mlp_dim,
        patch_size,
        temporal_patch_size,
        in_channels,
        llm_vocab_size
    )
    
    # 计算激活内存
    vision_activation_memory = cal_vision_activation_memory(
        1, 
        vision_seq_len_list[0], 
        vision_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim
    )
    llm_activation_memory = cal_llama_activation_memory(
        1, 
        llm_seq_len_list[0], 
        llm_num_layers, 
        llm_hidden_size, 
        llm_mlp_dim
    )

    
    # 输出结果
    print("模型配置")
    print("--------------------------------")
    print(f"vision_seq_len_list: {vision_seq_len_list}")
    print(f"llm_seq_len_list: {llm_seq_len_list}")
    print(f"vision_num_layers: {vision_num_layers}")
    print(f"llm_num_layers: {llm_num_layers}")
    print(f"vision_hidden_size: {vision_hidden_size}")
    print(f"llm_hidden_size: {llm_hidden_size}")
    print(f"llm_vocab_size: {llm_vocab_size}")
    print(f"in_channels: {in_channels}")
    print(f"patch_size: {patch_size}")
    print(f"temporal_patch_size: {temporal_patch_size}")
    print("--------------------------------")
    print(f"Vision模型FLOPs: {total_vision_flops/1e12:.2f} TFLOPs")
    print(f"LLM模型FLOPs: {total_llm_flops/1e12:.2f} TFLOPs")
    print(f"总FLOPs: {total_flops/1e12:.2f} TFLOPs")
    
    print(f"\nVision模型参数内存: {vision_model_memory/1e9:.2f} GB")
    print(f"LLM模型参数内存: {llm_model_memory/1e9:.2f} GB")
    print(f"总模型参数内存: {total_model_memory/1e9:.2f} GB")
    
    print(f"\n混合精度训练内存: {mixed_training_memory/1e9:.2f} GB")
    print(f"激活内存: {vision_activation_memory/1e9:.2f} GB")
    print(f"激活内存: {llm_activation_memory/1e9:.2f} GB")
    
    print("--------------------------------") 


def main():

    vision_seq_len_list = [8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 7680, 7168, 7808, 2688]
    llm_seq_len_list = [4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 4224, 3968, 3968, 3712, 3712, 3968, 4096, 1664, 1280]
    vision_num_layers = 32
    llm_num_layers = 32
    vision_hidden_size = 1280
    llm_hidden_size = 2560
    llm_vocab_size = 50304
    in_channels = 3
    patch_size = 14
    # temporal_patch_size: 2

    total_vision_flops = 0
    total_llm_flops = 0
    for vision_seq_len in vision_seq_len_list:
        # 计算FLOPs
        vision_flops = cal_vision_flops(
            1, 
            vision_seq_len, 
            vision_num_layers, 
            vision_hidden_size, 
            vision_mlp_dim,
            llm_hidden_size,
            patch_size,
        temporal_patch_size,
            in_channels
        )
        total_vision_flops += vision_flops
    for llm_seq_len in llm_seq_len_list:
        llm_flops = cal_llama_flops(
            1, 
            llm_seq_len, 
            llm_num_layers, 
            llm_hidden_size, 
            llm_mlp_dim,
            llm_vocab_size
        )
        total_llm_flops += llm_flops
    total_flops = total_vision_flops + total_llm_flops
    
    # 计算模型参数内存
    vision_model_memory = cal_vision_memory(
        vision_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim,
        llm_hidden_size,
        patch_size,
        temporal_patch_size,
        in_channels
    )
    llm_model_memory = cal_llama_memory(
        llm_num_layers, 
        llm_hidden_size, 
        llm_mlp_dim,
        llm_vocab_size
    )
    total_model_memory = vision_model_memory + llm_model_memory
    
    # 计算混合精度训练内存
    mixed_training_memory = cal_mixed_training_memory(
        vision_num_layers, 
        llm_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim,
        llm_hidden_size, 
        llm_mlp_dim,
        patch_size,
        temporal_patch_size,
        in_channels,
        llm_vocab_size
    )
    
    # 计算激活内存
    activation_memory = cal_activation_memory(
        1, 
        vision_seq_len_list[0], 
        llm_seq_len_list[0], 
        vision_num_layers, 
        llm_num_layers, 
        vision_hidden_size, 
        vision_mlp_dim,
        llm_hidden_size, 
        llm_mlp_dim
    )
    
    # 输出结果
    print("模型配置")
    print("--------------------------------")
    print(f"vision_seq_len_list: {vision_seq_len_list}")
    print(f"llm_seq_len_list: {llm_seq_len_list}")
    print(f"vision_num_layers: {vision_num_layers}")
    print(f"llm_num_layers: {llm_num_layers}")
    print(f"vision_hidden_size: {vision_hidden_size}")
    print(f"llm_hidden_size: {llm_hidden_size}")
    print(f"llm_vocab_size: {llm_vocab_size}")
    print(f"in_channels: {in_channels}")
    print(f"patch_size: {patch_size}")
    print(f"temporal_patch_size: {temporal_patch_size}")
    print("--------------------------------")
    print(f"Vision模型FLOPs: {total_vision_flops/1e12:.2f} TFLOPs")
    print(f"LLM模型FLOPs: {total_llm_flops/1e12:.2f} TFLOPs")
    print(f"总FLOPs: {total_flops/1e12:.2f} TFLOPs")
    
    print(f"\nVision模型参数内存: {vision_model_memory/1e9:.2f} GB")
    print(f"LLM模型参数内存: {llm_model_memory/1e9:.2f} GB")
    print(f"总模型参数内存: {total_model_memory/1e9:.2f} GB")
    
    print(f"\n混合精度训练内存: {mixed_training_memory/1e9:.2f} GB")
    print(f"激活内存: {activation_memory/1e9:.2f} GB")
    
    print("--------------------------------") 


if __name__ == "__main__":
    main()
