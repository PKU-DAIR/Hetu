# Activation Recomputation (Gradient Checkpointing)

## Introduction

Activation recomputation, also known as gradient checkpointing, is a memory optimization technique that trades off computational overhead for reduced memory consumption during training. It achieves this by discarding intermediate activations during the forward pass and recomputing them during the backward pass.

## Configuration

### recompute_granularity
Controls the scope of activations to recompute within a Transformer block.

* `selective`: Recomputes only specific submodules (`RMSNorm` and `Swiglu` for Llama) in the Transformer block.
  - Reduces memory usage for targeted submodules while preserving the rest for memory-speed trade-off
* `full`: Recomputes all activations in the Transformer block.
  - Maximizes memory savings at the cost of higher computational overhead.
* **Default**: Not set (user must explicitly specify).

### recompute_method
Defines the strategy for partitioning and recomputing Transformer blocks.

* `uniform`: Divides the total number of Transformer blocks into partitions and recomputes contiguous blocks in each partition.
  - Example: If there are 24 blocks and `recompute_num_layers`=2, each partition will recompute 2 contiguous blocks, preserving the activations of the first block in each partition.
* `block`: Recomputes the first `recompute_num_layers` individual Transformer blocks in each pipeline stage, leaving the rest untouched.
* **Default**: Not set (user must explicitly specify).

### recompute_num_layers
Specifies the number of Transformer blocks to recompute per partition or stage. Interpretation depends on `recompute_method`.

* `uniform`: the number of contiguous Transformer blocks to recompute in each partition
* `block`: the number of individual Transformer blocks to recompute in each pipeline stage
* **Default**: 1

### recompute_layer_idxs
Manually specifies the indices of Transformer blocks to recompute.

* Overrides `recompute_method` and `recompute_num_layers` when set.
* **Default**: all Transformer blocks.

## Usage Notes

### Compatibility

1. **Selective Recomputation Constraints**: `recompute_granularity=selective` is incompatible with `recompute_method`. Please use `recompute_layer_idxs` to manually specify blocks for selective recomputation.

2. **Parameter Precedence**: If `recompute_layer_idxs` is set, `recompute_method` and `recompute_num_layers` are ignored. 