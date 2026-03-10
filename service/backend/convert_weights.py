#!/usr/bin/env python3
"""Convert SingleHDR TensorFlow checkpoints to PyTorch state dicts.

Run this on a machine with TensorFlow installed (e.g., the GPU instance).
It reads TF checkpoints and produces .pt files loadable by the PyTorch inference backend.

Usage:
    # Convert basic pipeline (3 separate checkpoints)
    python convert_weights.py --mode basic \
        --ckpt_deq /path/to/ckpt_deq/model.ckpt \
        --ckpt_lin /path/to/ckpt_lin/model.ckpt \
        --ckpt_hal /path/to/ckpt_hal/model.ckpt \
        --output_dir /path/to/pytorch_weights/

    # Convert refinement pipeline (single joint checkpoint)
    python convert_weights.py --mode refinement \
        --ckpt_ref /path/to/ckpt_deq_lin_hal_ref/model.ckpt \
        --output_dir /path/to/pytorch_weights/
"""

import argparse
import os
import sys
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    print("ERROR: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)


def load_tf_vars(ckpt_path):
    """Load all variables from a TF checkpoint as a dict {name: numpy_array}."""
    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = reader.get_variable_to_shape_map()
    result = {}
    for name in sorted(var_map.keys()):
        if 'Adam' in name or 'global_step' in name or 'beta1_power' in name or 'beta2_power' in name:
            continue
        result[name] = reader.get_tensor(name)
    return result


def conv_w(tf_w):
    """TF conv weight [H,W,Cin,Cout] -> PyTorch [Cout,Cin,H,W]."""
    return torch.from_numpy(tf_w.transpose(3, 2, 0, 1).copy())


def fc_w(tf_w):
    """TF dense weight [in,out] -> PyTorch [out,in]."""
    return torch.from_numpy(tf_w.T.copy())


def to_tensor(arr):
    return torch.from_numpy(arr.copy())


# ============================================================================
# Dequantization-Net mapping
# ============================================================================

def convert_dequantization(tf_vars, prefix="Dequantization_Net"):
    """Convert dequantization net weights."""
    # TF uses auto-numbered conv2d, conv2d_1, ... conv2d_18
    # Order: conv_in1, conv_in2, down1_c1, down1_c2, down2_c1, down2_c2,
    #        down3_c1, down3_c2, down4_c1, down4_c2,
    #        up1_c1, up1_c2, up2_c1, up2_c2, up3_c1, up3_c2, up4_c1, up4_c2,
    #        conv_out

    pt_names = [
        'conv_in1', 'conv_in2',
        'down1_c1', 'down1_c2', 'down2_c1', 'down2_c2',
        'down3_c1', 'down3_c2', 'down4_c1', 'down4_c2',
        'up1_c1', 'up1_c2', 'up2_c1', 'up2_c2',
        'up3_c1', 'up3_c2', 'up4_c1', 'up4_c2',
        'conv_out',
    ]

    state_dict = {}
    for i, pt_name in enumerate(pt_names):
        tf_suffix = f"conv2d{'_' + str(i) if i > 0 else ''}"
        k_name = f"{prefix}/{tf_suffix}/kernel"
        b_name = f"{prefix}/{tf_suffix}/bias"
        state_dict[f"{pt_name}.weight"] = conv_w(tf_vars[k_name])
        state_dict[f"{pt_name}.bias"] = to_tensor(tf_vars[b_name])

    return state_dict


# ============================================================================
# Refinement-Net mapping (same structure, different prefix)
# ============================================================================

def convert_refinement(tf_vars, prefix="Refinement_Net"):
    """Convert refinement net weights."""
    pt_names = [
        'conv_in1', 'conv_in2',
        'down1_c1', 'down1_c2', 'down2_c1', 'down2_c2',
        'down3_c1', 'down3_c2', 'down4_c1', 'down4_c2',
        'up1_c1', 'up1_c2', 'up2_c1', 'up2_c2',
        'up3_c1', 'up3_c2', 'up4_c1', 'up4_c2',
        'conv_out',
    ]

    state_dict = {}
    for i, pt_name in enumerate(pt_names):
        tf_suffix = f"conv2d{'_' + str(i) if i > 0 else ''}"
        k_name = f"{prefix}/{tf_suffix}/kernel"
        b_name = f"{prefix}/{tf_suffix}/bias"
        state_dict[f"{pt_name}.weight"] = conv_w(tf_vars[k_name])
        state_dict[f"{pt_name}.bias"] = to_tensor(tf_vars[b_name])

    return state_dict


# ============================================================================
# Linearization-Net mapping
# ============================================================================

def convert_crf_feature_net(tf_vars, prefix="crf_feature_net"):
    """Convert CrfFeatureNet weights."""
    state_dict = {}

    # conv1 (has bias, has BN)
    _map_conv_bn(state_dict, tf_vars, prefix, 'conv1', 'crf_feature_net.conv1', has_bias=True)

    # ResNet blocks
    res_blocks = [
        # (block_name, has_branch1, stride)
        ('res2a', True), ('res2b', False), ('res2c', False),
        ('res3a', True), ('res3b', False),
    ]

    for block_name, has_branch1 in res_blocks:
        if has_branch1:
            _map_conv_bn(state_dict, tf_vars, prefix,
                         f'{block_name}_branch1', f'crf_feature_net.{block_name}_b1',
                         has_bias=False)
        for sub in ['branch2a', 'branch2b', 'branch2c']:
            _map_conv_bn(state_dict, tf_vars, prefix,
                         f'{block_name}_{sub}', f'crf_feature_net.{block_name}_b2{sub[-1]}',
                         has_bias=False)

    return state_dict


def _map_conv_bn(state_dict, tf_vars, tf_prefix, tf_conv_name, pt_prefix, has_bias=True):
    """Map a conv + bn pair from TF to PyTorch."""
    # Conv weights
    k = f"{tf_prefix}/{tf_conv_name}/weights"
    state_dict[f"{pt_prefix}.conv.weight"] = conv_w(tf_vars[k])
    if has_bias:
        b = f"{tf_prefix}/{tf_conv_name}/biases"
        if b in tf_vars:
            state_dict[f"{pt_prefix}.conv.bias"] = to_tensor(tf_vars[b])

    # BatchNorm
    bn_prefix = f"{tf_prefix}/bn_{tf_conv_name}" if not tf_conv_name.startswith('res') else f"{tf_prefix}/bn{tf_conv_name[len('res'):]}"
    # Handle naming: conv1 -> bn_conv1, res2a_branch1 -> bn2a_branch1
    if tf_conv_name == 'conv1':
        bn_prefix = f"{tf_prefix}/bn_conv1"
    else:
        # res2a_branch1 -> bn2a_branch1
        bn_prefix = f"{tf_prefix}/bn{tf_conv_name[3:]}"

    bn_base = f"{bn_prefix}/BatchNorm"
    if f"{bn_base}/gamma" in tf_vars:
        state_dict[f"{pt_prefix}.bn.weight"] = to_tensor(tf_vars[f"{bn_base}/gamma"])
        state_dict[f"{pt_prefix}.bn.bias"] = to_tensor(tf_vars[f"{bn_base}/beta"])
        state_dict[f"{pt_prefix}.bn.running_mean"] = to_tensor(tf_vars[f"{bn_base}/moving_mean"])
        state_dict[f"{pt_prefix}.bn.running_var"] = to_tensor(tf_vars[f"{bn_base}/moving_variance"])
        state_dict[f"{pt_prefix}.bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)


def convert_ae_invcrf(tf_vars, prefix="ae_invcrf_decode_net"):
    """Convert AEInvcrfDecodeNet weights."""
    state_dict = {}
    state_dict["ae_invcrf_decode_net.fc.weight"] = fc_w(tf_vars[f"{prefix}/dense/kernel"])
    state_dict["ae_invcrf_decode_net.fc.bias"] = to_tensor(tf_vars[f"{prefix}/dense/bias"])
    return state_dict


# ============================================================================
# Hallucination-Net mapping
# ============================================================================

def convert_hallucination(tf_vars, prefix="Hallucination_Net"):
    """Convert HallucinationNet weights."""
    state_dict = {}

    # Encoder conv layers (VGG16)
    enc_layers = [
        ('encoder/h1/conv_1', 'enc_h1_c1'),
        ('encoder/h1/conv_2', 'enc_h1_c2'),
        ('encoder/h2/conv_1', 'enc_h2_c1'),
        ('encoder/h2/conv_2', 'enc_h2_c2'),
        ('encoder/h3/conv_1', 'enc_h3_c1'),
        ('encoder/h3/conv_2', 'enc_h3_c2'),
        ('encoder/h3/conv_3', 'enc_h3_c3'),
        ('encoder/h4/conv_1', 'enc_h4_c1'),
        ('encoder/h4/conv_2', 'enc_h4_c2'),
        ('encoder/h4/conv_3', 'enc_h4_c3'),
        ('encoder/h5/conv_1', 'enc_h5_c1'),
        ('encoder/h5/conv_2', 'enc_h5_c2'),
        ('encoder/h5/conv_3', 'enc_h5_c3'),
    ]
    for tf_name, pt_name in enc_layers:
        state_dict[f"{pt_name}.weight"] = conv_w(tf_vars[f"{prefix}/{tf_name}/W_conv2d"])
        state_dict[f"{pt_name}.bias"] = to_tensor(tf_vars[f"{prefix}/{tf_name}/b_conv2d"])

    # Encoder h6 conv + BN
    state_dict["enc_h6_conv.weight"] = conv_w(tf_vars[f"{prefix}/encoder/h6/conv/W_conv2d"])
    state_dict["enc_h6_conv.bias"] = to_tensor(tf_vars[f"{prefix}/encoder/h6/conv/b_conv2d"])
    _map_hal_bn(state_dict, tf_vars, prefix, "encoder/h6/batch_norm", "enc_h6_bn")

    # Decoder layers
    dec_deconv = [
        ('decoder/h1/decon2d', 'dec_h1_conv', 'dec_h1_bn'),
        ('decoder/h2/decon2d', 'dec_h2_conv', 'dec_h2_bn'),
        ('decoder/h3/decon2d', 'dec_h3_conv', 'dec_h3_bn'),
        ('decoder/h4/decon2d', 'dec_h4_conv', 'dec_h4_bn'),
        ('decoder/h5/decon2d', 'dec_h5_conv', 'dec_h5_bn'),
    ]
    for tf_name, pt_conv, pt_bn in dec_deconv:
        state_dict[f"{pt_conv}.weight"] = conv_w(tf_vars[f"{prefix}/{tf_name}/W_conv2d"])
        state_dict[f"{pt_conv}.bias"] = to_tensor(tf_vars[f"{prefix}/{tf_name}/b_conv2d"])
        _map_hal_bn(state_dict, tf_vars, prefix, f"{tf_name}/batch_norm_dc", pt_bn)

    # Skip connections
    skip_layers = [
        ('decoder/h2/fuse_skip_connection', 'dec_h2_skip'),
        ('decoder/h3/fuse_skip_connection', 'dec_h3_skip'),
        ('decoder/h4/fuse_skip_connection', 'dec_h4_skip'),
        ('decoder/h5/fuse_skip_connection', 'dec_h5_skip'),
        ('decoder/h6/fuse_skip_connection', 'dec_h6_skip'),
        ('decoder/h7/fuse_skip_connection', 'dec_h7_skip'),
    ]
    for tf_name, pt_name in skip_layers:
        state_dict[f"{pt_name}.weight"] = conv_w(tf_vars[f"{prefix}/{tf_name}/W_conv2d"])
        state_dict[f"{pt_name}.bias"] = to_tensor(tf_vars[f"{prefix}/{tf_name}/b_conv2d"])

    # h7 final conv + BN
    state_dict["dec_h7_conv.weight"] = conv_w(tf_vars[f"{prefix}/decoder/h7/conv2d/W_conv2d"])
    state_dict["dec_h7_conv.bias"] = to_tensor(tf_vars[f"{prefix}/decoder/h7/conv2d/b_conv2d"])
    _map_hal_bn(state_dict, tf_vars, prefix, "decoder/h7/batch_norm", "dec_h7_bn")

    return state_dict


def _map_hal_bn(state_dict, tf_vars, prefix, tf_bn_name, pt_bn_name):
    """Map hallucination net batch norm."""
    base = f"{prefix}/{tf_bn_name}"
    state_dict[f"{pt_bn_name}.weight"] = to_tensor(tf_vars[f"{base}/gamma"])
    state_dict[f"{pt_bn_name}.bias"] = to_tensor(tf_vars[f"{base}/beta"])
    state_dict[f"{pt_bn_name}.running_mean"] = to_tensor(tf_vars[f"{base}/moving_mean"])
    state_dict[f"{pt_bn_name}.running_var"] = to_tensor(tf_vars[f"{base}/moving_variance"])
    state_dict[f"{pt_bn_name}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)


# ============================================================================
# Main conversion logic
# ============================================================================

def convert_basic(ckpt_deq, ckpt_lin, ckpt_hal, output_dir):
    """Convert 3 separate checkpoints for basic pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dequantization checkpoint...")
    tf_deq = load_tf_vars(ckpt_deq)
    deq_sd = convert_dequantization(tf_deq)
    torch.save(deq_sd, os.path.join(output_dir, "dequantization.pt"))
    print(f"  Saved {len(deq_sd)} parameters")

    print("Loading linearization checkpoint...")
    tf_lin = load_tf_vars(ckpt_lin)
    lin_sd = {}
    lin_sd.update(convert_crf_feature_net(tf_lin))
    lin_sd.update(convert_ae_invcrf(tf_lin))
    torch.save(lin_sd, os.path.join(output_dir, "linearization.pt"))
    print(f"  Saved {len(lin_sd)} parameters")

    print("Loading hallucination checkpoint...")
    tf_hal = load_tf_vars(ckpt_hal)
    hal_sd = convert_hallucination(tf_hal)
    torch.save(hal_sd, os.path.join(output_dir, "hallucination.pt"))
    print(f"  Saved {len(hal_sd)} parameters")

    print(f"\nBasic pipeline weights saved to {output_dir}/")


def convert_refinement_ckpt(ckpt_path, output_dir):
    """Convert single joint checkpoint for refinement pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading joint checkpoint...")
    tf_vars = load_tf_vars(ckpt_path)

    # Print all variable names for debugging
    print(f"\nFound {len(tf_vars)} variables:")
    for name in sorted(tf_vars.keys()):
        print(f"  {name}: {tf_vars[name].shape}")

    deq_sd = convert_dequantization(tf_vars)
    torch.save(deq_sd, os.path.join(output_dir, "dequantization.pt"))
    print(f"\nDequantization: {len(deq_sd)} parameters")

    lin_sd = {}
    lin_sd.update(convert_crf_feature_net(tf_vars))
    lin_sd.update(convert_ae_invcrf(tf_vars))
    torch.save(lin_sd, os.path.join(output_dir, "linearization.pt"))
    print(f"Linearization: {len(lin_sd)} parameters")

    hal_sd = convert_hallucination(tf_vars)
    torch.save(hal_sd, os.path.join(output_dir, "hallucination.pt"))
    print(f"Hallucination: {len(hal_sd)} parameters")

    ref_sd = convert_refinement(tf_vars)
    torch.save(ref_sd, os.path.join(output_dir, "refinement.pt"))
    print(f"Refinement: {len(ref_sd)} parameters")

    print(f"\nRefinement pipeline weights saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert SingleHDR TF checkpoints to PyTorch")
    parser.add_argument("--mode", choices=["basic", "refinement", "both"], required=True)
    parser.add_argument("--ckpt_deq", help="Dequantization checkpoint path")
    parser.add_argument("--ckpt_lin", help="Linearization checkpoint path")
    parser.add_argument("--ckpt_hal", help="Hallucination checkpoint path")
    parser.add_argument("--ckpt_ref", help="Joint refinement checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Output directory for .pt files")
    args = parser.parse_args()

    if args.mode in ("basic", "both"):
        if not all([args.ckpt_deq, args.ckpt_lin, args.ckpt_hal]):
            parser.error("--ckpt_deq, --ckpt_lin, --ckpt_hal required for basic mode")
        basic_dir = os.path.join(args.output_dir, "basic")
        convert_basic(args.ckpt_deq, args.ckpt_lin, args.ckpt_hal, basic_dir)

    if args.mode in ("refinement", "both"):
        if not args.ckpt_ref:
            parser.error("--ckpt_ref required for refinement mode")
        ref_dir = os.path.join(args.output_dir, "refinement")
        convert_refinement_ckpt(args.ckpt_ref, ref_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
