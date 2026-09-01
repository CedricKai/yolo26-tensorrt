import sys  # noqa: F401
import argparse
import io
import os
import struct
import torch.nn as nn
import numpy as np
import re
import torch
from rich.tree import Tree
from rich import print as rprint
from rich.console import Console
from ultralytics import YOLO

# FP16_MIN_NORMAL = np.finfo(np.float16).tiny
# FP32_MIN_NORMAL = np.finfo(np.float32).tiny

def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='detect', choices=['detect', 'cls', 'seg', 'pose'],
        help='determines the model is detection/classification')
    parser.add_argument(
        '--fp16', action='store_true', default=False,
        help='Clamp FP16 subnormal values to 0 (enable when deploying in FP16)')

    parser.add_argument(
        '--forward-txt', default="./file/model_info.txt",
        help="Txt path for forward hook layer information")
    parser.add_argument(
        '--raw-txt', default="./file/model_raw.txt",
        help="Txt path for raw model repr string")
    parser.add_argument(
        '--tree-txt', default="./file/model_tree.txt",
        help="Txt path for ordered rich tree structure")

    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, os.path.splitext(os.path.basename(args.weights))[0] + '.wts')

    # Create output directory ./file/
    out_dir = "./file"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return (args.weights, args.output, args.type, args.fp16,
            args.forward_txt, args.raw_txt, args.tree_txt)



def tensor_stat(t: torch.Tensor):
    if t is None:
        return None, None, False, False
    t = t.detach().cpu()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    return t.min().item(), t.max().item(), has_nan, has_inf

def fmt_range(vmin, vmax):
    if vmin is None or vmax is None:
        return "None"
    return f"[{vmin:.6f}, {vmax:.6f}]"

forward_order_modules = []
visited_id = set()

def track_forward(module, inputs, outputs):
    mid = id(module)
    if mid not in visited_id:
        visited_id.add(mid)
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        in_shape = tuple(inp.shape) if torch.is_tensor(inp) else "multi-input"
        out_shape = tuple(out.shape) if torch.is_tensor(out) else "multi-output"
        forward_order_modules.append({
            "name": module._name,
            "type": type(module).__name__,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "module": module
        })

def export_layer_info(model, dummy_input, save_path):
    model.eval()
    handles = []
    for name, m in model.named_modules():
        m._name = name
        h = m.register_forward_hook(track_forward)
        handles.append(h)

    with torch.no_grad():
        model(dummy_input)
    for h in handles:
        h.remove()

    with open(save_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(forward_order_modules):
            name = item["name"]
            m = item["module"]
            typ = item["type"]
            in_sh = item["in_shape"]
            out_sh = item["out_shape"]

            line = f"[{idx:03d}] {typ:<10} {name:<40} {in_sh} --> {out_sh}\n"
            # Conv2d
            if isinstance(m, nn.Conv2d):
                w_min, w_max, w_nan, w_inf = tensor_stat(m.weight)
                b_min, b_max, b_nan, b_inf = tensor_stat(m.bias)
                line += (
                    f"      Param: in={m.in_channels},out={m.out_channels},k={m.kernel_size},s={m.stride},p={m.padding},g={m.groups}\n"
                    f"      Weight: {fmt_range(w_min,w_max)} NAN={w_nan} INF={w_inf}\n"
                    f"      Bias:   {fmt_range(b_min,b_max)} NAN={b_nan} INF={w_inf}\n"
                )
            # BN2d
            elif isinstance(m, nn.BatchNorm2d):
                g_min,g_max,g_nan,g_inf = tensor_stat(m.weight)
                b_min,b_max,b_nan,b_inf = tensor_stat(m.bias)
                rm_min,rm_max,_,_ = tensor_stat(m.running_mean)
                rv_min,rv_max,_,_ = tensor_stat(m.running_var)
                line += (
                    f"      eps={m.eps}, momentum={m.momentum}\n"
                    f"      γ:{fmt_range(g_min,g_max)}  β:{fmt_range(b_min,b_max)}\n"
                    f"      run_mean:{fmt_range(rm_min,rm_max)} run_var:{fmt_range(rv_min,rv_max)}\n"
                )
            line += "\n"
            f.write(line)


def write_wts(wts_file, model, use_fp16=False):
    sub_count  = 0
    with open(wts_file, 'w', encoding='utf-8') as f:
        total_keys = len(model.state_dict().keys())
        f.write(f'{total_keys}\n')
        print(f"===== total layers（fp16={use_fp16}）：{total_keys} =====")
        print(f"waiting...")
        for k, v in model.state_dict().items():
            v_out  = v.clone()
            # if use_fp16:
            #     mask = (torch.abs(v_out) < FP16_MIN_NORMAL) & (v_out != 0)
            #     v_out [mask] = 0.0
            #     sub_count += mask.sum().item()

            vr = v_out .reshape(-1).cpu().numpy()
            elem_num = len(vr)

            f.write(f'{k} {elem_num}')
            for vv in vr:
                f.write(' ' + struct.pack('>f', float(vv)).hex())
            f.write('\n')
    #
    # print(f"Total subnormal FP16 values clamped to 0: {sub_count }")



def extract_num(s: str):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else -1

def build_tree(module: torch.nn.Module, tree: Tree):
    children = list(module.named_children())
    children.sort(key=lambda x: extract_num(x[0]))

    for name, child in children:
        label = f"{name} | {child.__class__.__name__}"
        sub = tree.add(label)
        build_tree(child, sub)


if __name__ == "__main__":
    pt_file, wts_file, m_type, use_fp16, forward_txt_path, raw_txt_path, tree_txt_path = parse_args()

    print(f'Generating .wts for {m_type} model')
    print(f'Loading {pt_file}')

    # Load model
    device = 'cpu'
    model = torch.load(pt_file, map_location=device, weights_only=False)['model'].float()  # load to FP32
    if m_type in ['detect', 'seg', 'pose']:
        anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
        delattr(model.model[-1], 'anchors')
    model.to(device).eval()

    # Export .wts file
    write_wts(wts_file, model, use_fp16)

    # Export layer info via forward hook
    dummy = torch.randn(1,3,640,640,device=device)
    export_layer_info(model, dummy, forward_txt_path)

    # Load model for rich tree visualization
    yaml_file = os.path.splitext(pt_file)[0] + ".yaml"
    yolo_model = YOLO(yaml_file).load(pt_file)
    net = yolo_model.__dict__['_modules']['model']

    # Raw network repr string
    raw_model_str = str(net)

    # Build rich tree structure
    root = Tree("model")
    build_tree(net, root)

    # Capture plain text of tree
    console = Console(file=io.StringIO(), record=True)
    console.print(root)
    tree_str = console.export_text()

    # Save raw model structure
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(raw_model_str)

    # Save ordered tree structure
    with open(tree_txt_path, "w", encoding="utf-8") as f:
        f.write(tree_str)

    print("All files saved:")
    print(f"  -> {wts_file} (pt >>> wts )")
    print(f"  -> {forward_txt_path} (forward execution order layer info)")
    print(f"  -> {raw_txt_path} (raw model print output)")
    print(f"  -> {tree_txt_path} (ordered tree structure)")

