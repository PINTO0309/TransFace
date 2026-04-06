import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


class FeatureOnlyWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        feat, _, _ = self.net(x)
        return feat


def fold_bn_after_matmul(model, bn_node_names):
    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    node_map = {node.name: node for node in graph.node}
    replacement_nodes = {}

    for bn_name in bn_node_names:
        bn_node = node_map.get(bn_name)
        if bn_node is None or bn_node.op_type != "BatchNormalization":
            continue

        matmul_output, scale_name, bias_name, mean_name, var_name = bn_node.input
        matmul_node = None
        for node in graph.node:
            if node.op_type == "MatMul" and node.output and node.output[0] == matmul_output:
                matmul_node = node
                break
        if matmul_node is None:
            continue

        weight_name = matmul_node.input[1]
        weight = numpy_helper.to_array(initializer_map[weight_name]).copy()
        scale = numpy_helper.to_array(initializer_map[scale_name])
        bias = numpy_helper.to_array(initializer_map[bias_name])
        mean = numpy_helper.to_array(initializer_map[mean_name])
        var = numpy_helper.to_array(initializer_map[var_name])

        epsilon = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f
                break

        alpha = scale / np.sqrt(var + epsilon)
        folded_weight = weight * alpha
        folded_bias = bias - mean * alpha

        initializer_map[weight_name].CopyFrom(
            numpy_helper.from_array(folded_weight.astype(weight.dtype), name=weight_name)
        )

        bias_init_name = f"{bn_name}_bias"
        graph.initializer.append(
            numpy_helper.from_array(folded_bias.astype(np.float32), name=bias_init_name)
        )
        add_node = helper.make_node(
            "Add",
            inputs=[matmul_output, bias_init_name],
            outputs=list(bn_node.output),
            name=f"{bn_name}_folded_add",
        )
        replacement_nodes[bn_name] = add_node

    if not replacement_nodes:
        return model

    updated_nodes = []
    for node in graph.node:
        updated_nodes.append(replacement_nodes.get(node.name, node))
    del graph.node[:]
    graph.node.extend(updated_nodes)
    return model


def remove_unused_initializers(model):
    used_names = set()
    for node in model.graph.node:
        used_names.update(node.input)
    kept_initializers = [init for init in model.graph.initializer if init.name in used_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    return model

def convert_onnx(net, path_module, output, opset=17, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()
    export_net = FeatureOnlyWrapper(net)
    export_net.eval()
    torch.onnx.export(
        model=export_net,
        args=img,
        f=output,
        input_names=["data"],
        output_names=["feat"],
        dynamic_axes={"data": {0: "N"}, "feat": {0: "N"}},
        opset_version=opset,
        dynamo=False,
        external_data=False,
    )
    model = onnx.load(output)
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    model = fold_bn_after_matmul(
        model,
        bn_node_names=[
            "/net/feature/feature.1/BatchNormalization",
            "/net/feature/feature.3/BatchNormalization",
        ],
    )
    model = remove_unused_initializers(model)
    model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    onnx.save(model, output)


if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=True, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify)
