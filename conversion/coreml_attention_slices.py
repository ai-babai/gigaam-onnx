"""Freeze GigaAM Zapformer projection slice boundaries for ORT Core ML."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx
from onnx import external_data_helper, helper, numpy_helper


PROJECTION_BIASES = (
    ".self_attn.qk_in_proj.bias",
    ".self_attn.vg_in_proj.bias",
    ".conv_module.in_proj.bias",
)


def projection_bias(node, producers, initializers):
    if node.op_type != "Add":
        return None
    for name in node.input:
        if name.endswith(PROJECTION_BIASES):
            if name not in initializers:
                cast = producers[name]
                if cast.op_type != "Cast":
                    raise ValueError(f"Unexpected projection bias producer: {cast.name}")
                name = cast.input[0]
            bias = initializers[name]
            if len(bias.dims) != 1 or bias.dims[0] <= 1:
                raise ValueError(f"Unexpected projection bias shape: {bias.name}")
            return bias
    return None


def slice_index(name, projection, width, producers, initializers):
    if name in initializers:
        value = numpy_helper.to_array(initializers[name])
    else:
        node = producers[name]
        if node.op_type == "Constant":
            value = numpy_helper.to_array(helper.get_attribute_value(node.attribute[0]))
        elif node.op_type == "Gather":
            shape = producers[node.input[0]]
            index = slice_index(node.input[1], projection, width, producers, initializers)
            if (
                shape.op_type == "Shape"
                and not shape.attribute
                and shape.input[0] == projection
                and index in (-1, 2)
            ):
                return width
            raise ValueError(f"Slice depends on a dimension other than projection channels: {node.name}")
        else:
            left, right = (
                slice_index(arg, projection, width, producers, initializers)
                for arg in node.input
            )
            if node.op_type == "Add":
                return left + right
            if node.op_type == "Mul":
                return left * right
            if node.op_type == "Div" and left >= 0 and right > 0:
                return left // right
            raise ValueError(f"Unexpected slice-index operation: {node.name}")
    if value.dtype.kind not in "iu" or value.size != 1:
        raise ValueError(f"Expected one integer slice index: {name}")
    return int(value.item())


def freeze_projection_slices(model):
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    count = 0
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        projection = node.input[0]
        parent = producers.get(projection)
        if parent is None:
            continue
        bias = projection_bias(parent, producers, initializers)
        if bias is None:
            continue

        width = bias.dims[0]
        start, end, axis, *steps = (
            slice_index(name, projection, width, producers, initializers)
            for name in node.input[1:]
        )
        if axis not in (-1, 2) or steps not in ([], [1]) or not 0 <= start < end <= width:
            raise ValueError(f"Unexpected projection slice: {node.name}")
        # ORT Core ML can return a wrong channel count for the dynamic split.
        for slot, value in enumerate((start, end, 2), start=1):
            name = f"{node.output[0]}.coreml_index_{slot}"
            if name in initializers or name in producers:
                raise ValueError(f"Slice index already exists: {name}")
            tensor = numpy_helper.from_array(np.array([value], dtype=np.int64), name)
            model.graph.initializer.append(tensor)
            node.input[slot] = name
        count += 1
    if count == 0:
        raise ValueError("No GigaAM projection slices found")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.absolute()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    model = onnx.load(source, load_external_data=False)
    weight_files = {
        external_data_helper.ExternalDataInfo(tensor).location
        for tensor in model.graph.initializer
        if external_data_helper.uses_external_data(tensor)
    }
    for name in sorted(weight_files):
        original, companion = source.parent / name, output.parent / name
        if not companion.is_file() or not original.samefile(companion):
            parser.error(f"Link the source weight file to {companion} before conversion")
    count = freeze_projection_slices(model)
    with NamedTemporaryFile(dir=output.parent, suffix=".onnx") as temp:
        onnx.save(model, temp.name)
        onnx.checker.check_model(temp.name)
        output.hardlink_to(temp.name)
    print(f"{count} constant projection slices: {output}")


if __name__ == "__main__":
    main()
