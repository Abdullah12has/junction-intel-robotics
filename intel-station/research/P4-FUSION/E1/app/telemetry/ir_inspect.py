import collections
import json
import sys
import xml.etree.ElementTree as ET


def inspect(xml_path):
    root = ET.parse(xml_path).getroot()
    layers = root.find("layers")
    params, results = [], []
    types = collections.Counter()
    const_prec = collections.Counter()
    for layer in layers.findall("layer"):
        t = layer.get("type")
        types[t] += 1
        if t == "Parameter":
            port = layer.find("output/port")
            params.append({"name": layer.get("name"), "shape": [d.text for d in port.findall("dim")], "precision": port.get("precision")})
        elif t == "Result":
            port = layer.find("input/port")
            results.append({"name": layer.get("name"), "shape": [d.text for d in port.findall("dim")], "precision": port.get("precision")})
        elif t == "Const":
            const_prec[layer.find("output/port").get("precision")] += 1
    return {
        "file": xml_path,
        "layer_count": len(layers.findall("layer")),
        "parameters": params,
        "results": results,
        "const_precision_counts": dict(const_prec),
        "op_types_top15": types.most_common(15),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(inspect(p), indent=1))
