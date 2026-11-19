
import onnx
from onnx import numpy_helper

model_path = "/Users/humbertosmac/Documents/work/ML-likelihoods/OLLL/ONNX/ATLAS-SUSY-2018-04_V1_20260910.onnx"
model = onnx.load(model_path)

onnx.checker.check_model(model)

print("Graph name:", model.graph.name)
print("Number of nodes:", len(model.graph.node))

print("\nInputs:")
for inp in model.graph.input:
    print(inp.name, inp.type)

print("\nOutputs:")
for out in model.graph.output:
    print(out.name, out.type)

print("\nInitializers (stored weights):")
for weight in model.graph.initializer:
    arr = numpy_helper.to_array(weight)
    print(weight.name, arr.shape)

print("\nOperations:")
for i, node in enumerate(model.graph.node):
    print(i, node.op_type, node.name)
    print("  Inputs:", list(node.input))
    print("  Outputs:", list(node.output))

