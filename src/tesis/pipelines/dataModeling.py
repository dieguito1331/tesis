from kedro.pipeline import node, Pipeline
from tesis.nodes import nodes

def create_pipeline(**kwargs):
    node1 = node(func=nodes.trainModels, inputs = "parameters", outputs= None)
    node2 = node(func=nodes.modelSelection, inputs="parameters", outputs= None)

    return Pipeline([node1, node2])
