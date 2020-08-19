from kedro.pipeline import node, Pipeline
from tesis.nodes import nodes


def create_pipeline(**kwargs):
    node1 = node(func=nodes.createMetaReco, inputs="parameters", outputs="terminoMetaReco")

    return Pipeline([node1])
