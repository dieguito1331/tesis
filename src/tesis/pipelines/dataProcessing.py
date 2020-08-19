from kedro.pipeline import node, Pipeline
from tesis.nodes import nodes

def create_pipeline(**kwargs):
    node1 = node(func=nodes.createSimpleVariables, inputs="train", outputs="trainDataframe")
    node2 = node(func=nodes.createSimpleVariables, inputs="test", outputs="testDataframe")
    node3 = node(func=nodes.createBayesianSingleVariables, inputs=["trainDataframe", "testDataframe", "parameters"], 
                outputs = ["trainDataframeBayesian", "testDataframeBayesian"])
    node4 = node(func=nodes.createBayesianMultipleVariables, inputs=["trainDataframeBayesian", "testDataframeBayesian", "parameters"], 
                outputs = ["finalTrain", "finalTest"] )
    node5 = node(func=nodes.createFolds, inputs=["finalTrain", "finalTest", "parameters"], 
                outputs = "terminoDataProcessing")
    return Pipeline([node1, node2, node3, node4, node5])

