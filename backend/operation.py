import numpy as np

registry = {}


class Graph:
    """
    # Create a new graph
    """

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        registry[self.__class__] = self


def get_default_graph():
    return registry[Graph]


class Operation:
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes

        self.consumers = []

        for input_node in self.input_nodes:
            input_node.consumers.append(self)

        get_default_graph().operations.append(self)

    def compute(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class add(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, x, y):
        return x + y


class matmul(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, x, y):
        return x @ y


class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        get_default_graph().placeholders.append(self)


class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        get_default_graph().variables.append(self)


def traverse_postorder(operation):
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:
    """
    >>> Graph().as_default()
    >>> A = Variable([[1, 0], [0, -1]])
    >>> b = Variable([1, 1])
    >>> x = placeholder()
    >>> y = matmul(A, x)
    >>> z = add(y, b)
    >>> session = Session()
    >>> output = session.run(z, {x: [1, 2]})
    >>> output
    array([ 2, -1])
    """

    def run(self, operation, feed_dict=None):
        feed_dict = feed_dict or {}

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if isinstance(node, placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

            # Return the requested node value
        return operation.output
