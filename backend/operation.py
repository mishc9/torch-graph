import numpy as np
import torch

_graph_registry = {}


class Graph:
    """
    Create a new graph
    """

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        _graph_registry[self.__class__] = self


def get_default_graph():
    return _graph_registry[Graph]


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


class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return torch.log(x_value)


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        return x_value * y_value


class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        if self.axis is None:
            return torch.sum(A_value.view(A_value.numel()))
        return torch.sum(A_value, self.axis)


class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self, dtype=None):
        """Construct placeholder
        """
        if dtype is None:
            self.dtype = torch.float32
        self.consumers = []
        self._output = None

        # Append this placeholder to the list of placeholders in the currently active default graph
        get_default_graph().placeholders.append(self)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = torch.tensor(value, dtype=self.dtype)


class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None, dtype=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.value = torch.tensor(initial_value, dtype=self.dtype)
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        get_default_graph().variables.append(self)


def traverse_postorder(operation):
    """
    >>> Graph().as_default()
    >>> A = placeholder()
    >>> x = placeholder()
    >>> y = matmul(A, x)
    >>> z = matmul(A, x)
    >>> o = add(z, y)
    >>> session = Session()
    >>> session.run(o, {A: [[1, 0], [0, -1]], x: [1, 2]})
    tensor([ 2., -4.])
    """
    nodes_postorder = []
    nodes_set = set()

    def recurse(node):
        if node not in nodes_set:
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_set.add(node)
            nodes_postorder.append(node)

    for op in operation:
        recurse(op)
    # nodes_postorder
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
    tensor([ 2., -1.])
    >>> type(output)
    <class 'torch.Tensor'>
    """

    def run(self, operation, feed_dict=None, **kwargs):
        feed_dict = feed_dict or {}

        operation = to_list(operation)

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if isinstance(node, placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

        return unpack_singleton([op.output for op in operation])


class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value


class softmax(Operation):
    """
    >>> red_points = (np.random.randn(50, 2) - 2 * np.ones((50, 2))).astype(float)
    >>> blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2)).astype(float)
    >>> Graph().as_default()
    >>> X = placeholder()
    >>> c = placeholder()
    >>> W = Variable([[1, -1], [1, -1]])
    >>> b = Variable([0, 0])
    >>> p = softmax(add(matmul(X, W), b))
    >>> J = negative(reduce_sum(multiply(c, log(p)), axis=1))
    >>> session = Session()
    >>> c_value = np.array([[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points), dtype=float)
    >>> x_value = np.concatenate((blue_points, red_points))

    """

    def __init__(self, a):
        """
        Construct softmax

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """
        Compute the output of the softmax operation
        Args:
          a_value: Input value
        """
        return torch.exp(a_value) / torch.sum(torch.exp(a_value), dim=1)[:, None]


def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]


def unpack_singleton(x):
    """Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x


def get_session():
    # https://github.com/tensorflow/tensorflow/blob/81012dcd91770dc8113cd5beb4f854968c27e272/tensorflow/python/keras/_impl/keras/backend.py#L345
    return Session()


class Function(object):
    """Runs a computation graph.
    # https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/backend.py#L2425
    Arguments:
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
    """

    def __init__(self, inputs, outputs, updates=None, name=None,
                 **session_kwargs):
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with ops.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(state_ops.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = control_flow_ops.group(*updates_ops)
        self.name = name
        self.session_kwargs = session_kwargs

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            # if is_sparse(tensor):
            #     sparse_coo = value.tocoo()
            #     indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
            #                               np.expand_dims(sparse_coo.col, 1)), 1)
            #     value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value
        session = get_session()
        updated = session.run(
            self.outputs + [self.updates_op],
            feed_dict=feed_dict,
            **self.session_kwargs)
        return updated[:len(self.outputs)]


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.
    Arguments:
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.
    Returns:
        Output values as Numpy arrays.
    Raises:
        ValueError: if invalid kwargs are passed in.
    """
    if kwargs:
        for key in kwargs:
            if (key not in tf_inspect.getargspec(session_module.Session.run)[0] and
                    key not in tf_inspect.getargspec(Function.__init__)[0]):
                msg = ('Invalid argument "%s" passed to K.function with Tensorflow '
                       'backend') % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)
