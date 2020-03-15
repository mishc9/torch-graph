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
    try:
        return _graph_registry[Graph]
    except KeyError:
        Graph().as_default()
        return _graph_registry[Graph]


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
    >>> output = session.run([z], {x: [1, 2]})
    >>> output
    [tensor([ 2., -1.])]
    """

    def run(self, operation, feed_dict=None, **kwargs):
        """
        >>> var = Variable([1.], dtype=torch.float32)
        >>> ph = placeholder()
        >>> _ = var.assign(ph)
        >>> Session().run(var, feed_dict={ph: [0.]})
        tensor([0.])
        """
        feed_dict = feed_dict or {}

        return_as = None
        if isinstance(operation, list) or isinstance(operation, tuple):
            return_as = True

        operation = to_list(operation)

        nodes_postorder = traverse_postorder(operation)

        # Note #3: assign operation is specific. It's unary operation. And it does
        # the following: placeholder -> (assign) -> Variable.
        # So, I use a trick to ignore assign operations which have placeholders, but
        # these placeholder just have no relative values in a feed_dict.
        #
        # If there's no value for a given placeholder in a dict, you should skip next
        # assign operation and use current value of the Variable.
        skip_next_assign = False
        for index, node in enumerate(nodes_postorder):
            if isinstance(node, placeholder):
                try:
                    node.output = feed_dict[node]
                except KeyError as e:
                    if index < len(nodes_postorder) - 1 and isinstance(nodes_postorder[index + 1], assign):
                        skip_next_assign = True
                    else:
                        raise e
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                if (isinstance(node, assign) and skip_next_assign):
                    skip_next_assign = False
                    continue
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
        result = [op.output for op in operation]

        if return_as is not None:
            return result
        else:
            return unpack_singleton(result)


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


class assign(Operation):
    def __init__(self, assignee, source):
        super().__init__([source])
        self.assignee = assignee
        # assignee.input_nodes.append(self)

    def compute(self, value):
        self.assignee.value = value
        return self.assignee.value

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self, dtype=None, shape=None):
        """Construct placeholder
        """
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        self.consumers = []
        self.shape = shape
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
        if initial_value is not None:
            self.value = torch.tensor(initial_value, dtype=self.dtype)
        else:
            # TODO: what should we do now? Probably I should create initializers
            pass
        self.consumers = []
        # Note #2: there are no such operations by default as _assign_placeholder
        # But they could be set later in set_value() or another vai another function
        # Append this variable to the list of variables in the currently active default graph
        get_default_graph().variables.append(self)

    def assign(self, assign_placeholder):
        """
        :param assign_placeholder:
        :return:
        >>> graph = Graph().as_default()
        >>> var = Variable([1.], dtype=torch.float32)
        >>> ph = placeholder()
        >>> _ = var.assign(ph)
        >>> Session().run(var, feed_dict={ph: [0.]})
        tensor([0.])
        """
        # It is should be an operation, so we couldn't just do
        # self.value = value
        # But I don't know how to reassign value of tensor inplace in pytorch
        # self.value = torch.tensor(value, dtype=self.dtype)
        # source.consumers.append(self)
        # I'm not sure if it's necessary to set private properties here. Because the will
        # be set up in the set_value()
        self._assign_op = assign(self, assign_placeholder)
        self._assign_placeholder = assign_placeholder
        return self._assign_op


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
    # Todo: use ordered set
    nodes_postorder = []
    nodes_set = set()

    def recurse(node):
        # How to connect assign operations (source of variables) to the main graph?
        if node not in nodes_set:
            # if isinstance(node, assign):

            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            if isinstance(node, Variable) and hasattr(node, '_assign_op'):
                recurse(node._assign_op)
            nodes_set.add(node)
            nodes_postorder.append(node)

    for op in operation:
        recurse(op)
    return nodes_postorder


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
        # with ops.control_dependencies(self.outputs):
        #     updates_ops = []
        #     for update in updates:
        #         if isinstance(update, tuple):
        #             p, new_p = update
        #             updates_ops.append(state_ops.assign(p, new_p))
        #         else:
        #             # assumed already an op
        #             updates_ops.append(update)
        #     self.updates_op = control_flow_ops.group(*updates_ops)
        # Todo: recover updates_ops. For a while I use patch.
        self.updates_op = None
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
            self.outputs + ([self.updates_op] if self.updates_op is not None else []),
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
    >>> Graph().as_default()
    >>> A = Variable([[1, 0], [0, -1]])
    >>> b = Variable([1, 1])
    >>> x = placeholder()
    >>> y = matmul(A, x)
    >>> z = add(y, b)
    >>> f = function([x], [z])
    >>> f([[1, 2]])
    [tensor([ 2., -1.])]
    """
    # if kwargs:
    #     for key in kwargs:
    #         if (key not in tf_inspect.getargspec(session_module.Session.run)[0] and
    #                 key not in tf_inspect.getargspec(Function.__init__)[0]):
    #             msg = ('Invalid argument "%s" passed to K.function with Tensorflow '
    #                    'backend') % key
    #             raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)


def _convert_string_dtype(dtype):
    """Get the type from a string.

    # Arguments
        dtype: A string representation of a type.

    # Returns
        The type requested.

    # Raises
        ValueError: if `dtype` is not supported.
    """
    if dtype == 'float16':
        return torch.float16
    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float64':
        return torch.float64
    elif dtype == 'int16':
        return torch.int16
    elif dtype == 'int32':
        return torch.int32
    elif dtype == 'int64':
        return torch.int64
    elif dtype == 'uint8':
        return torch.int8
    elif dtype == 'uint16':
        return torch.uint16  # is it not allowed? # TODO
    else:
        raise ValueError('Unsupported dtype:', dtype)


def set_value(x, value):
    """Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    >>> var = Variable([0.])
    >>> set_value(var, [1.])
    >>> Session().run(var)
    tensor([1.])
    """
    value = np.asarray(value)
    # TODO: make pytorch-specific dtype processing
    tf_dtype = _convert_string_dtype(str(x.dtype).split('.')[1])
    # Note #1
    # tf backend uses _assign_placeholder of Variable to set new values.
    # Guess I don't need this as for now torchy Varibales are much simpler and don't depend
    # on tf's (over?)-complicated conceptions
    # ...
    # Heck NO. Thinking a little I've got that it was probably a simplest solution.
    # BUT as a bonus I can use my own Variable class .assign() operation, so I will do it.
    if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
    else:
        assign_placeholder = placeholder(dtype=tf_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op

    get_session().run(assign_op, feed_dict={assign_placeholder: value})


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    >>> batch = [(Variable([0.]), [1.]), (Variable([1.]), [0.])]
    >>> batch_set_value(batch)
    >>> Session().run([v for v, _ in batch])
    [tensor([1.]), tensor([0.])]
    """
    # Note #4: batch_set_value use same operations as set_value
    if tuples:
        assign_ops = []
        feed_dict = {}
        for x, value in tuples:
            value = np.asarray(value)
            tf_dtype = _convert_string_dtype(str(x.dtype).split('.')[1])
            if hasattr(x, '_assign_placeholder'):
                assign_placeholder = x._assign_placeholder
                assign_op = x._assign_op
            else:
                assign_placeholder = placeholder(dtype=tf_dtype,
                                                 shape=value.shape)
                assign_op = x.assign(assign_placeholder)
                x._assign_placeholder = assign_placeholder
                x._assign_op = assign_op
            assign_ops.append(assign_op)
            feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)
