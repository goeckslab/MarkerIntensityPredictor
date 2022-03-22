class CustomLayer:
    # The activation for the layer
    activation: str
    # The dimensions of the layer
    dimensions: int
    # The name of the layer
    name: str

    def __init__(self, name: str, dimensions: int, activation: str = "relu"):
        self.activation = activation
        self.dimensions = dimensions
        self.name = name
