class ModelContext:

    def __init__(self):

        self.data = {}

    def update(self, **kwargs):

        self.data = kwargs

    def get_context(self):

        return self.data