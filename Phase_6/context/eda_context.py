class EDAContext:

    def __init__(self):
        self.logs = []

    def add_step(self, step):

        self.logs.append(step)

    def get_context(self):

        return self.logs