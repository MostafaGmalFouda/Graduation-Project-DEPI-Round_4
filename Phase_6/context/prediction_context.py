class PredictionContext:

    def __init__(self):

        self.history = []

    def add_prediction(self, prediction):

        self.history.append(prediction)

    def get_context(self):

        return self.history