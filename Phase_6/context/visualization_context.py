class VisualizationContext:

    def __init__(self):

        self.charts = []

    def add_chart(
        self,
        chart_name,
        description
    ):

        self.charts.append({

            "chart": chart_name,
            "description": description

        })

    def get_context(self):

        return self.charts