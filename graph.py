import numpy as np


class Graph:
    def __init__(self, graphVertex: int) -> None:
        self.vertices = [i for i in range(graphVertex)]
        self.graph = np.zeros((graphVertex, graphVertex))

    def generate_edges(self):
        connection_distibution = np.random.normal(
            100 * 0.5 , np.sqrt(100 * 0.5 * 0.5), size=len(self.vertices)).round(0).astype(int)

        for i, _ in enumerate(self.vertices):
            subset = np.random.choice(
                self.vertices, np.clip(connection_distibution[i], 0, len(self.vertices) - 1))

            def set_number(
                    x): self.graph[i][x] = np.random.randint(10, 200)

            [set_number(subset_index)
             for _, subset_index in enumerate(subset)]

    def print(self):
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                if self.graph[i][j] != 0:
                    print(self.vertices[i], " -> ", self.vertices[j],
                          " edge weight: ", self.graph[i][j])

    def move(self, start: int):
        indexes = np.nonzero(self.graph[start])[0]
        startPoint = np.argmin(self.graph[start, indexes])

        return (start, indexes[startPoint], self.graph[start][indexes[startPoint]])

    def move_ext(self, start, moveFn):
        startPoint = moveFn(start, np.nonzero(self.graph[start])[0], self.graph[start][np.nonzero(self.graph[start])[0]])
        return (start, startPoint, self.graph[start][startPoint])


class TimedGraph:
    def __init__(self, time_span: int, graph_vertex: int) -> None:
        self.graphs = [Graph(graph_vertex) for _ in range(time_span)]
        self.time_span = time_span

    def generate_edges(self) -> None:
        for graph in self.graphs:
            graph.generate_edges()

    def go_through_time(self):
        for _, graph in enumerate(self.graphs):
            yield graph

    def print(self) -> None:
        for time, graph in enumerate(self.graphs):
            print(f"Timespan: {time}")
            graph.print()
