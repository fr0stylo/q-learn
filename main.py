import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, graphVertex: int) -> None:
        self.vertices = [i for i in range(graphVertex)]
        self.graph = np.zeros((graphVertex, graphVertex))

    def generate_edges(self):
        connection_distibution = np.random.normal(
            100*0.5*0.2, np.sqrt(100*0.5*0.5*0.2), size=len(self.vertices)).round(0).astype(int)

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
        startPoint = moveFn(start, np.nonzero(self.graph[start])[0])
        return (start, startPoint, self.graph[start][startPoint])


class TimedGraph:
    def __init__(self, timespan: int, graphVertex: int) -> None:
        self.graphs = [Graph(graphVertex) for _ in range(timespan)]

    def generate_edges(self) -> None:
        for graph in self.graphs:
            graph.generate_edges()

    def go_through_time(self) -> Graph:
        for _, graph in enumerate(self.graphs):
            yield graph

    def print(self) -> None:
        for time, graph in enumerate(self.graphs):
            print(f"Timespan: {time}")
            graph.print()


class QLearning:
    def __init__(self,
                 vertexLength: int,
                 graphGenerator: TimedGraph,
                 start: int,
                 end: int,
                 epochs: int,
                 dispersion: float,
                 epsilion: float,
                 timespan: int,
                 learning_rate: float):
        self.start = start
        self.end = end
        self.generator = graphGenerator
        self.epochs = epochs
        self.rewards = np.full(vertexLength, -1)
        self.rewards[end] = 1
        self.discount = dispersion
        self.epsilion = epsilion
        self.learning_rate = learning_rate
        self.q = np.random.rand(timespan+1, vertexLength, vertexLength)

    def calculate_next_move(self, time):
        def wrapper(start, pointsAvailable):
            if np.random.randn() < self.epsilion:
                index = np.argmax(self.q[time, start, pointsAvailable])
                return pointsAvailable[index]
            else:
                return np.random.choice(pointsAvailable, 1)[0]

        return wrapper

    def get_next_move(self, time):
        def wrapper(start, pointsAvailable):
            index = np.argmax(self.q[time, start, pointsAvailable])
            return pointsAvailable[index]

        return wrapper

    def train(self):
        for epoch in range(self.epochs):
            start = self.start

            for time, graph in enumerate(self.generator.go_through_time(), 1):
                (old_start, new_start, weight) = graph.move_ext(
                    start, self.calculate_next_move(time))

                reward = self.rewards[new_start]
                old_q_value = self.q[time - 1, old_start, new_start]
                temporal_difference = reward + \
                    (self.discount *
                     np.max(self.q[time, new_start])) - old_q_value
                new_q_value = old_q_value + \
                    (self.learning_rate * temporal_difference * (1 - (weight / 1000)))

                self.q[time-1, old_start, new_start] = new_q_value

                # print(f"{old_start}\t---{weight}--->\t{new_start}")

                if new_start == self.end:
                    # print(f"Epoch {epoch} reched end on day {time}")
                    break

                start = new_start
                pass

            if epoch % 50 == 0:
                print(f"Epoch {epoch} done")

    def get_best_path(self, start, end):
        weights = 0
        for time, graph in enumerate(self.generator.go_through_time()):
            (old_start, new_start, weight) = graph.move_ext(
                start, self.get_next_move(time))
            weights += weight
            start = new_start
            print(f"{old_start}\t---{weight}--->\t{new_start}")
            if new_start == end:
                print(f"Reched end on day {time} with route weight {weights}")
                break


if __name__ == "__main__":
    np.random.seed(1)
    start_point = 0
    end_point = 49
    graph_size = 50
    graph_time = 300

    epochs = 1000
    learning_rate = 0.9
    epsilion = 0.9
    discount = 0.9

    gf = TimedGraph(graph_time, graph_size)

    gf.generate_edges()

    model = QLearning(graph_size, gf, start_point, end_point,
                      epochs, discount, epsilion, graph_time, learning_rate)

    model.train()
    model.get_best_path(start_point, end_point)

    start = start_point
    end = end_point
    weightSum = 0
    for i, a in enumerate(gf.go_through_time(), 1):
        (oldStart, start, weight) = a.move(start)
        weightSum += weight
        print(f"{oldStart} ---{weight}--> {start}")
        if start == end_point:
            print(f"Finished in {i} with weight sum {weightSum}")
            break
