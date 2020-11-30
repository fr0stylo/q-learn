from collections import deque
import random

from graph import Graph, TimedGraph

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, verticesSize):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(REPLAY_BATCH,  out_features=REPLAY_BATCH * H),
            nn.ReLU(),
            nn.Linear(REPLAY_BATCH * H, REPLAY_BATCH * H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(REPLAY_BATCH * H, REPLAY_BATCH)
        )

    def forward(self, x):
        print(x.shape)
        return self.layers(x.flatten())


H = 100
REPLAY_MEM_SIZE = 50_00
REPLAY_BATCH = 32
DISCOUNT = 0.8
UPDATE_TARGET_EVERY = 5


class DQN:
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
        self.rewards = np.full(vertexLength, 0.1)
        self.rewards[end] = 1
        self.discount = dispersion
        self.epsilion = epsilion
        self.learning_rate = learning_rate
        self.q = np.random.rand(timespan+1, vertexLength, vertexLength)

        self.model = Model(vertexLength)
        self.model_optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, amsgrad=True)
        self.criterion_model = nn.MSELoss()

        self.target_model = Model(vertexLength)
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)
        self.target_update_counter = 0

    def upadate_replay_memory(self, transition):
        self.replay_memory.append(transition)

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
            if self.epsilion > np.random.rand():
                index = random.choice(range(len(pointsAvailable)))

                return pointsAvailable[index]
            else:
                return self.get_qs(pointsAvailable)

        return wrapper

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def trainEpoch(self):
        if len(self.replay_memory) < REPLAY_BATCH:
            return

        batch = random.sample(self.replay_memory, REPLAY_BATCH)

        current_state = np.array([state[0] for _, state in enumerate(batch)])
        current_q_list = self.model(torch.Tensor(current_state))

        new_current_states = np.array([state[2]
                                       for _, state in enumerate(batch)])
        future_qs_list = self.target_model(torch.Tensor(new_current_states))

        X = []
        y = []

        for index, (current_state, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[new_current_state])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_q_list[index]
            current_qs[new_current_state] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        self.model_optimizer.zero_grad()   # zero the gradient buffers
        output = self.model(torch.Tensor(X))
        loss = self.criterion_model(output, y)
        loss.backward()
        self.model_optimizer.step()

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def train(self):
        for epoch in range(self.epochs):
            start = self.start
            episode_reward = 0
            for time, graph in enumerate(self.generator.go_through_time(), 1):

                (old_start, new_start, weight) = graph.move_ext(
                    start, self.calculate_next_move(time))
                done = new_start == self.end

                self.upadate_replay_memory(
                    (old_start, self.rewards[new_start], new_start, done))
                self.trainEpoch()

                start = new_start

                # reward = self.rewards[new_start]
                # old_q_value = self.q[time - 1, old_start, new_start]
                # temporal_difference = reward + \
                #     (self.discount *
                #      np.max(self.q[time, new_start])) - old_q_value
                # new_q_value = old_q_value + \
                #     (self.learning_rate * temporal_difference * (1 - (weight / 1000)))

                # self.q[time-1, old_start, new_start] = new_q_value

                # # print(f"{old_start}\t---{weight}--->\t{new_start}")

                if new_start == self.end:
                    episode_reward += 100
                    break
                start = new_start
                pass

            if(start != self.end):
                episode_reward -= 200

            if epoch % 50 == 0:
                print(f"Epoch {epoch} done")

    # def get_best_path(self, start, end):
    #     weights = 0
    #     for time, graph in enumerate(self.generator.go_through_time()):
    #         (old_start, new_start, weight) = graph.move_ext(
    #             start, self.get_next_move(time))
    #         weights += weight
    #         start = new_start
    #         print(f"{old_start}\t---{weight}--->\t{new_start}")
    #         if new_start == end:
    #             print(f"Reched end on day {time} with route weight {weights}")
    #             break


if __name__ == "__main__":
    np.random.seed(1)
    start_point = 0
    end_point = 49
    graph_size = 50
    graph_time = 30

    epochs = 1000
    learning_rate = 0.9
    epsilion = 0.9
    discount = 0.9

    gf = TimedGraph(graph_time, graph_size)

    gf.generate_edges()

    model = DQN(graph_size, gf, start_point, end_point,
                epochs, discount, epsilion, graph_time, learning_rate)

    model.train()
    # model.get_best_path(start_point, end_point)

    # start = start_point
    # end = end_point
    # weightSum = 0
    # for i, a in enumerate(gf.go_through_time(), 1):
    #     (oldStart, start, weight) = a.move(start)
    #     weightSum += weight
    #     print(f"{oldStart} ---{weight}--> {start}")
    #     if start == end_point:
    #         print(f"Finished in {i} with weight sum {weightSum}")
    #         break
