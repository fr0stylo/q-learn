from collections import deque
import random

from graph import TimedGraph

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, feat_in):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(feat_in),
            # nn.LSTMCell(feat_in, 2),
            nn.Linear(feat_in, feat_in * H),
            nn.Tanh(),
            nn.Linear(feat_in * H, feat_in * H),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(feat_in * H, 1),
            nn.Tanhshrink()
        )

    def forward(self, x):
        result = self.layers(x)
        return result


class DQN:
    def __init__(self,
                 vertex_length: int,
                 graph_generator: TimedGraph,
                 start: int,
                 end: int,
                 timespan: int):
        self.start = start
        self.end = end
        self.time_span = timespan
        self.generator = graph_generator
        self.rewards = np.full(vertex_length, 0.1)
        self.rewards[end] = 1

        self.model = Model(4)
        self.model_optimizer = optim.AdamW(
            self.model.parameters(), lr=0.0001, amsgrad=True)
        self.criterion_model = nn.MSELoss()

        self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)

    def upadate_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_next_move(self, state):
        def wrapper(start, pointsAvailable, weights):
            if epsilon > np.random.rand():
                index = random.choice(range(len(pointsAvailable)))

                return pointsAvailable[index]
            else:
                (total_weight, episode_reward, time) = state
                mapped_state = torch.Tensor(
                    [[total_weight, episode_reward, weights[i], time] for i, _ in enumerate(weights)])

                return pointsAvailable[np.argmax(self.model(mapped_state).detach().numpy())]

        return wrapper

    def train_epoch(self):
        global epsilon
        if len(self.replay_memory) < REPLAY_BATCH:
            return

        batch = random.sample(self.replay_memory, REPLAY_BATCH)

        x = np.array([state[0:4] for _, state in enumerate(batch)])
        y = np.array([state[4] for _, state in enumerate(batch)])

        self.model_optimizer.zero_grad()  # zero the gradient buffers
        output = self.model(torch.Tensor(x))
        loss = self.criterion_model(output.flatten(), torch.Tensor(y))
        loss.backward()
        self.model_optimizer.step()

        epsilon *= EPSILON_DECAY

    def calculate_move_quality(self, total_weight, episode_reward, weight, time_left):
        return (1 / np.clip(total_weight, 0.1, 999999999999) - 1 / np.clip(weight, 1e-50, 5e3) * (1 / np.clip(time_left, 1, 999999999999))) * (episode_reward * DISCOUNT)

    def train(self):
        global epsilon
        for epoch in range(EPOCH_COUNT):
            start = self.start
            episode_reward = 0
            total_weight = 0

            for time, graph in enumerate(self.generator.go_through_time(), 1):

                (_, new_start, weight) = graph.move_ext(
                    start, self.get_next_move((total_weight, episode_reward, self.generator.time_span - time)))

                episode_reward += self.rewards[new_start]
                q = self.calculate_move_quality(
                    total_weight, episode_reward, weight, self.generator.time_span - time)
                if weight == 0:
                    episode_reward -= 1000
                    q -= 2e9
                # print(q)
                total_weight += weight

                if new_start == self.end:
                    episode_reward += 100
                    print(f"Epoch {epoch} found way, fitment {episode_reward}")

                self.upadate_replay_memory(
                    (total_weight, episode_reward, weight, self.generator.time_span - time, q))
                self.train_epoch()

                if new_start == self.end:
                    break

                start = new_start

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} done with epsilon {epsilon}, end {self.end}, fitment: {episode_reward}")

    def get_best_model_steps(self, f_start, end_point):
        global epsilon
        exepselon = epsilon
        epsilon = 0
        episode_reward = 0
        total_weight = 0
        start = f_start
        for time, graph in enumerate(self.generator.go_through_time(), 1):
            (old_start, new_start, weight) = graph.move_ext(
                start, self.get_next_move((total_weight, episode_reward, self.generator.time_span - time)))
            episode_reward += self.rewards[new_start]
            total_weight += weight

            print(f"Day {time}: {old_start} --{weight}--> {new_start}")

            if new_start == end_point:
                print(
                    f"Finished in day {time} with weight {total_weight}, model fitness {episode_reward}")
                break

            start = new_start
        epsilon = exepselon


H = 100
REPLAY_MEM_SIZE = 50_00
REPLAY_BATCH = 32
DISCOUNT = 0.8
UPDATE_TARGET_EVERY = 5
epsilon = 1
EPSILON_DECAY = 0.99975
EPOCH_COUNT = 1001

if __name__ == "__main__":
    np.random.seed(1)
    start_point = 0
    end_point = 49
    graph_size = 60
    graph_time = 50

    gf = TimedGraph(graph_time, graph_size)

    gf.generate_edges()

    model = DQN(graph_size, gf, start_point, end_point, graph_time)

    model.train()

    model.get_best_model_steps(start_point, end_point)

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
