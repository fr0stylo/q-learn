from collections import deque
import random

from graph import TimedGraph

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use gpu: {torch.cuda.is_available()}, device {device}")


class Model(nn.Module):
    def __init__(self, feat_in):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(feat_in),
            # nn.LSTMCell(feat_in, 2),
            nn.Linear(feat_in, H),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(H, 1),
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
        self.rewards = np.full(vertex_length, -1)
        self.rewards[end] = 100

        self.model = Model(FEATURES_COUNT)
        self.model = self.model.to(device)
        self.criterion_model = nn.MSELoss()
        self.model_optimizer = optim.Adam(
            self.model.parameters(), lr=1e-4)

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

                return pointsAvailable[np.argmax(self.model(mapped_state.to(device)).cpu().detach().numpy())]

        return wrapper

    def train_epoch(self):
        global epsilon
        if len(self.replay_memory) < REPLAY_BATCH:
            return

        batch = random.sample(self.replay_memory, REPLAY_BATCH)
        random.shuffle(batch)
        x = np.array([state[0:FEATURES_COUNT]
                      for _, state in enumerate(batch)])
        y = np.array([state[FEATURES_COUNT] for _, state in enumerate(batch)])

        output = self.model(torch.Tensor(x).to(device))
        loss = self.criterion_model(
            output.flatten(), torch.Tensor(y).to(device))
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        self.model.eval()
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY

        return loss

    def calculate_move_quality(self, total_weight, episode_reward, weight, time_left):
        return (1 / np.clip(weight, 1e-1, 5e3) * (1/np.clip(time_left, 1, 9e10))) * (episode_reward * DISCOUNT) / (np.clip(total_weight, 1e-1, 9e10) / (np.clip(time_left, 1, 9e10)))

    def train(self):
        global epsilon
        times_found_end = 0
        avg_loss = []
        for epoch, _ in enumerate(range(EPOCH_COUNT), 1):
            start = self.start
            episode_reward = 0
            total_weight = 0
            for time, graph in enumerate(self.generator.graphs, 1):

                (_, new_start, weight) = graph.move_ext(
                    start, self.get_next_move((total_weight, episode_reward, self.generator.time_span - time)))

                episode_reward += self.rewards[new_start]
                q = self.calculate_move_quality(
                    total_weight, episode_reward, weight, self.generator.time_span - time)
                # print(q)
                total_weight += weight

                self.upadate_replay_memory(
                    (total_weight, episode_reward, weight, self.generator.time_span - time, q))
                loss = self.train_epoch()

                if loss:
                    avg_loss.append(loss.cpu().detach().numpy())

                if new_start == self.end:
                    # print(f"Epoch {epoch} found way, fitment {episode_reward}")
                    times_found_end = times_found_end + 1
                    break

                start = new_start

            if epoch % 100 == 0:
                al = np.array(avg_loss).mean()
                print(
                    f"Epoch {epoch} done with epsilon {epsilon} loss: {al} found {times_found_end}")
                torch.save(
                    self.model, f"models/{EPOCH_COUNT}-{H}-{MIN_EPSILON}-{epoch}-{al}.pt")

                avg_loss = []
                times_found_end = 0

    def get_best_model_steps(self, f_start, end_point):
        global epsilon
        exepselon = epsilon
        epsilon = 0
        episode_reward = 0
        total_weight = 0
        start = f_start
        for time, graph in enumerate(self.generator.graphs, 1):
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


H = 256
REPLAY_MEM_SIZE = 50_00
REPLAY_BATCH = 10 * 128
DISCOUNT = 0.7
epsilon = 1
EPSILON_DECAY = 1-25e-5
EPOCH_COUNT = 100
MIN_EPSILON = 0.05
FEATURES_COUNT = 4

if __name__ == "__main__":
    np.random.seed(1)
    start_point = 0
    end_point = 49
    graph_size = 50
    graph_time = 100

    gf = TimedGraph(graph_time, graph_size)

    gf.generate_edges()

    model = DQN(graph_size, gf, start_point, end_point, graph_time)

    model.train()

    # model.get_best_model_steps(start_point, end_point)
    # model.get_best_model_steps(8, end_point)
    # model.get_best_model_steps(9, end_point)
    # model.get_best_model_steps(20, end_point)
    # model.get_best_model_steps(20, end_point)
    # model.get_best_model_steps(20, end_point)

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
