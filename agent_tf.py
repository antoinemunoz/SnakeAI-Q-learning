import random
import numpy as np
import tensorflow as tf
from collections import deque
from game import SnakeGameAI, Direction, Point
from model_tf import QNetwork, QTrainer

class AgentConfig:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001
    GAMMA = 0.9
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = AgentConfig.EPSILON
        self.memory = deque(maxlen=AgentConfig.MAX_MEMORY)
        self.model = QNetwork(11, 256, 3)
        self.trainer = QTrainer(self.model, AgentConfig.LR, AgentConfig.GAMMA)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        batch = random.sample(self.memory, min(len(self.memory), AgentConfig.BATCH_SIZE))
        states, actions, rewards, next_states, dones = zip(*batch)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            tf.expand_dims(state, axis=0),
            tf.expand_dims(action, axis=0),
            tf.expand_dims(tf.cast(reward, tf.float32), axis=0),
            tf.expand_dims(next_state, axis=0),
            tf.expand_dims(tf.cast(done, tf.float32), axis=0)
        )

    def get_action(self, state):
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            prediction = self.model(state_tensor)
            move = tf.argmax(prediction[0]).numpy()
            final_move[move] = 1
        if self.epsilon > AgentConfig.EPSILON_MIN:
            self.epsilon *= AgentConfig.EPSILON_DECAY
        return final_move


def train():
    agent = Agent()
    game = SnakeGameAI()
    record = 0
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_weights('best_model_tf.weights.h5')

            print(f"Game {agent.n_games}, Score {score}, Record: {record}")

if __name__ == '__main__':
    train()