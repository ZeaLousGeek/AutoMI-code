import numpy as np
import json
import os


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}

        self.actions = ['parameter_evolution', 'structure_update', 'continue_current']

    def _get_state_key(self, state):
        return str(state)

    def select_action(self, state):
        state_key = self._get_state_key(state)

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return int(np.argmax(self.q_table[state_key]))

    def update_q_value(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]

        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def get_action_name(self, action_index):
        if action_index < len(self.actions):
            return self.actions[action_index]
        return f'unknown_action_{action_index}'

    def save(self, filepath):
        data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration_rate': self.min_exploration_rate,
            'q_table': {k: v.tolist() for k, v in self.q_table.items()}
        }
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"RL状态文件不存在: {filepath}，使用默认参数")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        saved_action_size = data['action_size']
        self.state_size = data['state_size']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.exploration_rate = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        self.min_exploration_rate = data['min_exploration_rate']

        if saved_action_size != self.action_size:
            print(f"RL action_size 变更 ({saved_action_size} -> {self.action_size})，迁移 Q-table")
            self.q_table = {
                k: np.array(v)[:self.action_size]
                for k, v in data['q_table'].items()
            }
        else:
            self.q_table = {k: np.array(v) for k, v in data['q_table'].items()}


def create_state_representation(accuracy, accuracy_history, iteration_count):
    accuracy_bucket = max(0, min(int(accuracy * 10), 9))

    if len(accuracy_history) >= 3:
        recent_trend = accuracy - np.mean(accuracy_history[-3:])
    else:
        recent_trend = 0

    trend_bucket = 0 if recent_trend < -0.01 else 2 if recent_trend > 0.01 else 1
    iteration_bucket = max(0, min(iteration_count // 10, 9))

    return (accuracy_bucket, trend_bucket, iteration_bucket)


def calculate_reward(old_accuracy, new_accuracy):
    improvement = new_accuracy - old_accuracy

    if improvement > 0.02:
        return 10
    elif improvement > 0.01:
        return 5
    elif improvement > 0:
        return 2
    elif improvement == 0:
        return 0
    elif improvement > -0.01:
        return -1
    elif improvement > -0.02:
        return -3
    else:
        return -5
