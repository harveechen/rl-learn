import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable

from tqdm import tqdm

class Bandit:
    def __init__(self, k: int, non_stationary=False):
        self.k = k
        self.non_stationary = non_stationary
        self.q_star = np.random.normal(0.0, 1.0, size=k)

    def pull(self, action: int) -> float:
        return np.random.normal(self.q_star[action], 1.0)

    def update(self):
        if self.non_stationary:
            self.q_star += np.random.normal(0.0, 0.01, size=self.k)


class Agent:
    def __init__(self, k: int):
        self.k = k

    def select_action(self):
        raise NotImplementedError

    def update(self, action: int, reward: float):
        raise NotImplementedError

    def reset(self):
        pass


class EpsilonGreedyAgent(Agent):
    def __init__(self, k: int, epsilon: float, alpha=0, optimistic=False):
        super().__init__(k)
        self.epsilon = epsilon
        self.optimistic = optimistic
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        if self.optimistic:
            self.Q += 5
        self.alpah = alpha


    def select_action(self):
        if self.epsilon > 0 and np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        if self.alpah > 0:
            self.Q[action] += (reward - self.Q[action]) * self.alpah
        else:
            self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        if self.optimistic:
            self.Q += 5


class UCBAgent(Agent):
    def __init__(self, k: int, c: float):
        super().__init__(k)
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0
        

    def select_action(self):
        self.t += 1
        with np.errstate(divide='ignore'):
            ucb_values = self.Q + self.c * np.sqrt(np.log(self.t + 1) / (self.N + 1e-5))
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0


class SoftmaxAgent(Agent):
    def __init__(self, k: int, tau: float):
        super().__init__(k)
        self.tau = tau
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_action(self):
        prefs = self.Q / self.tau
        prefs -= np.max(prefs)  # 数值稳定
        probs = np.exp(prefs)
        probs /= np.sum(probs)
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)


class GradientBanditAgent(Agent):
    def __init__(self, k: int, alpha: float, use_baseline=True):
        super().__init__(k)
        self.alpha = alpha
        self.H = np.zeros(k)  # preferences
        self.pi = np.ones(k) / k
        self.avg_reward = 0
        self.time = 0
        self.use_baseline = use_baseline

    def select_action(self):
        exp_h = np.exp(self.H - np.max(self.H))  # 数值稳定
        self.pi = exp_h / np.sum(exp_h)
        return np.random.choice(self.k, p=self.pi)

    def update(self, action, reward):
        self.time += 1
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.time
            baseline = self.avg_reward
        else:
            baseline = 0
        for a in range(self.k):
            self.H[a] += self.alpha * (reward - baseline) * ((1 if a == action else 0) - self.pi[a])

    def reset(self):
        self.H = np.zeros(self.k)
        self.pi = np.ones(self.k) / self.k
        self.avg_reward = 0
        self.time = 0


class Experiment:
    def __init__(self, k=10, steps=1000, runs=2000, non_stationary=False):
        self.k = k
        self.steps = steps
        self.runs = runs
        self.non_stationary = non_stationary
        self.agents: Dict[str, Callable[[], Agent]] = {}

    def add_agent(self, name: str, agent_factory):
        self.agents[name] = agent_factory

    def run(self):
        self.avg_rewards = {name: np.zeros(self.steps) for name in self.agents}
        self.opt_action_percents = {name: np.zeros(self.steps) for name in self.agents}

        for name, create_agent in self.agents.items():
            for run in tqdm(range(self.runs), total=self.runs):
                bandit = Bandit(self.k, non_stationary=self.non_stationary)
                optimal_action = np.argmax(bandit.q_star)
                agent = create_agent()
                agent.reset()
                for t in range(self.steps):
                    action = agent.select_action()
                    reward = bandit.pull(action)
                    agent.update(action, reward)
                    self.avg_rewards[name][t] += reward
                    if action == optimal_action:
                        self.opt_action_percents[name][t] += 1
                    bandit.update()
                    if self.non_stationary:
                        optimal_action = np.argmax(bandit.q_star)

            self.avg_rewards[name] /= self.runs
            self.opt_action_percents[name] = (self.opt_action_percents[name] / self.runs) * 100

    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        for name, rewards in self.avg_rewards.items():
            axs[0].plot(rewards, label=name)
        axs[0].set_ylabel("Average Reward")
        axs[0].legend()
        axs[0].grid(True)

        for name, opt_actions in self.opt_action_percents.items():
            axs[1].plot(opt_actions, label=name)
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("% Optimal Action")
        axs[1].legend()
        axs[1].grid(True)

        title = f"{self.k}-Armed Bandit ({self.runs} runs)"
        if self.non_stationary:
            title += " [Non-Stationary]"
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


def exp2_8():
    exp = Experiment(k=10, steps=1000, runs=2000, non_stationary=False)
    exp.add_agent("UCB c=2", lambda: UCBAgent(10, c=2))
    exp.add_agent("ε=0.1", lambda: EpsilonGreedyAgent(10, epsilon=0.1, alpha=0.1))
    exp.run()
    exp.plot()

def exp2_5():
    exp = Experiment(k=10, steps=1000, runs=2000, non_stationary=False)
    exp.add_agent("Gradient alpha=0.1", lambda: GradientBanditAgent(10, alpha=0.1))
    exp.add_agent("Gradient alpha=0.1 no baseline", lambda: GradientBanditAgent(10, alpha=0.1, use_baseline=False))
    exp.add_agent("Gradient alpha=0.4", lambda: GradientBanditAgent(10, alpha=0.4))
    exp.add_agent("Gradient alpha=0.4 no baseline" , lambda: GradientBanditAgent(10, alpha=0.4, use_baseline=False))
    exp.run()
    exp.plot()

if __name__ == "__main__":
    # exp = Experiment(k=10, steps=1000, runs=2000, non_stationary=False)
    # exp.add_agent("optimistic, ε=0", lambda: EpsilonGreedyAgent(10, epsilon=0, alpha=0.1, optimistic=True))
    # exp.add_agent("ε=0.1", lambda: EpsilonGreedyAgent(10, epsilon=0.1, alpha=0.1))
    # exp.add_agent("ε=0.1 alpha=0.1", lambda: EpsilonGreedyAgent(10, epsilon=0.1, alpha=0.1))
    # exp.add_agent("ε=0.01", lambda: EpsilonGreedyAgent(10, epsilon=0.01))
    # exp.add_agent("UCB c=2", lambda: UCBAgent(10, c=2))
    # exp.add_agent("Softmax τ=0.1", lambda: SoftmaxAgent(10, tau=0.1))
    # exp.add_agent("Gradient α=0.1", lambda: GradientBanditAgent(10, alpha=0.1))

    # exp.run()
    # exp.plot()

    exp2_5()
