import numpy as np
from mlproj_manager.problems.reinforcement_learning.abstract_rl_environment import RLEnvironment


class MountainCar(RLEnvironment):
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float32
    Reward = -1 at every step

    Summary Name: steps_per_episode
    """

    def __init__(self, normalize_state=False):
        super(MountainCar, self).__init__(normalize_state=normalize_state)

        # current state of the environment
        self.step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float32)
        # environment constants
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.position_limits = [-1.2, 0.5]
        self.velocity_limits = [-0.07, 0.07]
        self.action_dictionary = {0: -1,    # accelerate backwards
                                  1: 0,     # coast
                                  2: 1}     # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        self.step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        new_state = self.normalize(self.current_state) if self.normalize_state else self.current_state
        return new_state

    " Update environment "
    def step(self, action):
        self.step_count += 1

        if action not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[action]
        reward = -1.0
        terminate = False

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * np.cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        self.current_state = np.array((position, velocity), dtype=np.float64)

        if self.normalize_state:
            return self.normalize(self.current_state), reward, terminate
        else:
            return self.current_state, reward, terminate

    def normalize(self, state):
        """ normalize to [-1, 1] """
        temp_state = np.zeros(shape=2, dtype=np.float64)
        den = (self.position_limits[1] - self.position_limits[0]) / 2
        shift = (self.position_limits[0] + self.position_limits[1]) / 2
        temp_state[0] = (state[0] - shift) / den

        temp_state[1] = (state[1]) / self.velocity_limits[1]
        return temp_state


def main():
    import matplotlib.pyplot as plt

    def pumping_action(state: list, epsilon=0.5):
        """ this policy moves in the direction of the velocity """
        p = np.random.rand()
        if p < epsilon:
            return np.random.randint(0,3)
        if state[1] < 0:
            return 0    # accelerate backwards
        elif state[1] > 0:
            return 2    # accelerate forward
        else:
            return 1    # coast

    # initialize environment
    env = MountainCar(normalize_state=True)

    # training variables
    num_steps = 10000

    # for summaries
    reward_per_episode = []
    current_cumulative_reward = 0
    epsilon_decrease_rate = 0.9999
    print_limit = np.inf

    for i in range(num_steps):
        # get current state and select action
        current_state = env.get_current_state()
        action = pumping_action(current_state, epsilon_decrease_rate ** i)

        # execute action and print transition
        new_state, reward, terminate = env.step(action)
        if i < print_limit:
            print("Old state:", np.round(current_state, 3), "-->",
                  "Action:", action, "-->",
                  "New state:", np.round(new_state, 3))

        # update summaries
        current_cumulative_reward += reward
        if terminate:
            reward_per_episode.append(current_cumulative_reward)
            current_cumulative_reward *= 0
            env.reset()
            print_limit = i

    # print and plot summaries
    num_episodes = len(reward_per_episode)
    print("Number of episodes completed: {0}".format(num_episodes))
    print("Largest reward:\t{0}\tSmallest reward:\t{1}".format(max(reward_per_episode), min(reward_per_episode)))
    if num_episodes > 1:
        plt.plot(np.arange(num_episodes), reward_per_episode)
        plt.show(); plt.close()

if __name__ == '__main__':
    main()

# if __name__ == "__main__":
#     config = Config()
#     config.store_summary = True
#     config.max_actions = 100
#
#     summary = {}
#     actions = 3
#
#     env = MountainCar(config, summary=summary)
#
#     steps = 100
#     cumulative_reward = 0
#     terminations = 0
#     successful_episode_steps = []
#
#     for i in range(steps):
#         action = np.random.randint(actions)
#         old_state = env.get_current_state()
#         new_state, reward, terminate, timeout = env.step(action)
#         print("Old state:", np.round(old_state, 3), "-->",
#               "Action:", action, "-->",
#               "New state:", np.round(new_state, 3))
#         cumulative_reward += reward
#         if terminate or timeout:
#             print("\n## Reset ##\n")
#             if terminate:
#                 terminations += 1
#                 successful_episode_steps.append(env.step_count)
#             env.reset()
#
#     print("Number of steps per episode:", summary['steps_per_episode'])
#     print("Number of episodes that reached the end:", terminations)
#     average_length = np.average(successful_episode_steps) if len(successful_episode_steps) > 0 else np.inf
#     print("The average number of steps per episode was:", average_length)
#     print("Cumulative reward:", cumulative_reward)