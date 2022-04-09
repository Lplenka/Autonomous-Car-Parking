import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics


class generate_graphs:
    def __init__(self):
        self.helper_method()

    def helper_method(self):
        plt.rc('font', size=18)
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3f}".format(x)})
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def process_data(self, data):
        sac_velocities_ep1, sac_rewards_ep1, sac_avg_velocity, sac_number_of_steps, episode_list = self.process_sac_data(
            data)
        imi_velocities_ep1, imi_rewards_ep1, imi_avg_velocity, imi_number_of_steps = self.process_imi_data(
            data)
        gen_velocities_ep1, gen_rewards_ep1, gen_avg_velocity, gen_number_of_steps = self.process_gen_data(
            data)

        self.plot_graphs(sac_velocities_ep1, sac_rewards_ep1, imi_velocities_ep1, imi_rewards_ep1, gen_velocities_ep1, gen_rewards_ep1,
                         sac_avg_velocity, sac_number_of_steps, imi_avg_velocity, imi_number_of_steps, gen_avg_velocity, gen_number_of_steps, episode_list)

    def process_sac_data(self, data):
        sac_velocities_ep1 = []
        sac_rewards_ep1 = []
        sac_avg_velocity = []
        sac_number_of_steps = []
        episode_list = []
        if 'SAC' in data.keys():
            sac_data = data["SAC"]
            sac_velocities_ep1 = sac_data[1][0]
            sac_rewards_ep1 = sac_data[1][1]
            count = 1
            for each_episode in sac_data.values():
                sac_avg_velocity.append(statistics.mean(each_episode[0]))
                sac_number_of_steps.append(len(each_episode[1]))
                episode_list.append(count)
                count += 1

        return sac_velocities_ep1, sac_rewards_ep1, sac_avg_velocity, sac_number_of_steps, episode_list

    def process_imi_data(self, data):
        imi_velocities_ep1 = []
        imi_rewards_ep1 = []
        imi_avg_velocity = []
        imi_number_of_steps = []
        if 'IMI' in data.keys():
            imi_data = data["IMI"]
            imi_velocities_ep1 = imi_data[1][0]
            imi_rewards_ep1 = imi_data[1][1]
            for each_episode in imi_data.values():
                imi_avg_velocity.append(statistics.mean(each_episode[0]))
                imi_number_of_steps.append(len(each_episode[1]))

        return imi_velocities_ep1, imi_rewards_ep1, imi_avg_velocity, imi_number_of_steps

    def process_gen_data(self, data):
        gen_velocities_ep1 = []
        gen_rewards_ep1 = []
        gen_avg_velocity = []
        gen_number_of_steps = []
        if 'GEN' in data.keys():
            gen_data = data["GEN"]
            gen_velocities_ep1 = gen_data[1][0]
            gen_rewards_ep1 = gen_data[1][1]
            for each_episode in gen_data.values():
                gen_avg_velocity.append(statistics.mean(each_episode[0]))
                gen_number_of_steps.append(len(each_episode[1]))

        return gen_velocities_ep1, gen_rewards_ep1, gen_avg_velocity, gen_number_of_steps

    def plot_graphs(self, sac_velocities_ep1, sac_rewards_ep1, imi_velocities_ep1, imi_rewards_ep1, gen_velocities_ep1, gen_rewards_ep1,
                    sac_avg_velocity, sac_number_of_steps, imi_avg_velocity, imi_number_of_steps, gen_avg_velocity, gen_number_of_steps, episode_list):
        self.plot_velocity_vs_steps(
            sac_velocities_ep1, imi_velocities_ep1, gen_velocities_ep1)
        self.plot_rewards_vs_steps(
            sac_rewards_ep1, imi_rewards_ep1, gen_rewards_ep1)
        self.plot_metrics(sac_avg_velocity, sac_number_of_steps, imi_avg_velocity,
                          imi_number_of_steps, gen_avg_velocity, gen_number_of_steps, episode_list)

    def plot_metrics(self, sac_avg_velocity, sac_number_of_steps, imi_avg_velocity, imi_number_of_steps, gen_avg_velocity, gen_number_of_steps, episode_list):
        print('Average Velocity For SAC ==> ', sac_avg_velocity)
        print('Average Velocity For IMI ==> ', imi_avg_velocity)
        print('Average Velocity For GEN ==> ', gen_avg_velocity)

        plt.figure()
        gen_avg_velocity_new = gen_avg_velocity*len(sac_number_of_steps)
        plt.plot(episode_list, sac_avg_velocity,  color='blue', label='SAC')
        plt.plot(episode_list, imi_avg_velocity,
                 color='orange', label='Imitation')
        plt.plot(episode_list, gen_avg_velocity_new,
                 color='green', label='Genetic')
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Average Velocity', fontsize=16)
        plt.title('Avergae Velocity vs Episodes', fontsize=16)
        plt.grid(True)
        h, l = plt.gca().get_legend_handles_labels()
        o = [0, 1, 2]
        plt.legend([h[i] for i in o], [l[i] for i in o])
        plt.savefig("Avergae Velocity vs Episodes.png")
        # plt.show()

        plt.figure()
        gen_number_of_steps_new = gen_number_of_steps*len(sac_number_of_steps)
        plt.plot(episode_list, sac_number_of_steps, color='blue', label='SAC')
        plt.plot(episode_list, imi_number_of_steps,
                 color='orange', label='Imitation')
        plt.plot(episode_list, gen_number_of_steps_new,
                 color='green', label='Genetic')
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Total Number of steps', fontsize=16)
        plt.title('Total Number of steps vs Episodes', fontsize=16)
        plt.grid(True)
        h, l = plt.gca().get_legend_handles_labels()
        o = [0, 1, 2]
        plt.legend([h[i] for i in o], [l[i] for i in o])
        plt.savefig("Total Number of steps vs Episodes.png")
        # plt.show()

    def plot_velocity_vs_steps(self, sac_velocities_ep1, imi_velocities_ep1, gen_velocities_ep1):
        plt.figure()
        plt.plot(sac_velocities_ep1, color='blue', label='SAC')
        plt.plot(imi_velocities_ep1, color='orange', label='Imitation')
        plt.plot(gen_velocities_ep1, color='green', label='Genetic')
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('Velocity', fontsize=16)
        plt.title('Velocity vs steps', fontsize=16)
        plt.grid(True)
        h, l = plt.gca().get_legend_handles_labels()
        o = [0, 1, 2]
        plt.legend([h[i] for i in o], [l[i] for i in o])
        plt.savefig("Velocity vs steps.png")
        # plt.show()

    def plot_rewards_vs_steps(self, sac_rewards_ep1, imi_rewards_ep1, gen_rewards_ep1):
        print('Cumalative reward for SAC ==> ', np.cumsum(sac_rewards_ep1))
        print('Cumalative reward for IMI ==> ', np.cumsum(imi_rewards_ep1))
        print('Cumalative reward for GEN ==> ', np.cumsum(gen_rewards_ep1))
        plt.figure()
        plt.plot(sac_rewards_ep1, color='blue', label='SAC')
        plt.plot(imi_rewards_ep1, color='orange', label='Imitation')
        plt.plot(gen_rewards_ep1, color='green', label='Genetic')
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('Rewards', fontsize=16)
        plt.title('Rewards vs steps', fontsize=16)
        plt.grid(True)
        h, l = plt.gca().get_legend_handles_labels()
        o = [0, 1, 2]
        plt.legend([h[i] for i in o], [l[i] for i in o])
        plt.savefig("Rewards vs steps.png")
        # plt.show()
