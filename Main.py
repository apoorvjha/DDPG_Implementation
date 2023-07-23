import utility
import Network_Architecture
import Experience_Replay
import Noise_Perturbation
import environment
import numpy as np
import torch

if __name__ == '__main__':
    # Read configuration
    config = utility.read_configuration()
    # Get device
    device = utility.get_device()
    # Initializing Critic Network
    critic_network = torch.nn.DataParallel(Network_Architecture.Network(config, mode = 'critic')).to(device)
    # Initializing Actor Network
    actor_network = torch.nn.DataParallel(Network_Architecture.Network(config)).to(device)
    # Initializing Target counterparts of Actor and Critic Networks (Enhanced Learning Stability)
    critic_target_network = torch.nn.DataParallel(Network_Architecture.Network(config, mode = 'critic')).to(device)
    actor_target_network = torch.nn.DataParallel(Network_Architecture.Network(config)).to(device)
    # Initializing Experience Replay Buffer
    buffer = Experience_Replay.ExperienceReplyBuffer(config)
    # Initilize the Simulation Environment
    env = environment.Environment(config)
    overall_reward = []
    # Loop over the number of Episodes
    for episode in range(config["n_episodes"]):
        # Defining Random Process (ETA) for action exploration
        # Mean = 0 and standard deviation = 0.2    
        ETA = Noise_Perturbation.OrnsteinUlembeckNoise(mean = np.zeros(1), std = float(config['std']) * np.ones(1))
        # Observe Initial State
        current_state, info = env.reset()
        current_state = utility.zero_pad_state(current_state, config)
        # Instantiate Timestep variable
        T = 1
        # Loop over Timesteps
        is_final = True
        episodal_reward = []
        while(is_final):
            if T % config['max_steps_per_episode'] == 0:
                break
            # Select action for current timestep
            current_action = actor_network(torch.from_numpy(current_state).float()).cpu().detach().numpy().reshape(-1) + ETA.sample()
            # Observe next state and current reward
            next_state, reward, is_final, truncated, info = env.step(current_action)
            next_state = utility.zero_pad_state(next_state, config)
            episodal_reward.append(reward)
            is_final = not(is_final or truncated)
            # Store the transition information in buffer
            buffer.store(current_state, current_action, next_state, reward)
            current_state = next_state
            # Minibatch training
            if len(buffer) > config["experience_replay_buffer_cutoff"]:
                # Get Minibatch data from buffer
                current_states, current_actions, next_states, rewards = buffer.sample(config["experience_replay_buffer_cutoff"])
                # Get Future Reward estimate from Bellman Equation
                Y = rewards + config["GAMMA"] * critic_target_network(torch.from_numpy(next_states).float(), torch.from_numpy(current_actions).float())
                # Update the weights of Critic Network (Squared Mean Loss) 
                critic_network.update(Y, critic_network(torch.from_numpy(current_states).float(), torch.from_numpy(current_actions).float()))
                # Update Actor Network (Mean of Sampled Policy Gradients)
                actor_network.update(critic_network(torch.from_numpy(current_states).float(), torch.from_numpy(current_actions).float()))
                # Update the target Networks
                critic_target_network.update_parameters(config["TAU"], critic_network.state_dict())
                actor_target_network.update_parameters(config["TAU"], actor_network.state_dict())
            # Increment Timestep variable
            T += 1
        if len(episodal_reward) > 0:
            avg_reward = sum(episodal_reward) / len(episodal_reward)
            print(f"[EPISODE - {episode}] - Average Reward = {avg_reward}")
            overall_reward.append(avg_reward)
        utility.plot(episodal_reward, "Timesteps", "Reward", f"Episodal_Reward_Progression_{episode}", config)
    utility.plot(overall_reward, "Episode", "Average Reward", "Average_Reward_Per_Episode", config)