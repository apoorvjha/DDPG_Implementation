import utility
import Network_Architecture
import Experience_Replay
import Noise_Perturbation
import environment

if __name__ == '__main__':
    # Read configuration
    config = utility.read_configuration()
    # Initializing Critic Network
    critic_network = Network_Architecture.Network(...)
    # Initializing Actor Network
    actor_network = Network_Architecture.Network(...)
    # Initializing Target counterparts of Actor and Critic Networks (Enhanced Learning Stability)
    critic_target_network = Network_Architecture.Network(...)
    actor_target_network = Network_Architecture.Network(...)
    # Initializing Experience Replay Buffer
    buffer = Experience_Replay.ExperienceReplyBuffer(...)
    # Initilize the Simulation Environment
    env = environment.Environment(...)
    # Loop over the number of Episodes
    for episode in range(config["n_episodes"]):
        # Defining Random Process (ETA) for action exploration
        # Mean = 0 and standard deviation = 0.2    
        ETA = Noise_Perturbation.OrnsteinUlembeckNoise(mean = np.zeros(1), std = float(config['std']) * np.ones(1))
        # Observe Initial State
        current_state, is_final = env.reset()
        # Instantiate Timestep variable
        T = 1
        # Loop over Timesteps
        while(is_final):
            # Select action for current timestep
            current_action = actor_network(current_state) + ETA.sample()
            # Observe next state and current reward
            next_state, reward, is_final = env.step()
            # Store the transition information in buffer
            buffer.store(current_state, current_action, next_state, reward)
            # Minibatch training
            if len(buffer) > config["experience_replay_buffer_cutoff"]:
                # Get Minibatch data from buffer
                current_states, current_actions, next_states, rewards = buffer.sample(config["experience_replay_buffer_cutoff"])
                # Get Future Reward estimate from Bellman Equation
                Y = rewards + config["GAMMA"] * critic_target_network([next_states, next_actions])
                # Update the weights of Critic Network (Squared Mean Loss) 
                critic_network.update(Y, critic_network([current_states, current_actions]))
                # Update Actor Network (Mean of Sampled Policy Gradients)
                actor_network.update(critic_network([current_states, current_actions]))
                # Update the target Networks
                critic_target_network.update_parameters(config["TAU"], critic_network.parameters())
                actor_target_network.update_parameters(config["TAU"], actor_network.parameters())
            # Increment Timestep variable
            T += 1
