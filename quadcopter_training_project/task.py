import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_x = self.target_pos[0]
        self.target_y = self.target_pos[1]
        self.target_z = self.target_pos[2]
        
        self.greatest_distance_to_target = 0.0

    def get_reward(self, debug):
        """Uses current pose of sim to return reward."""
        '''
        ### Standard get_reward ###
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
        '''
        
        # Calculate relative distance to target in comparison to greatest distance 
        # to target recorded
        distances_xyz = abs(self.sim.pose[:3] - self.target_pos)
        distance_to_target = distances_xyz.sum()
        if distance_to_target > self.greatest_distance_to_target:
            self.greatest_distance_to_target = distance_to_target
            
        # Range of 'relative_distance_distance_to_target': 0-1
        relative_distance_to_target = distance_to_target / self.greatest_distance_to_target
        
        reward_distance = 1.0 - relative_distance_to_target # range: 0-1
        
        
        # Calculate if the distribution of velocities corresponds to the distribution of distances (representing if the quadcopter moves into the right direction
        distances_xyz = abs(self.sim.pose[:3] - self.target_pos)
        distances_xyz_normalized = (distances_xyz-min(distances_xyz))/(max(distances_xyz)-min(distances_xyz))
        
        velocities_normalized = (self.sim.v-min(self.sim.v))/(max(self.sim.v)-min(self.sim.v))
        
        diff_between_distances_velocities = abs(distances_xyz_normalized - velocities_normalized).sum()
        
        reward_velocity = 2.0 - diff_between_distances_velocities # range: 0-2
        
        
        # Add to toral reward
        reward = reward_distance + reward_velocity # range: 0-3
        
        
        
        
        
        
        
        
        '''
        current_x = self.sim.pose[0]
        current_y = self.sim.pose[1]
        current_z = self.sim.pose[2]

        # Weigh velocities according to their normalized distance (and therefore importance that Quadcopter goes into according direction
        velocity_rewards_xyz = (self.sim.pose[:3] - self.target_pos) * self.sim.v * -distances_xyz_normalized
        
        reward_elements = [distances_xyz.sum(), velocity_rewards_xyz.sum()]
        reward_elements_normalized = (reward_elements-min(reward_elements))/(max(reward_elements)-min(reward_elements))
        
        reward = 1.0 + reward_elements_normalized[0]*reward_elements[0] + reward_elements_normalized[1]*reward_elements[1]
        '''
        
        
        
        if debug:
            print("### GET_REWARD OUTPUT:")
            print("self.sim.pose: {}".format(self.sim.pose))
            print("self.sim.v: {}".format(self.sim.v))
            print("self.target_pos: {}".format(self.target_pos))
            print("relative_distance_to_target: {}".format(relative_distance_to_target))
            print("diff_between_distances_velocities: {}".format(diff_between_distances_velocities))
            print("reward: {}".format(reward))
        
        return reward

    def step(self, rotor_speeds, debug=False):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(debug) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state