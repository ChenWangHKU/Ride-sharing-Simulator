from array import array
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .PrioritizedMemory import Memory


# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class MODEL(nn.Module):
    def __init__(self,
                cfg,
                total_grid_num,
                total_time_step):
        super(MODEL, self).__init__()
        
        self.cfg = cfg
        self.total_grid_num = total_grid_num
        self.total_time_step = total_time_step
        self.loc_embed_num = self.cfg.MODEL.LOCATION_EMBED_NUM
        self.time_embed_num = self.cfg.MODEL.TIME_EMBED_NUM
        self.max_capacity = self.cfg.VEHICLE.MAXCAPACITY


        self.path_input_dim = (2 * self.max_capacity + 1) * (self.loc_embed_num + 1)
        
        self.embedding1 = nn.Embedding(self.total_grid_num + 1, self.loc_embed_num)
        self.fc1 = nn.Sequential(
            nn.Linear(self.path_input_dim, 200),
            nn.ReLU()
        )

        self.embedding2 = nn.Embedding(total_time_step, self.time_embed_num)
        
        self.fc2 = nn.Sequential(
            nn.Linear(200+self.loc_embed_num+self.time_embed_num, 300),
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2*total_grid_num, 300),
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU()
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(300*2, 100),
            nn.ReLU(),
            nn.Linear(100,1)
        )

    # params
    # veh_grid_list: the current grid id list of the vehicle: batch_size * (2 * max_capacity + 1)
    # veh_t_delay: the delay time of each node in the current state
    # cur_loc: the current grid id of the vehicle
    # cur_t: the current timepoint of the simulation system
    # veh_dis: the distribution of vehicles in the previous 15 minutes
    # req_dis: the distribution of requests in the previous 15 minutes
    def forward(self, state):
        veh_grid_list, veh_t_delay, cur_loc, cur_t, veh_dis, req_dis = state
        # # data type
        # veh_grid_list = veh_grid_list.type(torch.int)
        # veh_t_delay = veh_t_delay.type(torch.float32)
        # cur_loc = cur_loc.type(torch.int)
        # cur_t = cur_t.type(torch.int)
        # veh_dis = veh_dis.type(torch.float32)
        # req_dis = req_dis.type(torch.float32)

        batch_size = veh_grid_list.shape[0]
        '''matching'''
        path_emedb = self.embedding1(veh_grid_list) # batch_size * (2 * max_capacity + 1) * loc_embed_num
        path_ori_inp = torch.cat((path_emedb, veh_t_delay.unsqueeze(-1)), axis = -1)
        path_ori_inp = path_ori_inp.view(batch_size, -1)
        path_ori = self.fc1(path_ori_inp) # batch_size * 300
        

        # the current location's embbeding
        cur_loc_embed = self.embedding1(cur_loc).squeeze() # batch_size * loc_embed_num
        # the current time's embbeding
        cur_t_embed = self.embedding2(cur_t).squeeze() # batch_size * time_embed_num

        matching_input = torch.cat((path_ori, cur_loc_embed, cur_t_embed), axis = -1)
        m_inp = self.fc2(matching_input) # batch_size * 300

        '''repositioning'''
        veh_dis = veh_dis.view(batch_size, -1)
        req_dis = req_dis.view(batch_size, -1) # batch_size * total_grid_num
        repositioning_inp = torch.cat((veh_dis, req_dis), axis = 1) # batch_size * (2*total_grid_num)
        r_inp = self.fc3(repositioning_inp) # batch_size * 300
        
        '''combination'''
        inp = torch.cat((m_inp, r_inp), axis = 1).type(torch.float) # batch_size * 600

        value = self.fc4(inp) # batch_size * 1

        return value



# Agent for the Ride-pooling
# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class Agent():
    def __init__(self, 
                cfg,
                total_grid_num,
                total_time_step):
        self.cfg = cfg
        self.total_grid_num = total_grid_num
        self.total_time_step = total_time_step

        # These are hyper parameters for the DQN
        self.discount_factor = self.cfg.MODEL.DISCOUNT_FACTOR
        self.learning_rate = self.cfg.MODEL.LEARNING_RATE
        self.memory_size = self.cfg.MODEL.MEMORY_SIZE
        self.batch_size = self.cfg.MODEL.BATCH_SIZE
        self.train_frequency = self.cfg.MODEL.TRAIN_FREQUENCY
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = 'cpu'

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = MODEL(cfg,
                        total_grid_num,
                        total_time_step)
        self.model = self.model.to(self.device)
        self.model.apply(self.weights_init)
        self.target_model = MODEL(cfg,
                            total_grid_num,
                            total_time_step)
        self.target_model = self.target_model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Convert the components of the state to tensor and send them to the specific device
    def state2tensor(self, state, device = None):
        if device is None:
            device = self.device
        
        state_tensor = []
        for item in state:
            if not isinstance(item, array):
                item = np.array(item)
            if item.dtype == 'float64':
                item = item.astype(np.float32)
            item = torch.from_numpy(item)
            item = item.to(device)
            state_tensor.append(item)
        
        return state_tensor


    # Score the state during the simulation
    def get_value(self, state):
        state = self.state2tensor(state)
        value = self.model(state)
        value = value.detach().cpu().numpy()

        return value


    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, states, actions, rewards, next_states, done):
        states_torch = self.state2tensor(states, self.device)
        next_states_torch = self.state2tensor(next_states, self.device)
        
        value = self.model(states_torch).detach().cpu().numpy()
        target_value = self.target_model(next_states_torch).detach().cpu().numpy()
        
        rewards = np.array(rewards).reshape(len(rewards), 1)
        
        if done.any():
            target_value = rewards
        else:
            target_value = rewards + self.discount_factor * target_value
        
        # We use the mean TD Difference of all vehicles to calculate the priority of the experience
        error = np.mean(abs(value - target_value))

        self.memory.add(error, [states, actions, rewards, next_states, done])


    # Convert the samples to formats that can be handeled by the model
    def FormatSampleBatch(self, batch):
        # [[1,2],[3,4]] --> [[1,3], [2,4]]
        def TransposeList(batch):
            new_batch = [[] for _ in range(len(batch[0]))]
            for sample in batch:
                for i in range(len(sample)):
                    new_batch[i].append(sample[i])
            
            return new_batch
        
        # concatenate state batch
        def FormatState(state):
            for i in range(len(state)):
                item = np.array(state[i])
                state[i] = np.vstack(item)
            
            return state


        # states, actions, rewards, next_states, dones
        batch = TransposeList(batch)
        # veh_grid_list, veh_t_delay, cur_loc, cur_t, veh_dis, req_dis 
        batch[0] = FormatState(TransposeList(batch[0])) # states
        batch[3] = FormatState(TransposeList(batch[3])) # next_states

        return batch


    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.FormatSampleBatch(mini_batch)
        
        rewards = np.vstack(np.array(rewards, dtype = np.float32))
        # bool to binary
        dones = np.vstack(np.array(dones, dtype = np.float32))
        
        # Value of current states and next states
        states_torch = self.state2tensor(states, self.device)
        next_states_torch = self.state2tensor(next_states, self.device)
        pred = self.model(states_torch)
        target = self.target_model(next_states_torch)
        
        # Convert data format
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        is_weights = torch.from_numpy(is_weights).to(self.device)
        
        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * target
        
        errors = torch.abs(pred - target).detach().cpu().numpy()
        errors = errors.reshape(self.batch_size, -1)
        errors = np.mean(errors, axis = 1) # batch_size

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()
        
        # MSE Loss function
        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = is_weights * loss.view(self.batch_size,-1).mean(axis = 1)
        loss = loss.mean()
        #loss = (is_weights * F.mse_loss(pred, target, reduction = 'none').view(self.batch_size,-1).mean(axis = 0)).mean()
        loss.backward()

        # and train
        self.optimizer.step
        