import numpy as np
import math

'''Object of the data distribution including requests and vehicles'''
class Distribution():
    '''parameters:
    type: the type of distribution (requests or vehicles)
    step_time: The step time of the simulation system
    record_time: The previous time needed to be recorded
    x_grid_num: the horizontal grid number
    y_grid_num: the vertical grid number
    '''
    def __init__(self,
                type,
                step_time,
                record_time,
                x_grid_num,
                y_grid_num,
                node_coord_to_grid):
        self.type = type
        self.record_steps = int(record_time / step_time)
        self.current_step = 0
        self.y_grid_num = y_grid_num
        self.x_grid_num = x_grid_num
        self.node_coord_to_grid = node_coord_to_grid

        self.distribution = []


    # Update the distribution
    def Update(self, data_list):
        dis_tmp = np.zeros((self.y_grid_num, self.x_grid_num))
        for data in data_list:
            # Update each data
            if self.type == 'requests':
                coord = data.pickup_position
            else:
                coord = data.current_position
            (x_num, y_num) = self.node_coord_to_grid[coord]
            dis_tmp[y_num, x_num] += 1
        
        if self.current_step < self.record_steps:
            self.distribution.append(dis_tmp)
        else:
            self.distribution[self.current_step % self.record_steps] = dis_tmp
        
        self.current_step += 1
        
    
    # Get the mean distribution in the previous record time
    def GetDistribution(self):
        if len(self.distribution) == 0:
            return np.zeros((self.y_grid_num, self.x_grid_num), dtype = np.float32)
        
        dis = np.array(self.distribution, dtype = np.float32)
        dis = np.sum(dis, axis = 0)
        dis = (dis - np.mean(dis)) / (np.std(dis) + 1e-6)
        return dis


'''objective of states'''
class States():
    def __init__(self,
                cfg,
                node_coord_to_grid,
                requests_record_time = 1800,
                vehicle_record_time = None):
        self.cfg = cfg
        self.node_coord_to_grid = node_coord_to_grid
        #self.nodes_coordinate_grid = nodes_coordinate_grid
        self.x_grid_num = self.cfg.ENVIRONMENT.NY.X_GRID_NUM
        self.y_grid_num = self.cfg.ENVIRONMENT.NY.Y_GRID_NUM
        self.step_time = self.cfg.SIMULATION.STEP_TIME
        
        # We input the mean distribution of the requests in the previous 30 minutes
        self.requests_distribution = Distribution(type = 'requests',
                                                step_time = self.step_time,
                                                record_time = requests_record_time,
                                                x_grid_num = self.x_grid_num,
                                                y_grid_num = self.y_grid_num,
                                                node_coord_to_grid = self.node_coord_to_grid)
        # We input the current distribution of the vehicles
        self.vehicles_distribution = Distribution(type = 'vehicles',
                                                step_time = self.step_time,
                                                record_time = self.step_time,
                                                x_grid_num = self.x_grid_num,
                                                y_grid_num = self.y_grid_num,
                                                node_coord_to_grid = self.node_coord_to_grid)
    
    # Get the vehicles' states: veh_grid_list, veh_t_delay, cur_loc
    def Vehicles2States(self, vehicles):
        max_capacity = self.cfg.VEHICLE.MAXCAPACITY
        
        veh_grid_list = np.zeros((len(vehicles), 2*max_capacity+1), dtype = int)
        veh_t_delay = np.zeros((len(vehicles), 2*max_capacity+1), dtype = float)
        cur_loc = np.ones((len(vehicles), 1), dtype = int)

        for idx, vehicle in enumerate(vehicles):
            # current position
            cur_loc[idx] = self.node_coord_to_grid_id(vehicle.current_position)
            veh_grid_list[idx, 0] = self.node_coord_to_grid_id(vehicle.current_position)
            
            if vehicle.path is not None:
                # time delay
                time_delay = vehicle.path.time_delay_to_each_position
                veh_t_delay[idx, 1:len(time_delay)+1] = time_delay
                # grid list
                next_positions = vehicle.path.next_positions
                for ip, pos in enumerate(next_positions):
                    veh_grid_list[idx, ip+1] = self.node_coord_to_grid_id(pos)
        
        assert veh_grid_list.any() >=0 and veh_grid_list.any() <= self.x_grid_num * self.y_grid_num
        
        return [veh_grid_list, veh_t_delay, cur_loc]
    

    # Convert grid coordinate to grid id
    def node_coord_to_grid_id(self, coord):
        (x_num, y_num) = self.node_coord_to_grid[coord]
        grid_id = int(y_num * self.x_grid_num + x_num + 1)
        
        return grid_id

    
    # Get states
    def GetStates(self, vehicles, step):
        states = []
        states_veh = self.Vehicles2States(vehicles)
        states.extend(states_veh)

        cur_t = math.floor(step * self.step_time / self.cfg.MODEL.TIME_INTERVAL)
        cur_t = np.ones((len(vehicles), 1), dtype = int) * cur_t
        veh_dis = self.vehicles_distribution.GetDistribution()
        veh_dis = np.repeat(veh_dis.reshape(1, veh_dis.shape[0], veh_dis.shape[1]), len(vehicles), axis = 0)
        req_dis = self.requests_distribution.GetDistribution()
        req_dis = np.repeat(req_dis.reshape(1, req_dis.shape[0], req_dis.shape[1]), len(vehicles), axis = 0)
        
        states.extend([cur_t, veh_dis, req_dis])
        
        return states
