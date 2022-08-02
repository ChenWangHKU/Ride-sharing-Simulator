from cmath import cos, sin
import numpy as np
import osmnx as ox
import math
import random


'''
The object of the environment of interest
The object only provides the static physical information
The dynamic process is realized in the control cnter
'''
class EnvironmentNYModel:
    def __init__(self,
                network_file_path = None,
                vehicle_velocity = 25 / 3.6,
                travel_time_file_path = None,
                travel_distance_file_path = None,
                consider_itinerary = False,
                cfg = None
                ):
        self.cfg = cfg
        self.vehicle_velocity = vehicle_velocity
        
        # get file path
        self.network_file_path = network_file_path
        self.travel_time_file_path = travel_time_file_path
        self.travel_distance_file_path =travel_distance_file_path
        
        # We do not consider congestion at the first stage
        self.consider_congestion = self.cfg.ENVIRONMENT.CONSIDER_CONGESTION
        self.x_grid_num = self.cfg.ENVIRONMENT.NY.X_GRID_NUM
        self.y_grid_num = self.cfg.ENVIRONMENT.NY.Y_GRID_NUM
        
        # Note: Using osmnx' API to generate itinerary nodes may be too slow to run the simulation for many epochs
        # We can generate the itinerary nodes between each pair of nodes ahead of time and save the results at mongodb
        # that we can call when we want to generate itinerary nodes, which will accelerate the simulation
        self.consider_itinerary = consider_itinerary
        
        # Initialize road network
        self.road_network, self.node_coord_to_id, self.node_id_to_coord, self.nodes_coordinate, self.nodes_connection = self.InitializeEnvironment()
        self.node_coord_to_grid, self.nodes_coordinate_grid = self.SplitGrids()
        self.node_lnglat_to_xy = self.LngLat2xy()

        


    # function: Initialize road network, including shortest path, traval time, travel distance, etc.
    # params: data director or database API
    # return: converting node id -> node coordinate and back
    def InitializeEnvironment(self):
        # Load road network: nodes and edges from graphml file
        G = ox.load_graphml(self.network_file_path)
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        # Load nodes' id and coordinate
        nodes_id = gdf_nodes.index.tolist()
        nodes_lng = gdf_nodes['x'].tolist()
        nodes_lat = gdf_nodes['y'].tolist()
        # mutual conversion
        node_id_to_coord, node_coord_to_id = {}, {}
        nodes_coordinate = []
        for idx in range(len(nodes_lng)):
            node_coord_to_id[(nodes_lng[idx], nodes_lat[idx])] = nodes_id[idx]
            node_id_to_coord[nodes_id[idx]] = (nodes_lng[idx], nodes_lat[idx])
            nodes_coordinate.append((nodes_lng[idx], nodes_lat[idx]))
        
        # connections
        nodes_connection = []
        nodes_u, nodes_v, _ = list(zip(*gdf_edges.index))
        for node_u, node_v in zip(nodes_u, nodes_v):
            nodes_connection.append((node_u, node_v)) # node_id

        return G, node_coord_to_id, node_id_to_coord, nodes_coordinate, nodes_connection

    
    # function: Split the environment to grids
    # params: The horizontal and vertical grid number of the environment
    # return: node coordinate to grid N.o. & node coordinate in each grid
    def SplitGrids(self):
        node_coord_to_grid = {}
        nodes_coordinate_grid = np.zeros((self.y_grid_num, self.x_grid_num), dtype = list)
        nodes_coord_np = np.array(self.nodes_coordinate)
        # The border of the environment
        lng_max, lng_min = np.max(nodes_coord_np[:,0]), np.min(nodes_coord_np[:,0])
        lat_max, lat_min = np.max(nodes_coord_np[:,1]), np.min(nodes_coord_np[:,1])
        # distance of each grid
        delta_x = (lng_max - lng_min) / self.x_grid_num
        delta_y = (lat_max - lat_min) / self.y_grid_num
        # associate each node of the environment to a grid
        for (lng, lat) in self.nodes_coordinate:
            x_num = math.floor((lng - lng_min) / delta_x)
            y_num = math.floor((lat - lat_min) / delta_y)
            
            if x_num == self.x_grid_num:
                x_num -= 1
            if y_num == self.y_grid_num:
                y_num -= 1
            
            node_coord_to_grid[(lng, lat)] = (x_num, y_num)
            # connect each node with grid
            if nodes_coordinate_grid[y_num, x_num] == 0:
                nodes_coordinate_grid[y_num, x_num] = [(lng,lat)]
            else:
                nodes_coordinate_grid[y_num, x_num].append((lng, lat))
        
        return node_coord_to_grid, nodes_coordinate_grid

    
    # function: Convert coordinate system from longtitude & latitude to x & y
    # params: None
    # return: dict of converting (lng, lat) to (x,y)
    # Note: It's computing expensive to calculate distance usiing longtitude and latitude, so that we convert nodes' coordinate to xy
    def LngLat2xy(self):
        node_lnglat_to_xy = {}
        x0, y0 = np.mean(np.array(self.nodes_coordinate), axis = 0)

        # Convert angle system to radian system
        ori_lng, ori_lat = x0 * math.pi / 180., y0 * math.pi / 180.
        Earth_R = 6371393 # unit: meter
        
        for (lng, lat) in self.nodes_coordinate:
            des_lng, des_lat = lng * math.pi / 180., lat * math.pi / 180.
            # Distance
            dis_EW = Earth_R * math.acos(min(1, math.cos(ori_lat)**2 * math.cos(ori_lng - des_lng) + math.sin(ori_lat)**2))
            dis_NS = Earth_R * abs(ori_lat - des_lat)
            x = dis_EW * np.sign(des_lng - ori_lng)
            y = dis_NS * np.sign(des_lat - ori_lat)

            node_lnglat_to_xy[(lng, lat)] = (x, y)
            
        return node_lnglat_to_xy


    # function: get 8 (or less 8) repositioning locations nearby according to the vehicle's location
    # params: the vehicle's location
    # return: repositioning coordinates, grid coordinates, repositioning distance and time
    def GetRepositionLocation(self, vehicle_location):
        reposition = []
        #vehicle_location[0] = round(vehicle_location[0], 7)
        vx, vy = self.node_coord_to_grid[vehicle_location]
        v_grid = vy * self.x_grid_num + vx + 1 # current(vehicle) grid id
        grids = [(vy-1, vx), (vy-1, vx+1), (vy, vx+1), (vy+1, vx+1), (vy+1, vx), (vy+1, vx-1), (vy, vx-1), (vy-1, vx-1)]
        for (ry, rx) in grids:
            # Check the bound
            if ry >=0 and ry < self.y_grid_num and rx >= 0 and rx < self.x_grid_num:
                # Choose a repositioning node randomly in the grid
                coord_list = self.nodes_coordinate_grid[ry, rx]
                if isinstance(coord_list, list): # Make sure there exists a node in the grid
                    lng, lat = coord_list[int(random.random()*len(coord_list))] # repositioning node coordinate
                    distance, time = self.GetDistanceandTime(vehicle_location, (lng, lat))
                    r_grid = ry * self.x_grid_num + rx + 1 # repositioning grid id
                    reposition.append((lng, lat, v_grid, r_grid, distance, time))
        
        return reposition


    # function: Given a position, find the nearest road node
    # params: The position's coordinate (lng, lat)
    # return: the nearest road node' coordinate (lng, lat)
    def GetNearestNode(self, node):
        nearest_dis = 99999999
        nearest_node = None
        for road_noad in self.nodes_coordinate:
            dis, _ = self.GetDistanceandTime(node, road_noad)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_node = road_noad
        assert nearest_node is not None
        
        return nearest_node
    
    
    # function: Calculate the travel distance and time between origin and destination according to the type
    # params: The origin and destination position, type: 'Linear', 'Manhattan' or 'Itinerary'
    # return: the travel distance and time
    ''' 
        Note: Since it is not necessary to consider itinerary nodes, i.e., Manhattan distance is good enough (e.g., check constraints),
        we allow users to choose whether considering itinerary nodes or not when calculating distance.
    '''
    def GetDistanceandTime(self, origin, destination, type = 'Linear', congestion_factor = 1.0):
        # if the input origin and destination are node id, then we convert them to coordinate
        if not isinstance(origin, tuple):
            origin = self.node_id_to_coord[origin]
        if not isinstance(destination, tuple):
            destination = self.node_id_to_coord[destination]
        
        # if len(str(origin[0]).split('.')[1]) > 7 or len(str(origin[1]).split('.')[1]) > 7:
        #     origin = (round(origin[0], 7), round(origin[1], 7))
        # if len(str(destination[0]).split('.')[1]) > 7 or len(str(destination[1]).split('.')[1]) > 7:
        #     destination = (round(destination[0], 7), round(destination[1], 7))
        
        x1, y1 = self.node_lnglat_to_xy[origin]
        x2, y2 = self.node_lnglat_to_xy[destination]
        
        # convert lat and lng to distance
        if type == 'Linear': # linear distance
            dis = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type == 'Manhattan': # Manhattan distance
            dis = abs(x1 - x2) + abs(y1 - y2)
        else:
            raise NotImplementedError
        # Here congestion_factoe = 1 means we don't consider congestion
        time = dis / self.vehicle_velocity

        return dis, time

        

    # function: Calculate or export the travel path between the origin and the destination
    # params: The origin and destination position
    # return: The travel path (intersections) between two positions
    def GetItinerary(self, origin, destination):
        if origin == destination: # The origin is same as destination so that distance = 0, time = 0
            return [origin, destination], [0], [0]
        
        if isinstance(origin, tuple):
            origin = self.node_coord_to_id[origin]
        if isinstance(destination, tuple):
            destination = self.node_coord_to_id[destination]
            # origin = ox.get_nearest_node(self.road_network, origin)
            # destination = ox.get_nearest_node(self.road_network, destination)
        itinerary = ox.distance.shortest_path(self.road_network, origin, destination, weight='length', cpus=16)
        
        if itinerary is None: # We only consider the origin and destination if the API can't find the itinerary
            itinerary = [origin , destination]
        
        # Calculate distance and time
        dis , time = [], []
        for node_idx in range(len(itinerary) - 1):
            d, t = self.GetDistanceandTime(itinerary[node_idx], itinerary[node_idx + 1])
            dis.append(d)
            time.append(t)

        return list(itinerary), dis, time
        

    # function: Simulate or export road congestion 
    # params: todo...
    # return: todo...
    # Note: it may be too complicated to consider the road congestion, so that we may not consider the congestion at the first development stage
    def GetCongestion(self):
        raise NotImplementedError




'''
The object of the environment of interest
The object only provides the static physical information
The dynamic process is realized in the control cnter
'''
class EnvironmentToyModel:
    def __init__(self,
                num_nodes = 10,
                distance_per_line = 1000,
                vehicle_velocity = 20/3.6,
                consider_congestion = False
                ):
        self.num_nodes = num_nodes
        self.distance_per_line = distance_per_line
        self.vehicle_velocity = vehicle_velocity
        self.consider_congestion = consider_congestion
        self.nodes_coordinate, self.nodes_connection = None, None
        # Initialize network
        self.nodes_coordinate, self.nodes_connection = self.InitializeEnvironment()

    
    # function: Initialize road network, including shortest path, traval time, travel distance, etc.
    # params: data director or database API
    # return: num_nodes * 3: [node_id, x, y]; List[(node_id, node_id)]: connection
    def InitializeEnvironment(self):
        total_num_nodes = self.num_nodes ** 2
        nodes_coordinate = np.zeros((total_num_nodes, 3))
        nodes_connection = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                nodes_coordinate[i*self.num_nodes + j, 0] = i * self.num_nodes + j + 1 # node_id
                nodes_coordinate[i*self.num_nodes + j, 1] = j * self.distance_per_line # x coordinate (positive to right)
                nodes_coordinate[i*self.num_nodes + j, 2] = i * self.distance_per_line # y coordinate (positive to down)
        for i in range(total_num_nodes-1):
            for j in range(i+1, total_num_nodes):
                if self.GetTravelDistance(nodes_coordinate[i,0], nodes_coordinate[j,0]) <= self.distance_per_line:
                    nodes_connection.append((i,j))
        return nodes_coordinate, nodes_connection


    # function: Calculate or export the shortest travel time between the origin and the destination
    # params: The origin and destination position
    # return: The shortest travel time between two positions
    def GetTravelTime(self, origin, destination, consider_itinerary = None, dis = None, congestion_factor = 1.0):
        # assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        # if origin == destination:
        #     return 0
        if dis is not None:
            total_distance = dis
        else:
            ori_row, des_row = np.ceil(origin / self.num_nodes), np.ceil(destination / self.num_nodes)
            ori_col, des_col = origin - (ori_row - 1) * self.num_nodes , destination - (des_row - 1) * self.num_nodes
            total_distance = (abs(ori_row - des_row) + abs(ori_col - des_col)) * self.distance_per_line
            
        return total_distance / self.vehicle_velocity



    # function: Calculate or export the shortest travel distance between the origin and the destination
    # params: The origin and destination position
    # return: The shortest travel distance between two positions
    def GetTravelDistance(self, origin, destination, consider_itinerary = None):
        assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        if origin == destination:
            return 0
        
        ori_row, des_row = np.ceil(origin / self.num_nodes), np.ceil(destination / self.num_nodes)
        ori_col, des_col = origin - (ori_row - 1) * self.num_nodes , destination - (des_row - 1) * self.num_nodes
        total_distance = (abs(ori_row - des_row) + abs(ori_col - des_col)) * self.distance_per_line

        return total_distance



    # function: Calculate or export the travel path between the origin and the destination
    # params: The origin and destination position
    # return: The travel path (intersections) between two positions
    def GetItineraryNodeList(self, origin, destination):
        assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        assert origin != destination

        # Calculate row and column of origin and destination
        ori_row, des_row = int(np.ceil(origin / self.num_nodes)), int(np.ceil(destination / self.num_nodes))
        ori_col, des_col = int(origin - (ori_row - 1) * self.num_nodes) , int(destination - (des_row - 1) * self.num_nodes)

        itinerary_node_list = [] # Note: We skip the current node (the first node)
        
        if ori_row == des_row: # If the rows of the origin and the destination are same, we only calculate the column difference
            delta_col = ori_col - des_col
            for i in range(1, abs(delta_col)+1):
                node_id = origin - delta_col / abs(delta_col) * i
                itinerary_node_list.append(node_id)
        
        else: # We calculate the row difference first
            delta_row = ori_row - des_row
            for i in range(1, abs(delta_row) + 1):
                node_id_row = origin - delta_row / abs(delta_row) * i * self.num_nodes
                itinerary_node_list.append(node_id_row)
            if ori_col != des_col: # Then we calculate the column difference
                delta_col = ori_col - des_col
                for i in range(1, abs(delta_col)+1):
                    node_id_col = node_id_row - delta_col / abs(delta_col) * i
                    itinerary_node_list.append(node_id_col)
        
        return itinerary_node_list

        

    # function: Simulate or export road congestion 
    # params: todo...
    # return: todo...
    # Note: it may be too complicated to consider the road congestion, so that we may not consider the congestion at the first development stage
    def GetCongestion(self):
        pass