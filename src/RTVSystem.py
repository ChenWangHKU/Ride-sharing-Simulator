from .utils.Trip import Trip, Path
from .Vehicle import Vehicle
from .Request import Request
from .utils.PlanPath import PlanPath
from .utils.VirtualRequest import VirtualRequest
import random
import pickle
import heapq
import itertools as it
from tqdm import tqdm
import numpy as np
import time
import math

'''
The subsystem of the control center that handles requests, vehicles and trips
'''
class RTVSystem:
    def __init__(self,
                environment,
                start_timepoint = 0,
                end_timepoint = 3600,
                step_time = 6,
                consider_itinerary = True,
                cfg = None):
        self.cfg = cfg
        self.environment = environment
        self.start_timepoint = start_timepoint
        self.end_timepoint = end_timepoint
        self.step_time = step_time
        self.consider_itinerary = consider_itinerary
        
        # To accelerate the the simulation process, we don't consider itinerary nodes when we chenk constraints
        self.check_itinerary = self.cfg.REQUEST.CHECK_ITINERARY

        self.total_steps = int((self.end_timepoint - self.start_timepoint) / self.step_time + 1)

        self.PlanPath = PlanPath(environment=environment,
                                check_itinerary=self.check_itinerary,
                                method=self.cfg.VEHICLE.PlanPathMethod)

        # Initialize requests and vehicles
        # self.requests_all = self.InitializeRequests()
        # self.vehicles_all = self.InitializeVehicles()
    

    # function: Initialize all requests
    # params: dataset director or database API
    # return: list[request]
    # Note: If no dataset is provided, the function is supposed to generate requests according to a specific process 
    def InitializeRequests(self, request_data_dir = None):
        #print('Initialize requests')
        requests_all = [[] for _ in range(self.total_steps)]
        num_requests = 0
        # We load requests from the pickle file
        # Note: the data is not only used for our ride-pooling simulator, but other simulators.
        # Therefore, we need to convert the data from the pickle file to our request format
        if request_data_dir:
            '''
            request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'timestamp','start_time', 'date', 'origin_grid_id','dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob', 'fare']
            trip_distance: consider itinerary, unit: km
            itinerary_segment_dis_list: distance between each pair of itinerary nodes
            '''

            requests_raw = pickle.load(open(request_data_dir, 'rb')) # dict
            for idx in tqdm(requests_raw, desc = 'Initialize requests'):
                
                for request_raw in requests_raw[idx]:
                    # Calculate the corresponding step
                    # Note: the start_time's unit is second. e.g., 100 means 00:01:40 and 3600 means 01:00:00
                    timepoint = request_raw[8]
                    # The request is not within the target tiem interval
                    # Note: We assume that there are no new requests in the last 30 mins to complete the simulation
                    if timepoint < self.start_timepoint or timepoint > self.end_timepoint - 1800:
                        continue
                    # we filter the trip whose distance is less 0.1 km
                    if request_raw[7] < 0.1:
                        continue
                    # Assign each request to a simulation step
                    step = round((timepoint - self.start_timepoint) / self.step_time)
                    # pick-up and drop-off position (longitude, latitude)
                    pickup_position = (request_raw[3], request_raw[2])
                    dropoff_position = (request_raw[6], request_raw[5])
                    # pickup_position = self.environment.GetNearestNode((request_raw[3], request_raw[2]))
                    # dropoff_position = self.environment.GetNearestNode((request_raw[6], request_raw[5]))
                    if self.consider_itinerary:
                        travel_time = 0
                        travel_distance = 0
                        iti_nodes = request_raw[-5]
                        for idx in range(len(iti_nodes) - 1):
                            dis, t = self.environment.GetDistanceandTime(iti_nodes[idx], iti_nodes[idx+1])
                            travel_distance += dis
                            travel_time += t
                    else:
                        travel_distance, travel_time = self.environment.GetDistanceandTime(pickup_position, dropoff_position, type = 'Manhattan')
                    request = Request(cfg = self.cfg,
                                    id = request_raw[0],
                                    send_request_timepoint = step * self.step_time + self.start_timepoint,
                                    pickup_position = pickup_position,
                                    dropoff_position = dropoff_position, # We use (lng, lat) to represent position
                                    pickup_grid_id = request_raw[9],
                                    dropoff_grid_id = request_raw[10],
                                    original_travel_time = travel_time,
                                    original_travel_distance = travel_distance,
                                    num_person = 1)
                    requests_all[step].append(request)
                    
                    num_requests += 1
                    
                
        
        # If there is no request file, we generate requests for the toy model randomly
        else:
            num_requests = 60 # We will generate 60 requests randomly
            for request_id in range(num_requests):
                # the status of all requests are random
                step = int(random.random() * (self.end_timepoint - 600) / self.step_time) 
                send_request_timepoint = step * self.step_time
                pickup_position = int(random.random() * len(self.environment.nodes_coordinate)) + 1
                dropoff_position = int(random.random() * len(self.environment.nodes_coordinate)) + 1
                
                # Filter the requests that have same pickup and dropoff positions
                while dropoff_position == pickup_position:
                    dropoff_position = int(random.random() * len(self.environment.nodes_coordinate)) + 1
                
                original_travel_time = self.environment.GetTravelTime(pickup_position, dropoff_position)
                original_travel_distance = self.environment.GetTravelDistance(pickup_position, dropoff_position)
                
                # Initial the request
                request = Request(cfg = self.cfg,
                            id = request_id,
                            send_request_timepoint = send_request_timepoint,
                            pickup_position = pickup_position,
                            dropoff_position = dropoff_position,
                            original_travel_time = original_travel_time,
                            original_travel_distance = original_travel_distance,
                            num_person = 1)
                requests_all[step].append(request)

            # # Demo case   
            # req1 = Request(id = 0,
            #                 send_request_timepoint = 0,
            #                 pickup_position = 19,
            #                 dropoff_position = 7,
            #                 original_travel_time = self.environment.GetTravelTime(19, 7),
            #                 original_travel_distance = self.environment.GetTravelDistance(19, 7),
            #                 num_person = 1)
            # req2 = Request(id = 1,
            #                 send_request_timepoint = 0,
            #                 pickup_position = 23,
            #                 dropoff_position = 8,
            #                 original_travel_time = self.environment.GetTravelTime(23, 8),
            #                 original_travel_distance = self.environment.GetTravelDistance(23, 8),
            #                 num_person = 1)
            # requests_ll[0].append(req1)
            # requests_ll[0].append(req2)
        
        return requests_all, num_requests


    # function: Initialize all vehicles
    # params: dataset director or database API
    # return: list[vehicle]
    # Note: If no dataset is provided, the function is supposed to generate vehicles according to a specific distuibution 
    def InitializeVehicles(self, vehicle_data_dir, num_vehicles = 1000):
        vehicles_all = []

        # We load requests from the pickle file
        # Note: (1) the data is not only used for our ride-pooling simulator, but other simulators.
        # Therefore, we need to convert the data from the pickle file to our vechile format
        # (2) There are so many vehicles in the pickle file (i.e., 20,000 vehicles) that out pc may not be able to run them all.
        # Therefore, we need to downsample vehicles (e.g., 1,000 vehicles)
        if vehicle_data_dir:
            '''
            elf.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list',
                               'node_id', 'grid_id']
            '''
            vehicles_raw = pickle.load(open(vehicle_data_dir, 'rb'))
            # Downsample vehicles
            num_vehilces_raw = len(vehicles_raw)
            ds_gap = int(num_vehilces_raw / num_vehicles)
            vehicle_id = 0
            for idx in tqdm(range(0, num_vehilces_raw, ds_gap), desc = 'Initialize vehicles'):
                # Initialize vehicles
                # We allocate vehicles to nearest intersections
                # current_position = self.environment.GetNearestNode((vehicles_raw['lng'][idx], vehicles_raw['lat'][idx]))
                current_position = (vehicles_raw['lng'][idx], vehicles_raw['lat'][idx])
                vehicle = Vehicle(cfg=self.cfg,
                                id = vehicles_raw['driver_id'][idx],
                                current_position = current_position, # We use coordinate to represent position
                                current_grid_id = vehicles_raw['grid_id'][idx],
                                start_time = vehicles_raw['start_time'][idx],
                                end_time = vehicles_raw['end_time'][idx],
                                online = True,
                                open2request = True)
                vehicles_all.append(vehicle)

        # Generate vehicles for the toy model
        else:
            num_vehicles = 10
            for vehicle_id in range(num_vehicles):
                current_position = int(random.random() * len(self.environment.nodes_coordinate)) + 1
                veh = Vehicle(id = vehicle_id,
                            current_position = current_position,
                            start_time = 0,
                            end_time = 999999,
                            online = True,
                            open2request = True,
                            max_capacity = 4)
                vehicles_all.append(veh)
           
            # # Demo case   
            # veh = Vehicle(id = 0,
            #                 current_position = 19,
            #                 start_time = 0,
            #                 end_time = 999999,
            #                 online = True,
            #                 open2request = True,
            #                 max_capacity = 4)
            # vehicles_all.append(veh)

        return vehicles_all



    # function: Allocate each request to 30 nearest vehicles
    # params: 1) requests of the current step; 2) vehicles of the current step; 
    #         3) The maximum number of vehicles that a request will be allocated; 4) The maximum distance between a vehicle and the allocated request
    # return: list[list[request]], length = num_vehicles
    def AllocateRequest2Vehicles(self, requests_step, vehicles_step, max_num_vehicles = 30, max_match_distance = None):
        if max_match_distance is None:
            max_match_distance = 999999
        
        max_num_vehicles = min(len(vehicles_step), max_num_vehicles)
        requests_for_each_vehicle = [[] for _ in range(len(vehicles_step))] # list[list[request]], length = num_vehicles

        for request in requests_step:
            # Calculte the pickup time and distance for all vehicles to pick the request
            for vehicle_idx, vehicle in enumerate(vehicles_step):
                # Check if the vehicle is online and open to requests
                if not vehicles_step[vehicle_idx].online or not vehicles_step[vehicle_idx].open2request:
                    continue
                # Check the constraints
                dis, t = self.environment.GetDistanceandTime(vehicle.current_position, request.pickup_position)
                if dis < max_match_distance and t < request.max_con_pickup_time:
                    requests_for_each_vehicle[vehicle_idx].append(request)
        
        return requests_for_each_vehicle


    # function: Generate feasible trips
    # params: vehicles for the current time step, request batch generated by the GetRequestBatch function
    # return: list[list[trip]], list[list[path]], length = num_vehicles
    def GenerateFeasibleTrips(self, vehicles_step, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS = 150, MAX_TRIPS = 30):
        # Get feasible trips for each vehicle
        feasible_trips = []
        feasible_paths = []
        for requests_for_vehicle, vehicle in zip(requests_for_each_vehicle, vehicles_step):
            trips = []
            paths = []
            tested_trips_requests = []
            num_is_feasible_calls = 0
            
            # append a null trip
            trips.append(Trip())
            paths.append(Path())
            
            # If the vehicle is empty and still online, but there is no requests nearby, then we consider repositioning the vehicle
            if len(requests_for_vehicle) == 0 and vehicle.path is None and vehicle.online:
                # Considering repositioning in the RL model
                if self.cfg.MODEL.REPOSITION:
                    # Repositioning idle vehicles to 8 (or less) grids nearby
                    reposition_locations = self.environment.GetRepositionLocation(vehicle.current_position)
                    for rep_loc in reposition_locations:
                        lng, lat, pickup_grid_id, dropoff_grid_id, distance, time = rep_loc
                        # Initialize request
                        virtual_request = VirtualRequest(pickup_position = vehicle.current_position,
                                                        dropoff_position = (lng, lat),
                                                        pickup_grid_id = pickup_grid_id,
                                                        dropoff_grid_id = dropoff_grid_id,
                                                        original_travel_time = 0,
                                                        original_travel_distance = -distance * 0.5)
                        reposition_trip = Trip(virtual_request)
                        # Initialize path
                        reposition_path = Path(current_position = vehicle.current_position,
                                                next_positions = [vehicle.current_position, (lng, lat)],
                                                time_needed_to_next_position = np.array([0, time]),
                                                dis_to_next_position = np.array([0, distance]),
                                                time_delay_to_each_position = np.zeros((2)))
                        if self.consider_itinerary:
                            reposition_path = self.PlanPath.UpdateItineraryNodes(reposition_path)
                        
                        trips.append(reposition_trip)
                        paths.append(reposition_path)
                    
                    feasible_trips.append(trips)
                    feasible_paths.append(paths)
                    continue
            
            # No trip when repositioning or delivering passengers
            if len(requests_for_vehicle) == 0:
                feasible_trips.append(trips)
                feasible_paths.append(paths)
                continue

            # Check feasibility for individual requests
            for request in requests_for_vehicle:
                # If there exists requests nearby, we stop the repositioning process
                if vehicle.current_capacity < len(vehicle.current_requests):
                    vehicle.Status2Idle()
                
                trip = Trip(request)

                path = self.PlanPath.PlanPath(vehicle, trip) # Note: Any parameters of the vehicle should not be changed at this fuction
              
                if path is not None:
                    # print(path.current_position)
                    # print(path.next_itinerary_nodes)
                    trips.append(trip)
                    paths.append(path)

                tested_trips_requests.append(trip.requests)
                num_is_feasible_calls += 1
            
            # Non-ride-pooling
            if self.cfg.VEHICLE.MAXCAPACITY == 1:
                feasible_trips.append(trips)
                feasible_paths.append(paths)
                continue
            
            # We use the average travel distance of each request in the trips to determine the trip priority
            def TripPriority(trip):
                assert len(trip.requests) > 0
                return -sum(request.original_travel_distance for request in trip.requests) / len(trip.requests)
            
            # Get feasible trips of size > 1, with a fixed budget of MAX_IS_FEASIBLE_CALLS
            trips_tobe_combined = [(TripPriority(trip), trip_idx+1) for trip_idx, trip in enumerate(trips[1:])]
            heapq.heapify(trips_tobe_combined) # convert list to heap
            
            while len(trips_tobe_combined) > 0 and num_is_feasible_calls < MAX_IS_FEASIBLE_CALLS:
                _, trip_heap_idx = heapq.heappop(trips_tobe_combined) # pop the trip with maximum average travel distance
                
                for trip_list_idx in range(1, len(trips)):
                    pre_requests = trips[trip_heap_idx].requests
                    new_requests = trips[trip_list_idx].requests
                    combined_trip = Trip(list(set(pre_requests) | set(new_requests)))
                   
                    # We judge if the combined trip has been tested through the requests
                    if combined_trip.requests not in tested_trips_requests:
                        
                        path = self.PlanPath.PlanPath(vehicle, combined_trip)
                        
                        if path is not None:
                            trips.append(combined_trip)
                            paths.append(path)
                            heapq.heappush(trips_tobe_combined, (TripPriority(combined_trip), len(trips) - 1))

                        num_is_feasible_calls += 1
                        tested_trips_requests.append(combined_trip.requests)


                # Create only MAX_ACTIONS actions
                if (MAX_TRIPS >= 0 and len(trips) >= MAX_TRIPS):
                    break

            feasible_trips.append(trips)
            feasible_paths.append(paths)
        
        return feasible_trips, feasible_paths