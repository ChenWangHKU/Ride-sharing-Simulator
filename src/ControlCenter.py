from .ActionSystem import ActionSystem
from .RTVSystem import RTVSystem
from .EvaluationSystem import EvaluationSystem
from .PostProcessSystem import PostProcessSystem
import numpy as np


'''
The control center is just like a ride-hailing platform
All requests and vehicles are handled here
The control center consists of 4 subsystems:
1. RTV System: handles requests, vehicles and trips
2. Evaluation System: evaluates and chooses trips
3. Action System: simulates actions and manages all vehicles
4. Post Process System: count and visulize results

See each object for detailed information
'''
class ControlCenter:
    def __init__(self,
                environment,
                start_timepoint,
                end_timepoint,
                step_time,
                consider_itinerary = True, # Consider itinerary nodes or only origin and destination of requests
                cfg =None
                ):
        self.cfg = cfg
        self.environment = environment
        self.start_timepoint = start_timepoint
        self.end_timepoint = end_timepoint
        self.step_time = step_time
        self.total_steps = int((end_timepoint + self.cfg.SIMULATION.TIME2FINISH - start_timepoint) / step_time - 1)
        

        self.RTV_system = RTVSystem(environment = self.environment,
                                    start_timepoint = start_timepoint,
                                    end_timepoint = end_timepoint,
                                    step_time = step_time,
                                    consider_itinerary = consider_itinerary,
                                    cfg = cfg
                                    )
        self.evaluation_system = EvaluationSystem(cfg = cfg)

        self.current_timepoint = start_timepoint
        self.step = 0

        # Initialize requests and vehicles, see class RTVSystem for detailed information
        self.requests_all = None
        self.requests_step = None
        self.vehicles_all = None

        self.action_system = ActionSystem(cfg = self.cfg,
                                        vehicles = None,
                                        requests = None,
                                        environment = self.environment,
                                        current_timepoint = self.current_timepoint,
                                        step_time = self.step_time,
                                        RTV_system = self.RTV_system,
                                        consider_itinerary = consider_itinerary)

        self.post_process_system = PostProcessSystem(vehicles = None,
                                                    requests = None,
                                                    environment = self.environment,
                                                    current_timepoint = self.current_timepoint
                                                    )
    # Initialize the requests and vehicles
    def Initialize(self, requests, vehicles):
        self.requests_all = requests
        self.vehicles_all = vehicles
        self.requests_step = requests[self.step]
        self.action_system.vehicles = vehicles
        self.action_system.requests = requests[self.step]
        self.post_process_system.requests = requests[self.step]
        self.post_process_system.vehicles = vehicles


    # Update the system's parameters
    def UpdateParameters(self, timepoint, step):
        self.current_timepoint = timepoint
        self.action_system.current_timepoint = timepoint
        self.post_process_system.current_timepoint = timepoint
        self.step = step
    
    # Update requests at next time step
    # params: Unmatched_requests: requests that haven't been allocated to any vehicles and don't cancel
    def UpdateRequests(self, unmatched_requests):
        if self.step >= self.total_steps-1 or self.step >= len(self.requests_all) - 1:
            new_requests = []
        else:
            new_requests = self.requests_all[self.step + 1] # New requests at next time step
        requests = list(set(unmatched_requests) | set(new_requests)) # Union
        self.action_system.requests = requests
        self.post_process_system.requests = requests
        self.requests_step = requests


    '''RTV System'''
    # Allocate requests to each vehicle, see class RTVSystem for detailed information    
    def AllocateRequest2Vehicles(self, max_num_vehicles = 30, max_match_distance = 3000):
        requests_for_each_vehicle = self.RTV_system.AllocateRequest2Vehicles(self.requests_step, self.vehicles_all, max_num_vehicles, max_match_distance)
        return requests_for_each_vehicle
    
    # Generate feasible trips and the corresponding paths, see class RTVSystem for detailed information
    def GenerateFeasibleTrips(self, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS = 150, MAX_TRIPS = 30):
        feasible_trips, feasible_paths = self.RTV_system.GenerateFeasibleTrips(self.vehicles_all, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS, MAX_TRIPS)
        return feasible_trips, feasible_paths


    '''Evaluation System'''
    # Score feasible trips, see class EvaluationSystem for detailed information
    def ScoreTrips(self, feasible_trips, feasible_paths, pre_values):
        scored_feasible_trips = self.evaluation_system.ScoreTrips(feasible_trips, feasible_paths, pre_values)
        return scored_feasible_trips
    
    # Score feasible trips based on Reinforcement Learning, see class EvaluationSystem for detailed information
    # todo...
    def ScoreTripsRL(self, feasible_trips):
        return self.evaluation_system.ScoreTripsRL(feasible_trips)

    # Choose a trip and the corresponding path for each vehicle, see class EvaluationSystem for detailed information
    def ChooseTrips(self, scored_feasible_trips, feasible_paths):
        final_trips, final_paths, rewards = self.evaluation_system.ChooseTrips(scored_feasible_trips, feasible_paths)
        return final_trips, final_paths, rewards
    

    '''Action System'''
    # Update the trip and the path of the vehicle, see class ActionSystem for detailed information
    def UpdateVehicles(self, final_trips, final_paths, vehicles = None):
        self.action_system.UpdateVehicles(final_trips, final_paths, vehicles)
    
    # Simulate the action of each vehicle and manage all vehicles, see class ActionSystem for detailed information
    def SimulateVehicleAction(self, vehicles = None):
        self.action_system.SimulateVehicleActions(vehicles)

    # Remove the finished requests, and the unmatched requests are returned and will be merged with the requests at next time step, see class ActionSystem for detailed information
    def ProcessRequests(self):
        unmatched_requests = self.action_system.ProcessRequests()
        return unmatched_requests


    '''PostProcess System'''
    # Draw road network of New York model, see class PostProcessSystem for detailed information
    def DrawRoadNetworkNYModel(self, ax):
        return self.post_process_system.DrawRoadNetworkNYModel(ax)
    
    # Draw vehicles and requests of New York model, see class PostProcessSystem for detailed information
    def DrawVehiclesandRequestsNYModel(self, ax, v_size = 0.002, draw_route = True):
        ax = self.post_process_system.DrawVehiclesandReuqestsNYModel(ax, v_size = v_size, draw_route = draw_route)
        return ax
    
    # Draw the distribution of vehicles, see class PostProcessSystem for detailed information
    def DrawVehiclesDistributionNYModel(self, ax, v_size = 0.002):
        ax = self.post_process_system.DrawVehiclesDistributionNYModel(ax = ax, v_size = v_size)
        return 0

    def DrawRequestsDistributionNYModel(self, ax, requests_all, radius = 0.0005):
        ax = self.post_process_system.DrawRequestsDistributionNYModel(ax = ax, requests_all = requests_all, radius = radius)
        return ax

    # Draw road network of toy model, see class PostProcessSystem for detailed information
    def DrawRoadNetworkToyModel(self, ax):
        return self.post_process_system.DrawRoadNetworkToyModel(ax)

    # Draw vehicles and requests of toy model, see class PostProcessSystem for detailed information
    def DrawVehiclesandRequestsToyModel(self, ax):
        ax = self.post_process_system.DrawVehiclesandReuqestsToyModel(ax)
        return ax
    # Integrate all result images to a vedio, see class PostProcessSystem for detailed information
    def MakeVedio(self, imgs = None, img_path = 'Output/tmp', vedio_fps = 30, vedio_path = 'Output'):
        self.post_process_system.MakeVedio(imgs = imgs, img_path = img_path, vedio_fps = vedio_fps, vedio_path = vedio_path)


    ''' 
    Once the simulation finished, we calculate:
    (1) REQUEST
        1. The number of requests 
        2. Service rate
        3. The average assigning time
        4. The average waiting time of requests
        5. The average detour time
        6. The average detour distance
        7. Cancellation rate (assign)
        8. Cancallation rate (pickup)

    (2) VEHICLE
        1. The number of vehicles
        2. The average idle time
        3. The total income of all vehicles
        4. The total travel distance of all vehicles
    '''
    def CalculateResults(self):
        requests_results = np.zeros((8))
        vehicles_results = np.zeros((4))
        # Requests' results
        for requests in self.requests_all:
            for request in requests:
                requests_results[0] += 1
                # The request has been served
                if request.finish_pickup:
                    requests_results[1] += 1
                    requests_results[2] += request.assign_timepoint - request.send_request_timepoint
                    requests_results[3] += request.pickup_timepoint - request.assign_timepoint
                    #requests_results[4] += request.dropoff_timepoint - request.pickup_timepoint - request.original_travel_time
                    requests_results[5] += max(0, request.distance_on_vehicle - request.original_travel_distance)
                    requests_results[4] += max(0, request.time_on_vehicle - request.original_travel_time)
                    # requests_results[5] += request.distance_on_vehicle - request.original_travel_distance
                # The request has been cancelled
                # Note: Here, we assume that there is no passenger at any vehicles. In other words, all trips are finished at the end of simulation
                else:
                    if request.finish_assign:
                        requests_results[7] += 1
                    else:
                        requests_results[6] += 1
        # mean value
        requests_results[2:6] /= requests_results[1]
        requests_results[1] /= requests_results[0]
        requests_results[6:] /= requests_results[0]

        # Vehicles' results
        for vehicle in self.vehicles_all:
            vehicles_results[0] += 1
            vehicles_results[1] += vehicle.total_idle_time
            # Here, we calculate the total income and travel distance to evaluate the system's income and energy consumption
            vehicles_results[2] += vehicle.total_income
            vehicles_results[3] += vehicle.total_distance
        vehicles_results[1] /= vehicles_results[0]

        return requests_results, vehicles_results