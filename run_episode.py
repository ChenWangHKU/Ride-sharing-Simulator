import copy
from tqdm import tqdm
import numpy as np
import time

from src.RL.states import States

runiing_time_for_print = 20

'''
    (1) agent=None and train=False mean running the simulation without RL model
    (2) One can find anything in the control_centrol
'''
def RunEpisode(requests, vehicles, control_center, agent = None, train = False, train_step = 0):
    # Initialization
    control_center.Initialize(requests, vehicles)
    if agent:
        states = States(cfg = control_center.cfg,
                        node_coord_to_grid = control_center.environment.node_coord_to_grid,
                        requests_record_time = 1800) # We record requests in the previous 30 mins 
    
    # Run the simulation
    for step in tqdm(range(control_center.total_steps), desc = 'Running simulation steps: '):
        # Upadate parameters of the control center
        current_timepoint = control_center.start_timepoint + step * control_center.step_time
        control_center.UpdateParameters(current_timepoint, step)
        if agent:
            # Update the vehicles' distribution
            states.vehicles_distribution.Update(vehicles)
            # Get the current states
            cur_states = states.GetStates(vehicles, step)
        
        ts = time.time()

        # Allocate each rquest to the vehicles nearby
        requests_for_each_vehicle = control_center.AllocateRequest2Vehicles()
        
        if time.time() - ts > runiing_time_for_print:
            print(' AllocateRequest2Vehicles time: ', time.time() - ts)
        ts = time.time()

        # Filter requests that don't meet the system's contraints and combine requests together (ride-pooling) of each vehicle
        feasible_trips, feasible_paths = control_center.GenerateFeasibleTrips(requests_for_each_vehicle)
        
        if time.time() - ts > runiing_time_for_print:
            print(' GenerateFeasibleTrips time: ', time.time() - ts)
        ts = time.time()

        # For each vehicle, simulate action of each trip to get post-decision states
        if agent:
            next_vehicles = []
            for (vehicle, trips, paths) in zip(vehicles, feasible_trips, feasible_paths):
                for (trip, path) in zip(trips, paths):
                    # Here we should deepcopy the vehicle for each feasible trip
                    next_vehicle = copy.deepcopy(vehicle)
                    next_trip, next_path = copy.deepcopy(trip), copy.deepcopy(path)
                    # For each trip, simulate the action
                    control_center.UpdateVehicles([next_trip], [next_path], [next_vehicle])
                    control_center.SimulateVehicleAction([next_vehicle])
                    
                    next_vehicles.append(next_vehicle)
            
            if time.time() - ts > runiing_time_for_print:
                print(' Simulate virtual action time: ', time.time() - ts)
            ts = time.time()

            # Update requests' distribution
            states.requests_distribution.Update(requests[step])
            # Get the post-decision states
            post_states = states.GetStates(next_vehicles, step)
            
            # Score each decision by RL model
            pre_value = agent.get_value(post_states)
        else:
            pre_value = None

        if time.time() - ts > runiing_time_for_print:
            print(' Agent scoring time: ', time.time() - ts)
        ts = time.time()

        # Get the final score of each decision
        scored_feasible_trips = control_center.ScoreTrips(feasible_trips, feasible_paths, pre_value)

        # Choose a trip for aeach vehicle
        final_trips, final_paths, rewards = control_center.ChooseTrips(scored_feasible_trips, feasible_paths)
        
        if time.time() - ts > runiing_time_for_print:
            print('ChooseTrips time: ', time.time() - ts)
        ts = time.time()
        
        # Update the vehicles according to the final trips and paths
        control_center.UpdateVehicles(final_trips, final_paths)
        
        if time.time() - ts > runiing_time_for_print:
            print('Upadate vehicles time: ', time.time() - ts)
        ts = time.time()

        # Simulate actions 
        control_center.SimulateVehicleAction()

        if time.time() - ts > runiing_time_for_print:
            print('SimulateVehicleAction time: ', time.time() - ts)
        ts = time.time()
        
        if agent:
            # Get next states
            next_states = states.GetStates(vehicles, step+1)
            # Judge if it's the final step
            done = np.zeros((len(vehicles), 1)) if step < control_center.total_steps -1 else np.ones((len(vehicles), 1)) 
            # Save the experience to the memory
            agent.append_sample(cur_states, final_trips, rewards, next_states, done)
       
        if time.time() - ts > runiing_time_for_print:
            print('Append states time: ', time.time() - ts)
        ts = time.time() 
       
        # Process the requests that unassigned: cancel or wait
        unmatched_requests = control_center.ProcessRequests()
        # Update requests
        control_center.UpdateRequests(unmatched_requests)

        if time.time() - ts > runiing_time_for_print:
            print('Process requests time: ', time.time() - ts)
        ts = time.time() 

        # Train the model
        if agent is not None and train and (train_step + 1) % agent.train_frequency == 0:
            print("Train model...")
            agent.train_model()
            print('done!')
        
        train_step += 1

        if time.time() - ts > runiing_time_for_print:
            print('Train agent time: ', time.time() - ts)
    
    # every episode update the target model to be same with model
    if train:
        agent.update_target_model()
        return train_step