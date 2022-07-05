#####################################################
######         Written by Wang CHEN            ######
######    E-mail: u3008939@connect.hku.hk      ######
######     Copyright @ Smart Mobility Lab      ######
######    Department of Civil Engineering      ######
######      Thu University of Hong Kong        ######
#####################################################


from msilib.schema import Environment
from src.Environment import EnvironmentToyModel, EnvironmentNYModel
from src.ControlCenter import ControlCenter
from src.RL.Model import Agent
from run_episode import RunEpisode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
import argparse
import yaml
from easydict import EasyDict as edict
import copy
import math
import torch



def parse_args():
    parser = argparse.ArgumentParser(description='Ride-pooling simulator')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--OutputDir',
                        help='output directory',
                        type=str,
                        default='./exp')
    parser.add_argument('--device',
                        help='GPU or CPU',
                        type=str,
                        default='cuda')
    parser.add_argument('--SaveFre',
                        help='Save the model after each 50 epochs',
                        type=int,
                        default=10)
    parser.add_argument('--DrawResult',
                        help='Draw the result image of each step',
                        type=bool,
                        default=False)                  
    parser.add_argument('--DrawDistribution',
                        help='Draw the distribution of vehicles and requests',
                        type=bool,
                        default=False)  
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    # config file
    with open(args.cfg) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    # log file
    logger = logging.getLogger('')

    # New output filefold
    if not os.path.exists(args.OutputDir):
        os.makedirs(args.OutputDir)
    # New the output filefold of the current experiment
    cfg_file_name = os.path.basename(args.cfg).split('.')[0]
    if not os.path.exists(os.path.join(args.OutputDir, cfg_file_name)):
        os.makedirs(os.path.join(args.OutputDir, cfg_file_name))
    # New the image path
    img_path = os.path.join(args.OutputDir, cfg_file_name, 'tmp')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    # set the log file path
    filehandler = logging.FileHandler(os.path.join(args.OutputDir, cfg_file_name, 'training.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    # Write config information
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    
    # For convenience, we load the configrations ahead of time
    # For control center
    start_timepoint = cfg.SIMULATION.START
    end_timepoint = cfg.SIMULATION.END
    step_time = cfg.SIMULATION.STEP_TIME
    # For environment
    velocity = cfg.VEHICLE.VELOCITY
    consider_itinerary = cfg.ENVIRONMENT.CONSIDER_ITINERARY
    env_type = cfg.ENVIRONMENT.TYPE
    
    # Initialize environment
    if env_type == 'NY':
        # Initialize the New York Model
        environment = EnvironmentNYModel(network_file_path = cfg.ENVIRONMENT.NY.RoadFile,
                                        vehicle_velocity = velocity,
                                        travel_time_file_path = cfg.ENVIRONMENT.NY.TravelTimeFile,
                                        travel_distance_file_path = cfg.ENVIRONMENT.NY.TravelDisFile,
                                        consider_itinerary = consider_itinerary,
                                        cfg = cfg)
    elif env_type == 'TOY':
        # Initilize the Toy Model
        environment = EnvironmentToyModel(num_nodes = cfg.ENVIRONMENT.TOY.NumNode,
                                        distance_per_line = cfg.ENVIRONMENT.TOY.DisPerLine,
                                        vehicle_velocity = velocity,
                                        consider_congestion = False)
    else:
        raise NotImplementedError

    # Initilize the control center
    control_center = ControlCenter(environment = environment,
                                    start_timepoint = start_timepoint,
                                    end_timepoint = end_timepoint,
                                    step_time = step_time,
                                    consider_itinerary = consider_itinerary,
                                    cfg = cfg)

    # Record the number of requests and vehicles
    total_steps = int((end_timepoint - start_timepoint) / step_time - 1)
    total_grids = int(cfg.ENVIRONMENT.NY.X_GRID_NUM * cfg.ENVIRONMENT.NY.Y_GRID_NUM)
    logger.info('The number of steps: {}'.format(total_steps))
    logger.info('The number of grids: {}'.format(total_grids))
    logger.info('******************************')
    
    
    # Load requests for training
    train_data = []
    train_data_path = cfg.REQUEST.DATA.TRAIN
    train_data_names = os.listdir(train_data_path) # all file names in the 'train_data_path' filefold
    for day, train_data_name in enumerate(train_data_names):
        train_data_dir = os.path.join(train_data_path, train_data_name)
        train_data_one_day, num = control_center.RTV_system.InitializeRequests(train_data_dir)
        train_data.append(train_data_one_day)
        logger.info('The number of training requests (day {}): {} '.format(day+1, num))
    
    # # Load requests for validation
    # val_data = []
    # val_data_path = cfg.REQUEST.DATA.VALIDATION # all file names in the 'val_data_path' filefold
    # val_data_names = os.listdir(val_data_path)
    # for day, val_data_name in enumerate(val_data_names):
    #     val_data_dir = os.path.join(val_data_path, val_data_name)
    #     val_data_one_day, num = control_center.RTV_system.InitializeRequests(val_data_dir)
    #     val_data.append(val_data_one_day)
    #     logger.info('The number of validation requests: {} of day {}'.format(num, day+1))
    
    # Load vehicles
    vehicles = control_center.RTV_system.InitializeVehicles(cfg.VEHICLE.DATA, num_vehicles = cfg.VEHICLE.NUM)
    logger.info('The number of vehicles: {} '.format(len(vehicles)))
    logger.info('******************************')

    # Record the results
    def LogResults(logger, requests_results, vehicles_results):
        # Requests
        logger.info('Service rate:                     {}'.format(requests_results[1]))
        logger.info('The average assigning time (s):   {}'.format(requests_results[2]))
        logger.info('The average pick-up time (min):   {}'.format(requests_results[3] / 60))
        logger.info('The average detour time (min):    {}'.format(requests_results[4] / 60))
        logger.info('The average detour distance (km): {}'.format(requests_results[5] / 1000))
        logger.info('Cancellation rate (assign):       {}'.format(requests_results[6]))
        logger.info('Cancellation rate (pickup):       {}'.format(requests_results[7]))
        logger.info('******************************')
        
        # Vehicles
        logger.info('The average idle time(min):                     {}'.format(vehicles_results[1] / 60))
        logger.info('The total income of all vehicles (USD):         {}'.format(vehicles_results[2]))
        logger.info('The total travel distance of all vehicles (km): {}'.format(vehicles_results[3] / 1000))
        logger.info('******************************')

    # Initialize the agent
    time_pieces = math.ceil((end_timepoint - start_timepoint) / cfg.MODEL.TIME_INTERVAL) + 1
    agent = Agent(cfg = cfg,
                total_grid_num = total_grids,
                total_time_step = time_pieces)

    # train the model
    best_income = 0
    train_step = 0

    for episode in range(cfg.MODEL.EPISODES):
        # train
        for day, requests in enumerate(train_data):
            vehicles_tmp = copy.deepcopy(vehicles)
            requests_tmp = copy.deepcopy(requests)
            train_step = RunEpisode(requests_tmp, vehicles_tmp, control_center, agent, train=True, train_step = train_step)
            
            # Record the results
            requests_results, vehicles_results = control_center.CalculateResults()
            logger.info('*********** Episode {},  Training day {} ************'.format(episode+1, day+1))
            LogResults(logger, requests_results, vehicles_results)
        
        # # validate
        # for day, requests in enumerate(val_data):
        #     vehicles_tmp = copy.deepcopy(vehicles)
        #     requests_tmp = copy.deepcopy(requests)
        #     RunEpisode(requests_tmp, vehicles_tmp, control_center, agent, train=False)
        #     # Record the results
        #     requests_results, vehicles_results = control_center.CalculateResults()
        #     logger.info('*********** Episode {},  Validation day {} ************'.format(episode+1, day+1))
        #     LogResults(logger, requests_results, vehicles_results)
        

        # Save the best model
        if vehicles_results[2] > best_income:
            best_income = vehicles_results[2]
            torch.save(agent.model, os.path.join(args.OutputDir, cfg_file_name, 'best_model'))
        # Save the model every 50 episodes
        if (episode + 1) % args.SaveFre == 0:
            torch.save(agent.model, os.path.join(args.OutputDir, cfg_file_name, 'model_episode' + str(episode+1)))


if __name__ == '__main__':
    main()
