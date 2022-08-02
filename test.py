#####################################################
######         Written by Wang CHEN            ######
######     E-mail: wchen22@connect.hku.hk      ######
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
from tqdm import tqdm
import copy
import math
import numpy as np
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
    filehandler = logging.FileHandler(os.path.join(args.OutputDir, cfg_file_name, 'test.log'))
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
    
    
    # Load requests for test
    test_data = []
    test_data_path = cfg.REQUEST.DATA.TEST
    test_data_names = os.listdir(test_data_path) # all file names in the 'train_data_path' filefold
    for day, test_data_name in enumerate(test_data_names):
        test_data_dir = os.path.join(test_data_path, test_data_name)
        test_data_one_day, num = control_center.RTV_system.InitializeRequests(test_data_dir)
        test_data.append(test_data_one_day)
        logger.info('The number of test requests (day {}): {} '.format(day+1, num))
    
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
    agent.model = torch.load('./exp/best_model')

    
    # test
    requests_results_all = []
    vehicles_results_all = []
    for day, requests in enumerate(test_data):
        RunEpisode(requests, vehicles, control_center, agent, train=False)
        
        # Record the results
        requests_results, vehicles_results = control_center.CalculateResults()
        requests_results_all.append(requests_results)
        vehicles_results_all.append(vehicles_results)

        logger.info('********************* Test day {} ************************'.format(day+1))
        LogResults(logger, requests_results, vehicles_results)
    
    # Average results
    requests_results_all = np.array(requests_results_all).mean(axis = 0)
    vehicles_results_all = np.array(vehicles_results_all).mean(axis = 0)
    logger.info('****************** Test Average Results *********************')
    LogResults(logger, requests_results_all, vehicles_results_all)
    


if __name__ == '__main__':
    main()
