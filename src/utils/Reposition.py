import random
import numpy as np
from .Trip import  Path


'''
The subsystem of the control center that simulates actions and manages all vehicles
The parameters of all vehicles are changed here
'''
class Reposition:
    def __init__(self,
                environment,
                method,
                consider_itinerary):
       
        self.environment = environment
        self.method = method
        self.consider_itinerary = consider_itinerary
    
    
    # function: Repositioning idle vehicles according to the given method
    def Reposition(self, vehicle):
        if self.method == 'Random':
            self.RepositioningIdleVehiclesRandomly(vehicle)
        elif self.method == 'ToHotAera':
            self.RepositioningIdleVehicles2HotAera(vehicle, hot_positions = None)
        else:
            raise NotImplementedError


    # function: Manage idle vehicles to another node randomly (within 10 kms)
    # params: The vehicle needed to be relocated
    # return: Actions of idle vehicles or updated vehicles
    def RepositioningIdleVehiclesRandomly(self, vehicle):
        while True:
            node = self.environment.nodes_coordinate[int(random.random()*len(self.environment.nodes_coordinate))] # Coordinate
            if node != vehicle.current_position: # Repositioning to another position
                break
        dis, time = self.environment.GetDistanceandTime(node, vehicle.current_position, type = 'Manhattan')
        
        # Initialize path
        path = Path(current_position=vehicle.current_position,
                    next_positions=[vehicle.current_position, node],
                    time_needed_to_next_position=np.array([0,time]),
                    dis_to_next_position=np.array([0,dis]),
                    time_delay_to_each_position=np.zeros((2)))           
        
        # Update the vehicle's path
        vehicle.path = path

        return vehicle


    # function: Manage idle vehicles to hot areas
    # params: requests at the current time step or predicted demand at next time step
    # return: Actions of idle vehicles or updated vehicles
    def RepositioningIdleVehicles2HotAera(self, vehicle):
        raise NotImplementedError