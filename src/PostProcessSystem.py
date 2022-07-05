import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

'''
The subsystem of the control center that visualizes the results
'''
class PostProcessSystem:
    def __init__(self,
                vehicles,
                requests,
                environment,
                current_timepoint
                ):
        self.vehicles = vehicles
        self.requests = requests
        self.environment = environment
        self.current_timepoint = current_timepoint

        self.img_num = 0


    # function: Draw the intial road network of NY model
    # params: the predefined picture
    # return: picture
    # Note: the coordinates of nodes are (lng, lat)
    def DrawRoadNetworkNYModel(self, ax, TIME = True):
        nodes_coordinate = self.environment.nodes_coordinate
        nodes_connection = self.environment.nodes_connection

        # ax.set_xlim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,1])/1000 + 1)
        # ax.set_ylim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,2])/1000 + 1)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        # ax.xaxis.set_label_position('top')
        # ax.xaxis.set_ticks_position('top') 
        # ax.invert_yaxis() # positive to down

        for (u,v) in nodes_connection:
            x1, y1 = self.environment.node_id_to_coord[u]
            x2, y2 = self.environment.node_id_to_coord[v] # unit: degree
            plt.plot((x1,x2), (y1,y2), color = 'gray', linewidth = 1)
            
        if TIME:
            ax.set_title(self.GetTime(), y=0, fontsize = 22)

        return ax
    
    
    # Draw the distribution of vehicles
    def DrawVehiclesDistributionNYModel(self, ax, v_size):
        ax = self.DrawRoadNetworkNYModel(ax) # Draw the road network first
        # Draw the vehicles' distribution
        for vehicle in self.vehicles:
            if not isinstance(vehicle.current_position, tuple):
                xv, yv = self.environment.node_id_to_coord[vehicle.current_position]
            else:
                xv, yv = vehicle.current_position
            # We use rectangle to represent vehicles
            rec = plt.Rectangle((xv-v_size/2, yv-v_size/2), v_size, v_size, facecolor = 'green', alpha = 1)
            ax.add_patch(rec)
        
        return ax


    # function: Draw the distribution of all requests
    # params: The predefined picture, all requests
    # return: The picture of the distribution of all requests
    # Note: (1) It's so complicated to draw the vehicles and requests at each time step that we can draw the distribution of all requests, from
    # which we can have a general view of the simulation. (2) There may be more than one request at each node, so we use different colors to represent
    # the number of the requests at a node, i.e., green: [1,2]; bule: [3,4]; orange: [5,6]; yellow: [7,8]; red: [9, +]
    def DrawRequestsDistributionNYModel(self, ax, requests_all, radius = 0.0005):
        ax = self.DrawRoadNetworkNYModel(ax, False) # Draw the road network first
        coord_to_num = {} # used to count the number of requests
        for coord in self.environment.nodes_coordinate:
            coord_to_num[coord] = 0
        # Count the number of requests at each node
        for requests in requests_all:
            for request in requests:
                if not isinstance(request.pickup_position, tuple):
                    coord = self.environment.node_id_to_coord[request.pickup_position]
                else:
                    coord = request.pickup_position
                coord_to_num[coord] += 1

        node_color = ['green', 'blue', 'orange', 'yellow', 'red']  
        
        # Draw requests
        for coord in coord_to_num:
            if coord_to_num[coord] > 0:
                color = node_color[min(int(coord_to_num[coord] / 2.1), 4)]
                cir = plt.Circle(coord, radius = radius, color=color, fill=True, alpha=1)
                ax.add_patch(cir)
        
        return ax
        
    
    
    # function: Draw the vehicles and requests
    # params: None
    # return: picture
    def DrawVehiclesandReuqestsNYModel(self, ax, v_size = 0.002, draw_route = True):
        ax = self.DrawRoadNetworkNYModel(ax) # Draw the road network first

        nodes_coordinate = self.environment.nodes_coordinate
        v_size = v_size # the size of the vehicle

        # Draw the vehicles and request therein
        for vehicle in self.vehicles:
            if not isinstance(vehicle.current_position, tuple):
                xv, yv = self.environment.node_id_to_coord[vehicle.current_position]
            else:
                xv, yv = vehicle.current_position
            # We use rectangle to represent vehicles
            rec = plt.Rectangle((xv-v_size/2, yv-v_size/2), v_size, v_size, facecolor = 'slategrey')
            ax.add_patch(rec)

            # Draw routes of vehicles
            if draw_route:
                if vehicle.path is not None:
                    nodes = vehicle.path.next_itinerary_nodes
                    if nodes[0] != vehicle.current_position:
                        nodes.insert(0, vehicle.current_position)
                    for idx in range(len(nodes) - 1):
                        n1, n2 = nodes[idx], nodes[idx+1]
                        if not isinstance(n1, tuple):
                            x1, y1 = self.environment.node_id_to_coord[n1]
                        else:
                            x1, y1 = n1
                        if not isinstance(n2, tuple):
                            x2, y2 = self.environment.node_id_to_coord[n2]
                        else:
                            x2, y2 = n2
                        plt.plot((x1,x2), (y1,y2), color = 'green', linewidth = 1)
                    

            # We use circles to represent requests
            # green represents requests have been pick up
            for i, request in enumerate(vehicle.current_requests):
                cir = plt.Circle(self.GetCircleCenter(xv,yv,v_size,i), radius = v_size/4, color="green", fill=True, alpha=1)
                ax.add_patch(cir)
                # We also draw the destination of the request
                if not isinstance(request.dropoff_position, tuple):
                    xr_d, yr_d = self.environment.node_id_to_coord[request.dropoff_position]
                else:
                    xr_d,yr_d = request.dropoff_position
                cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="green", fill=False, alpha=0.5)
                ax.add_patch(cir)
            
            # orange represents request has been allocated to a vehicle but has not been pick up
            for i,request in enumerate(vehicle.next_requests):
                # The coordinate of the request
                if not isinstance(request.pickup_position, tuple):
                    xr_p,yr_p = self.environment.node_id_to_coord[request.pickup_position]
                else:
                    xr_p,yr_p = request.pickup_position
                if not isinstance(request.dropoff_position, tuple):
                    xr_d,yr_d = self.environment.node_id_to_coord[request.dropoff_position]
                else:
                    xr_d,yr_d = request.dropoff_position
                
                cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="orange", fill=True, alpha=0.5)
                ax.add_patch(cir)
                # cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="orange", fill=False, alpha=0.5)
                # ax.add_patch(cir)
        
        # Draw the new requests
        # red represents new requests
        for request in self.requests:
            # The coordinate of the request
            if not isinstance(request.pickup_position, tuple):
                xr_p,yr_p = self.environment.node_id_to_coord[request.pickup_position]
            else:
                xr_p,yr_p = request.pickup_position
            if not isinstance(request.dropoff_position, tuple):
                xr_d,yr_d = self.environment.node_id_to_coord[request.dropoff_position]
            else:
                xr_d,yr_d = request.dropoff_position
            cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="red", fill=True, alpha=0.5)
            ax.add_patch(cir)
            # cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="red", fill=False, alpha=0.5)
            # ax.add_patch(cir)
        return ax



    # function: Draw the intial road network of toy model
    # params: None
    # return: picture
    def DrawRoadNetworkToyModel(self, ax):
        nodes_coordinate = self.environment.nodes_coordinate
        nodes_connection = self.environment.nodes_connection

        ax.set_xlim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,1])/1000 + 1)
        ax.set_ylim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,2])/1000 + 1)
        ax.set_xlabel('km')
        ax.set_ylabel('km')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top') 
        ax.invert_yaxis() # positive to down


        for (i,j) in nodes_connection:
            x1, x2 = nodes_coordinate[i,1] / 1000, nodes_coordinate[j,1] / 1000
            y1, y2 = nodes_coordinate[i,2] / 1000, nodes_coordinate[j,2] / 1000 # unit: km
            plt.plot((x1,x2), (y1,y2), color = 'gray')
        
        
        ax.set_title(self.GetTime(), y=0, fontsize = 22)

        return ax
    

    # function: Draw the vehicles and requests
    # params: None
    # return: picture
    def DrawVehiclesandReuqestsToyModel(self, ax):
        #ax = self.DrawRoadNetworkToyModel(ax) # Draw the road network first

        nodes_coordinate = self.environment.nodes_coordinate
        v_size = self.environment.distance_per_line / 1000 # the size of the vehicle

        # Draw the vehicles and request therein
        for vehicle in self.vehicles:
            xv, yv = nodes_coordinate[int(vehicle.current_position - 1), 1:] / 1000
            # We use rectangle to represent vehicles
            rec = plt.Rectangle((xv-v_size/2, yv-v_size/2), v_size, v_size, facecolor = 'slategrey')
            ax.add_patch(rec)

            # We use circles to represent requests
            # green represents requests have been pick up
            for i, request in enumerate(vehicle.current_requests):
                cir = plt.Circle(self.GetCircleCenter(xv,yv,v_size,i), radius = v_size/4, color="green", fill=True, alpha=1)
                ax.add_patch(cir)
                # We also draw the destination of the request
                xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000
                cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="green", fill=False, alpha=0.5)
                ax.add_patch(cir)
            
            # orange represents request has been allocated to a vehicle but has not been pick up
            for i,request in enumerate(vehicle.next_requests):
                xr_p,yr_p = nodes_coordinate[int(request.pickup_position - 1), 1:] / 1000 
                xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000 # The coordinate of the request
                cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="orange", fill=True, alpha=0.5)
                ax.add_patch(cir)
                cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="orange", fill=False, alpha=0.5)
                ax.add_patch(cir)
        
        # Draw the new requests
        # red represents new requests
        for request in self.requests:
            xr_p,yr_p = nodes_coordinate[int(request.pickup_position - 1), 1:] / 1000 
            xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000 # The coordinate of the request
            cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="red", fill=True, alpha=0.5)
            ax.add_patch(cir)
            cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="red", fill=False, alpha=0.5)
            ax.add_patch(cir)
        return ax

    
    # function: convert current timepoint to real time
    # params: None
    # return: real time: string
    def GetTime(self):
        hour = int(self.current_timepoint / 3600)
        min = int((self.current_timepoint - hour * 3600) / 60)
        sec = self.current_timepoint - hour * 3600 - min * 60

        return f'{hour} h {min} m {sec} s'

    
    # function: Calculate the position of requests in the vehicle
    # params: the coordinate of the vehicle, the size of the vehicle, and the index of the request
    # return: the circle center of the request
    def GetCircleCenter(self, xv, yv, v_size, i):
        if i == 0:
            return (xv - v_size/4, yv - v_size/4)
        elif i == 1:
            return (xv + v_size/4, yv - v_size/4)
        elif i == 2:
            return (xv + v_size/4, yv + v_size/4)
        else:
            return (xv - v_size/4, yv + v_size/4)



    # function: Make all result images a vedio
    # params: the image path
    # return: None
    def MakeVedio(self, imgs = None, img_path = 'Output/tmp', vedio_fps=30, vedio_path = 'Output'):
        # read images
        if imgs:
            imgs = imgs
        else:
            img_names = os.listdir(img_path)
            for idx in range(len(img_names)):
                img_name = str(idx).zfill(6) + '.png'
                img = cv2.imread(os.path.join(img_path, img_name))
                if img is None:
                    continue
                imgs.append(img)
                os.remove(os.path.join(img_path, img_name)) # remove the images
        
        # make vedio
        height, width = imgs[0].shape[:2]
        vedio_name = 'result.mp4'
        i2v = image2video(width, height)    
        i2v.start(os.path.join(vedio_path, vedio_name), vedio_fps)
        for i in tqdm(range(len(imgs)), desc = 'Making video: '):
            img = imgs[i]
            i2v.record(img)
        i2v.end()



class image2video():
    def __init__(self, img_width, img_height):
        self.video_writer = None
        self.is_end = False
        self.img_width = img_width
        self.img_height = img_height 

    def start(self, file_name, fps):
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        img_size = (self.img_width, self.img_height)

        self.video_writer = cv2.VideoWriter()
        self.video_writer.open(file_name, four_cc, fps, img_size, True)

    def record(self, img):
        if self.is_end is False:
            self.video_writer.write(img)

    def end(self):
        self.is_end = True
        self.video_writer.release()


class video2image():
    def __init__(self, file, start_frame = 1330, end_frame = 2000):
        video = cv2.VideoCapture(file)
        self.n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)+0.5)
        self.fps = int(video.get(cv2.CAP_PROP_FPS)+0.5)
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
        
        print('start frame extraction ...')
        self.images = []
        if start_frame is None or end_frame is None:
            start_frame, end_frame = 0, self.n_frames
        for frame in range(end_frame):
            if (frame+1) % 50 == 0 and frame >= start_frame:
                print(f'complete {frame+1}/{self.n_frames}')
                #break
            _, image = video.read()
            if image is not None and frame >= start_frame:
                self.images.append(image)
        