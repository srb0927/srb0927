from geometry_msgs.msg import Pose, PoseArray, Quaternion, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.cluster.hierarchy import linkage, fcluster, fclusterdata
from . util import rotateQuaternion, getHeading
from . pf_base import PFLocaliserBase
from time import time
import numpy as np
import random
import rospy
import math
import copy

class PFLocaliser(PFLocaliserBase):
    
    # 初始化参数
    def __init__(self):

        super(PFLocaliser, self).__init__()
        
        # 设定运动模型参数
        self.ODOM_ROTATION_NOISE    = np.random.uniform(0.01, 0.3) # 里程计模型旋转噪声
        self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.01, 0.3) # 里程计模型x轴(正向)噪声
        self.ODOM_DRIFT_NOISE       = np.random.uniform(0.01, 0.3) # 里程计模型y轴(侧向)噪声

        # Sensor model parameters
        self.PARTICLE_COUNT = 200 # 粒子数包括用于定位机器人在其环境中的粒子滤波器
        self.NUMBER_PREDICTED_READINGS = 90 # 在计算粒子权重时，激光传感器扫描读数(观测值)的数量常常与预测值进行比较

        # 构成粒子云的粒子的噪声参数
        self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(75, 100) # 粒子云粒子位置噪声-高斯标准差(粒子位置扩散)
        self.PARTICLE_ANGULAR_NOISE    = np.random.uniform(1, 120) # 粒子云粒子角噪声-冯米塞斯标准偏差(粒子旋转扩展)

        # 位姿估计聚类参数
        # 姿态估计技术可用:全局均值，最佳粒子，hac聚类
        self.POSE_ESTIMATE_TECHNIQUE = "hac clustering" # 主动姿态估计技术用于确定机器人的位置和方向
        self.CLUSTER_DISTANCE_THRESHOLD = 0.35 # 群间平均距离阈值，适用于形成平坦群时


    def roulette_wheel_selection(self, probability_weights, cumulative_probability_weight):
        # 轮盘算法-选择权重高的粒子(信念的高概率)
        # 以便对更接近机器人实际姿态的粒子进行重采样
        pose_array = PoseArray() # 初始化一个pose数组对象

        # 对于所有组成粒子云的粒子
        for _ in range(len(self.particlecloud.poses)):
            stop_criterion = random.random() * cumulative_probability_weight # 初始化轮盘赌轮盘选择过程的停止条件，表示要超过的累积概率权重总和的一个百分点
            
            probability_weight_sum = 0 # 初始化概率权重和
            index = 0 # 初始化索引变量(选定的单个粒子)

            # 当概率权重之和小于停止准则时
            while probability_weight_sum < stop_criterion:
                probability_weight_sum += probability_weights[index] # 将被选个体的概率权重加到概率权重之和上并使之相等
                index += 1 # Increment the index of the selected individual
            
            pose_array.poses.append(copy.deepcopy(self.particlecloud.poses[index - 1])) # 将高权重选定个体(所有参数)的姿态配置附加到姿态数组
        
        return pose_array


    def particle_cloud_noise(self, pose_object):
        # 对噪声参数值进行重新采样
        # 如果里程计旋转噪声参数值大于0.1
        if self.ODOM_ROTATION_NOISE > 0.1:
            self.ODOM_ROTATION_NOISE -= 0.01 # 减小参数值(随时间收敛到机器人的位姿)
        
        # 如果里程计旋转噪声参数值不大于0.1
        else:
            self.ODOM_ROTATION_NOISE = np.random.uniform(0.01, 0.1) # 里程计模型旋转噪声
        
        # 如果里程计的平移噪声参数值大于0.1
        if self.ODOM_TRANSLATION_NOISE > 0.1:
            self.ODOM_TRANSLATION_NOISE -= 0.01 # 减小参数值(随时间收敛到机器人的位姿)
        
        # 如果里程计平移噪声参数值不大于0.1
        else:
            self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.01, 0.1) # 测程模型x轴(正向)噪声

        # 如果里程计航海噪声参数值大于0.1
        if self.ODOM_DRIFT_NOISE > 0.1:
            self.ODOM_DRIFT_NOISE -= 0.01 # 减小参数值(随时间收敛到机器人的位姿)
        
        # 如果里程计航海噪声参数值大于0.1
        else:
            self.ODOM_DRIFT_NOISE = np.random.uniform(0.01, 0.1) # 里程计模型y轴(侧向)噪声
        
        # 如果粒子位置噪声参数值大于2
        if self.PARTICLE_POSITIONAL_NOISE > 2.0:
            self.PARTICLE_POSITIONAL_NOISE -= 0.1 # 减小参数值(随时间收敛到机器人的位姿)
        
        # 如果粒子位置噪声参数值不大于2
        else:
            self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(0.01, 2) # 粒子云粒子位置噪声-高斯标准差(粒子位置扩散)

        # 如果粒子角噪声参数值大于90
        if self.PARTICLE_ANGULAR_NOISE > 90.0:
            self.PARTICLE_ANGULAR_NOISE -= 1.0 # 减小参数值(随时间收敛到机器人的位姿)
        
        # 如果粒子位置噪声参数值不大于90
        else:
           self.PARTICLE_ANGULAR_NOISE = np.random.uniform(0.01, 90) # 粒子云粒子角噪声-冯米塞斯标准偏差(粒子旋转扩展)

        # 添加位置噪声到姿态对象中粒子的x和y坐标
        pose_object.position.x += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_TRANSLATION_NOISE
        pose_object.position.y += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_DRIFT_NOISE
        
        # 添加圆形分布噪声(von Mises分布-基于高斯的圆形分布数据采样)
        # e.g. ([0, 270 (90 * 3 standard deviations of the mean)] - 180) * [0.01, 0.1] = [0, 9] (highest)
        angular_displacement_noise = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ODOM_ROTATION_NOISE

        # 添加旋转噪声到姿态对象中的粒子的方向
        pose_object.orientation = rotateQuaternion(pose_object.orientation, angular_displacement_noise)
        
        return pose_object

  
    def initialise_particle_cloud(self, initial_pose):
        """
        Called whenever an initial_pose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        
        :Args:
            | initial_pose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        pose_array = PoseArray() # 初始化一个pose数组对象

        for _ in range(self.PARTICLE_COUNT):                            
            pose_object = Pose() # 初始化一个pose对象

            # 通常采样并计算机器人初始观测的x和y坐标值的位置噪声
            positional_noise_x = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_TRANSLATION_NOISE
            positional_noise_y = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.ODOM_DRIFT_NOISE
            
            # 添加位置噪声到姿态对象中粒子的x和y坐标值
            pose_object.position.x = initial_pose.pose.pose.position.x + positional_noise_x
            pose_object.position.y = initial_pose.pose.pose.position.y + positional_noise_y

            # 添加圆形分布噪声(von Mises分布-基于高斯的圆形数据采样)
            angular_displacement_noise = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ODOM_ROTATION_NOISE
            
            # 添加旋转噪声到姿态对象中的粒子的方向
            pose_object.orientation = rotateQuaternion(initial_pose.pose.pose.orientation, angular_displacement_noise)

            pose_array.poses.append(pose_object)
            
        return pose_array
 

    def update_particle_cloud(self, scan): # 函数根据从传感器模型得到的粒子所获得的权值进行重采样
        """
        This should use the supplied laser scan to update the current
        particle cloud. 
        
        i.e. self.particlecloud should be updated
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update
        """
        
        global latest_scan # 初始化一个全局激光扫描变量
        latest_scan = scan # 存储机器人当前的观测数据

        probability_weights = [] # 初始化概率权重数组变量
        cumulative_probability_weight = 0 # 初始化累积概率权重变量
        
        # 对于粒子云中的每个姿态对象(粒子姿态)
        for pose_object in self.particlecloud.poses:
            probability_weight = self.sensor_model.get_weight(scan, pose_object) # 存储当前粒子的概率权值，精确地表示机器人当前的姿态
            probability_weights.append(probability_weight) # 将为当前粒子计算的概率权重附加到概率权重数组变量
            cumulative_probability_weight += probability_weight # 将当前粒子的概率权重加到累积概率权重变量并使其相等

        # 任务:根据粒子的概率，通过创建一个新的位置和方向的粒子来重新采样粒子云
        # 在旧的粒子云中，使用轮盘选择来选择高权重粒子，而不是低权重粒子(超时后收敛到机器人的当前姿态)
        pose_array = self.roulette_wheel_selection(probability_weights, cumulative_probability_weight) # 重新取样组成粒子云的粒子，这些粒子云被认为是最准确地表示机器人当前姿势的粒子
        
        # 对于pose数组中的每个pose对象(粒子pose)
        for pose_object in pose_array.poses:
            pose_object = self.particle_cloud_noise(pose_object) # 对位姿数组中当前粒子的位姿(机器人观察)施加噪声
        
        self.particlecloud = pose_array # 更新粒子云对象
        """
        Output the estimated position and orientation of the robot
        """
        robot_estimated_position = self.estimatedpose.pose.pose.position # 存储机器人当前姿态的估计位置
        robot_estimated_orientation = self.estimatedpose.pose.pose.orientation # 存储机器人当前姿态的估计方向
    
        # 将机器人估计的方向从四元数转换为欧拉坐标
        orientation_list = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        # 输出机器人的位置和方向估计值(估计姿态)
        print("/estimatedPose: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    x=robot_estimated_position.x, 
                                                                                                    y=robot_estimated_position.y, 
                                                                                                    yaw=math.degrees(yaw)))


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """

        # 如果主动姿态估计技术是全局均值
        if self.POSE_ESTIMATE_TECHNIQUE == "global mean":
            estimated_pose = Pose() # 初始化一个pose对象
            
            # 初始化位置和方向累计和变量
            position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))
            
            # F对于粒子云中的每个姿态对象(粒子姿态)
            for pose_object in self.particlecloud.poses:
                position_sum_x    += pose_object.position.x # 将当前粒子的x位置值与x位置之和相加并相等
                position_sum_y    += pose_object.position.y # 将当前粒子的y位置值与y位置之和相加并相等
                orientation_sum_z += pose_object.orientation.z # 将当前粒子的z旋转值与z方向之和相加并等于
                orientation_sum_w += pose_object.orientation.w # 使当前粒子的w旋转值与w方向之和相等
            
            estimated_pose.position.x    = position_sum_x    / self.PARTICLE_COUNT # S将机器人估计位姿的x位置值设置为粒子x位置值的平均值
            estimated_pose.position.y    = position_sum_y    / self.PARTICLE_COUNT # 将机器人估计位姿的y位置值设为粒子y位置值的平均值
            estimated_pose.orientation.z = orientation_sum_z / self.PARTICLE_COUNT # 将机器人估计位姿的z旋转值设置为粒子的平均z旋转值
            estimated_pose.orientation.w = orientation_sum_w / self.PARTICLE_COUNT # 将机器人姿态估计的w旋转值设为粒子w旋转值的平均值

        # 否则，如果主动姿态估计技术是最好的粒子
        elif self.POSE_ESTIMATE_TECHNIQUE == "best particle":
            estimated_pose = Pose() # 初始化一个pose对象
            
            # 实例化位置、方向和最高信念变量
            position_x, position_y, orientation_z, orientation_w, highest_belief = (0 for _ in range(5))

            # 对于粒子云中的每个姿态对象(粒子姿态)
            for pose_object in self.particlecloud.poses:
                probability_weight = self.sensor_model.get_weight(latest_scan, pose_object) # 存储表示机器人当前姿态的当前粒子的概率权值

                # 如果粒子云中粒子的最高记录信念没有设定，或者小于当前粒子的信念
                if highest_belief == 0 or highest_belief < probability_weight:
                    position_x    = pose_object.position.x # 存储当前粒子姿态的x位置值
                    position_y    = pose_object.position.y # 存储当前粒子姿态的y位置值
                    orientation_z = pose_object.orientation.z # 存储当前粒子姿态的z旋转值
                    orientation_w = pose_object.orientation.w # 存储当前粒子姿态的w旋转值

                    highest_belief = probability_weight # 更新构成粒子云的粒子的最高记录信念

            estimated_pose.position.x    = position_x # 将机器人估计姿态的x位置值设置为大多数人相信的粒子x位置值
            estimated_pose.position.y    = position_y # 将机器人估计姿态的y位置值设为大多数人相信的粒子y位置值
            estimated_pose.orientation.z = orientation_z # 将机器人估计姿态的z旋转值设置为大多数人相信的粒子z旋转值
            estimated_pose.orientation.w = orientation_w # 将机器人估计位姿的w旋转值设为大多数人相信的粒子w旋转值

        # 如果主动姿态估计技术是层次凝聚聚类(HAC)
        elif self.POSE_ESTIMATE_TECHNIQUE == "hac clustering":
            estimated_pose = Pose()	# Instantiate a pose object
            
            # 初始化位置和方向数组变量，以形成距离矩阵
            position_x, position_y, orientation_z, orientation_w = ([] for _ in range(4))
           
            # 对于粒子云中的每个姿态对象(粒子姿态)
            for pose_object in self.particlecloud.poses:     
                # 用公式表示HCA使用的距离矩阵的分量
                position_x.append(pose_object.position.x) # 将当前粒子的x位置附加到位置x数组
                position_y.append(pose_object.position.y) # 将当前粒子的y位置附加到位置y数组
                orientation_z.append(pose_object.orientation.z) # 将当前粒子姿态的z方向附加到z方向数组
                orientation_w.append(pose_object.orientation.w) # 将当前粒子姿态的w方向附加到w方向数组

            position_x    = np.array(position_x) # 将x位置数组强制转换为numpy类型数组
            position_y    = np.array(position_y) # 将y位置数组转换为numpy类型数组
            orientation_z = np.array(orientation_z) # 将z方向数组转换为numpy类型数组
            orientation_w = np.array(orientation_w) # 将w方向数组转换为numpy类型数组
            
            # 将每个粒子的位置和方向值按列堆叠，形成距离矩阵
            distance_matrix = np.column_stack((position_x, position_y, orientation_z, orientation_w))
            
            # 对压缩后的距离矩阵进行层次/凝聚聚类
            # 返回以链接矩阵编码的层次聚类
            # 中值:当两个簇合并成一个新的簇时，两个簇的质心的平均值给出新的质心
            # 又称WPGMC算法
            linkage_matrix = linkage(distance_matrix, method='median')    
                                       
            # 将粒子云中的粒子聚类(通过最小化它们之间的差异)，并为每个粒子分配一个身份
            # 对应其所属集群(返回表示粒子云中每个粒子所属集群的数字数组)
            particle_cluster_identities = fcluster(linkage_matrix, self.CLUSTER_DISTANCE_THRESHOLD, criterion='distance')
            #print(particle_cluster_identities) # 输出粒子集群标识(粒子与集群的关联)

            cluster_count = max(particle_cluster_identities) # 确定由粒子组成的簇的数目
            cluster_particle_counts = [0] * cluster_count # 初始化群集粒子数数组变量
            cluster_probability_weight_sums = [0] * cluster_count # 初始化一个聚类概率权重和数组变量

            # 对于粒子云中的每个粒子簇
            for i, particle_cluster_identity in enumerate(particle_cluster_identities):                 
                pose_object = self.particlecloud.poses[i] # 为粒子分配存储在粒子云中的姿态对象(访问姿态数据)
                
                probability_weight = self.sensor_model.get_weight(latest_scan, pose_object) # 存储表示机器人当前姿态的当前粒子的概率权值

                cluster_particle_counts[particle_cluster_identity - 1] += 1 # 增加组成群集的粒子数
                cluster_probability_weight_sums[particle_cluster_identity - 1] += probability_weight # 将当前粒子的概率权重存储在
                #print(particle_cluster_identity, cluster_probability_weight_sums[particle_cluster_identity - 1]) # 输出每个聚类的聚类概率权重和

            # 找出与所有其他星团相比，总体上最精确的粒子团，
            # 用于更准确地表示机器人当前的姿态
            cluster_highest_belief = cluster_probability_weight_sums.index(max(cluster_probability_weight_sums)) + 1 
            cluster_highest_belief_particle_count = cluster_particle_counts[cluster_highest_belief - 1] # 存储组成最精确星团的粒子数量
            #print(cluster_highest_belief, cluster_probability_weight_sums) # 输出最精确的聚类ID及其概率权重之和

            # 初始化估计的姿态平均操作的位置和方向累加和变量
            position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))

            # 对于粒子云中的每个粒子簇
            for i, particle_cluster_identity in enumerate(particle_cluster_identities):
                # 如果当前粒子属于最准确地表示机器人当前姿态的簇
                if (particle_cluster_identity == cluster_highest_belief):
                    pose_object = self.particlecloud.poses[i] # 为粒子分配存储在粒子云中的姿态对象(访问姿态数据)
                    
                    position_sum_x    += pose_object.position.x # 将当前粒子的x位置值与x位置之和相加并相等
                    position_sum_y    += pose_object.position.y # 将当前粒子的y位置值与y位置之和相加并相等
                    orientation_sum_z += pose_object.orientation.z # 将当前粒子的z旋转值与z方向之和相加并等于
                    orientation_sum_w += pose_object.orientation.w # 使当前粒子的w旋转值与w方向之和相等
                
            estimated_pose.position.x    = position_sum_x    / cluster_highest_belief_particle_count # 将机器人估计位姿的x位置值设置为粒子x位置值的平均值
            estimated_pose.position.y    = position_sum_y    / cluster_highest_belief_particle_count # 将机器人估计位姿的y位置值设为粒子y位置值的平均值
            estimated_pose.orientation.z = orientation_sum_z / cluster_highest_belief_particle_count # 将机器人估计位姿的z旋转值设置为粒子的平均z旋转值
            estimated_pose.orientation.w = orientation_sum_w / cluster_highest_belief_particle_count # 将机器人姿态估计的w旋转值设为粒子w旋转值的平均值

        """
        Output the estimated position and orientation of the robot
        """
        robot_estimated_position = estimated_pose.position # 存储机器人当前姿态的估计位置
        robot_estimated_orientation = estimated_pose.orientation # 存储机器人当前姿态的估计方向
    
        # 将机器人估计的方向从四元数转换为欧拉符号
        orientation_list = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        # 输出机器人的位置和方向估计值(估计姿态)
        print("{technique}: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    technique=self.POSE_ESTIMATE_TECHNIQUE.title(),
                                                                                                    x=robot_estimated_position.x, 
                                                                                                    y=robot_estimated_position.y, 
                                                                                                    yaw=math.degrees(yaw)))

        return estimated_pose # 返回机器人的估计姿态
