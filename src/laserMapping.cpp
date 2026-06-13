// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "scancontext/Scancontext.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>
#include <boost/filesystem.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN], s_plot12[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, gps_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       extrinT_gps(3, 0.0);
V3D                  GPS_T_wrt_IMU(Zero3d);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry::ConstPtr> gps_buffer;
V3D init_rtk_pos(Zero3d);
bool gps_inited = false;

// Loop Closure Variables
/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

SCManager scManager;
std::vector<gtsam::Pose3> keyframePoses;
std::vector<PointCloudXYZI::Ptr> keyframeClouds;
std::mutex mtxLoop;
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
int keyframe_count = 0;
V3D last_kf_pos(Zero3d);
M3D last_kf_rot(Eye3d);
bool loop_closure_en = true;
double loop_dist_threshold = 2.0;
double loop_angle_threshold = 0.2;

ros::Publisher pubOdomAftPGO;
ros::Publisher pubPathAftPGO;

void gps_odom_cbk(const nav_msgs::Odometry::ConstPtr &msg) 
{
    mtx_buffer.lock();
    gps_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

gtsam::Pose3 eigenToGtsamPose(const V3D &pos, const M3D &rot) {
    return gtsam::Pose3(gtsam::Rot3(rot), gtsam::Point3(pos));
}

// 辅助函数：将 PointCloudXYZI 转换为 PointCloudXYZRGB 并发布，指定颜色
void publish_cloud_with_color(const ros::Publisher &pub, const PointCloudXYZI::Ptr &cloud, int r, int g, int b) {
    if (pub.getNumSubscribers() == 0) return;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_rgb->resize(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        cloud_rgb->points[i].x = cloud->points[i].x;
        cloud_rgb->points[i].y = cloud->points[i].y;
        cloud_rgb->points[i].z = cloud->points[i].z;
        cloud_rgb->points[i].r = r;
        cloud_rgb->points[i].g = g;
        cloud_rgb->points[i].b = b;
    }
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud_rgb, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "camera_init";
    pub.publish(msg);
}

void perform_loop_closure() {
    int loop_id = -1;
    int curr_id = -1;

    mtxLoop.lock();
    // 关键帧数量过少时，跳过回环检测，避免在起始阶段产生误匹配
    if (keyframePoses.size() < (int)scManager.NUM_EXCLUDE_RECENT) {
        mtxLoop.unlock();
        return;
    }

    // 调用 ScanContext 进行回环检测
    // 返回值为 pair，first 为匹配到的历史关键帧 ID，second 为粗略的偏航角差 (yaw diff)
    auto detectResult = scManager.detectLoopClosureID();
    loop_id = detectResult.first;
    float yaw_diff = detectResult.second;
    curr_id = keyframePoses.size() - 1; // 当前最新的关键帧 ID

    // 如果未检测到回环，直接返回
    if (loop_id == -1) {
        mtxLoop.unlock();
        return;
    }

    // 获取外参 T_I_L (IMU Frame -> Lidar Frame 的逆，即 Lidar -> IMU)
    gtsam::Pose3 T_I_L(gtsam::Rot3(state_point.offset_R_L_I), gtsam::Point3(state_point.offset_T_L_I));

    // Scan2Map: 构建回环候选帧附近的局部地图
    PointCloudXYZI::Ptr cloud_target(new PointCloudXYZI());
    int window_size = 5; // 历史帧前后各取 5 帧
    gtsam::Pose3 pose_prev = isamCurrentEstimate.at<gtsam::Pose3>(loop_id);
    
    for (int i = -window_size; i <= window_size; ++i) {
        int idx = loop_id + i;
        if (idx < 0 || idx >= (int)keyframePoses.size()) continue;
        
        PointCloudXYZI::Ptr cloud_trans(new PointCloudXYZI());
        gtsam::Pose3 pose_idx = isamCurrentEstimate.at<gtsam::Pose3>(idx);
        // 将历史关键帧转换到 loop_id 的 IMU 局部坐标系下
        // p_Iprev = T_Iprev_W * T_W_Iidx * T_I_L * p_L
        gtsam::Pose3 T_Iprev_Lidx = pose_prev.inverse() * pose_idx * T_I_L;
        pcl::transformPointCloud(*keyframeClouds[idx], *cloud_trans, T_Iprev_Lidx.matrix().cast<float>());
        *cloud_target += *cloud_trans;
    }

    // 对局部地图进行降采样，提高 ICP 效率和鲁棒性
    PointCloudXYZI::Ptr cloud_target_down(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> ds;
    ds.setLeafSize(0.3, 0.3, 0.3); 
    ds.setInputCloud(cloud_target);
    ds.filter(*cloud_target_down);

    // 提取当前帧点云并转换到其对应的 IMU 系下
    PointCloudXYZI::Ptr cloud_curr(new PointCloudXYZI());
    pcl::transformPointCloud(*keyframeClouds[curr_id], *cloud_curr, T_I_L.matrix().cast<float>());
    
    // 使用后端优化后的位姿计算初始猜测
    gtsam::Pose3 pose_curr = isamCurrentEstimate.at<gtsam::Pose3>(curr_id);
    mtxLoop.unlock();

    ROS_INFO("Loop detected by SC: between %d and %d, local map size: %lu", loop_id, curr_id, cloud_target_down->size());
    
    // 使用 Generalized ICP (GICP) 算法提高匹配鲁棒性
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
    gicp.setMaxCorrespondenceDistance(2.0);     // 增大搜索距离以适应位姿偏差
    gicp.setMaximumIterations(100);             // 最大迭代次数
    gicp.setTransformationEpsilon(1e-6);        // 两次变换矩阵之间的差值阈值
    gicp.setEuclideanFitnessEpsilon(1e-6);      // 均方误差阈值
    gicp.setCorrespondenceRandomness(20);       // GICP 采样随机性

    gicp.setInputSource(cloud_curr);            // 设置源点云（当前帧，已在 IMU 系）
    gicp.setInputTarget(cloud_target_down);    // 设置目标点云（历史局部地图，已在 loop_id 的 IMU 系）
    
    // 使用关键帧位姿计算初始猜测 (Initial Guess)
    gtsam::Pose3 relative_pose_guess = pose_prev.inverse() * pose_curr;
    Eigen::Matrix4f init_guess = relative_pose_guess.matrix().cast<float>();


    PointCloudXYZI dummy;
    gicp.align(dummy, init_guess);

    ROS_INFO("GICP result: score %.4f, converged: %s", gicp.getFitnessScore(), gicp.hasConverged() ? "true" : "false");

    if (gicp.hasConverged()) {
        Eigen::Matrix4f T_final = gicp.getFinalTransformation();

        gtsam::Pose3 relative_pose_matched(gtsam::Rot3(T_final.block<3,3>(0,0).cast<double>()), 
                                          gtsam::Point3(T_final.block<3,1>(0,3).cast<double>()));
        
        // 计算先验位姿与匹配位姿之间的残差 (Delta / Residual)
        gtsam::Pose3 delta_res = relative_pose_guess.between(relative_pose_matched);
        V3D res_t = delta_res.translation();
        V3D res_r = delta_res.rotation().ypr();
        
        ROS_INFO("ICP Pose Residual (Match - Prior): T [%.4f, %.4f, %.4f], R [%.4f, %.4f, %.4f] (deg)", 
                 res_t(0), res_t(1), res_t(2), 
                 res_r(2) * 180.0 / M_PI, res_r(1) * 180.0 / M_PI, res_r(0) * 180.0 / M_PI);
    }

    // 如果 ICP 收敛且匹配得分小于设定阈值 (0.3)，认为回环有效
    if (gicp.hasConverged() && gicp.getFitnessScore() < 0.3) {
        // 保存匹配结果到 PCD 文件 (转换到全局坐标系以便可视化)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
        Eigen::Matrix4f T_W_Iprev = pose_prev.matrix().cast<float>();
        
        auto transform_point = [&](const auto& p_in, const Eigen::Matrix4f& T) {
            V3F pt_in(p_in.x, p_in.y, p_in.z);
            V3F pt_out = T.block<3,3>(0,0) * pt_in + T.block<3,1>(0,3);
            return pt_out;
        };

        for (size_t i = 0; i < cloud_target_down->size(); ++i) {
            pcl::PointXYZRGB p;
            V3F p_out = transform_point(cloud_target_down->points[i], T_W_Iprev);
            p.x = p_out(0); p.y = p_out(1); p.z = p_out(2);
            p.r = 255; p.g = 0; p.b = 0;
            cloud_rgb->push_back(p);
        }
        for (size_t i = 0; i < dummy.size(); ++i) {
            pcl::PointXYZRGB p;
            V3F p_out = transform_point(dummy.points[i], T_W_Iprev);
            p.x = p_out(0); p.y = p_out(1); p.z = p_out(2);
            p.r = 0; p.g = 255; p.b = 0;
            cloud_rgb->push_back(p);
        }
        std::string pcd_path = root_dir + "Log/loop_closure_" + std::to_string(loop_id) + "_" + std::to_string(curr_id) + ".pcd";
        pcl::io::savePCDFileBinary(pcd_path, *cloud_rgb);
        ROS_INFO("Saved loop closure PCD to %s", pcd_path.c_str());

        // 同步保存 scan2map 中的 map (转换为 PointXYZI 并对齐到全局位置)
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_map_save(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& p_in : cloud_target_down->points) {
            pcl::PointXYZI p_out;
            V3F pt_out = transform_point(p_in, T_W_Iprev);
            p_out.x = pt_out(0); p_out.y = pt_out(1); p_out.z = pt_out(2);
            p_out.intensity = p_in.intensity;
            cloud_map_save->push_back(p_out);
        }
        std::string map_pcd_path = root_dir + "Log/loop_map_" + std::to_string(loop_id) + "_" + std::to_string(curr_id) + ".pcd";
        pcl::io::savePCDFileBinary(map_pcd_path, *cloud_map_save);
        
        // 单独保存匹配后的当前帧点云 (对齐到全局位置)
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_scan_save(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& p_in : dummy.points) {
            pcl::PointXYZI p_out;
            V3F pt_out = transform_point(p_in, T_W_Iprev);
            p_out.x = pt_out(0); p_out.y = pt_out(1); p_out.z = pt_out(2);
            p_out.intensity = p_in.intensity;
            cloud_scan_save->push_back(p_out);
        }
        std::string scan_pcd_path = root_dir + "Log/loop_scan_" + std::to_string(loop_id) + "_" + std::to_string(curr_id) + ".pcd";
        pcl::io::savePCDFileBinary(scan_pcd_path, *cloud_scan_save);
        ROS_INFO("Saved map to %s and scan to %s", map_pcd_path.c_str(), scan_pcd_path.c_str());

        // 获取 ICP 计算出的变换矩阵 (IMU 系下的相对位姿)
        Eigen::Matrix4f T = gicp.getFinalTransformation();
        // 转换为 gtsam 格式的 Pose3 (提取旋转和平移部分)
        gtsam::Pose3 relative_pose(gtsam::Rot3(T.block<3,3>(0,0).cast<double>()), 
                                  gtsam::Point3(T.block<3,1>(0,3).cast<double>()));
        
        mtxLoop.lock();
        gtsam::Pose3 pose_before = isamCurrentEstimate.at<gtsam::Pose3>(curr_id);

        // 定义回环约束的噪声模型 (前三个为平移方差，后三个为旋转方差)
        gtsam::noiseModel::Diagonal::shared_ptr loopNoise = 
            gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
        // 在 GTSAM 因子图中添加 BetweenFactor（两节点间的相对位姿约束因子）
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(loop_id, curr_id, relative_pose, loopNoise));
        
        // 重新运行 ISAM2 优化以更新全局轨迹
        isam->update(gtSAMgraph, gtsam::Values()); // 传入新的因子和初始估计（无新变量，传空）
        isam->update();                            // 再次执行 update 确保充分收敛
        isamCurrentEstimate = isam->calculateEstimate(); // 获取优化后的最新状态估计结果

        gtsam::Pose3 pose_after = isamCurrentEstimate.at<gtsam::Pose3>(curr_id);
        gtsam::Pose3 delta = pose_before.between(pose_after);

        gtSAMgraph.resize(0);                      // 清空临时因子图，为下一次添加做准备
        mtxLoop.unlock();
        
        ROS_INFO("Loop optimized!");
        ROS_INFO("\033[1;33m[ISAM2 Result] Pose %d optimized: Shift [%.4f, %.4f, %.4f], Rot [%.4f, %.4f, %.4f]\033[0m", 
                 curr_id, delta.x(), delta.y(), delta.z(), 
                 delta.rotation().roll() * 180.0 / M_PI, 
                 delta.rotation().pitch() * 180.0 / M_PI, 
                 delta.rotation().yaw() * 180.0 / M_PI);
    }
}

void loop_closure_thread() {
    ros::Rate rate(1.0);
    while (ros::ok()) {
        rate.sleep();
        if (!loop_closure_en) continue;
        perform_loop_closure();
    }
}

void publish_pgo_path(const ros::Publisher &pubPath) {
    nav_msgs::Path pgo_path;
    pgo_path.header.frame_id = "camera_init";
    pgo_path.header.stamp = ros::Time::now();
    for (int i = 0; i < (int)isamCurrentEstimate.size(); i++) {
        if (!isamCurrentEstimate.exists(i)) continue;
        gtsam::Pose3 pose = isamCurrentEstimate.at<gtsam::Pose3>(i);
        geometry_msgs::PoseStamped ps;
        ps.header = pgo_path.header;
        ps.pose.position.x = pose.x();
        ps.pose.position.y = pose.y();
        ps.pose.position.z = pose.z();
        ps.pose.orientation.x = pose.rotation().toQuaternion().x();
        ps.pose.orientation.y = pose.rotation().toQuaternion().y();
        ps.pose.orientation.z = pose.rotation().toQuaternion().z();
        ps.pose.orientation.w = pose.rotation().toQuaternion().w();
        pgo_path.poses.push_back(ps);
    }
    pubPath.publish(pgo_path);
}

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    // boreas 数据
    // msg->angular_velocity.x = msg_in->angular_velocity.x;
    // msg->angular_velocity.y = msg_in->angular_velocity.y;
    // msg->angular_velocity.z = msg_in->angular_velocity.z;
    // msg->linear_acceleration.x = msg_in->linear_acceleration.x;
    // msg->linear_acceleration.y = msg_in->linear_acceleration.y;
    // msg->linear_acceleration.z = -msg_in->linear_acceleration.z;
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() -
                                            time_diff_lidar_to_imu);
    ROS_INFO("msg linear acceleration: %f %f %f, angular velocity: "
             "%f %f %f",
             msg->linear_acceleration.x, msg->linear_acceleration.y,
             msg->linear_acceleration.z, msg->angular_velocity.x,
             msg->angular_velocity.y, msg->angular_velocity.z);

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
      msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu +
                                              msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;

    /*** push gps data, and pop from gps buffer ***/
    meas.gps.clear();
    while (!gps_buffer.empty())
    {
        double gps_time = gps_buffer.front()->header.stamp.toSec();
        if (gps_time < lidar_end_time - 0.2) // Too old, discard
        {
            gps_buffer.pop_front();
        }
        else
        {
            break;
        }
    }

    if (!gps_buffer.empty())
    {
        double min_diff = 10.0;
        int closest_idx = -1;
        for (int i = 0; i < (int)gps_buffer.size(); i++)
        {
            double diff = std::abs(gps_buffer[i]->header.stamp.toSec() - lidar_end_time);
            if (diff < min_diff)
            {
                min_diff = diff;
                closest_idx = i;
            }
            else if (gps_buffer[i]->header.stamp.toSec() > lidar_end_time)
            {
                break;
            }
        }

        if (closest_idx != -1 && min_diff < 0.2)
        {
            meas.gps.push_back(gps_buffer[closest_idx]);
            // Pop everything up to the closest one to keep buffer clean
            for (int i = 0; i <= closest_idx; i++)
            {
                gps_buffer.pop_front();
            }
        }
    }

    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    int plane_fitted_num = 0;
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            plane_fitted_num++;
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;
    plane_fitted_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            plane_fitted_num++; // This counts points that initially passed ikdtree search
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;


    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    

    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/gps_topic", gps_topic,"/applanix/odom");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_T_gps", extrinT_gps, vector<double>(3, 0.0));

    if (extrinT_gps.size() == 3) {
        GPS_T_wrt_IMU << extrinT_gps[0], extrinT_gps[1], extrinT_gps[2];
    }

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gps_extrinsic(GPS_T_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_gps = nh.subscribe(gps_topic, 200000, gps_odom_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
    pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);

    // Initialize GTSAM
    isam = new gtsam::ISAM2();
    initialEstimate.clear();
    gtSAMgraph.resize(0);

    std::thread loop_thread(loop_closure_thread);

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (runtime_pos_log)
            {
                V3D ang_predict = SO3ToEuler(state_point.rot);
                printf("[ IMU Predict ]: time: %0.6f pos: %0.6f, %0.6f, %0.6f; ang: %0.6f, %0.6f, %0.6f; vel: %0.6f, %0.6f, %0.6f\n", 
                    Measures.lidar_beg_time,
                    state_point.pos(0), state_point.pos(1), state_point.pos(2), 
                    ang_predict(0), ang_predict(1), ang_predict(2),
                    state_point.vel(0), state_point.vel(1), state_point.vel(2));
            }

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            V3D pos_before = state_point.pos;
            V3D euler_before = SO3ToEuler(state_point.rot);
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

            // GPS Fusion after LiDAR Update
            if (!Measures.gps.empty())
            {
                for (const auto &gps_msg : Measures.gps)
                {
                    state_ikfom s = kf.get_x();
                    auto P = kf.get_P();

                    // GPS Measurement (Position only for RTK)
                    V3D z_p_raw(gps_msg->pose.pose.position.x, gps_msg->pose.pose.position.y, gps_msg->pose.pose.position.z);
                    if (!gps_inited)
                    {
                        init_rtk_pos = z_p_raw - s.rot * GPS_T_wrt_IMU;
                        gps_inited = true;
                        ROS_INFO("GPS Initialized. Init RTK (Origin) Pos: [%.3f, %.3f, %.3f]", init_rtk_pos(0), init_rtk_pos(1), init_rtk_pos(2));
                    }
                    V3D z_p = z_p_raw - init_rtk_pos;
                    
                    // 1. Residual (Innovation)
                    // predicted_z_p = s.pos + s.rot * GPS_T_wrt_IMU
                    V3D predicted_z_p = s.pos + s.rot * GPS_T_wrt_IMU;
                    V3D innovation_p = z_p - predicted_z_p;

                    // 2. Jacobian H (3 x 23)
                    // State: pos(3), rot(3), offset_R(3), offset_T(3), vel(3), bg(3), ba(3), grav(2)
                    Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
                    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // d(res)/d(delta_p)
                    // d(R * L)/d(delta_theta) = -R * [L]x
                    H.block<3, 3>(0, 3) = -s.rot.toRotationMatrix() * skew_sym_mat(GPS_T_wrt_IMU); 

                    // 3. Noise Covariance R (3 x 3)
                    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.01; // Default 0.1m std dev
                    if (gps_msg->pose.covariance[0] > 1e-9)
                    {
                        for (int i = 0; i < 3; i++) R(i, i) = gps_msg->pose.covariance[i * 7];
                    }

                    // 4. Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
                    Eigen::Matrix3d S = H * P * H.transpose() + R;
                    Eigen::Matrix<double, 23, 3> K = P * H.transpose() * S.inverse();

                    // 5. Update State and Covariance
                    Eigen::Matrix<double, 23, 1> dx = K * innovation_p;
                    s.boxplus(dx);
                    kf.change_x(s);
                    Eigen::Matrix<double, 23, 23> P_new = (Eigen::Matrix<double, 23, 23>::Identity() - K * H) * P;
                    kf.change_P(P_new);
                    
                    V3D euler = SO3ToEuler(s.rot);
                    ROS_INFO("GPS Position fused: res [%.3f, %.3f, %.3f], pos [%.3f, %.3f, %.3f], euler [%.3f, %.3f, %.3f]", 
                             innovation_p(0), innovation_p(1), innovation_p(2),
                             s.pos(0), s.pos(1), s.pos(2),
                             euler(0), euler(1), euler(2));
                }
            }

            if (effct_feat_num < 30 || res_mean_last > 0.1)
            {
                ROS_WARN_THROTTLE(1.0, "Poor LiDAR matching: effect_feat_num=%d, res_mean=%f", effct_feat_num, res_mean_last);
            }

            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            
            if (runtime_pos_log)
            {
                V3D pos_diff = state_point.pos - pos_before;
                V3D euler_diff = euler_cur - euler_before;
                for (int i = 0; i < 3; i++)
                {
                    if (euler_diff[i] > 180) euler_diff[i] -= 360;
                    else if (euler_diff[i] < -180) euler_diff[i] += 360;
                }
                printf("[ Matching Diff ]: Position Diff (m): [%.4f, %.4f, %.4f], Rotation Diff (deg): [%.4f, %.4f, %.4f]\n",
                    pos_diff(0), pos_diff(1), pos_diff(2),
                    euler_diff(0), euler_diff(1), euler_diff(2));
            }

            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** Loop Closure Keyframe Selection ***/
            V3D curr_pos = state_point.pos;
            M3D curr_rot = state_point.rot.toRotationMatrix();
            double dist = (curr_pos - last_kf_pos).norm();
            M3D rot_diff = last_kf_rot.transpose() * curr_rot;
            V3D rot_diff_v = Log(rot_diff);
            double angle = rot_diff_v.norm();

            if (dist > loop_dist_threshold || angle > loop_angle_threshold || keyframe_count == 0) {
                last_kf_pos = curr_pos;
                last_kf_rot = curr_rot;

                mtxLoop.lock();
                gtsam::Pose3 curr_pose = eigenToGtsamPose(curr_pos, curr_rot);
                if (keyframe_count == 0) {
                    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = 
                        gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, curr_pose, priorNoise));
                } else {
                    gtsam::Pose3 last_pose = keyframePoses.back();
                    gtsam::Pose3 relative_pose = last_pose.between(curr_pose);
                    gtsam::noiseModel::Diagonal::shared_ptr odomNoise = 
                        gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_count - 1, keyframe_count, relative_pose, odomNoise));
                }
                initialEstimate.insert(keyframe_count, curr_pose);
                isam->update(gtSAMgraph, initialEstimate);
                isam->update();
                isamCurrentEstimate = isam->calculateEstimate();
                gtSAMgraph.resize(0);
                initialEstimate.clear();

                keyframePoses.push_back(curr_pose);
                PointCloudXYZI::Ptr curr_cloud(new PointCloudXYZI());
                pcl::copyPointCloud(*feats_down_body, *curr_cloud);
                keyframeClouds.push_back(curr_cloud);

                pcl::PointCloud<pcl::PointXYZI> sc_cloud;
                pcl::copyPointCloud(*curr_cloud, sc_cloud);
                scManager.makeAndSaveScancontextAndKeys(sc_cloud);
                
                keyframe_count++;
                ROS_INFO("New keyframe added: ID %d, total %lu", keyframe_count-1, keyframePoses.size());
                publish_pgo_path(pubPathAftPGO);
                mtxLoop.unlock();
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                s_plot11[time_log_counter] = effct_feat_num;
                s_plot12[time_log_counter] = res_mean_last;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, effect point size, residual mean\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), int(s_plot11[i]), s_plot12[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
