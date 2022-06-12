#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/mls.h>
#include <Eigen/Core>
#include <set>

#include "npy.hpp"

struct PointXYZCluster
{
    double x;
    double y;
    double z;
    int cluster_id;
};

/**
 * @brief load npy file to std::vector
 *
 */
void load_npy(std::string path, std::vector<double> &data, std::vector<unsigned long> &shape)
{
    bool fortran_order;

    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    std::cout << "Loaded Numpy Array: \n";
    std::cout << "shape: ";
    for (size_t i = 0; i < shape.size(); i++)
        std::cout << shape[i] << ", ";
    std::cout << endl;
    std::cout << "fortran order: " << (fortran_order ? "+" : "-");
    std::cout << endl;
}

void insert_clusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, std::vector<PointXYZCluster> &vec_out, std::vector<pcl::PointIndices> &clusters)
{
    for (size_t i = 0; i < clusters.size(); i++)
    {
        pcl::PointIndices &cluster = clusters[i];
        for (size_t j = 0; j < cluster.indices.size(); j++)
        {
            pcl::PointXYZRGB &point = cloud_in->points[cluster.indices[j]];
            vec_out[cluster.indices[j]] = {point.x, point.y, point.z, (int)i};
        }
    }
}

void median_outliers(pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud, std::vector<PointXYZCluster> &vec_out, std::vector<pcl::PointIndices> &clusters)
{
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(colored_cloud);
    // Loop over all clusters and assign most common neighbor-cluster to points in clusters with < 40 elements
    int k = 40;
    for (int i = 0; i < clusters.size(); i++)
    {
        if (clusters[i].indices.size() > 40)
        {
            if (vec_out[clusters[i].indices[0]].cluster_id != i)
            {
                std::cout << vec_out[clusters[i].indices[0]].cluster_id << " " << i << std::endl;
            }
            continue;
        }
        for (int idx : clusters[i].indices)
        {
            Eigen::VectorXi cluster_count = Eigen::VectorXi::Zero(clusters.size());
            pcl::PointXYZRGB point = colored_cloud->points[idx];

            std::vector<int> pointIdxNKNSearch(k);
            std::vector<float> pointNKNSquaredDistance(k);
            if (kdtree.nearestKSearch(point, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (int j = 0; j < pointIdxNKNSearch.size(); j++)
                {
                    int cluster_id = vec_out[pointIdxNKNSearch[j]].cluster_id;
                    cluster_count[cluster_id] += 1;
                }
                Eigen::Index max_idx;
                int max = cluster_count.maxCoeff(&max_idx);
                vec_out[idx].cluster_id = max_idx;
            }
        }
    }

    std::set<int> cluster_unique;
    for (size_t i = 0; i < vec_out.size(); i++)
    {
        cluster_unique.insert(vec_out[i].cluster_id);
    }
    std::cout << "Unique clusters: " << cluster_unique.size() << endl;
}

/**
 * @brief convert region labeld pc to colored pointcloud
 *
 */
void vec_2_xyzrgbpcl(std::vector<PointXYZCluster> &vec_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{
    for (size_t i = 0; i < vec_in.size(); i++)
    {
        pcl::PointXYZRGB &point = cloud_out->points[i];
        point.x = vec_in[i].x;
        point.y = vec_in[i].y;
        point.z = vec_in[i].z;

        int id = vec_in[i].cluster_id;
        point.r = ((int)pow(2, id)) % 160 + 50;
        point.g = ((int)pow(4, id)) % 160 + 50;
        point.b = ((int)pow(8, id)) % 160 + 50;
    }
}

void save_to_npy(std::string path, std::vector<PointXYZCluster> vec_out, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    std::vector<double> data;
    int idx = path.find_last_of("/");
    std::string out_path = "../out/" + path.substr(idx + 1, path.length() - idx - 5) + "_segmented.npy";
    for (size_t i = 0; i < vec_out.size(); i++)
    {
        data.push_back(vec_out[i].x);
        data.push_back(vec_out[i].y);
        data.push_back(vec_out[i].z);
        data.push_back(normals->points[i].normal_x);
        data.push_back(normals->points[i].normal_y);
        data.push_back(normals->points[i].normal_z);
        data.push_back(vec_out[i].cluster_id);
    }

    std::array<long unsigned, 2> shape{{vec_out.size(), 7}};
    npy::SaveArrayAsNumpy(out_path, false, shape.size(), shape.data(), data);
    std::cout << "Saved Numpy Array to " << out_path << std::endl;
}

void label_surfaces(std::string path, bool visualize)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    std::vector<double> data;
    std::vector<unsigned long> shape;
    load_npy(path, data, shape);

    for (int i = 0; i < shape[0]; i++)
    {
        cloud->push_back(pcl::PointXYZ(data[i * 6], data[i * 6 + 1], data[i * 6 + 2]));
        normals->push_back(pcl::Normal(data[i * 6 + 3], data[i * 6 + 4], data[i * 6 + 5]));
    }

    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(1);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(20);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothModeFlag(true);
    reg.setSmoothnessThreshold(20.0 / 180.0 * M_PI);
    reg.setCurvatureTestFlag(false);
    reg.setCurvatureThreshold(10000);
    reg.setResidualTestFlag(false);
    reg.setResidualThreshold(0.0003);

    std::vector<pcl::PointIndices> clusters;
    std::cout << "Region Growth\n";
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();

    std::vector<PointXYZCluster> vec_out(colored_cloud->size());
    insert_clusters(colored_cloud, vec_out, clusters);

    median_outliers(colored_cloud, vec_out, clusters);

    vec_2_xyzrgbpcl(vec_out, colored_cloud);

    save_to_npy(path, vec_out, normals);

    if (visualize)
    {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addPointCloud(colored_cloud, path);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, path);

        viewer->addCoordinateSystem(0.1);
        viewer->setSize(640, 480);
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<std::string> paths = {
        "../npy/233190_8_seed_0/233190_shard_0.npy",
        "../npy/233190_8_seed_0/233190_shard_1.npy",
        "../npy/233190_8_seed_0/233190_shard_2.npy",
        "../npy/233190_8_seed_0/233190_shard_3.npy",
        "../npy/233190_8_seed_0/233190_shard_4.npy",
        "../npy/233190_8_seed_0/233190_shard_5.npy",
        "../npy/233190_8_seed_0/233190_shard_6.npy",
        "../npy/233190_8_seed_0/233190_shard_7.npy",
    };

    for (auto path : paths)
    {
        label_surfaces(path, true);
    }

    return (0);
}