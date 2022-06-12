#include <stdint.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <vector>
#include <string>

#include "npy.hpp"

void load_npy(std::string path, std::vector<double> &data, std::vector<unsigned long> &shape);

void visualize(std::vector<std::string> paths)
{
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);

  for (std::string path : paths)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::vector<double> data;
    std::vector<unsigned long> shape;
    load_npy(path, data, shape);

    for (int i = 0; i < shape[0]; i++)
    {
      // cloud->push_back(pcl::PointXYZ(data[i*6], data[i*6+1], data[i*6+2]));
      pcl::PointXYZRGB point((std::uint8_t)((data[i * 6 + 3] + 1) * 128), (std::uint8_t)((data[i * 6 + 4] + 1) * 128), (std::uint8_t)((data[i * 6 + 5] + 1) * 128));
      point.x = data[i * 6];
      point.y = data[i * 6 + 1];
      point.z = data[i * 6 + 2];
      cloud->push_back(point);
    }

    viewer->addPointCloud(cloud, path);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, path);
  }
  viewer->addCoordinateSystem(0.1);
  viewer->setSize(640, 480);
  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
  }
}

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

int main()
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

  visualize(paths);
  return 0;
}