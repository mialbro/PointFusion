

class BoundingBox3D {
public:
    BoundingBox3D(torch::Tensor, const cv::Mat&);
private:
    pointcloud;
    std::vector<cv::Point3d> corners;
}