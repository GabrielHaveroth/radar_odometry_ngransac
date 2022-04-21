#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <iostream>
#include "thread_rand.h"


void enforce_orthogonality(Eigen::MatrixXd &R) {
    if (R.cols() == 3) {
        const Eigen::Vector3d col1 = R.block(0, 1, 3, 1).normalized();
        const Eigen::Vector3d col2 = R.block(0, 2, 3, 1).normalized();
        const Eigen::Vector3d newcol0 = col1.cross(col2);
        const Eigen::Vector3d newcol1 = col2.cross(newcol0);
        R.block(0, 0, 3, 1) = newcol0;
        R.block(0, 1, 3, 1) = newcol1;
        R.block(0, 2, 3, 1) = col2;
    } else if (R.cols() == 2) {
        const double epsilon = 0.001;
        if (fabs(R(0, 0) - R(1, 1)) > epsilon || fabs(R(1, 0) + R(0, 1)) > epsilon) {
            std::cout << "ERROR: this is not a proper rigid transformation!" << std::endl;
        }
        double a = (R(0, 0) + R(1, 1)) / 2;
        double b = (-R(1, 0) + R(0, 1)) / 2;
        double sum = sqrt(pow(a, 2) + pow(b, 2));
        a /= sum;
        b /= sum;
        R(0, 0) = a; R(0, 1) = b;
        R(1, 0) = -b; R(1, 1) = a;
    }
}

void getInliers(Eigen::MatrixXd Tf, std::vector<int> &inliers, Eigen::MatrixXd p1, Eigen::MatrixXd p2, double tolerance) {
    int dim = p1.rows();
    Eigen::MatrixXd p1_prime = Eigen::MatrixXd::Ones(dim + 1, p1.cols());
    p1_prime.block(0, 0, dim, p1.cols()) = p1;
    p1_prime = Tf * p1_prime;
    inliers.clear();
    for (uint i = 0; i < p1_prime.cols(); ++i) {
        auto distance = (p1_prime.block(0, i, dim, 1) - p2.block(0, i, dim, 1)).norm();
        if (distance < tolerance)
            inliers.push_back(i);
    }
}

void get_rigid_transform(Eigen::MatrixXd p1, Eigen::MatrixXd p2, Eigen::MatrixXd &Tf) {
    assert(p1.cols() == p2.cols() && p1.rows() == p2.rows());
    const int dim = p1.rows();
    Eigen::VectorXd mu1 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd mu2 = mu1;
    // Calculate centroid of each point cloud
    for (int i = 0; i < p1.cols(); ++i) {
        mu1 += p1.block(0, i, dim, 1);
        mu2 += p2.block(0, i, dim, 1);
    }
    mu1 /= p1.cols();
    mu2 /= p1.cols();
    // Subtract centroid from each cloud
    Eigen::MatrixXd q1 = p1;
    Eigen::MatrixXd q2 = p2;
    for (int i = 0; i < p1.cols(); ++i) {
        q1.block(0, i, dim, 1) -= mu1;
        q2.block(0, i, dim, 1) -= mu2;
    }
    // Calculate rotation using SVD
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i < p1.cols(); ++i) {
        H += q1.block(0, i, dim, 1) * q2.block(0, i, dim, 1).transpose();
    }
    auto svd = H.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd R_hat = V * U.transpose();
    if (R_hat.determinant() < 0) {
        V.block(0, dim - 1, dim, 1) = -1 * V.block(0, dim - 1, dim, 1);
        R_hat = V * U.transpose();
    }
    if (R_hat.determinant() != 1.0)
        enforce_orthogonality(R_hat);
    // Calculate translation
    Eigen::VectorXd t = mu2 - R_hat * mu1;
    // Create the output transformation
    Tf = Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    Tf.block(0, 0, dim, dim) = R_hat;
    Tf.block(0, dim, dim, 1) = t;
}

double findTransform(
	at::Tensor p1_tensor,
	at::Tensor p2_tensor,
	at::Tensor probabilities,
	int randSeed,
	int hypCount,
	float inlierThresh,
	at::Tensor out_T,
	at::Tensor out_gradients
	)
{
	omp_set_num_threads(8);
    Eigen::setNbThreads(8);
	std::vector<int> best_inliers;
	int cCount = probabilities.size(1); // number of correspondences
	int subset_size = 2; // size of the minimal set
	// access PyTorch tensors
	at::TensorAccessor<float, 2> p1_access = p1_tensor.accessor<float, 2>();
	at::TensorAccessor<float, 2> p2_access = p2_tensor.accessor<float, 2>();
	at::TensorAccessor<float, 3> pAccess = probabilities.accessor<float, 3>();		
	at::TensorAccessor<float, 2> TAccess = out_T.accessor<float, 2>();		
	at::TensorAccessor<float, 3> gAccess = out_gradients.accessor<float, 3>();
	// read correspondences and weights
	std::vector<float> wPts;
	Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, cCount);
    Eigen::MatrixXd p2 = p1;

	for(int c = 0; c < cCount; c++)	
	{
	// 	pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
	// 	pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
		wPts.push_back(pAccess[0][c][0]);
		p1(0, c) = p1_access[0][c];
    	p1(1, c) = p1_access[1][c];
    	p2(0, c) = p2_access[0][c];
    	p2(1, c) = p2_access[1][c];	
	}
	// create categorial distribution from weights
	ThreadRand::init(randSeed);
	std::discrete_distribution<int> multinomialDist(wPts.begin(), wPts.end());
	
	cv::Mat_<double> K = cv::Mat_<double>::eye(3, 3); // dummy calibration matrix (assuming normalized coordinates)
	
	std::vector<int> inlierCounts(hypCount, -1); // hypothesis scores
	std::vector<std::vector<int>> minSets(hypCount); // minimal sets corresponding to each hypothesis
	uint max_inliers = 0;

	// main RANSAC loop
	for(int h = 0; h < hypCount; h++)
	{
#if defined(_OPENMP)
		unsigned threadID = omp_get_thread_num();
#else

		unsigned threadID = 0;
#endif
		//sample a minimal set
		minSets[h] = std::vector<int>(subset_size); // mark which correspondences were selected
		
		Eigen::MatrixXd p1small, p2small;
		p1small = Eigen::MatrixXd::Zero(2, subset_size);
        p2small = p1small;
		
		for(int j = 0; j < subset_size; j++)
		{
			// choose a correspondence based on the provided weights/probabilities
			int cIdx = multinomialDist(ThreadRand::generators[threadID]);
			p1small.block(0, j, 2, 1) = p1.block(0, cIdx, 2, 1);
			p2small.block(0, j, 2, 1) = p2.block(0, cIdx, 2, 1);
			minSets[h][j] = cIdx;
		}
		Eigen::MatrixXd T_current;
        get_rigid_transform(p1small, p2small, T_current);
		 // Check the number of inliers
        std::vector<int> inliers;
		getInliers(T_current, inliers, p1, p2, inlierThresh);
		inlierCounts[h] = inliers.size();

		if (inliers.size() > max_inliers) {
            best_inliers = inliers;
            max_inliers = inliers.size();
        }

	}
	int bestScore = -1; // best inlier count
	// Refine transformation using the inlier set
    Eigen::MatrixXd p1small, p2small;
    p1small = Eigen::MatrixXd::Zero(2, best_inliers.size());
    p2small = p1small;

    for (uint j = 0; j < best_inliers.size(); ++j) {
        p1small.block(0, j, 2, 1) = p1.block(0, best_inliers[j], 2, 1);
        p2small.block(0, j, 2, 1) = p2.block(0, best_inliers[j], 2, 1);
    }

	Eigen::MatrixXd T_best;
    get_rigid_transform(p1small, p2small, T_best);

	for(int h = 0; h < hypCount; h++)
	{
		// store best solution overall
		if(inlierCounts[h] > bestScore)
		{
			bestScore = inlierCounts[h];
		}

		// keep track of the minimal sets sampled in the gradient tensor
		for(unsigned c = 0; c < minSets[h].size(); c++)
		{
			int cIdx = minSets[h][c];
			gAccess[0][cIdx][0] += 1;
		}
	}
	for(unsigned i = 0; i < 3; i++)
	{
		for(unsigned j = 0; j < 3; j++)
		{
			TAccess[i][j] = T_best(i, j);
		}
	}
	return bestScore;
}


// register C++ functions for use in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("find_transform", &findTransform, "Computes fundamental matrix from given correspondences.");
}

