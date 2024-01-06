// C++ includes
// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <Eigen/Dense>
using namespace autodiff;

// The scalar function for which the gradient is needed
VectorXdual f(VectorXdual x, int a)
{
    //std::transform(x.begin(), x.end(), x.begin(), [](const real& r) { return r * exp(r); });
    autodiff::VectorXdual diff(3);
    diff(0) = x(0) * x(1) * a;
    diff(1) = x(0);
    diff(2) = 2 * x[0];
   // dual diff = std::accumulate(x.begin(), x.end(), dual(0.));
    return diff;
    //return x * x.sum();
}

int main()
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    Eigen::Vector3d A, B;
    A << 1, 2, 3;
    B << 4, 5, 6;
 

    // ´òÓ¡½á¹û
    std::cout << "Original Matrix:\n" << A.transpose() * B<< "\n";


    //VectorD x{ 1, 2, 3, 4, 5 };                  // the input array x with 5 variables

    //dual u;                                     // the output scalar u = f(x) evaluated together with gradient below

    //Eigen::VectorXd g = gradient(f, wrt(x), at(x), u); // evaluate the function value u and its gradient vector g = du/dx

    Vector3dual x(3);                  // the input array x with 5 variables
    x << 1, 2, 3;

    Vector3dual y(3);                  // the input array x with 5 variables
    y <<1, 2, 3;


    VectorXdual u;

    int s = 1;

    MatrixXd J = jacobian(f, wrt(x), at(x, s), u); // evaluate the output vector F and the Jacobian matrix dF/dx

    for (int i = 0; i < 3; ++i)
        std::cout << u(i).grad << std::endl;
    std::cout << "F = \n" << u << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << J << std::endl;    // print the evaluated Jacobian matrix dF/dx
}
