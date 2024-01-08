#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

// Function to perform FEM for Poisson's equation
void finiteElementMethod(int num_elements, double length) {
    // Number of nodes
    int num_nodes = num_elements + 1;

    // Mesh spacing
    double dx = length / num_elements;

    // Assembling stiffness matrix and load vector
    MatrixXd K = MatrixXd::Zero(num_nodes, num_nodes);
    VectorXd F = VectorXd::Zero(num_nodes);

    for (int i = 0; i < num_elements; ++i) {
        // Element stiffness matrix
        MatrixXd Ke(2, 2);
        Ke << 1 / dx, -1 / dx, -1 / dx, 1 / dx;

        // Assemble into global stiffness matrix
        K(i, i) += Ke(0, 0);
        K(i, i + 1) += Ke(0, 1);
        K(i + 1, i) += Ke(1, 0);
        K(i + 1, i + 1) += Ke(1, 1);
    }

    // Assemble load vector
    F(num_nodes - 1) = 1.0;

    // Applying Dirichlet boundary condition (u(0) = 0)
    K(0, 0) = 1.0;
    K(0, 1) = 0.0;
    F(0) = 0.0;

    // Solving the linear system Ku = F
    VectorXd u = K.colPivHouseholderQr().solve(F);

    // Display the solution
    std::cout << "Solution (Displacements):\n" << u << "\n";
}

int main() {
    // Parameters
    int num_elements = 5;
    double length = 1.0;

    // Solve Poisson's equation using FEM
    finiteElementMethod(num_elements, length);

    return 0;
}
