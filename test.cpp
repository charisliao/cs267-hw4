#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <numeric>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class CSRMatrix {
public:
    unsigned int nbrow;                 // number of rows
    unsigned int nbcol;                 // number of columns

    std::vector<int> row_ptrs;          // row pointers for the first non-zero value in each row
    std::vector<int> col_idxs;          // column indices of the non-zero values
    std::vector<double> values;         // non-zero values

public:
    /**
     * @brief Construct a new CSRMatrix object
     * 
     * @param nr number of rows
     * @param nc number of columns
    */
    CSRMatrix(const int& nr, const int& nc) : nbrow(nr), nbcol(nc) {
        row_ptrs.resize(nbrow + 1, 0);    // initialize the row pointers
    };

    /**
     * @brief Copy constructor
     * 
     * @param m CSRMatrix object to copy
    */
    CSRMatrix(const CSRMatrix& m) :
        nbrow(m.nbrow), nbcol(m.nbcol), row_ptrs(m.row_ptrs), 
        col_idxs(m.col_idxs), values(m.values) {};

    /**
     * @brief Assignment operator
     * 
     * @param m CSRMatrix object to assign
     * @return CSRMatrix&
    */
    CSRMatrix& operator=(const CSRMatrix& m) {
        if (this != &m) {
            nbrow = m.nbrow;
            nbcol = m.nbcol;
          
            row_ptrs = m.row_ptrs;
            col_idxs = m.col_idxs;
            values = m.values;
        }
        return *this;
    }

    int NbRow() const { return nbrow; }  // get number of rows
    int NbCol() const { return nbcol; }  // get number of columns


    /**
     * @brief Get the value at the specified row and column index
     * 
     * @param i row index
     * @param j column index
     * 
     * @return double value at the specified row and column index
    */
    double operator()(const int& i, const int& j) const {
        assert (i < NbRow() && j < NbCol());

        // iterate over the non-zero values in row i
        int start = row_ptrs[i];
        int end = row_ptrs[i + 1];
        for (int k = start; k < end; k++) {
            if (col_idxs[k] == j) {
                return values[k];
            }
        }
        return 0.;
    }
    
    /**
     * @brief Assign a value to the specified row and column index
     * 
     * @param i row index
     * @param j column index
    */
    double& Assign(const int& i, const int& j) {
        assert (i < NbRow() && j < NbCol());

        // iterate over the non-zero values in row i
        int start = row_ptrs[i];
        int end = row_ptrs[i + 1];
        for (int k = start; k < end; k++) {
            if (col_idxs[k] == j) {
                return values[k];
            }
        }
        // if the column does not exist yet, add it
        col_idxs.insert(col_idxs.begin() + end, j);
        values.insert(values.begin() + end, 0.);

        // update the row pointers
        for (int k = i + 1; k < row_ptrs.size(); k++) {
            row_ptrs[k]++;
        }
        return values[end];
    }

    // parallel matrix-vector product with distributed vector xi
    /**
     * @brief Perform matrix-vector multiplication with a distributed vector xi
     * 
     * @param xi distributed vector
     * @return std::vector<double> result of the matrix-vector multiplication
    */
    std::vector<double> operator*(const std::vector<double>& xi) const {

        std::vector<double> x(NbCol());
        std::copy(xi.begin(), xi.end(), x.begin());


        std::vector<double> b(NbRow(), 0.);

        // iterate over all the rows
        for (int i = 0; i < NbRow(); i++) {
            // iterate over the non-zero entries in row i
            for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
                int j = col_idxs[k];       // get column of non-zero entry k
                double Mij = values[k];    // get value of non-zero entry k
                b[i] += Mij * x[j];        
            }
        }
        return b;
    }

    /**
     * @brief Print the matrix
    */
   void print() {
        for (int i = 0; i < NbRow(); i++) {
            for (int j = 0; j < NbCol(); j++) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};


// parallel scalar product (u,v) (u and v are distributed)
/**
 * @brief Compute the scalar product of two distributed vectors u and v
 * 
 * @param u distributed vector
 * @param v distributed vector
 * @return double scalar product of the two distributed vectors
*/
double operator,(const std::vector<double>& u, const std::vector<double>& v) {
    assert(u.size() == v.size());
    double sp = 0.;
    for (int j = 0; j < u.size(); j++) { 
        sp += u[j] * v[j]; 
    }
    return sp;
}

// norm of a vector u
/**
 * @brief Compute the norm of vector u
 * 
 * @param u distributed vector
 * @return double norm 
*/
double Norm(const std::vector<double>& u) {
    return sqrt((u, u));
}

// addition of two vectors u+v
/**
 * @brief Add two vectors u and v
 * 
 * @param u vector
 * @param v vector
 * @return std::vector<double> result of the addition
*/
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v) {
    assert(u.size() == v.size());
    std::vector<double> w = u;
    for (int j = 0; j < u.size(); j++) { 
        w[j] += v[j]; 
    }
    return w;
}

// multiplication of a vector by a scalar a*u
/**
 * @brief Multiply a vector u by a scalar a
 * 
 * @param a scalar
 * @param u vector
 * @return std::vector<double> result of the multiplication
*/
std::vector<double> operator*(const double& a, const std::vector<double>& u) {
    std::vector<double> w(u.size());
    for (int j = 0; j < w.size(); j++) { 
        w[j] = a * u[j]; 
    }
    return w;
}

// addition assignment operator, add v to u
/**
 * @brief Add vector v to vector u
 * 
 * @param u vector
 * @param v vector
*/
void operator+=(std::vector<double>& u, const std::vector<double>& v) {
    assert(u.size() == v.size());
    for (int j = 0; j < u.size(); j++) { 
        u[j] += v[j]; 
    }
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
/**
 * @brief Jacbobi Preconditioner
 * 
 * @details Perform forward and backward substitution using the Cholesky factorization of the local diagonal block
 * 
 * @param P Cholesky factorization of the local diagonal block
 * @param u distributed vector
 * @return std::vector<double> result of the forward and backward substitution
*/
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u) {
    Eigen::VectorXd b(u.size());
    for (int i = 0; i < u.size(); i++) 
        b[i] = u[i];
    Eigen::VectorXd xe = P.solve(b);
    std::vector<double> x(u.size());
    for (int i = 0; i < u.size(); i++) 
        x[i] = xe[i];
    return x;
}

/**
 * @brief Distributed Conjugate Gradient
 * 
 * @details Solve the linear system Ax = b using the Conjugate Gradient method
 * 
 * @param A distributed sparse matrix
 * @param b distributed vector
 * @param x distributed vector
 * @param tol tolerance
*/
void CG(const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    double tol = 1e-6) {

    assert(b.size() == A.NbRow());
    x.assign(b.size(), 0.);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

    int n = A.NbRow();

    // get the local diagonal block of A
    std::vector<Eigen::Triplet<double>> coefficients;

    for (int i = 0; i < n; i++) {  
        for (int k = A.row_ptrs[i]; k < A.row_ptrs[i + 1]; k++) {   
            int j = A.col_idxs[k];  
            if (j >= 0 && j < n) 
                coefficients.push_back(Eigen::Triplet<double>(i, j, A.values[k]));
        }
    }

    // compute the Cholesky factorization of the diagonal block for the preconditioner
    Eigen::SparseMatrix<double> B(n, n);
    B.setFromTriplets(coefficients.begin(), coefficients.end());
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

    std::vector<double> r = b, z = prec(P, r), p = z, Ap = A * p;
    double np2 = (p, Ap), alpha = 0., beta = 0.;
    double nr = sqrt((z, r));
    double epsilon = tol * nr;

    std::vector<double> res = A * x;
    res += (-1) * b;

    double rres = sqrt((res, res));

    int num_it = 0;
    while (rres > 1e-5) {
        alpha = (nr * nr) / (np2);
        x += (+alpha) * p;
        r += (-alpha) * Ap;
        z = prec(P, r);
        nr = sqrt((z, r));
        beta = (nr * nr) / (alpha * np2);
        p = z + beta * p;
        Ap = A * p;
        np2 = (p, Ap);

        rres = sqrt((r, r));

        num_it++;
        if (rank == 0 && !(num_it % 1)) {
            std::cout << "iteration: " << num_it << "\t";
            std::cout << "residual:  " << rres << "\n";
        }
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize the MPI environment

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
        MPI_Finalize(); // Finalize MPI environment
        return 0;
    }

    int N = find_int_arg(argc, argv, "-N", 100000); // global size

    assert(N % size == 0);
    int n = N / size; // number of local rows

    // row-distributed matrix
    CSRMatrix A(n, N);

    int offset = n * rank;

    // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
    for (int i = 0; i < n; i++) {
        A.Assign(i, offset + i) = 2.0;
        if (offset + i - 1 >= 0) A.Assign(i, offset + i - 1) = -1;
        if (offset + i + 1 < N)  A.Assign(i, offset + i + 1) = -1;
        if (offset + i + N < N) A.Assign(i, offset + i + N) = -1;
        if (offset + i - N >= 0) A.Assign(i, offset + i - N) = -1;
    }

    // Print Matrix A
    std::cout << "Rank: " << rank << ", Local Matrix A: " << std::endl;
    A.print();

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    // Multiply A by a vector x
    std::vector<double> x(n, 1), b = A * x;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "wall time for CG: " << MPI_Wtime() - time << std::endl;

    // Print Vector X
    std::cout << "rank: " << rank << ", x: ";
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    // Print Vector B
    std::cout << "rank: " << rank << ", b: ";
    for (int i = 0; i < n; i++) {
        std::cout << b[i] << " ";
    }

    std::cout << std::endl;

    MPI_Finalize(); // Finalize the MPI environment

    return 0;
}




