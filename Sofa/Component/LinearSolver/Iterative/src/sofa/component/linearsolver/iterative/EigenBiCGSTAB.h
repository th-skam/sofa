/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include <memory>
#include <variant>

/// Eigen's BiCGSTAB stops on the absolute residual |r| from this version on, and on the
/// relative one |r|/|b| before it, while reporting |r|/|b| as its error in both cases.
#define SOFA_EIGEN_BICGSTAB_ABSOLUTE_TOLERANCE EIGEN_VERSION_AT_LEAST(3, 4, 90)

namespace sofa::component::linearsolver::iterative
{

MAKE_SELECTABLE_ITEMS(EigenBiCGSTABPreconditioner,
    sofa::helper::Item{"Identity", "No preconditioning"},
    sofa::helper::Item{"Diagonal", "Jacobi preconditioning: the inverse of the diagonal of the matrix"},
    sofa::helper::Item{"IncompleteLUT", "Incomplete LU factorization with dual threshold (ILUT)"});

/// Linear system solver using the stabilized bi-conjugate gradient (BiCGSTAB) of Eigen.
///
/// Unlike the conjugate gradient it does not assume A = Aᵗ, so the assembled system
/// keeps its non-symmetric contributions.
template<class TMatrix, class TVector>
class EigenBiCGSTAB : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(EigenBiCGSTAB, TMatrix, TVector), SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, TMatrix, TVector));

    using Matrix = TMatrix;
    using Vector = TVector;
    using Real = typename Matrix::Real;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations
    Data<Real> d_tolerance; ///< Desired accuracy, evaluating |r|/|b|
    Data<bool> d_warmStart; ///< Use the previous solution as the initial estimate
    Data<EigenBiCGSTABPreconditioner> d_preconditioner; ///< Preconditioner of the iterations
    Data<Real> d_dropTolerance; ///< IncompleteLUT only: threshold below which entries of the factors are dropped
    Data<int> d_fillFactor; ///< IncompleteLUT only: maximum number of entries kept per row of the factors, as a multiple of the average number per row of the matrix

    void init() override;
    void solve(Matrix& A, Vector& x, Vector& b) override;
    void invert(Matrix& A) override;

    bool supportNonSymmetricSystem() const override { return true; }

    static const std::string GetCustomTemplateName()
    {
        return TMatrix::Name();
    }

protected:
    EigenBiCGSTAB();

    using EigenSparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
    using EigenSparseMatrixMap = Eigen::Map<EigenSparseMatrix>;
    using EigenVectorMap = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    template<class Preconditioner>
    using Solver = Eigen::BiCGSTAB<EigenSparseMatrix, Preconditioner>;

    /// Alternatives in the order of the items of EigenBiCGSTABPreconditioner
    std::variant<Solver<Eigen::IdentityPreconditioner>,
                 Solver<Eigen::DiagonalPreconditioner<Real>>,
                 Solver<Eigen::IncompleteLUT<Real>>> m_solver;

    bool m_preconditionerComputed { false };

    sofa::linearalgebra::CompressedRowSparseMatrix<Real> m_filtered;
    std::unique_ptr<EigenSparseMatrixMap> m_map;

    /// Pattern of the last analysed matrix: the preconditioner analysis only runs again
    /// when it, or the preconditioner, changes.
    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex m_analysedRowBegin;
    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex m_analysedColsIndex;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_EIGENBICGSTAB_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API EigenBiCGSTAB<linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal>>;
#endif

} // namespace sofa::component::linearsolver::iterative
