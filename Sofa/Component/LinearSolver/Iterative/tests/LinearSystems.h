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

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

#include <algorithm>
#include <cmath>

/// Reference linear systems used to exercise the iterative solvers. Each builder
/// fills A and an exact solution x, then derives b = A x, so a solver can be
/// checked against the answer and not only against its own residual.
namespace sofa::component::linearsolver::iterative::testing
{

using Matrix = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
using Vector = sofa::linearalgebra::FullVector<SReal>;

/// x_i = 1 + i/n, so no component is zero and the scale varies along the vector
inline void makeReferenceSolution(sofa::Size n, Vector& x)
{
    x.resize(n);
    for (sofa::Size i = 0; i < n; ++i)
    {
        x[i] = 1.0 + static_cast<SReal>(i) / static_cast<SReal>(n);
    }
}

inline void computeRHS(const Matrix& A, const Vector& x, Vector& b)
{
    b.resize(x.size());
    b.clear();
    A.mul(b, x);
}

/// Upwind-differenced 1D convection-diffusion, tridiagonal with
/// (-1-w(i), diag, -1+w(i)). w(i) is the local convection strength: 0 gives the
/// symmetric Laplacian, and the further from 0 the more unequal the two
/// off-diagonals.
template<class ConvectionField>
inline void makeConvectionDiffusionField(sofa::Size n, SReal diag, ConvectionField w,
                                         Matrix& A, Vector& x, Vector& b)
{
    A.resize(n, n);
    for (sofa::Size i = 0; i < n; ++i)
    {
        if (i > 0)
        {
            A.add(i, i - 1, -1.0 - w(i));
        }
        A.add(i, i, diag);
        if (i + 1 < n)
        {
            A.add(i, i + 1, -1.0 + w(i));
        }
    }
    A.compress();

    makeReferenceSolution(n, x);
    computeRHS(A, x, b);
}

/// Constant convection strength.
///
/// Careful what this is used for. A constant-coefficient tridiagonal is turned
/// symmetric by the diagonal similarity D = diag(((1+beta)/(1-beta))^(i/2)), whose
/// condition number is ((1+beta)/(1-beta))^(n/2) -- about 1e32 at beta = 0.9,
/// n = 50. The eigenvalues stay real and well separated, but the eigenvector basis
/// does not, and every Krylov method reacts to that with enormous transient
/// residual growth before it recovers, losing as many digits as the growth spans.
/// So this family is a *hard case* to be documented, not the system to measure
/// ordinary convergence on. Use makeVariableConvectionDiffusion for that.
inline void makeConvectionDiffusion(sofa::Size n, SReal beta, Matrix& A, Vector& x, Vector& b)
{
    makeConvectionDiffusionField(n, 2.0, [beta](sofa::Size) { return beta; }, A, x, b);
}

/// Spatially varying convection strength that changes sign along the domain, as a
/// physical velocity field would. Because no two neighbouring rows share the same
/// coefficients, there is no diagonal similarity that symmetrises the operator, so
/// it stays strongly non-symmetric without becoming pathologically non-normal.
/// This is the system the convergence tests are written against.
inline void makeVariableConvectionDiffusion(sofa::Size n, SReal beta, Matrix& A, Vector& x, Vector& b)
{
    const SReal twoPiOverN = 2.0 * std::acos(static_cast<SReal>(-1.0)) / static_cast<SReal>(n);
    makeConvectionDiffusionField(n, 2.0,
        [beta, twoPiOverN](sofa::Size i) { return beta * std::sin(twoPiOverN * static_cast<SReal>(i)); },
        A, x, b);
}

/// The variable convection-diffusion operator with row i multiplied by a factor
/// spanning several orders of magnitude. Scaling a row of A and the matching entry
/// of b by the same number leaves x untouched, so the exact solution is unchanged
/// while the diagonal now varies wildly -- which is the situation a diagonal
/// preconditioner exists for, and which it should visibly improve.
inline void makeBadlyScaledConvectionDiffusion(sofa::Size n, SReal beta, SReal decades,
                                              Matrix& A, Vector& x, Vector& b)
{
    const SReal twoPiOverN = 2.0 * std::acos(static_cast<SReal>(-1.0)) / static_cast<SReal>(n);
    const auto rowScale = [n, decades](sofa::Size i)
    {
        // sweeps from 10^-decades up to 10^+decades along the domain
        const SReal t = 2.0 * static_cast<SReal>(i) / static_cast<SReal>(n - 1) - 1.0;
        return std::pow(static_cast<SReal>(10.0), decades * t);
    };

    A.resize(n, n);
    for (sofa::Size i = 0; i < n; ++i)
    {
        const SReal w = beta * std::sin(twoPiOverN * static_cast<SReal>(i));
        const SReal scale = rowScale(i);
        if (i > 0)
        {
            A.add(i, i - 1, scale * (-1.0 - w));
        }
        A.add(i, i, scale * 2.0);
        if (i + 1 < n)
        {
            A.add(i, i + 1, scale * (-1.0 + w));
        }
    }
    A.compress();

    makeReferenceSolution(n, x);
    computeRHS(A, x, b);
}

/// Symmetric tridiagonal (-1, 2, -1) plus one extra entry per row at a constant
/// column offset. The asymmetry here is structural rather than a graded rescaling
/// of the two off-diagonals, so it exercises a different failure mode from the
/// convection-diffusion family.
inline void makeShiftedCirculant(sofa::Size n, SReal diag, sofa::Size shift, SReal gamma,
                                 Matrix& A, Vector& x, Vector& b)
{
    A.resize(n, n);
    for (sofa::Size i = 0; i < n; ++i)
    {
        if (i > 0)
        {
            A.add(i, i - 1, -1.0);
        }
        A.add(i, i, diag);
        if (i + 1 < n)
        {
            A.add(i, i + 1, -1.0);
        }
        A.add(i, (i + shift) % n, gamma);
    }
    A.compress();

    makeReferenceSolution(n, x);
    computeRHS(A, x, b);
}

/// Small non-symmetric system, used where the assertion is on the number of
/// iterations rather than on the accuracy: BiCGSTAB must terminate in at most
/// n steps in exact arithmetic.
inline void makeNonSymmetric3x3(Matrix& A, Vector& x, Vector& b)
{
    A.resize(3, 3);
    A.add(0, 0, 4.0); A.add(0, 1, 1.0);
    A.add(1, 0, 2.0); A.add(1, 1, 5.0); A.add(1, 2, 1.0);
                      A.add(2, 1, 2.0); A.add(2, 2, 6.0);
    A.compress();

    makeReferenceSolution(3, x);
    computeRHS(A, x, b);
}

/// Symmetric positive definite tridiagonal (-1, 2, -1), for cross-checking
/// against CGLinearSolver on ground both solvers are valid on.
inline void makeSymmetricPositiveDefinite(sofa::Size n, Matrix& A, Vector& x, Vector& b)
{
    makeConvectionDiffusion(n, 0.0, A, x, b);
}

/// I + gamma S with S skew-symmetric. As gamma grows the symmetric part becomes
/// negligible, and a purely skew A breaks BiCGSTAB by construction: with the usual
/// shadow residual r0hat = r0, the coefficient r0hat . A r0 is identically zero.
/// This is the family that exercises the breakdown guards rather than the
/// convergence path.
inline void makeSkewDominant(sofa::Size n, SReal gamma, Matrix& A, Vector& x, Vector& b)
{
    A.resize(n, n);
    for (sofa::Size i = 0; i < n; ++i)
    {
        A.add(i, i, 1.0);
        if (i + 1 < n)
        {
            A.add(i, i + 1, gamma);
            A.add(i + 1, i, -gamma);
        }
    }
    A.compress();

    makeReferenceSolution(n, x);
    computeRHS(A, x, b);
}

/// ||A x - b|| / ||b||, the quantity every convergence assertion is made on
inline SReal relativeResidual(const Matrix& A, const Vector& x, const Vector& b)
{
    Vector residual(b.size());
    residual.clear();
    A.mul(residual, x);
    residual.peq(b, -1.0);
    return static_cast<SReal>(residual.norm() / b.norm());
}

/// max_i |a_i - b_i|, so a failure reports one number instead of n
inline SReal maxAbsoluteDifference(const Vector& a, const Vector& b)
{
    SReal maxDiff = 0.0;
    for (Vector::Index i = 0; i < a.size(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
    }
    return maxDiff;
}

} // namespace sofa::component::linearsolver::iterative::testing
