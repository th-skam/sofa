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
#include <sofa/testing/BaseTest.h>

#include <sofa/component/linearsolver/iterative/PBiCGStabLinearSolver.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>

#include "LinearSystems.h"

#include <algorithm>

namespace sofa::component::linearsolver::iterative::testing
{

using Solver = PBiCGStabLinearSolver<Matrix, Vector>;

/// Every test wants the same four Datas set away from their scene-oriented
/// defaults: many more iterations and a much tighter tolerance than a time step
/// would ever ask for.
static Solver::SPtr makeSolver(unsigned iterations, SReal tolerance, bool warmStart = false)
{
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    solver->d_maxIter.setValue(iterations);
    solver->d_tolerance.setValue(tolerance);
    solver->d_warmStart.setValue(warmStart);
    solver->init();
    return solver;
}

static const sofa::type::vector<SReal>& residualHistory(const Solver::SPtr& solver)
{
    return solver->d_graph.getValue().at("Error 1");
}

/// The system CGLinearSolver cannot solve (see NonSymmetricSystem_test).
TEST(PBiCGStabLinearSolver, SolvesNonSymmetricSystem)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(50, 0.9, A, x, b);

    const Solver::SPtr solver = makeSolver(200, 1e-10);

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_LT(relativeResidual(A, solution, b), 1e-10);
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-8);
}

/// A second non-symmetric system whose asymmetry is structural -- an extra entry
/// per row at a fixed column offset -- rather than a rescaling of the two
/// off-diagonals. Passing both means the solver is not tuned to one operator shape.
TEST(PBiCGStabLinearSolver, SolvesShiftedCirculantSystem)
{
    Matrix A;
    Vector x, b;
    makeShiftedCirculant(50, 3.0, 7, 1.0, A, x, b);

    const Solver::SPtr solver = makeSolver(200, 1e-10);

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_LT(relativeResidual(A, solution, b), 1e-10);
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-8);
}

/// A limitation worth recording rather than hiding. A constant-coefficient
/// convection-diffusion operator at beta = 0.9, n = 50 has real, well separated
/// eigenvalues but an eigenvector basis conditioned around 1e32. The recursive
/// residual and the true residual agree at every iteration, so nothing here is
/// inconsistent -- but the residual grows transiently to about 1e12 before coming
/// back down, and the digits spent on that excursion are not recovered. A direct
/// LU factorisation is unaffected.
///
/// If this test starts passing at a tighter bound, the solver got better; if it
/// starts failing loosely, something regressed in the recurrence.
TEST(PBiCGStabLinearSolver, ExtremeNonNormalityCostsAccuracy)
{
    Matrix A;
    Vector x, b;
    makeConvectionDiffusion(50, 0.9, A, x, b);

    const Solver::SPtr solver = makeSolver(400, 1e-12);

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    const auto& history = residualHistory(solver);
    const SReal worst = *std::max_element(history.begin(), history.end());
    const SReal trueResidual = relativeResidual(A, solution, b);

    msg_info("PBiCGStab") << "extreme non-normality: peak recursive residual " << worst
                          << ", final recursive " << history.back()
                          << ", final true " << trueResidual;

    EXPECT_GT(worst, 1e6) << "the transient growth that makes this case hard is gone; "
                             "the operator or the test system changed";
    EXPECT_LT(trueResidual, 1e-1) << "no progress at all was made";
}

/// BiCGSTAB terminates in at most n steps in exact arithmetic, so on a 3x3 system
/// three iterations must reach machine precision. Any sign error or misordering in
/// the recurrence still converges eventually but not this fast, which makes this
/// the sharpest cheap check on the recurrence itself.
TEST(PBiCGStabLinearSolver, ExactTerminationOnSmallSystem)
{
    Matrix A;
    Vector x, b;
    makeNonSymmetric3x3(A, x, b);

    const Solver::SPtr solver = makeSolver(3, 1e-14);

    Vector solution(3);
    solution.clear();
    solver->solve(A, solution, b);

    EXPECT_LT(relativeResidual(A, solution, b), 1e-13);
    EXPECT_LE(residualHistory(solver).size(), 4u) << "took more than 3 iterations on a 3x3 system";
}

/// On ground where both methods are valid they must agree. This is what makes the
/// CG-fails test a statement about symmetry rather than about this solver being
/// different in some unspecified way.
TEST(PBiCGStabLinearSolver, MatchesConjugateGradientOnSymmetricSystem)
{
    Matrix A;
    Vector x, b;
    makeSymmetricPositiveDefinite(50, A, x, b);

    const Solver::SPtr bicgstab = makeSolver(200, 1e-12);
    Vector bicgstabSolution(b.size());
    bicgstabSolution.clear();
    bicgstab->solve(A, bicgstabSolution, b);

    using CG = CGLinearSolver<Matrix, Vector>;
    const CG::SPtr cg = sofa::core::objectmodel::New<CG>();
    cg->d_maxIter.setValue(1000);
    cg->d_tolerance.setValue(1e-12);
    cg->d_smallDenominatorThreshold.setValue(1e-30);
    cg->init();
    Vector cgSolution(b.size());
    cgSolution.clear();
    cg->solve(A, cgSolution, b);

    EXPECT_LT(maxAbsoluteDifference(bicgstabSolution, cgSolution), 1e-9);
    EXPECT_LT(maxAbsoluteDifference(bicgstabSolution, x), 1e-9);
}

/// The residual curve in d_graph is the only per-iteration observable the solver
/// publishes, so it has to mean what it claims: |r|/|b| starting at 1 and ending
/// under the tolerance, one entry per iteration.
TEST(PBiCGStabLinearSolver, ResidualHistoryIsConsistent)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(50, 0.9, A, x, b);

    const Solver::SPtr solver = makeSolver(200, 1e-10);

    Vector solution(b.size());
    solution.clear();
    solver->solve(A, solution, b);

    const auto& history = residualHistory(solver);
    ASSERT_GT(history.size(), 1u);
    EXPECT_NEAR(history.front(), 1.0, 1e-12) << "x starts at zero, so the first residual must be |b|/|b|";
    EXPECT_LE(history.back(), 1e-10);
    EXPECT_NEAR(history.back(), relativeResidual(A, solution, b), 1e-9)
        << "the last recorded residual must be the residual of the returned solution";
}

/// Handed the exact answer, the solver must recognise it and return immediately
/// instead of taking a step away from it.
TEST(PBiCGStabLinearSolver, WarmStartUsesTheInitialEstimate)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(20, 0.9, A, x, b);

    const Solver::SPtr solver = makeSolver(200, 1e-10, true);

    Vector solution(x);
    solver->solve(A, solution, b);

    EXPECT_EQ(residualHistory(solver).size(), 1u) << "no iteration should have run";
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-12);
}

/// Without warmStart the incoming estimate is discarded, not used. A solver that
/// silently kept it would pass every other test here.
TEST(PBiCGStabLinearSolver, ColdStartDiscardsTheInitialEstimate)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(20, 0.9, A, x, b);

    const Solver::SPtr solver = makeSolver(200, 1e-10, false);

    Vector solution(b.size());
    for (Vector::Index i = 0; i < solution.size(); ++i)
    {
        solution[i] = 1000.0;   // deliberately far from the answer
    }
    solver->solve(A, solution, b);

    EXPECT_NEAR(residualHistory(solver).front(), 1.0, 1e-12);
    EXPECT_LT(maxAbsoluteDifference(solution, x), 1e-8);
}

/// b = 0 has x = 0 as its solution and no defined residual ratio; the loop must
/// not run and must not divide by |b|.
TEST(PBiCGStabLinearSolver, ZeroRightHandSide)
{
    Matrix A;
    Vector x, b;
    makeVariableConvectionDiffusion(20, 0.9, A, x, b);
    b.clear();

    const Solver::SPtr solver = makeSolver(200, 1e-10);

    Vector solution(b.size());
    for (Vector::Index i = 0; i < solution.size(); ++i)
    {
        solution[i] = 1.0;
    }
    solver->solve(A, solution, b);

    for (Vector::Index i = 0; i < solution.size(); ++i)
    {
        EXPECT_EQ(solution[i], 0.0);
    }
}

} // namespace sofa::component::linearsolver::iterative::testing
