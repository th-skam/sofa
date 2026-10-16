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
#include <sofa/component/linearsolver/iterative/EigenBiCGSTAB.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <type_traits>

namespace sofa::component::linearsolver::iterative
{

template<class TMatrix, class TVector>
EigenBiCGSTAB<TMatrix, TVector>::EigenBiCGSTAB()
    : d_maxIter(initData(&d_maxIter, 25u, "iterations", "Maximum number of iterations"))
    , d_tolerance(initData(&d_tolerance, static_cast<Real>(1e-5), "tolerance", "Desired accuracy, evaluating |r|/|b| (ratio of current residual norm over right-hand side norm)"))
    , d_warmStart(initData(&d_warmStart, false, "warmStart", "Use the previous solution as the initial estimate, instead of starting from zero"))
    , d_preconditioner(initData(&d_preconditioner, EigenBiCGSTABPreconditioner("Diagonal"), "preconditioner", ("Preconditioner of the iterations:\n" + EigenBiCGSTABPreconditioner::dataDescription()).c_str()))
    , d_dropTolerance(initData(&d_dropTolerance, static_cast<Real>(1e-4), "dropTolerance", "IncompleteLUT only: threshold below which entries of the factors are dropped"))
    , d_fillFactor(initData(&d_fillFactor, 10, "fillFactor", "IncompleteLUT only: maximum number of entries kept per row of the factors, as a multiple of the average number per row of the matrix"))
{
}

template<class TMatrix, class TVector>
void EigenBiCGSTAB<TMatrix, TVector>::init()
{
    Inherit1::init();

    if (d_fillFactor.getValue() < 1)
    {
        msg_error() << "fillFactor must be at least 1, got " << d_fillFactor.getValue();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class TMatrix, class TVector>
void EigenBiCGSTAB<TMatrix, TVector>::invert(Matrix& A)
{
    m_preconditionerComputed = false;

    if (this->isComponentStateInvalid())
    {
        return;
    }

    {
        SCOPED_TIMER_VARNAME(copyTimer, "copyMatrixData");
        m_filtered.copyNonZeros(A);
        m_filtered.compress();
    }

    // Eigen expects one row offset per row, plus one, and copyNonZeros() drops the
    // rows which are entirely zero.
    m_filtered.fullRows();

    if (m_filtered.rowBegin.size() != static_cast<std::size_t>(m_filtered.rows()) + 1)
    {
        msg_error() << "Cannot solve a matrix of size " << m_filtered.rows() << "x"
                    << m_filtered.cols() << " described by " << m_filtered.rowBegin.size()
                    << " row offsets (" << m_filtered.rows() + 1 << " expected)"
                    << ": it likely contains entries outside its bounds.";
        return;
    }

    m_map = std::make_unique<EigenSparseMatrixMap>(m_filtered.rows(), m_filtered.cols(), m_filtered.getColsValue().size(),
                                                   reinterpret_cast<typename EigenSparseMatrixMap::StorageIndex*>(m_filtered.rowBegin.data()),
                                                   reinterpret_cast<typename EigenSparseMatrixMap::StorageIndex*>(m_filtered.colsIndex.data()),
                                                   m_filtered.colsValue.data());

    bool analyzePattern = m_analysedRowBegin != m_filtered.rowBegin || m_analysedColsIndex != m_filtered.colsIndex;

    const auto selected = static_cast<std::size_t>(d_preconditioner.getValue());
    if (selected != m_solver.index())
    {
        switch (selected)
        {
            case 0: m_solver.template emplace<0>(); break;
            case 1: m_solver.template emplace<1>(); break;
            default: m_solver.template emplace<2>(); break;
        }
        analyzePattern = true;
    }

    std::visit([this, analyzePattern](auto& solver)
    {
        using Preconditioner = typename std::decay_t<decltype(solver)>::Preconditioner;
        if constexpr (std::is_same_v<Preconditioner, Eigen::IncompleteLUT<Real>>)
        {
            solver.preconditioner().setDroptol(d_dropTolerance.getValue());
            solver.preconditioner().setFillfactor(d_fillFactor.getValue());
        }

        if (analyzePattern)
        {
            SCOPED_TIMER_VARNAME(patternAnalysisTimer, "patternAnalysis");
            solver.analyzePattern(*m_map);
        }

        {
            SCOPED_TIMER_VARNAME(preconditionerTimer, "computePreconditioner");
            solver.factorize(*m_map);
        }

        // Eigen reports a row with no entry at all as a numerical issue
        if (solver.info() != Eigen::Success)
        {
            msg_error() << "Cannot compute the preconditioner: the matrix has an empty row.";
            return;
        }

        m_preconditionerComputed = true;
    }, m_solver);

    if (analyzePattern)
    {
        m_analysedRowBegin = m_filtered.rowBegin;
        m_analysedColsIndex = m_filtered.colsIndex;
    }
}

template<class TMatrix, class TVector>
void EigenBiCGSTAB<TMatrix, TVector>::solve(Matrix& A, Vector& x, Vector& b)
{
    SOFA_UNUSED(A);

    if (!m_preconditionerComputed)
    {
        return;
    }

    EigenVectorMap xMap(x.ptr(), x.size());
    EigenVectorMap bMap(b.ptr(), b.size());

    std::visit([this, &xMap, &bMap](auto& solver)
    {
        const Real tolerance = d_tolerance.getValue();
        solver.setMaxIterations(d_maxIter.getValue());
#if SOFA_EIGEN_BICGSTAB_ABSOLUTE_TOLERANCE
        solver.setTolerance(tolerance * bMap.stableNorm());
#else
        solver.setTolerance(tolerance);
#endif

        {
            SCOPED_TIMER_VARNAME(solveTimer, "EigenBiCGSTAB::solve");
            if (d_warmStart.getValue())
            {
                xMap = solver.solveWithGuess(bMap, xMap);
            }
            else
            {
                xMap = solver.solve(bMap);
            }
        }

        msg_info() << "Solved in " << solver.iterations() << " iterations, relative residual " << solver.error();
        msg_warning_when(solver.error() > tolerance)
            << "Did not converge: relative residual " << solver.error() << " after "
            << solver.iterations() << " iterations, for a tolerance of " << tolerance;
    }, m_solver);
}

} // namespace sofa::component::linearsolver::iterative
