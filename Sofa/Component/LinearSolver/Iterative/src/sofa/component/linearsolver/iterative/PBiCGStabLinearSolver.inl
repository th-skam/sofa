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
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/component/linearsolver/iterative/PBiCGStabLinearSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/map.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <cmath>
#include <string>

namespace sofa::component::linearsolver::iterative
{

template<class TMatrix, class TVector>
PBiCGStabLinearSolver<TMatrix,TVector>::PBiCGStabLinearSolver()
    : d_maxIter(initData(&d_maxIter, 25u, "iterations", "Maximum number of iterations after which the iterative descent of BiCGSTAB must stop"))
    , d_tolerance(initData(&d_tolerance, 1e-5_sreal, "tolerance", "Desired accuracy of the solution, evaluating |r|/|b| (ratio of current residual norm over right-hand side norm)"))
    , d_threshold(initData(&d_threshold, 1e-30_sreal, "threshold", "Floor below which a scalar denominator of the recurrence is treated as a breakdown rather than divided by"))
    , d_warmStart(initData(&d_warmStart, false, "warmStart", "Use the previous solution as the initial estimate, instead of starting from zero"))
    , d_use_precond(initData(&d_use_precond, true, "use_precond", "Use a preconditioner"))
    , l_preconditioner(initLink("preconditioner", "Link towards the linear solver used to precondition BiCGSTAB"))
    , d_graph(initData(&d_graph, "graph", "Graph of residuals at each iteration"))
{
    d_graph.setWidget("graph");
    this->f_listening.setValue(true);
}

template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::init()
{
    Inherit1::init();

    if (l_preconditioner.empty())
    {
        msg_info() << "No preconditioner is set: this solver will act as an unpreconditioned BiCGSTAB.";
    }
    else if (l_preconditioner.get() == nullptr)
    {
        msg_error() << "No preconditioner found at path: " << l_preconditioner.getLinkedPath();
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }
    else if (l_preconditioner->getTemplateName() == "GraphScattered")
    {
        // a preconditioner has to expose M⁻¹, which a matrix-free solver cannot
        msg_error() << "Cannot use the preconditioner " << l_preconditioner->getName()
                    << " because it is templated on GraphScatteredType, so it assembles no matrix.";
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        msg_info() << "Preconditioner path used: '" << l_preconditioner.getLinkedPath() << "'";
    }

    ensureRequiredLinearSystemType();
    if (this->isComponentStateInvalid())
        return;

    this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
}

/// Assembled instantiations work with whichever linear system the base class chose.
template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::checkLinearSystem()
{
    Inherit1::checkLinearSystem();
}

template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::ensureRequiredLinearSystemType()
{
}

/// Nothing to wire on the assembled path: an assembled preconditioner owns its own
/// linear system and that system is assembled by the scene like any other.
template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::bwdInit()
{
}

template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        // one residual curve per Newton iteration, discarded at the start of the step
        m_newtonIteration = 0;
        d_graph.beginEdit()->clear();
        d_graph.endEdit();
    }
}

/// Generic path: the preconditioner exposes its right-hand side and solution as
/// linearalgebra::BaseVector, so the values have to be copied in and out.
template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::applyPreconditioner(Vector& out, Vector& in)
{
    auto* system = l_preconditioner->getLinearSystem();
    auto* rhs = system->getSystemRHSBaseVector();
    auto* solution = system->getSystemSolutionBaseVector();

    if (!rhs || !solution)
    {
        msg_error() << "The preconditioner (" << l_preconditioner->getPathName()
                    << ") does not expose a right-hand side and a solution vector; "
                       "the preconditioner cannot be applied.";
        out = in;
        return;
    }

    if (rhs->size() != in.size() || solution->size() != in.size())
    {
        msg_error() << "The preconditioner system is sized " << rhs->size()
                    << " but the solver operates on " << in.size()
                    << " unknowns; the preconditioner cannot be applied.";
        out = in;
        return;
    }

    for (typename Vector::Index i = 0; i < in.size(); ++i)
    {
        rhs->set(i, in[i]);
    }

    l_preconditioner->solveSystem();

    for (typename Vector::Index i = 0; i < in.size(); ++i)
    {
        out[i] = solution->element(i);
    }
}

/// Solve A x = b with the preconditioned stabilized bi-conjugate gradient.
///
/// The recurrence follows van der Vorst (1992). Only products with A appear, never
/// with Aᵗ, which is what allows the same code to run on the matrix-free operator.
template<class TMatrix, class TVector>
void PBiCGStabLinearSolver<TMatrix,TVector>::solve(Matrix& A, Vector& x, Vector& b)
{
    SCOPED_TIMER_VARNAME(solveTimer, "PBiCGStabLinearSolver::solve");

    std::map<std::string, sofa::type::vector<Real> >& graph = *d_graph.beginEdit();
    m_newtonIteration++;
    sofa::type::vector<Real>& graphError = graph[std::string("Error ") + std::to_string(m_newtonIteration)];
    graphError.clear();

    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, A, x, b);

    Vector& r    = *vtmp.createTempVector(); ///< residual b - A x
    Vector& rhat = *vtmp.createTempVector(); ///< shadow residual, fixed at r₀
    Vector& p    = *vtmp.createTempVector(); ///< search direction
    Vector& v    = *vtmp.createTempVector(); ///< A·M⁻¹p
    Vector& s    = *vtmp.createTempVector(); ///< residual after the first half-step
    Vector& t    = *vtmp.createTempVector(); ///< A·M⁻¹s

    const bool applyPrecond = l_preconditioner.get() != nullptr && d_use_precond.getValue();

    // M⁻¹p and M⁻¹s only need storage of their own when there is an M to apply.
    // Unpreconditioned they *are* p and s, and copying them every iteration would
    // be pure waste.
    Vector* yStorage = applyPrecond ? vtmp.createTempVector() : nullptr;
    Vector* zStorage = applyPrecond ? vtmp.createTempVector() : nullptr;
    Vector& y = applyPrecond ? *yStorage : p;
    Vector& z = applyPrecond ? *zStorage : s;

    // On the assembled path the temporaries come back unsized, and nothing below
    // would size them: eq() and peq() iterate over the destination's own size, so
    // they silently do nothing on an empty vector, and a product with an unsized
    // vector reads past its end. Assignment is the only sizing operation both
    // vector families accept -- GraphScatteredVector::resize is deliberately
    // unimplemented -- so every temporary is sized from b up front.
    r = b;
    rhat = b;
    p = b;
    v = b;
    s = b;
    t = b;
    if (applyPrecond)
    {
        y = b;
        z = b;
    }

    const Real bNorm = static_cast<Real>(b.norm());
    const Real tolerance = d_tolerance.getValue();
    const Real threshold = d_threshold.getValue();

    unsigned iter = 0;
    const char* endcond = "iterations";

    // Nothing to solve against: any x is a solution of A x = 0 up to the kernel of
    // A, and the residual ratio the loop tests is not even defined.
    if (bNorm == 0.0)
    {
        x.clear();
        graphError.push_back(0);
        endcond = "null norm of vector b";
    }
    else
    {
        if (d_warmStart.getValue())
        {
            r = A * x;
            r.eq(b, r, -1.0);        // r = b - A x
        }
        else
        {
            x.clear();
            r = b;                   // r = b - A·0
        }

        rhat = r;
        p.clear();
        v.clear();

        Real rhoPrevious = 1.0;
        Real alpha = 1.0;
        Real omega = 1.0;

        Real residual = static_cast<Real>(r.norm()) / bNorm;
        graphError.push_back(residual);

        while (iter < d_maxIter.getValue() && residual > tolerance)
        {
            ++iter;

            const Real rho = rhat.dot(r);
            if (!std::isfinite(rho) || std::abs(rho) <= threshold)
            {
                msg_warning() << "Breakdown at iteration " << iter << ": the shadow residual has become "
                              << "orthogonal to the residual (rho = " << rho << "). Returning the current estimate.";
                endcond = "breakdown (rho)";
                break;
            }

            const Real beta = (rho / rhoPrevious) * (alpha / omega);
            if (!std::isfinite(beta))
            {
                msg_warning() << "Breakdown at iteration " << iter << ": the direction update overflowed "
                              << "(beta = " << beta << "). Returning the current estimate.";
                endcond = "breakdown (beta)";
                break;
            }

            p.peq(v, -omega);        // p = p - omega v
            p.eq(r, p, beta);        // p = r + beta p

            if (applyPrecond)
            {
                SCOPED_TIMER_VARNAME(applyPrecondTimer, "PBiCGStabLinearSolver::apply Precond");
                applyPreconditioner(y, p);
            }

            v = A * y;

            const Real sigma = rhat.dot(v);
            if (!std::isfinite(sigma) || std::abs(sigma) <= threshold)
            {
                msg_warning() << "Breakdown at iteration " << iter << ": the shadow residual has become "
                              << "orthogonal to A p (sigma = " << sigma << "). Returning the current estimate.";
                endcond = "breakdown (sigma)";
                break;
            }

            // alpha is checked before it reaches x: an absolute floor on sigma
            // cannot catch every overflow, since rho/sigma can exceed the range of
            // Real while both operands are perfectly finite
            const Real alphaCandidate = rho / sigma;
            if (!std::isfinite(alphaCandidate))
            {
                msg_warning() << "Breakdown at iteration " << iter << ": the step length overflowed "
                              << "(alpha = " << alphaCandidate << "). Returning the current estimate.";
                endcond = "breakdown (alpha)";
                break;
            }
            alpha = alphaCandidate;

            x.peq(y, alpha);         // x = x + alpha y
            s.eq(r, v, -alpha);      // s = r - alpha v

            // Half-step convergence. Taking the stabilization step from here would
            // cost an extra product with A and divide by a t·t that is about to be
            // zero, so the loop exits with x as it stands.
            residual = static_cast<Real>(s.norm()) / bNorm;
            if (!std::isfinite(residual))
            {
                msg_warning() << "Breakdown at iteration " << iter
                              << ": the residual is no longer finite. Returning the current estimate.";
                endcond = "breakdown (residual)";
                break;
            }
            if (residual <= tolerance)
            {
                r = s;
                graphError.push_back(residual);
                endcond = "tolerance";
                break;
            }

            if (applyPrecond)
            {
                SCOPED_TIMER_VARNAME(applyPrecondTimer, "PBiCGStabLinearSolver::apply Precond");
                applyPreconditioner(z, s);
            }

            t = A * z;

            // omega minimises ||s - omega t||, which is the stabilization that
            // separates BiCGSTAB from the bi-conjugate gradient it derives from
            const Real tau = t.dot(t);
            if (!std::isfinite(tau) || tau <= threshold)
            {
                msg_warning() << "Breakdown at iteration " << iter << ": A M⁻¹s vanished (t·t = " << tau
                              << "). Returning the current estimate.";
                endcond = "breakdown (t.t)";
                break;
            }

            const Real omegaCandidate = t.dot(s) / tau;
            if (!std::isfinite(omegaCandidate))
            {
                msg_warning() << "Breakdown at iteration " << iter << ": the stabilization coefficient "
                              << "overflowed (omega = " << omegaCandidate << "). Returning the current estimate.";
                endcond = "breakdown (omega)";
                break;
            }
            omega = omegaCandidate;

            x.peq(z, omega);         // x = x + omega z
            r.eq(s, t, -omega);      // r = s - omega t

            residual = static_cast<Real>(r.norm()) / bNorm;
            graphError.push_back(residual);
            if (!std::isfinite(residual))
            {
                msg_warning() << "Breakdown at iteration " << iter
                              << ": the residual is no longer finite. Returning the current estimate.";
                endcond = "breakdown (residual)";
                break;
            }

            rhoPrevious = rho;
        }

        if (residual <= tolerance && endcond == std::string("iterations"))
        {
            endcond = "tolerance";
        }
    }

    // Backstop. The guards above stop the recurrence before a non-finite scalar can
    // reach x, but the vectors they are applied to are themselves built from earlier
    // iterates, and on a divergent system those can overflow between two checks.
    // Handing inf or NaN back would put it straight into the mechanical state, where
    // it is unrecoverable and hard to trace; a zero correction is survivable and
    // reported. norm() is the only finiteness probe both vector families support,
    // since GraphScatteredVector has no element access.
    if (!std::isfinite(static_cast<Real>(x.norm())))
    {
        msg_warning() << "The computed solution is not finite; returning a zero solution instead. "
                      << "The system is likely singular or the operator is diverging.";
        x.clear();
        endcond = "non-finite solution";
    }

    d_graph.endEdit();

    if (applyPrecond)
    {
        vtmp.deleteTempVector(zStorage);
        vtmp.deleteTempVector(yStorage);
    }
    vtmp.deleteTempVector(&t);
    vtmp.deleteTempVector(&s);
    vtmp.deleteTempVector(&v);
    vtmp.deleteTempVector(&p);
    vtmp.deleteTempVector(&rhat);
    vtmp.deleteTempVector(&r);

    sofa::helper::AdvancedTimer::valSet("PBiCGStab iterations", iter);

    msg_info() << "solve, nbiter = " << iter << " stop because of " << endcond;
}

} // namespace sofa::component::linearsolver::iterative
