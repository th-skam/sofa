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

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * An infinitesimal rotation, in whichever dimension, and the two matrices a corotational tangent
 * builds out of it.
 *
 * A rotation has one component per independent plane: an angle in 2D, an axis-angle vector in 3D.
 * Everything below is written so that a derivation done with skew matrices in 3D reads the same in
 * 2D, where the "skew matrix" degenerates to a single column and the axial vector to a scalar.
 */

/// Independent components of an infinitesimal rotation in N dimensions: 1 in 2D, 3 in 3D.
template <sofa::Size N>
static constexpr sofa::Size NumberOfRotationComponents = N * (N - 1) / 2;

/// generator(w) is the matrix taking a rotation to its action: dR = R generator(dtheta). It is the
/// skew matrix [w]x in 3D and the angle times [[0,-1],[1,0]] in 2D.
template <class Real>
constexpr sofa::type::Mat<2, 2, Real> generator(const sofa::type::Vec<1, Real>& w) noexcept
{
    return {{Real(0), -w[0]}, {w[0], Real(0)}};
}

template <class Real>
constexpr sofa::type::Mat<3, 3, Real> generator(const sofa::type::Vec<3, Real>& w) noexcept
{
    return sofa::type::crossProductMatrix(w);
}

/// The vector w for which generator(w) == m - m^T, so axial(generator(w)) is 2w. In 3D this is
/// twice sofa::helper::Decompose::skewVec; in 2D it is the one entry an antisymmetric 2x2 has.
template <class Real>
constexpr sofa::type::Vec<1, Real> axial(const sofa::type::Mat<2, 2, Real>& m) noexcept
{
    return {m(1, 0) - m(0, 1)};
}

template <class Real>
constexpr sofa::type::Vec<3, Real> axial(const sofa::type::Mat<3, 3, Real>& m) noexcept
{
    return {m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1)};
}

/**
 * crossOperator(v) is the cross product as a matrix: crossOperator(v) * w == v x w, which is a
 * rotation in both dimensions -- a vector in 3D, a scalar in 2D. So it is the skew matrix [v]x in
 * 3D and the single row (generator(1) v)^T in 2D.
 *
 * Its transpose is the companion the tangent needs, since crossOperator(v)^T * w == generator(w) * v
 * holds in both dimensions.
 */
template <class Real>
constexpr sofa::type::Mat<1, 2, Real> crossOperator(const sofa::type::Vec<2, Real>& v) noexcept
{
    return {{-v[1], v[0]}};
}

template <class Real>
constexpr sofa::type::Mat<3, 3, Real> crossOperator(const sofa::type::Vec<3, Real>& v) noexcept
{
    return sofa::type::crossProductMatrix(v);
}

}  // namespace sofa::component::solidmechanics::fem::elastic
