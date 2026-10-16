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

#include <sofa/core/trait/DataTypes.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/InfinitesimalRotation.h>
#include <sofa/helper/decompose.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
struct PolarDecomposition
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

    template<std::size_t NumberOfNodesInElement>
    sofa::Coord_t<DataTypes> computeCentroid(const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodes)
    {
        sofa::Coord_t<DataTypes> centroid;
        for (const auto& node : nodes)
        {
            centroid += node;
        }
        centroid /= static_cast<sofa::Real_t<DataTypes>>(NumberOfNodesInElement);
        return centroid;
    }

    /**
     * The rest offsets q_j = X_j - t0, mapped by the inverse of their spread M = sum_j q_j q_j^T.
     *
     * With them, H = sum_j (x_j - t) q_j^T M^-1 is the deformation gradient of an affine element,
     * whatever its shape. Without M^-1, H = F M, and polar(F M) differs from polar(F) under strain
     * unless M is isotropic, as it is for a square but not for a triangle. Plain offsets where M is
     * singular, as for an element of lower dimension than the space.
     */
    template<std::size_t NumberOfNodesInElement>
    std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> computeRestOffsets(
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        const auto t0 = computeCentroid(nodesRestPosition);

        std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> offsets;
        RotationMatrix spread;
        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
        {
            offsets[j] = nodesRestPosition[j] - t0;
            spread += sofa::type::dyad(offsets[j], offsets[j]);
        }

        RotationMatrix inverseSpread(sofa::type::NOINIT);
        if (inverseSpread.invert(spread))
        {
            for (auto& offset : offsets)
            {
                offset = inverseSpread * offset;
            }
        }
        return offsets;
    }

    template<std::size_t NumberOfNodesInElement>
    void computeRotation(RotationMatrix& rotationMatrix, const RotationMatrix& initialRotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        SOFA_UNUSED(initialRotationMatrix);

        const auto t = computeCentroid(nodesPosition);
        const auto restOffsets = computeRestOffsets(nodesRestPosition);

        sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> P(sofa::type::NOINIT);
        sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> Q(sofa::type::NOINIT);

        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
        {
            P[j] = nodesPosition[j] - t;
            Q[j] = restOffsets[j];
        }

        const auto H = P.multTranspose(Q);

        sofa::helper::Decompose<sofa::Real_t<DataTypes>>::polarDecomposition(H, rotationMatrix);
    }

    static constexpr sofa::Size Dimension = DataTypes::spatial_dimensions;
    static constexpr sofa::Size RotationComponents = NumberOfRotationComponents<Dimension>;

    /// Q's block for one node: a rotation's components against a node's coordinates.
    using RotationJacobianBlock =
        sofa::type::Mat<RotationComponents, Dimension, sofa::Real_t<DataTypes>>;

    /// Whether the extracted rotation can be differentiated. 1D has no rotation to speak of.
    static constexpr bool hasRotationJacobian = (Dimension == 2 || Dimension == 3);

    /**
     * Q = dtheta/dx, the derivative of the extracted rotation with respect to the element nodes.
     *
     * From H = R S, with dR = R generator(dtheta) and dS symmetric,
     *
     *     A dtheta = axial( R^T dH ),      A = tr(S) I - S   in 3D,   A = tr(S)   in 2D
     *
     * the two forms coming from generator(w) S + S generator(w) = generator((tr(S) I - S) w) in 3D
     * and = tr(S) generator(w) in 2D, for symmetric S.
     *
     * H = sum_a p_a q_a^T, q_a the mapped rest offsets of computeRestOffsets, is linear in the nodes
     * and the centroid drops out of it, since sum_a q_a = 0. So dH = sum_a dx_a q_a^T, and since
     * axial(v w^T) = w x v the block for node a is A^-1 crossOperator(q_a) R^T -- one row in 2D,
     * three in 3D.
     *
     * In 3D the eigenvalues of A are the pairwise sums of the singular values of H, and in 2D A is
     * their sum, so A is nonsingular for every element that has not degenerated. Returns false
     * where it has, leaving the jacobian zeroed -- which is the frozen-rotation tangent, and the
     * caller's cue to say so.
     */
    template<std::size_t NumberOfNodesInElement>
    bool computeRotationJacobian(
        std::array<RotationJacobianBlock, NumberOfNodesInElement>& jacobian,
        const RotationMatrix& rotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        const auto t = computeCentroid(nodesPosition);
        const auto restOffsets = computeRestOffsets(nodesRestPosition);

        // The same H that computeRotation decomposed, rebuilt rather than carried: its symmetric
        // factor S = R^T H is what A is built from, and computeRotation does not return it.
        RotationMatrix H;
        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
        {
            H += sofa::type::dyad(nodesPosition[j] - t, restOffsets[j]);
        }
        const RotationMatrix S = rotationMatrix.multTranspose(H);

        sofa::type::Mat<RotationComponents, RotationComponents, sofa::Real_t<DataTypes>> A;
        if constexpr (Dimension == 3)
        {
            A = -S;
            for (sofa::Size i = 0; i < Dimension; ++i)
            {
                A(i, i) += sofa::type::trace(S);
            }
        }
        else
        {
            A(0, 0) = sofa::type::trace(S);
        }

        decltype(A) inverseA(sofa::type::NOINIT);
        if (!inverseA.invert(A))
        {
            for (auto& block : jacobian)
            {
                block.clear();
            }
            return false;
        }

        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
        {
            jacobian[j] = inverseA * crossOperator(restOffsets[j]).multTransposed(rotationMatrix);
        }
        return true;
    }

    static constexpr sofa::helper::Item getItem()
    {
        return {"polar", "Polar decomposition"};
    }
};

}
