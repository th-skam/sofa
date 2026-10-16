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
#include <sofa/component/solidmechanics/fem/elastic/CorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
CorotationalFEMForceField<DataTypes, ElementType>::CorotationalFEMForceField()
    : d_checkElementTangent(initData(&d_checkElementTangent, false, "checkElementTangent",
        "Diagnostic: report the analytic element tangent against a central difference of the element "
        "force, and its asymmetry, at every force evaluation. Expensive."))
    , m_rotationMethods(this)
{
    this->addUpdateCallback("selectRotationMethod", {&this->m_rotationMethods.d_rotationMethod},
        [this](const sofa::core::DataTracker&)
        {
            m_rotationMethods.selectRotationMethod();
            computeInitialRotations();
            return this->getComponentState();
        },
        {});

    m_rotationMethods.selectRotationMethod();
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::init()
{
    BaseElementLinearFEMForceField<DataTypes, ElementType>::init();
    FEMForceField<DataTypes, ElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        m_rotationMethods.selectRotationMethod();
        computeInitialRotations();
    }

    if (!this->isComponentStateInvalid())
    {
        const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::beforeElementForce(
    const sofa::core::MechanicalParams* mparams, sofa::type::vector<ElementGradient>& f,
    const sofa::VecCoord_t<DataTypes>& x)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    m_rotations.resize(elements.size(), RotationMatrix::Identity());

    // Q and W are formed only for a rotation method that can be differentiated, so the methods that
    // cannot pay neither the memory nor the work and keep the frozen-rotation tangent they had.
    m_consistentTangent = m_rotationMethods.providesRotationJacobian();
    if (m_consistentTangent)
    {
        m_rotationJacobians.resize(elements.size());
        m_rotationCouplings.resize(elements.size());
    }

    // Here rather than in the element loop because it has to be sequential to report a maximum, and
    // because it depends on x alone -- so no cache can make it agree with a tangent that does not.
    if (d_checkElementTangent.getValue())
    {
        checkElementTangents(x);
    }
}

template <class DataTypes, class ElementType>
auto CorotationalFEMForceField<DataTypes, ElementType>::computeElementLocalDisplacement(
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes,
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& restNodes,
    const RotationMatrix& rotation) const -> ElementDisplacement
{
    const auto t = translation(nodes);
    const auto t0 = translation(restNodes);

    ElementDisplacement displacement{ sofa::type::NOINIT };

    for (sofa::Size j = 0; j < trait::NumberOfNodesInElement; ++j)
    {
        displacement.setsub(j * trait::spatial_dimensions,
            rotation.multTranspose(nodes[j] - t) - (restNodes[j] - t0));
    }

    return displacement;
}

template <class DataTypes, class ElementType>
auto CorotationalFEMForceField<DataTypes, ElementType>::computeElementForce(
    const ElementNodes& nodes, const ElementNodes& restNodes,
    const typename trait::ElementHessian& stiffness,
    const RotationMatrix& initialRotationTransposed,
    RotationMatrix& rotation,
    ElementDisplacement& localDisplacement,
    ElementGradient& localForce) -> ElementGradient
{
    static constexpr auto DIM = trait::spatial_dimensions;

    m_rotationMethods.computeRotation(rotation, initialRotationTransposed, nodes, restNodes);
    localDisplacement = computeElementLocalDisplacement(nodes, restNodes, rotation);
    localForce = stiffness * localDisplacement;

    ElementGradient force = localForce;
    sofa::type::Vec<DIM, sofa::Real_t<DataTypes>> nodeForce { sofa::type::NOINIT };
    for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
    {
        force.getsub(n * DIM, nodeForce);
        force.setsub(n * DIM, rotation * nodeForce);
    }
    return force;
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::computeElementsForces(
    const sofa::simulation::Range<std::size_t>& range, const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementGradient>& elementForces, const sofa::VecCoord_t<DataTypes>& nodePositions)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto restPositionAccessor = this->mstate->readRestPositions();
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];

        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, nodePositions);
        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> restElementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionAccessor.ref());

        auto& elementInitialRotationTransposed = this->m_initialRotationsTransposed[elementId];
        auto& elementRotation = this->m_rotations[elementId];

        const auto& stiffnessMatrix = elementStiffness[elementId];

        ElementDisplacement displacement { sofa::type::NOINIT };
        ElementGradient localForce { sofa::type::NOINIT };
        elementForces[elementId] = computeElementForce(
            elementNodesCoordinates, restElementNodesCoordinates, stiffnessMatrix,
            elementInitialRotationTransposed, elementRotation, displacement, localForce);

        // The rotation's derivative is formed on the same schedule as the rotation itself, which is
        // what the tangent needs: R is frozen for the whole linear solve that follows.
        if (m_consistentTangent)
        {
            const bool differentiated = m_rotationMethods.computeRotationJacobian(
                m_rotationJacobians[elementId], elementRotation, elementNodesCoordinates,
                restElementNodesCoordinates);
            if (!differentiated)
            {
                msg_warning_once() << "An element rotation could not be differentiated, which takes "
                    "an element deformed past degeneracy. Its tangent falls back to a frozen "
                    "rotation.";
            }

            computeRotationCoupling(m_rotationCouplings[elementId], stiffnessMatrix, displacement,
                                    localForce, restElementNodesCoordinates);
        }
    }
}


template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::computeRotationCoupling(
    CouplingBlocks& coupling,
    const typename trait::ElementHessian& stiffness,
    const ElementDisplacement& localDisplacement,
    const ElementGradient& localForce,
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& restNodes) const
{
    static constexpr auto DIM = trait::spatial_dimensions;

    // B_x, node by node. Since crossOperator(v)^T w == generator(w) v, the block that takes dtheta
    // to the local displacement it induces at node a is -crossOperator(x_l,a)^T, with
    // x_l,a = q_a + u_l,a the local position about the centroid.
    const auto t0 = translation(restNodes);
    CouplingBlocks localPosition;
    sofa::Coord_t<DataTypes> nodeValue { sofa::type::NOINIT };
    for (sofa::Size a = 0; a < trait::NumberOfNodesInElement; ++a)
    {
        localDisplacement.getsub(a * DIM, nodeValue);
        localPosition[a] = -crossOperator(nodeValue + (restNodes[a] - t0)).transposed();
    }

    // W = K_l B_x - B_f, node by node.
    RotationMatrix stiffnessBlock { sofa::type::NOINIT };
    for (sofa::Size a = 0; a < trait::NumberOfNodesInElement; ++a)
    {
        coupling[a].clear();
        for (sofa::Size c = 0; c < trait::NumberOfNodesInElement; ++c)
        {
            stiffness.getsub(a * DIM, c * DIM, stiffnessBlock);
            coupling[a] += stiffnessBlock * localPosition[c];
        }
        localForce.getsub(a * DIM, nodeValue);
        coupling[a] += crossOperator(nodeValue).transposed();
    }
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::checkElementTangents(
    const sofa::VecCoord_t<DataTypes>& x)
{
    static constexpr auto DIM = trait::spatial_dimensions;
    static constexpr auto NDOF = trait::NumberOfDofsInElement;

    using Real = sofa::Real_t<DataTypes>;
    using ElementTangent = typename trait::ElementHessian;

    const auto frobenius = [](const auto& m)
    {
        Real sum {};
        for (sofa::Size i = 0; i < m.nbLines; ++i)
            for (sofa::Size j = 0; j < m.nbCols; ++j)
                sum += m(i, j) * m(i, j);
        return std::sqrt(sum);
    };

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto restPositionAccessor = this->mstate->readRestPositions();
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);
    if (m_rotations.size() < elements.size() || elementStiffness.size() < elements.size())
    {
        return;
    }

    Real worstError {}, worstAsymmetry {}, worstStrain {};
    std::size_t worstElement {};
    sofa::Coord_t<DataTypes> worstCentroid;

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        const ElementNodes nodes = extractNodesVectorFromGlobalVector(element, x);
        const ElementNodes restNodes =
            extractNodesVectorFromGlobalVector(element, restPositionAccessor.ref());
        const auto& stiffness = elementStiffness[elementId];
        const auto& initialRotationTransposed = m_initialRotationsTransposed[elementId];

        RotationMatrix rotation { sofa::type::NOINIT };
        ElementDisplacement displacement { sofa::type::NOINIT };
        ElementGradient localForce { sofa::type::NOINIT };
        computeElementForce(nodes, restNodes, stiffness, initialRotationTransposed, rotation,
                            displacement, localForce);

        JacobianBlocks jacobian;
        CouplingBlocks coupling;
        m_rotationMethods.computeRotationJacobian(jacobian, rotation, nodes, restNodes);
        computeRotationCoupling(coupling, stiffness, displacement, localForce, restNodes);

        // The analytic tangent, block by block: R K_l R^T + R W Q.
        ElementTangent analytic { sofa::type::NOINIT };
        RotationMatrix stiffnessBlock { sofa::type::NOINIT };
        for (sofa::Size a = 0; a < trait::NumberOfNodesInElement; ++a)
        {
            for (sofa::Size b = 0; b < trait::NumberOfNodesInElement; ++b)
            {
                stiffness.getsub(a * DIM, b * DIM, stiffnessBlock);
                analytic.setsub(a * DIM, b * DIM,
                    (rotation * stiffnessBlock).multTransposed(rotation)
                        + rotation * coupling[a] * jacobian[b]);
            }
        }

        // A central difference of the same element force, column by degree of freedom. The step is
        // relative to the element, so it means the same thing at every level of a sweep.
        Real elementScale {};
        const auto t0 = translation(restNodes);
        for (const auto& restNode : restNodes)
        {
            elementScale = std::max(elementScale, (restNode - t0).norm());
        }
        const Real step = static_cast<Real>(1e-6) * elementScale;

        ElementTangent difference { sofa::type::NOINIT };
        for (sofa::Size j = 0; j < NDOF; ++j)
        {
            ElementNodes forward = nodes, backward = nodes;
            forward[j / DIM][j % DIM] += step;
            backward[j / DIM][j % DIM] -= step;

            RotationMatrix perturbedRotation { sofa::type::NOINIT };
            ElementDisplacement perturbedDisplacement { sofa::type::NOINIT };
            ElementGradient perturbedLocalForce { sofa::type::NOINIT };
            const auto forwardForce = computeElementForce(forward, restNodes, stiffness,
                initialRotationTransposed, perturbedRotation, perturbedDisplacement,
                perturbedLocalForce);
            const auto backwardForce = computeElementForce(backward, restNodes, stiffness,
                initialRotationTransposed, perturbedRotation, perturbedDisplacement,
                perturbedLocalForce);

            for (sofa::Size i = 0; i < NDOF; ++i)
            {
                difference(i, j) = (forwardForce[i] - backwardForce[i]) / (2 * step);
            }
        }

        const auto scale = frobenius(difference);
        if (scale > 0)
        {
            const Real error = frobenius(analytic - difference) / scale;
            if (error > worstError)
            {
                worstError = error;
                worstElement = elementId;
                worstCentroid = t0;
            }
            worstAsymmetry = std::max(worstAsymmetry,
                                      frobenius(analytic - analytic.transposed()) / scale);
        }
        worstStrain = std::max(worstStrain, displacement.norm() / elementScale);
    }

    msg_info() << "element tangent over " << elements.size() << " elements: relative error against a central "
        "difference " << worstError << " at element " << worstElement << " (rest centroid "
        << worstCentroid << "), relative asymmetry " << worstAsymmetry
        << ", local strain ||u_l||/h " << worstStrain;
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::computeElementsForcesDeriv(
    const sofa::simulation::Range<std::size_t>& range,
    const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementGradient>& elementForcesDeriv,
    const sofa::VecDeriv_t<DataTypes>& nodeDx)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);
    const bool consistent = m_consistentTangent
        && m_rotationJacobians.size() >= elements.size()
        && m_rotationCouplings.size() >= elements.size();

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];

        const auto& elementRotation = m_rotations[elementId];

        // dtheta = Q dx: how far the element's own frame turns under dx, which the frozen tangent
        // takes as zero. One vector per element, both terms below then being linear in it.
        RotationVector dtheta;
        if (consistent)
        {
            const auto& jacobian = m_rotationJacobians[elementId];
            for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
            {
                dtheta += jacobian[n] * nodeDx[element[n]];
            }
        }

        sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>> element_dx(sofa::type::NOINIT);

        for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
        {
            element_dx.setsub(n * trait::spatial_dimensions, elementRotation.multTranspose(nodeDx[element[n]]));
        }

        const auto& stiffnessMatrix = elementStiffness[elementId];

        auto& df = elementForcesDeriv[elementId];
        df = stiffnessMatrix * element_dx;

        // df = R ( K_l R^T dx + W dtheta ), the second term being K_R dx.
        sofa::type::Vec<trait::spatial_dimensions, sofa::Real_t<DataTypes>> nodedForce { sofa::type::NOINIT };
        for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
        {
            df.getsub(n * trait::spatial_dimensions, nodedForce);
            if (consistent)
            {
                nodedForce += m_rotationCouplings[elementId][n] * dtheta;
            }
            df.setsub(n * trait::spatial_dimensions, elementRotation * nodedForce);
        }
    }
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->sofa::core::behavior::ForceField<DataTypes>::mstate)
        .withRespectToPositionsIn(this->sofa::core::behavior::ForceField<DataTypes>::mstate);

    sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>> localMatrix(sofa::type::NOINIT);

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    if (m_rotations.size() < elements.size() || elementStiffness.size() < elements.size())
    {
        return;
    }

    const bool consistent = m_consistentTangent
        && m_rotationJacobians.size() >= elements.size()
        && m_rotationCouplings.size() >= elements.size();

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        const auto& elementRotation = m_rotations[elementId];
        const auto elementRotation_T = elementRotation.transposed();

        const auto& stiffnessMatrix = elementStiffness[elementId];

        for (sofa::Index n1 = 0; n1 < trait::NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < trait::NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getsub(trait::spatial_dimensions * n1, trait::spatial_dimensions * n2, localMatrix);  // extract the submatrix corresponding to the coupling of nodes n1 and n2
                // R K_l R^T, plus the dR/dx term K_R = R W Q, which is not symmetric.
                auto block = elementRotation * localMatrix * elementRotation_T;
                if (consistent)
                {
                    block += elementRotation * m_rotationCouplings[elementId][n1]
                        * m_rotationJacobians[elementId][n2];
                }
                dfdx(element[n1] * trait::spatial_dimensions, element[n2] * trait::spatial_dimensions) += -block;
            }
        }
    }
}

template <class DataTypes, class ElementType>
SReal CorotationalFEMForceField<DataTypes, ElementType>::getPotentialEnergy(
    const sofa::core::MechanicalParams*,
    const sofa::DataVecCoord_t<DataTypes>& x) const
{
    if (this->isComponentStateInvalid())
        return 0;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    if (m_rotations.size() < elements.size())
        return 0;

    const auto positionAccessor = sofa::helper::getReadAccessor(x);
    const auto restPositionAccessor = this->mstate->readRestPositions();

    sofa::Real_t<DataTypes> energy {};

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];

        const auto elementNodesCoordinates = extractNodesVectorFromGlobalVector(element, positionAccessor.ref());
        const auto restElementNodesCoordinates = extractNodesVectorFromGlobalVector(element, restPositionAccessor.ref());

        const auto displacement = computeElementLocalDisplacement(
            elementNodesCoordinates, restElementNodesCoordinates, m_rotations[elementId]);

        // Quadratic form of strain energy: 1/2 d^T K d
        energy += displacement * (elementStiffness[elementId] * displacement);
    }

    return static_cast<SReal>(0.5 * energy);
}

template <class DataTypes, class ElementType>
auto CorotationalFEMForceField<DataTypes, ElementType>::translation(
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) const -> sofa::Coord_t<DataTypes>
{
    // return nodes[0];
    return computeCentroid(nodes);
}

template <class DataTypes, class ElementType>
auto CorotationalFEMForceField<DataTypes, ElementType>::computeCentroid(
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) -> sofa::Coord_t<DataTypes>
{
    sofa::Coord_t<DataTypes> centroid;
    for (const auto node : nodes)
    {
        centroid += node;
    }
    centroid /= static_cast<sofa::Real_t<DataTypes>>(trait::NumberOfNodesInElement);
    return centroid;
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::computeRotations(
    sofa::type::vector<RotationMatrix>& rotations,
    const sofa::VecCoord_t<DataTypes>& nodePositions,
    const sofa::VecCoord_t<DataTypes>& nodeRestPositions)
{
    if (!this->l_topology)
        return;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    sofa::helper::IotaView indices {static_cast<decltype(elements.size())>(0ul), elements.size()};

    rotations.resize(elements.size(), RotationMatrix::Identity());
    if (m_initialRotationsTransposed.size() < elements.size())
    {
        m_initialRotationsTransposed.resize(elements.size(), RotationMatrix::Identity());
    }

    std::for_each(indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];

            const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesCoordinates =
                extractNodesVectorFromGlobalVector(element, nodePositions);
            const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> restElementNodesCoordinates =
                extractNodesVectorFromGlobalVector(element, nodeRestPositions);

            m_rotationMethods.computeRotation(rotations[elementId], m_initialRotationsTransposed[elementId], elementNodesCoordinates, restElementNodesCoordinates);
        }
    );
}

template <class DataTypes, class ElementType>
void CorotationalFEMForceField<DataTypes, ElementType>::computeInitialRotations()
{
    auto restPositionAccessor = this->sofa::core::behavior::ForceField<DataTypes>::mstate->readRestPositions();
    computeRotations(m_initialRotationsTransposed, restPositionAccessor.ref(), restPositionAccessor.ref());

    for (auto& rotation : m_initialRotationsTransposed)
    {
        rotation.transpose();
    }
}

template <class DataTypes, class ElementType>
auto CorotationalFEMForceField<DataTypes, ElementType>::computeStress(
    const DeformationGradient& F, sofa::Size elementId) -> StressVoigtVector
{
    const auto R = (elementId < m_rotations.size()) ? m_rotations[elementId] : RotationMatrix::Identity();
    const auto corotatedF = R.transposed() * F;
    const auto strainTensor = static_cast<Real_t<DataTypes>>(1)/2 * (corotatedF + corotatedF.transposed()) - DeformationGradient::Identity();

    sofa::type::Vec<type::NumberOfIndependentElements<trait::spatial_dimensions>, Real_t<DataTypes>> strainVoigt;
    for (sofa::Size i = 0; i < type::NumberOfIndependentElements<trait::spatial_dimensions>; ++i)
    {
        const auto [p, q] = type::toTensorIndices<trait::spatial_dimensions>(i);
        strainVoigt[i] = (p == q) ? strainTensor(p, q) : strainTensor(p, q) + strainTensor(q, p);
    }

    const auto youngModulus = this->getYoungModulusInElement(elementId);
    const auto poissonRatio = this->getPoissonRatioInElement(elementId);

    LameLambda<Real_t<DataTypes>> lambda { 0 };
    LameMu<Real_t<DataTypes>> mu { 0 };

    sofa::component::solidmechanics::fem::elastic::toLameParameters<DataTypes::spatial_dimensions, Real_t<DataTypes>>(
        YoungModulus<Real_t<DataTypes>>(youngModulus), PoissonRatio<Real_t<DataTypes>>(poissonRatio),
        lambda, mu);

    const auto elasticityTensor = makeIsotropicElasticityTensor<DataTypes::spatial_dimensions, Real_t<DataTypes>>(mu, lambda);

    const StressVoigtVector localStressVoigt = elasticityTensor.toVoigtMatSym().toMat() * strainVoigt;

    sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, Real_t<DataTypes>> localStressMatrix;
    for (sofa::Size i = 0; i < type::NumberOfIndependentElements<trait::spatial_dimensions>; ++i)
    {
        const auto [p, q] = type::toTensorIndices<trait::spatial_dimensions>(i);
        localStressMatrix(p, q) = localStressVoigt[i];
        localStressMatrix(q, p) = localStressVoigt[i];
    }

    const auto globalStressMatrix = R * localStressMatrix * R.transposed();

    StressVoigtVector stressVoigt;
    for (sofa::Size i = 0; i < type::NumberOfIndependentElements<trait::spatial_dimensions>; ++i)
    {
        const auto [p, q] = type::toTensorIndices<trait::spatial_dimensions>(i);
        stressVoigt[i] = globalStressMatrix(p, q);
    }

    return stressVoigt;
}

}  // namespace sofa::component::solidmechanics::fem::elastic
