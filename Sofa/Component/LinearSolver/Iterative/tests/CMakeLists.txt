cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.LinearSolver.Iterative_test)

set(SOURCE_FILES
    NonSymmetricSystem_test.cpp
    PBiCGStabLinearSolver_test.cpp
    PBiCGStabPreconditioner_test.cpp
    PBiCGStabBreakdown_test.cpp
    PBiCGStabScene_test.cpp
)

set(HEADER_FILES
    LinearSystems.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing
    Sofa.Component.LinearSolver.Iterative
    Sofa.Component.LinearSolver.Direct
    Sofa.Component.LinearSolver.Ordering
    Sofa.Component.LinearSolver.Preconditioner
    Sofa.Simulation.Graph
    Sofa.SimpleApi
)
