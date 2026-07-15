import Sofa

def createScene(root):
    root.addObject("RequiredPlugin", pluginName=[
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Topology.Container.Grid',
        'Sofa.Component.Engine.Transform',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual',
    ])

    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels showCollisionModels hideMappings hideForceFields showDetectionOutputs")

    root.gravity = [0, -5e-2, 0]
    root.dt = 1e-2

    # 3 livers with collision models. Each one belongs to a different sub-collision pipeline
    with root.addChild("Livers") as Livers:
        Livers.addObject("MeshOBJLoader", name="liverLoader", filename="mesh/liver.obj")
        for i in range(3):
            dx = [i*10, 0, 0]
            liver = Livers.addChild("liver"+str(i))
            liver.addObject("TransformEngine", name="engine", input_position="@../liverLoader.position", translation=dx)
            liver.addObject("TriangleSetTopologyContainer", name="topology", triangles="@../liverLoader.triangles", position="@engine.output_position")
            liver.addObject("MechanicalObject", name="dofs", template="Vec3d")
            liver.addObject("UniformMass", name="mass", totalMass=1.0)
            liver.addObject("TriangleCollisionModel", name="tri", contactStiffness=1)
            liver.addObject("LineCollisionModel", name="lin", contactStiffness=1)
            liver.addObject("PointCollisionModel", name="pt", contactStiffness=1)
            liver.addObject("EulerImplicitSolver", name="odeSolver")
            liver.addObject("CGLinearSolver", name="linearSolver", iterations=250, tolerance=1e-9, threshold=1e-12)

    # The floor object, shared between the 3 sub-collision pipelines
    root.addObject("RegularGridTopology", name="grid", min=[-10, 0, -5], max=[25, -10, 5], n=[30, 1, 20])
    with root.addChild("Floor") as Floor:
        Floor.addObject("TriangleSetTopologyContainer", name="topology", src="@../grid")
        Floor.addObject("MechanicalObject", name="dofs", position="@topology.position")
        Floor.addObject("UniformMass", name="mass", totalMass=1.0)
        Floor.addObject("TriangleCollisionModel", name="tri", contactStiffness=1)
        Floor.addObject("LineCollisionModel", name="lin", contactStiffness=1)
        Floor.addObject("PointCollisionModel", name="pt", contactStiffness=1)
        Floor.addObject("EulerImplicitSolver", name="odeSolver")
        Floor.addObject("CGLinearSolver", name="linearSolver", iterations=250, tolerance=1e-9, threshold=1e-9)
        Floor.addObject("FixedProjectiveConstraint", name="bc", fixAll=True)

    # 3 Pipelines, each liver vs floor
    sub_pipelines = []
    for i in range(3):
        root.addObject("BruteForceBroadPhase", name="bfbb"+str(i))
        root.addObject("BVHNarrowPhase", name="bvh"+str(i))
        root.addObject("NewProximityIntersection", name="intersection"+str(i), alarmDistance=5, contactDistance=2)
        root.addObject("CollisionResponse", name="response"+str(i), response="PenalityContactForceField")
        pipeline_name = "pipeline"+str(i)
        root.addObject("SubCollisionPipeline", name=pipeline_name
            , collisionModels=[
                "@/Livers/liver"+str(i)+"/tri", "@/Livers/liver"+str(i)+"/lin", "@/Livers/liver"+str(i)+"/pt",
                "@/Floor/tri", "@/Floor/lin", "@/Floor/pt",
            ]
            , broadPhaseDetection="@bfbb"+str(i)
            , narrowPhaseDetection="@bvh"+str(i)
            , intersectionMethod="@intersection"+str(i)
            , contactManager="@response"+str(i)
        )
        sub_pipelines.append("@"+pipeline_name)

    # Making this parallel causes crashes
    root.addObject("CompositeCollisionPipeline", name="compositePipeline", subCollisionPipelines=sub_pipelines
                   , parallelDetection=False)
