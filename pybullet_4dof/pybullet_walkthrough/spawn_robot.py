import pybullet as p
import pybullet_data
import os
import time

def spawn_robot(create_connection=True):
    # Connect to PyBullet only if requested
    physicsClient = None
    if create_connection:
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    # Load ground plane
    planeId = p.loadURDF("plane.urdf")

    # Load our robot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "..", "urdf", "robot.urdf")
    
    # Set initial pose
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Load robot with fixed base
    robot = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0,
        cameraPitch=0,
        cameraTargetPosition=[0, 0, 0]
    )
    
    return robot, physicsClient

def main():
    robot, physics_client = spawn_robot()
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)  # 240 Hz simulation
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main() 