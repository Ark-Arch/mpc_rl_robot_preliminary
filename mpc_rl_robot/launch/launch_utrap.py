import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    world_file = os.path.expanduser('~/mpc_rl_robot_preliminary/mpc_rl_robot/worlds/utrap_world.sdf')

    return LaunchDescription([

        # Launch Gazebo with our custom world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file,
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # Spawn TurtleBot3 at position (0, 0) facing the U-trap
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'burger',
                '-file', os.path.join(
                    get_package_share_directory('turtlebot3_gazebo'),
                    'models', 'turtlebot3_burger', 'model.sdf'
                ),
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.01'
            ],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': open(
                    os.path.expanduser(
                        '/opt/ros/humble/share/turtlebot3_gazebo/urdf/turtlebot3_burger.urdf'
                    )
                ).read()
            }]
        ),
    ])
