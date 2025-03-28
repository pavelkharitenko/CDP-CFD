# Configuration

Through configuation files (in YAML format) you describe your robots, environments and controllers in a scene.

## Structure of a config.yml

```yml
---
  world:
    ...
  rigid_bodies:
    ...
  fluid:
    ...
  coupling:
    ...
  virtual_propellers:
    ...
  actuators:
    ...
  sensors:
    ...
```


## Drones


## Environments

## Controllers

## Full example of a config.yml


The following YAML file will load a flat terrain, a drone with four rotors over it.

```yml
---
  world:
    gravity:
      - 0.0
      - 0.0
      - -9.81

  rigid_bodies:
    skeletons:
      - file_path: "p600.urdf"
        joint_velocities:
          p600__rotor_0_joint:
            - 400.0
          p600__rotor_1_joint:
            - 400.0
          p600__rotor_2_joint:
            - -400.0
          p600__rotor_3_joint:
            - -400.0
      - file_path: "floor.urdf"

  fluid:
    blocks:
      max_block_count: 128
    fluid:
      resolution: 0.05
      max_velocity: 40.0
      initial_density: 1.225
      initial_velocity:
        - 0.0
        - 0.0
        - 0.0
      kinematic_shear_viscosity: 1.8e-5

  coupling:
    coupling:
      search_center_target_skeleton_name: "p600"

  virtual_propellers:
    coupling:
      search_center_target_skeleton_name: "p600"
      default_propeller_radius: 0.195
      propeller_models:
        - name: "propeller-cw"
          blades: 2
          direction: "Reversed"
        - name: "propeller-ccw"
          blades: 2
          direction: "Forward"
      configurations:
        - target_skeleton_name: "p600"
          target_joint_name: "p600__rotor_0_joint" # front right, forward
          center_offset: 0.008
          model: "propeller-ccw"
        - target_skeleton_name: "p600"
          target_joint_name: "p600__rotor_1_joint" # back left, forward
          center_offset: 0.008
          model: "propeller-ccw"
        - target_skeleton_name: "p600"
          target_joint_name: "p600__rotor_2_joint" # front left, reversed
          center_offset: 0.008
          model: "propeller-cw"
        - target_skeleton_name: "p600"
          target_joint_name: "p600__rotor_3_joint" # back right, reversed
          center_offset: 0.008
          model: "propeller-cw"

  actuators:
    - type: "DcMotor"
      name: "bl"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_1_joint"
      model:
        stall_torque: 0.4
        zero_load_rpm: 20000.0
      direction: "Forward" # Options: Forward, Reversed
    - type: "DcMotor"
      name: "br"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_3_joint"
      model:
        stall_torque: 0.4
        zero_load_rpm: 20000.0
      direction: "Reversed" # Options: Forward, Reversed
    - type: "DcMotor"
      name: "fl"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_2_joint"
      model:
        stall_torque: 0.4
        zero_load_rpm: 20000.0
      direction: "Reversed" # Options: Forward, Reversed
    - type: "DcMotor"
      name: "fr"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_0_joint"
      model:
        stall_torque: 0.4
        zero_load_rpm: 20000.0
      direction: "Forward" # Options: Forward, Reversed
    - type: "Follow"
      name: "floor_controller"
      follower_skeleton_name: "floor"
      followed_target_skeleton_name: "p600"
      constraint: "Plane"
      plane_origin:
        - 0
        - 0
        - 0
      plane_normal:
        - 0
        - 0
        - 1
    - type: "MulticopterController"
      name: "controller"
      imu_name: "imu"
      fl_motor_name: "fl"
      fr_motor_name: "fr"
      bl_motor_name: "bl"
      br_motor_name: "br"
      mixer:
        thrust_to_pwm_power: 1.0
        yaw_weight: 1.0
      rate_controller:
        control_frequency: 1000.0
        k_proportional_gain:
          roll: 0.3
          pitch: 0.3
          yaw: 0.4
        k_integral_gain:
          roll: 1.0
          pitch: 1.0
          yaw: 1.0
        k_derivative_gain:
          roll: 0.05
          pitch: 0.05
          yaw: 0.0
        integral_limit:
          roll: 0.3
          pitch: 0.3
          yaw: 0.3
      attitude_controller:
        control_frequency: 250.0
        proportional_gain:
          roll: 5.0
          pitch: 5.0
          yaw: 4.0
        max_output_angular_rate:
          roll: 5.0
          pitch: 5.0
          yaw: 4.0
        yaw_weight: 0.4
      position_controller:
        control_frequency: 50.0
        velocity_proportional_gain:
          - 1.8
          - 1.8
          - 4.0
        velocity_integral_gain:
          - 0.4
          - 0.4
          - 1.0
        velocity_derivative_gain:
          - 0.2
          - 0.2
          - 0.4
        max_tilt: 0.4363 # 25 degrees
        hover_thrust: 0.45
        min_thrust: 0.1
        max_thrust: 0.9
        horizontal_thrust_margin: 0.6
        position_proportional_gain:
          - 0.95
          - 0.95
          - 1.0
        max_horizontal_velocity: 12.0
        max_up_velocity: 3.0
        max_down_velocity: 1.5

  sensors:
    - type: "Blocks"
      name: "blocks"
    - type: "Body"
      name: "imu"
      target_skeleton_name: "p600"
      target_body_name: "p600__imu_link"
      offset:
        - 0.0
        - 0.0
        - 0.0
      up_axis: "ZPositive"
      forward_axis: "XPositive"
    - type: "Joint"
      name: "joint_bl"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_1_joint"
      mode: "Velocity" # Options: Position, Velocity, Acceleration
    - type: "Joint"
      name: "joint_br"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_3_joint"
      mode: "Velocity" # Options: Position, Velocity, Acceleration
    - type: "Joint"
      name: "joint_fl"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_2_joint"
      mode: "Velocity" # Options: Position, Velocity, Acceleration
    - type: "Joint"
      name: "joint_fr"
      target_skeleton_name: "p600"
      target_joint_name: "p600__rotor_0_joint"
      mode: "Velocity" # Options: Position, Velocity, Acceleration

```
