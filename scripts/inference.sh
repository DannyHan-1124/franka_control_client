# kill old connection
# pgrep -af 'ssh.*(-L)'

python examples/pi05_policy_inference_franka.py \
    --task "put red cylinder on yellow cube and put green cylinder on blue cube" \
    --task_after_first_release "put green cylinder on blue cube" \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17725 \
    --max_position_step_m 0.0 \
    --max_rotation_step_rad 0.0 \
    --static_camera static_cam \
    --wrist_camera wrist_cam \
    --rtc_enabled \
    --rtc_execution_horizon 25 \
    --rtc_delay_steps 4
