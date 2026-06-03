# python examples/pi05_policy_inference_franka.py \
#     --task "put red cylinder on yellow cube and put green cylinder on blue cube" \
#     --pyzlc_name policy_inference \
#     --pyzlc_host 141.3.53.25 \
#     --pyzlc_group_name robot_lab_robotiq_202 \
#     --pyzlc_group_port 7725 \
#     --static_camera static_cam \
#     --wrist_camera wrist_cam \
#     --fps 5 \
#     --control_hz 50

python examples/pi05_policy_inference_franka.py \
    --task "put red cylinder on yellow cube and put green cylinder on blue cube" \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17725 \
    --static_camera static_cam \
    --wrist_camera wrist_cam \
    --fps 20 \
    --control_hz 100
