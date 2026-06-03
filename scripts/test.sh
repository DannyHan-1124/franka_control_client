# python -m franka_control_client.policy.pi05_policy_node \
#     --checkpoint_path /home/jjiang/utphd/pretrained_model \
#     --dataset_path /home/jjiang/jing/dataset/lerobot/cylinder_full \
#     --device cuda \
#     --policy_dtype bfloat16 \
#     --obs_topic pi05/observation \
#     --action_topic pi05/action \
#     --fps 20 \
#     --default_task "put red cylinder on yellow cube and put green cylinder on blue cube" \
#     --pyzlc_host 141.3.53.25 \
#     --pyzlc_group_name robot_lab_robotiq_202 \
#     --pyzlc_group_port 7725

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_pGbs16_20k_accumulation4/checkpoints/010000/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_cube_full \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put red cylinder on yellow cube and put green cylinder on blue cube" \
    --direct_zmq_bind tcp://0.0.0.0:40023
