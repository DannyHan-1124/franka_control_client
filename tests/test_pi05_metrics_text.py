from franka_control_client.policy_inference.pi05_policy_inference import Pi05PolicyInference


def test_metrics_text_columns_align_for_all_chunk_kinds(tmp_path):
    inference = object.__new__(Pi05PolicyInference)
    path = tmp_path / "metrics.txt"
    record = {
        "summary": {
            "task": "test",
            "abpolicy_enabled": True,
            "total_time_s": 1.0,
            "inference_calls": 2,
            "completed_chunks": 2,
            "actions_applied": 2,
            "empty_action_steps": 0,
            "avg_request_latency_s": 0.1,
            "avg_first_action_latency_s": 0.1,
            "avg_chunk_execution_duration_s": 0.1,
            "recommended_delay_steps": 2,
        },
        "chunks": [
            {
                "request_id": 1,
                "kind": "abpolicy_initial",
                "action_count": 32,
                "executed_action_count": 1,
                "predicted_delay_steps": None,
                "observed_delay_steps": None,
                "request_latency_s": 0.1,
                "first_action_latency_s": 0.1,
                "execution_duration_s": 0.1,
            },
            {
                "request_id": 2,
                "kind": "abpolicy_async",
                "action_count": 32,
                "executed_action_count": 1,
                "predicted_delay_steps": None,
                "observed_delay_steps": 2,
                "request_latency_s": 0.1,
                "first_action_latency_s": 0.1,
                "execution_duration_s": 0.1,
            },
        ],
    }

    inference._write_metrics_text(path, record)
    lines = path.read_text().splitlines()
    header, initial, asynchronous = lines[6:9]

    assert len(header) == len(initial) == len(asynchronous)
    assert initial.index("0.100") == asynchronous.index("0.100")
    assert header.split() == [
        "request",
        "kind",
        "actions",
        "executed",
        "pred_d",
        "obs_d",
        "request_s",
        "first_s",
        "duration_s",
    ]
