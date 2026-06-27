# Inference Pipeline Diagrams

These diagrams show the client-server timing patterns used in the inference
pipeline. They are written in Mermaid so they can be pasted directly into
Markdown slides, GitHub, or a Mermaid renderer.

## FASTER Streaming Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Robot<br/>franka_control_client
    participant S as Server / GPU<br/>pi05_policy_node
    participant M as Pi0.5 Model<br/>LeRobot

    rect rgb(235, 245, 255)
        Note over C: Control loop runs at fixed FPS<br/>e.g. 20 Hz
        C->>C: Capture cameras, arm state, gripper state
        C->>C: Build observation + task language
        C->>C: Add prefix metadata if delay > 0
        C->>S: Send observation request<br/>request_id = k
    end

    rect rgb(245, 245, 245)
        S->>S: Preprocess observation
        S->>S: Attach previous raw actions as prefix<br/>if prefix_request_id is provided
        S->>M: Start streaming denoising<br/>HAS schedule + action prefix
    end

    loop Denoising steps
        M->>M: Update noisy action chunk x_t
        M->>M: Mark action indices ready<br/>when scheduled time is low enough
        M-->>S: Yield newly ready indices + action chunk
        S-->>C: Stream action_delta<br/>indices = ready actions
        C->>C: Store actions by chunk index
        C->>C: Execute next available action at control tick
    end

    alt early_stop_actions reached
        M-->>S: Stop denoising early
    else full chunk finished
        M-->>S: Return remaining actions
    end

    S-->>C: Send final=True for request_id = k
    C->>C: Continue executing until execution_horizon
    C->>C: Preserve delay actions as next prefix
    C->>S: Send next observation request<br/>request_id = k + 1
```

## Synchronous Inference

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Robot
    participant S as Server / GPU
    participant M as Pi0.5 Model

    rect rgb(235, 245, 255)
        C->>C: Capture observation
        C->>S: Send observation
    end

    rect rgb(255, 245, 235)
        Note over C: Client waits<br/>robot may hold previous command
        S->>M: Predict full action chunk
        M->>M: Run all denoising steps
        M-->>S: Return complete action chunk
        S-->>C: Send full chunk
    end

    rect rgb(235, 255, 235)
        loop Execute chunk
            C->>C: Apply action a_i at control tick
        end
    end

    C->>S: Send next observation<br/>only after replanning point
```

## Asynchronous Continuous Inference

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Robot
    participant S as Server / GPU
    participant M as Pi0.5 Model

    rect rgb(235, 245, 255)
        loop Every control step
            C->>C: Capture latest observation
            C->>S: Publish observation<br/>with request_start_step
            C->>C: Execute action for current global step<br/>if available
        end
    end

    rect rgb(245, 245, 245)
        loop Server worker
            S->>S: Keep newest pending observation
            S->>S: Discard superseded observations
            S->>M: Predict full action chunk
            M-->>S: Return chunk for request_start_step
            S-->>C: Send chunk + request_start_step
        end
    end

    rect rgb(255, 250, 230)
        C->>C: Map action index i to target step<br/>target_step = request_start_step + i
        C->>C: Drop stale actions<br/>target_step < current_global_step
        C->>C: Newer chunks overwrite older future actions
    end
```

## Short Comparison

| Mode | Observation sending | Server response | Client behavior | Main benefit | Main risk |
| --- | --- | --- | --- | --- | --- |
| Synchronous | One observation per chunk | Full chunk after inference completes | Waits, then executes chunk | Simple and stable | Highest waiting latency |
| FASTER streaming | One observation per execution horizon | Partial action deltas as actions become ready | Starts executing ready actions before full chunk finishes | Lower time to first action | Needs careful delay/prefix tuning |
| Continuous async | Observation every control step | Full chunks from newest observations | Drops stale actions and prioritizes newest plans | More reactive to scene changes | Can hesitate if switching chunks too often |

