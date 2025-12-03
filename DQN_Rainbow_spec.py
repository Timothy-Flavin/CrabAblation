import itertools
import torch
from DQN_Rainbow import RainbowDQN, EVRainbowDQN

if __name__ == "__main__":
    print("Starting Integration Tests for RainbowDQN and EVRainbowDQN...")

    # Hyperparameters
    OBS_DIM = 16
    ACTION_BINS = 4
    HIDDEN_LAYER_SIZES = [128, 128]

    # Grid Search Parameters
    SOFT_AND_MUNCHAUSEN = [[True, True], [False, False]]
    DUELING = [True, False]
    POPART = [True, False]
    BETA = [0.0, 0.2]
    D_ACTION_DIMS = [1, 3]

    # Data Generation
    buffer_size = 100
    torch.manual_seed(42)
    # torch.autograd.set_detect_anomaly(True) # Optional for debugging

    single_obs = torch.rand(size=[OBS_DIM])

    # Create "Replay Buffer"
    buffer_obs = torch.rand(size=[buffer_size, OBS_DIM])
    buffer_next_obs = torch.rand(size=[buffer_size, OBS_DIM])
    buffer_rewards = torch.randn(size=[buffer_size]) * 5.0 + 2.0
    buffer_terminated = torch.randint(0, 2, [buffer_size])

    # Actions for D=1 and D=3
    buffer_actions_1 = torch.randint(0, ACTION_BINS, [buffer_size])
    buffer_actions_3 = torch.randint(0, ACTION_BINS, [buffer_size, 3])

    # Iterate over models
    models_to_test = [RainbowDQN, EVRainbowDQN]

    total_tests = (
        len(models_to_test)
        * len(SOFT_AND_MUNCHAUSEN)
        * len(DUELING)
        * len(POPART)
        * len(BETA)
        * len(D_ACTION_DIMS)
    )
    current_test = 0
    n_extrinsic_tests = 0
    n_extrinsic_run = 0
    n_extrinsic_passed = 0

    n_rnd_tests = 0
    n_rnd_run = 0
    n_rnd_pass = 0
    for ModelClass in models_to_test:
        print(f"\n{'='*20} Testing {ModelClass.__name__} {'='*20}")

        # Grid Search
        combinations = itertools.product(
            SOFT_AND_MUNCHAUSEN, DUELING, POPART, BETA, D_ACTION_DIMS
        )

        for sm, dueling, popart, beta, d_dim in combinations:
            current_test += 1
            soft, munchausen = sm

            print(
                f"\nTest {current_test}/{total_tests}: Soft={soft}, Munch={munchausen}, Dueling={dueling}, PopArt={popart}, Beta={beta}, D_Dim={d_dim}"
            )

            # Initialize Model
            try:
                agent = ModelClass(
                    input_dim=OBS_DIM,
                    n_action_dims=d_dim,
                    n_action_bins=ACTION_BINS,
                    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                    soft=soft,
                    munchausen=munchausen,
                    dueling=dueling,
                    popart=popart,
                    Beta=beta,
                    rnd_lr=1e-2,  # Higher LR to see RND convergence faster
                    intrinsic_lr=1e-3,
                )
                # Pre-warm running stats to avoid non-stationary input distribution for RND
                # This prevents the RND loss from spiking due to shifting normalization statistics
                agent.update_running_stats(buffer_obs)
            except Exception as e:
                print(f"FAILED to initialize model: {e}")
                raise e

            # 1. Test sample_action
            try:
                action = agent.sample_action(single_obs, eps=0.1, step=0)
                assert (
                    len(action) == d_dim
                ), f"Sampled action dim mismatch. Expected {d_dim}, got {len(action)}"
            except Exception as e:
                print(f"FAILED in sample_action: {e}")
                raise e

            # 2. Test update loop (10 steps)
            actions = buffer_actions_1 if d_dim == 1 else buffer_actions_3

            initial_rnd_loss = None
            initial_sigma = None

            # Get initial sigma if popart
            if popart:
                if hasattr(agent.online, "output_layer"):
                    layer = agent.online.output_layer
                    initial_sigma = layer.sigma.mean().item()
            n_extrinsic_run += 1
            if beta > 0:
                n_rnd_run += 1
            try:
                initial_ext_loss = 0
                loss = torch.zeros(1)
                for step in range(50):
                    # Pass step=buffer_size to allow sampling from the full buffer
                    loss = agent.update(
                        buffer_obs,
                        actions,
                        buffer_rewards,
                        buffer_next_obs,
                        buffer_terminated,
                        batch_size=None,
                        step=buffer_size,
                    )
                    if step == 0:
                        print(f"Initial Extrinsic loss: {loss:.4f}")
                    # Update running stats (usually done in runner, but needed for RND/PopArt to see shifts)
                    # We use a random batch from the buffer for this simulation
                    agent.update_running_stats(buffer_next_obs)

                    if step == 0:
                        initial_rnd_loss = agent.last_losses.get("rnd", 0.0)
                        initial_ext_loss = loss
                # Post-loop checks
                print(f"  Final Extrinsic Loss: {loss:.4f}")
                if loss < initial_ext_loss:
                    n_extrinsic_passed += 1
                # Check RND
                if beta > 0:
                    final_rnd = agent.last_losses.get("rnd", 0.0)
                    print(f"  RND Loss: {initial_rnd_loss:.4f} -> {final_rnd:.4f}")
                    if initial_rnd_loss > final_rnd:
                        n_rnd_pass += 1
                # Check PopArt
                if popart:
                    layer = agent.online.output_layer
                    final_sigma = layer.sigma.mean().item()
                    print(f"  PopArt Sigma: {initial_sigma:.4f} -> {final_sigma:.4f}")

                    # Check if sigma updated (it starts at 1.0)
                    if abs(final_sigma - 1.0) < 1e-6:
                        print("  WARNING: PopArt sigma did not change from 1.0!")
                    else:
                        print("  PopArt sigma updated successfully.")

            except Exception as e:
                print(f"FAILED in update loop: {e}")
                raise e
    print(
        f"Extrinsic experiments tried: {n_extrinsic_run} run: {n_extrinsic_run} passed: {n_extrinsic_passed}"
    )
    print(f"RND experiments tried: {n_rnd_tests} run: {n_rnd_run} passed: {n_rnd_pass}")
