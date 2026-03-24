import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def get_device_name():
    try:
        with open("device_name.txt", "r") as f:
            return f.read().strip()
    except Exception:
        try:
            return os.getlogin()
        except Exception:
            import getpass
            return getpass.getuser()

# Map per-dimension discrete bin indices to continuous actions (-1, 0, 1)
def bins_to_continuous(action_bins):
    return np.array([b - 1.0 for b in action_bins], dtype=np.float32)

class obs_transformer:
    def __init__(self):
        # 7x7 image, 3 channels (one-hot for IDs 1, 2, 8)
        self.image_flat_size = 7 * 7 * 2
        # Direction is one-hot encoded (4 values)
        self.direction_size = 4
        self.frame_size = self.image_flat_size + self.direction_size

        # Last obs includes image and direction
        self.last_obs = np.zeros(self.frame_size)

    def transform(self, obs):
        # Extract object ID channel
        img = obs["image"][:, :, 0]
        direction = obs["direction"]

        # One-hot encode IDs: 1=Empty, 2=Wall, 8=Goal
        one_hot_img = np.zeros((7, 7, 2), dtype=np.float32)
        # one_hot_img[:, :, 0] = img == 1
        one_hot_img[:, :, 0] = img == 2
        one_hot_img[:, :, 1] = img == 8

        # One-hot encode direction
        one_hot_dir = np.zeros(4, dtype=np.float32)
        one_hot_dir[direction] = 1.0

        current_obs = one_hot_img.flatten()

        # Combine current image and direction
        current_full = np.concatenate([current_obs, one_hot_dir])

        # Stack current and last frame
        transformed_obs = np.concatenate([current_full, self.last_obs])

        self.last_obs = current_full
        return transformed_obs

    def reset(self):
        self.last_obs = np.zeros(self.frame_size)
        return np.concatenate([self.last_obs, self.last_obs])


class FastObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transformer = obs_transformer()
        dummy_obs = self.transformer.reset()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(dummy_obs),), dtype=np.float32
        )

    def observation(self, obs):
        return self.transformer.transform(obs)

    def reset(self, **kwargs):
        self.transformer.reset()
        return super().reset(**kwargs)


def make_env_thunk(fully_obs, env_name):
    def thunk():
        if env_name == "minigrid":
            env = gym.make("MiniGrid-FourRooms-v0")
            env = FastObsWrapper(env)
        elif env_name == "cartpole":
            env = gym.make("CartPole-v1")
        elif env_name == "mujoco":
            env = gym.make("HalfCheetah-v5")
        return env

    return thunk


def plot_results(results, args, model_name):
    runner_name = args.env_name
    results_dir = os.path.join("results", model_name, runner_name)
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}.npy"),
        results["rhist"],
    )
    np.save(
        os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}.npy"),
        results["eval_hist"],
    )
    np.save(
        os.path.join(results_dir, f"loss_hist_{args.run}_{args.ablation}.npy"),
        results["lhist"],
    )
    np.save(
        os.path.join(
            results_dir, f"smooth_train_scores_{args.run}_{args.ablation}.npy"
        ),
        results["smooth_rhist"],
    )

    plt.plot(results["rhist"])
    plt.plot(results["smooth_rhist"])
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.title(f"Training rewards, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}"))
    plt.close()
    
    plt.plot(results["eval_hist"])
    plt.grid()
    plt.title(f"eval scores, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}"))
    plt.close()

    # Save total wall clock training time
    train_time_seconds = results["train_time"]
    np.save(
        os.path.join(results_dir, f"train_time_{args.run}_{args.ablation}.npy"),
        train_time_seconds,
    )
    print(f"Training wall clock time: {train_time_seconds:.2f} seconds")