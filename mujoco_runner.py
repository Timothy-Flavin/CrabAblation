import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from DQN_Rainbow import RainbowDQN


if __name__ == "__main__":

    def eval(agent: RainbowDQN):
        lenv = gym.make("CartPole-v1")
        obs, info = lenv.reset()
        done = False
        reward = 0.0
        while not done:
            with torch.no_grad():
                logits = agent.online(torch.from_numpy(obs))
                ev = agent.online.expected_value(logits)
                action = torch.argmax(ev, dim=-1).item()
            obs, r, term, trunc, info = lenv.step(action)
            reward += float(r)
            done = term or trunc
        return reward

    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    assert obs_dim is not None

    dqn = RainbowDQN(
        obs_dim,
        2,
        zmin=0,
        zmax=200,
        n_atoms=51,
        soft=False,
        munchausen=False,
        Thompson=False,
        Beta=0.0,
        dueling=True,
    )

    n_steps = 50000
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []
    r_ep = 0.0
    smooth_r = 0.0
    ep = 0
    blen = 10000
    update_every = 4

    # Memory buffer
    buff_actions = torch.zeros((blen,), dtype=torch.long)
    buff_obs = torch.zeros(size=(blen, int(obs_dim)), dtype=torch.float32)
    buff_next_obs = torch.zeros(size=(blen, int(obs_dim)), dtype=torch.float32)
    buff_term = torch.zeros(size=(blen,), dtype=torch.float32)
    buff_r = torch.zeros(size=(blen,), dtype=torch.float32)

    for i in range(n_steps):
        eps_current = 1 - i / n_steps
        action = dqn.sample_action(
            torch.from_numpy(obs), eps=eps_current, step=i, n_steps=n_steps
        )
        next_obs, r, term, trunc, info = env.step(action)
        # Update running stats with the freshly collected transition (single step)
        dqn.update_running_stats(torch.from_numpy(next_obs))

        # Save transition to memory buffer
        buff_actions[i % blen] = action
        buff_obs[i % blen] = torch.from_numpy(obs)
        buff_next_obs[i % blen] = torch.from_numpy(next_obs)
        buff_term[i % blen] = term
        buff_r[i % blen] = float(r)

        if i > 512:
            lhist.append(
                dqn.update(
                    buff_obs,
                    buff_actions,
                    buff_r,
                    buff_next_obs,
                    buff_term,
                    batch_size=64,
                    step=min(i, blen),
                )
            )
            if i % update_every == 0:
                dqn.update_target()

        if i % 1000 == 0:
            evalr = 0.0
            for k in range(5):
                evalr += eval(dqn)
            print(f"eval mean: {evalr/5}")
            eval_hist.append(evalr / 5)

        r_ep += float(r)

        if term or trunc:
            next_obs, info = env.reset()
            rhist.append(r_ep)
            if len(rhist) < 20:
                print(f"reward for episode: {ep}: {r_ep}")
                smooth_r = sum(rhist) / len(rhist)
            else:
                smooth_r = 0.05 * rhist[-1] + 0.95 * smooth_r
                if ep % 10 == 0:
                    print(
                        f"smooth reward for episode: {ep}: {smooth_r} at eps: {eps_current}"
                    )
            smooth_rhist.append(smooth_r)
            r_ep = 0.0
            ep += 1

        obs = next_obs

    plt.plot(rhist)
    plt.plot(smooth_rhist)
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.show()

    plt.plot(eval_hist)
    plt.grid()
    plt.title("eval scores")
    plt.show()
