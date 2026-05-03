from __future__ import annotations

import argparse
from collections import deque
from typing import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eaf_twin.config.defaults import scenario_configs
from eaf_twin.config.loader import load_config

from agents.controller import EAFController
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.heuristic import HeuristicParams
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.q_learning_policy import QLearningPolicy
from agents.policies.rl_common import ACTION_NAMES, Discretizer, normalized_obs_vec, safe_discrete_action
from agents.policies.safe_ppo_agentic_bc import SafePPOAgenticBCPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.episode_runner import run_episode


def _controller(base_cfg, scenario: str, seed: int) -> EAFController:
    sc = scenario_configs(base_cfg)
    return EAFController(replace(sc[scenario], random_seed=seed), enhanced_model=True)




def _episode_success(obs: dict, cfg) -> bool:
    return float(obs.get("bath_temp_c", 0.0)) >= float(cfg.steel_melt_temp_c)


def _should_early_stop(successes: deque, min_episodes: int = 80) -> bool:
    if len(successes) < min_episodes:
        return False
    return sum(successes) / len(successes) >= 0.92


def _action_index_from_action(obs: dict, action: dict) -> int:
    best_name = min(
        ACTION_NAMES,
        key=lambda k: abs(safe_discrete_action(k, obs)["power_mw"] - float(action.get("power_mw", 0.0))),
    )
    return ACTION_NAMES.index(best_name)
def _failure_reason(obs: dict) -> str:
    if float(obs.get("cum_tapped_kg", 0.0)) <= 1e-6:
        return "no_tap"
    if float(obs.get("bath_temp_c", 0.0)) < float(obs.get("_config_obj").steel_melt_temp_c if obs.get("_config_obj") else 0.0):
        return "cold_bath"
    if bool(obs.get("bath_temp_c", 0.0) > 1750.0):
        return "overheat"
    if bool(obs.get("can_tap", False)) and not bool(obs.get("tapping_started", False)):
        return "invalid_tap"
    return "max_steps_reached"


def train_q_learning(base_cfg, episodes: int, seed: int, output_dir: Path, max_steps: int) -> None:
    rng = np.random.default_rng(seed)
    policy = QLearningPolicy(epsilon=1.0, seed=seed)
    alpha, gamma = 0.15, 0.99
    logs = []
    recent_success: deque[int] = deque(maxlen=40)
    best_reward = float("-inf")
    for ep in range(episodes):
        controller = _controller(base_cfg, "base_case", seed + ep)
        obs = controller.reset(); total = 0.0
        eps = max(0.05, 1.0 - ep / max(episodes, 1))
        policy.epsilon = eps
        for _ in range(max_steps):
            if rng.random() < eps:
                a_name = ACTION_NAMES[int(rng.integers(0, len(ACTION_NAMES)))]
            else:
                a_name = policy.greedy_action_name(obs)
            action = safe_discrete_action(a_name, obs)
            res = controller.step(action)
            executed_name = min(ACTION_NAMES, key=lambda k: abs(safe_discrete_action(k, obs)["power_mw"] - float(res.info["safe_action"]["power_mw"])))
            policy.update(obs, executed_name, res.reward, res.observation, res.done, alpha=alpha, gamma=gamma)
            obs = res.observation; total += res.reward
            if res.done:
                break
        success = int(_episode_success(obs, controller.config))
        recent_success.append(success)
        if total > best_reward:
            best_reward = total
            output_dir.mkdir(parents=True, exist_ok=True)
            policy.save(output_dir / "q_table.json")
        logs.append({"episode": ep, "reward": total, "epsilon": eps, "success": success, "tap_success_rate": success, "final_temperature_c": float(obs.get("bath_temp_c", 0.0)), "cum_tapped_kg": float(obs.get("cum_tapped_kg", 0.0)), "energy_per_ton": (float(obs.get("cum_electric_mwh", 0.0)) * 1000.0 / max(1e-9, float(obs.get("cum_tapped_kg", 0.0)) / 1000.0)) if float(obs.get("cum_tapped_kg", 0.0)) > 0 else np.nan, "failure_reason": _failure_reason(obs)})
        if _should_early_stop(recent_success):
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save(output_dir / "final_policy.json")
    pd.DataFrame(logs).to_csv(output_dir / "training_curve.csv", index=False)


def train_dqn(base_cfg, episodes: int, seed: int, output_dir: Path, max_steps: int) -> None:
    rng = np.random.default_rng(seed)
    policy = DQNPolicy()
    target = DQNPolicy()
    target.q_network.load_state_dict(policy.q_network.state_dict())
    replay: deque = deque(maxlen=50000)
    gamma, lr = 0.99, 0.0005
    optimizer = torch.optim.Adam(policy.q_network.parameters(), lr=lr)
    logs = []
    recent_success: deque[int] = deque(maxlen=40)
    best_reward = float("-inf")
    for ep in range(episodes):
        ctrl = _controller(base_cfg, "base_case", seed + ep)
        obs = ctrl.reset()
        total = 0.0
        global_step = ep * max_steps
        for step in range(max_steps):
            eps = max(0.05, 1.0 - (global_step + step) / 10000.0 * (1.0 - 0.05))
            if rng.random() < eps:
                a_idx = int(rng.integers(0, len(ACTION_NAMES)))
            else:
                a_idx = int(np.argmax(policy.q_values(obs)))
            action = safe_discrete_action(ACTION_NAMES[a_idx], obs)
            res = ctrl.step(action)
            executed_idx = _action_index_from_action(obs, res.info["safe_action"])
            replay.append((np.asarray(normalized_obs_vec(obs)), executed_idx, res.reward, np.asarray(normalized_obs_vec(res.observation)), float(res.done), executed_idx))
            obs = res.observation
            total += res.reward
            if len(replay) >= 64:
                idx = rng.choice(len(replay), size=64, replace=False)
                batch = [replay[i] for i in idx]
                x = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
                a = torch.tensor([b[1] for b in batch], dtype=torch.int64)
                r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                nx = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
                done = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                q = policy.q_network(x).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a = policy.q_network(nx).argmax(dim=1)
                    next_q = target.q_network(nx).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = r + (1.0 - done) * gamma * next_q
                loss = F.smooth_l1_loss(q, y)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.q_network.parameters(), 1.0)
                optimizer.step()
            if (global_step + step) % 500 == 0:
                target.q_network.load_state_dict(policy.q_network.state_dict())
            if res.done:
                break
        success = int(_episode_success(obs, ctrl.config))
        recent_success.append(success)
        if total > best_reward:
            best_reward = total
            output_dir.mkdir(parents=True, exist_ok=True)
            policy.save(output_dir / "best_policy.pt")
        logs.append({"episode": ep, "reward": total, "epsilon": eps, "success": success, "tap_success_rate": success, "final_temperature_c": float(obs.get("bath_temp_c", 0.0)), "cum_tapped_kg": float(obs.get("cum_tapped_kg", 0.0)), "energy_per_ton": (float(obs.get("cum_electric_mwh", 0.0)) * 1000.0 / max(1e-9, float(obs.get("cum_tapped_kg", 0.0)) / 1000.0)) if float(obs.get("cum_tapped_kg", 0.0)) > 0 else np.nan, "failure_reason": _failure_reason(obs), "executed_safe_action_idx": executed_idx})
        if _should_early_stop(recent_success):
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save(output_dir / "final_policy.pt")
    pd.DataFrame(logs).to_csv(output_dir / "training_curve.csv", index=False)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.maximum(np.sum(e), 1e-12)


def train_ppo(base_cfg, episodes: int, seed: int, output_dir: Path, max_steps: int, safe_hybrid: bool = False, hybrid_name: str = "safe_ppo_agentic_mpc", learning_rate: float = 0.0001, gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.15, entropy_coef: float = 0.02, value_coef: float = 0.5, rollout_steps: int = 128, epochs: int = 4, batch_size: int = 32, init_actor_w: np.ndarray | None = None, init_value_w: np.ndarray | None = None) -> None:
    rng = np.random.default_rng(seed)
    policy = PPOPolicy(actor_w=init_actor_w, value_w=init_value_w)
    best = -1e18
    reward_ma: deque[float] = deque(maxlen=20)
    degrade_patience = 12
    degrade_count = 0
    logs = []
    recent_success: deque[int] = deque(maxlen=40)
    for ep in range(episodes):
        ctrl = _controller(base_cfg, "base_case", seed + ep)
        obs = ctrl.reset()
        traj = []
        total = 0.0
        entropy_coef_ep = max(0.001, entropy_coef * (0.995 ** ep))
        for _ in range(max_steps):
            x = np.asarray(normalized_obs_vec(obs))
            p = _softmax(policy.actor_w @ x)
            a = int(rng.choice(len(ACTION_NAMES), p=p))
            old_logp = float(np.log(max(p[a], 1e-12)))
            action = safe_discrete_action(ACTION_NAMES[a], obs)
            executed_idx = a
            res = ctrl.step(action)
            v = float(policy.value_w @ x)
            traj.append((x, executed_idx, old_logp, float(np.clip(res.reward, -5.0, 5.0)), float(res.done), v))
            obs = res.observation
            total += res.reward
            if res.done or len(traj) >= rollout_steps:
                break
        if total > best:
            best = total
            ckpt_name = f"best_{hybrid_name}_policy.pt" if safe_hybrid else "best_policy.pt"
            policy.save(output_dir / ckpt_name)
        # GAE
        adv, ret = [], []
        gae = 0.0
        next_v = 0.0
        for x, a, old_logp, r, done, v in reversed(traj):
            delta = r + gamma * next_v * (1.0 - done) - v
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            adv.insert(0, gae)
            ret.insert(0, gae + v)
            next_v = v
        adv = np.asarray(adv, dtype=float)
        ret = np.asarray(ret, dtype=float)
        if len(ret) > 1:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stop_for_kl = False
        for _ in range(epochs):
            if not traj:
                break
            idx = rng.permutation(len(traj))
            kl_values = []
            for start in range(0, len(idx), batch_size):
                bi = idx[start : start + batch_size]
                for j in bi:
                    x, a, old_logp, _, _, _ = traj[j]
                    probs = _softmax(policy.actor_w @ x)
                    logp = float(np.log(max(probs[a], 1e-12)))
                    ratio = np.exp(np.clip(logp - old_logp, -5.0, 5.0))
                    approx_kl = old_logp - logp
                    kl_values.append(float(max(0.0, approx_kl)))
                    s1 = ratio * adv[j]
                    s2 = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv[j]
                    pg_scale = -min(s1, s2)
                    onehot = np.zeros(len(ACTION_NAMES)); onehot[a] = 1.0
                    grad_logits = (probs - onehot) * pg_scale - entropy_coef_ep * (-np.log(np.maximum(probs, 1e-12)) - 1.0)
                    policy.actor_w -= learning_rate * np.outer(grad_logits, x)
                    vpred = float(policy.value_w @ x)
                    policy.value_w -= learning_rate * value_coef * 2.0 * (vpred - ret[j]) * x
                    policy.actor_w = np.clip(policy.actor_w, -10.0, 10.0)
                    policy.value_w = np.clip(policy.value_w, -10.0, 10.0)
            if kl_values and float(np.mean(kl_values)) > 0.02:
                stop_for_kl = True
                break
        if stop_for_kl:
            pass
        success = int(_episode_success(obs, ctrl.config))
        recent_success.append(success)
        reward_ma.append(total)
        moving_avg = float(np.mean(reward_ma))
        if moving_avg < best - 15.0:
            degrade_count += 1
        else:
            degrade_count = 0
        logs.append({"episode": ep, "reward": total, "success": success, "tap_success_rate": success, "final_temperature_c": float(obs.get("bath_temp_c", 0.0)), "cum_tapped_kg": float(obs.get("cum_tapped_kg", 0.0)), "energy_per_ton": (float(obs.get("cum_electric_mwh", 0.0)) * 1000.0 / max(1e-9, float(obs.get("cum_tapped_kg", 0.0)) / 1000.0)) if float(obs.get("cum_tapped_kg", 0.0)) > 0 else np.nan, "failure_reason": _failure_reason(obs), "moving_avg_reward": moving_avg})
        if ep > 30 and degrade_count >= degrade_patience:
            break
        if _should_early_stop(recent_success):
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_{hybrid_name}_policy.pt" if safe_hybrid else "best_policy.pt"
    if not (output_dir / ckpt_name).exists():
        policy.save(output_dir / ckpt_name)
    # Save only the best checkpoint to avoid selecting a degraded final model.
    pd.DataFrame(logs).to_csv(output_dir / "training_curve.csv", index=False)


def train_behavior_cloning(base_cfg, episodes: int, seed: int, output_dir: Path, max_steps: int) -> None:
    from agents.policies.mpc_policy import MPCPolicy

    expert = MPCPolicy(horizon=8)
    disc = Discretizer()
    counts: dict[str, dict[str, int]] = {}
    logs = []
    for ep in range(episodes):
        ctrl = _controller(base_cfg, "base_case", seed + ep)
        obs = ctrl.reset()
        total = 0.0
        for _ in range(max_steps):
            action = expert.act(obs)
            best = min(ACTION_NAMES, key=lambda k: abs(action["power_mw"] - safe_discrete_action(k, obs)["power_mw"]))
            key = disc.encode(obs)
            counts.setdefault(key, {})[best] = counts.setdefault(key, {}).get(best, 0) + 1
            res = ctrl.step(action)
            obs = res.observation
            total += res.reward
            if res.done:
                break
        logs.append({"episode": ep, "reward": total})
    mapping = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in counts.items()}
    bc = BehaviorCloningPolicy(mapping)
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.save(output_dir / "policy.json")
    pd.DataFrame(logs).to_csv(output_dir / "training_curve.csv", index=False)



def train_safe_ppo_agentic_bc(base_cfg, episodes: int, seed: int, output_dir: Path, max_steps: int, learning_rate: float = 0.0001, gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.15, entropy_coef: float = 0.02, value_coef: float = 0.5, rollout_steps: int = 128, epochs: int = 4, batch_size: int = 32) -> None:
    bc_dir = output_dir.parent / "behavior_cloning"
    if not (bc_dir / "policy.json").exists():
        train_behavior_cloning(base_cfg, episodes, seed, bc_dir, max_steps)
    bc_policy = BehaviorCloningPolicy.load(bc_dir / "policy.json")
    policy = PPOPolicy()
    # BC initialization/regularization: bias PPO logits toward BC-selected actions over sampled states.
    for ep in range(min(episodes, 50)):
        ctrl = _controller(base_cfg, "base_case", seed + 10000 + ep)
        obs = ctrl.reset()
        for _ in range(max_steps):
            x = np.asarray(normalized_obs_vec(obs), dtype=float)
            a = bc_policy.act(obs)
            idx = _action_index_from_action(obs, a)
            policy.actor_w[idx] += 0.01 * x
            policy.actor_w = np.clip(policy.actor_w, -10.0, 10.0)
            res = ctrl.step(a)
            obs = res.observation
            if res.done:
                break
    train_ppo(base_cfg, episodes, seed, output_dir, max_steps, safe_hybrid=True, hybrid_name="safe_ppo_agentic_bc", learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda, clip_epsilon=clip_epsilon, entropy_coef=entropy_coef, value_coef=value_coef, rollout_steps=rollout_steps, epochs=epochs, batch_size=batch_size, init_actor_w=policy.actor_w, init_value_w=policy.value_w)


def train_trainable_adaptive_controller(base_cfg, iterations: int, seed: int, output_dir: Path, max_steps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_params = HeuristicParams()
    logs = []
    for i in range(1, iterations + 1):
        candidate = HeuristicParams(float(rng.uniform(80, 120)), float(rng.uniform(55, 95)), float(rng.uniform(15, 45)), float(rng.uniform(0.8, 1.25)), float(rng.uniform(0.75, 1.3)))
        ctrl = _controller(base_cfg, "base_case", seed + i)
        out_ep = run_episode(ctrl, TrainablePolicy(candidate), policy_name="train_eval", max_steps=max_steps)
        score = out_ep.total_reward
        logs.append({"episode": i, "reward": score})
        if score > best_score:
            best_score = score
            best_params = candidate
            TrainablePolicy(params=best_params).save(output_dir / "best_policy.json")
    pd.DataFrame(logs).to_csv(output_dir / "training_curve.csv", index=False)


def _collect_training_report(base_dir: Path, trained: list[str]) -> None:
    def _svg_line_plot(df: pd.DataFrame, y_col: str, title: str, color: str) -> str:
        if y_col not in df.columns or df.empty:
            return f"<p>{title}: n/a</p>"
        x = pd.to_numeric(df["episode"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(x) < 2:
            return f"<p>{title}: insufficient data</p>"
        w, h, pad = 560.0, 220.0, 24.0
        x_span = max(1e-9, float(x.max() - x.min()))
        y_span = max(1e-9, float(y.max() - y.min()))
        pts = []
        for xi, yi in zip(x, y):
            px = pad + (xi - x.min()) / x_span * (w - 2 * pad)
            py = h - pad - (yi - y.min()) / y_span * (h - 2 * pad)
            pts.append(f"{px:.2f},{py:.2f}")
        return (
            f"<div><h4>{title}</h4>"
            f"<svg width='{int(w)}' height='{int(h)}' style='border:1px solid #ddd;background:#fff'>"
            f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"</svg></div>"
        )

    rows = []
    expected = {"trainable_adaptive_controller": "best_policy.json", "behavior_cloning": "policy.json", "q_learning": "q_table.json", "dqn": "best_policy.pt", "ppo": "best_policy.pt", "safe_ppo_agentic_mpc": "best_safe_ppo_agentic_mpc_policy.pt", "safe_ppo_agentic_sac": "best_safe_ppo_agentic_sac_policy.pt", "safe_ppo_agentic_td3": "best_safe_ppo_agentic_td3_policy.pt", "safe_ppo_agentic_bc": "best_safe_ppo_agentic_bc_policy.pt"}
    chart_blocks: list[str] = []
    for m, f in expected.items():
        p = base_dir / m / f
        curve = base_dir / m / "training_curve.csv"
        reward_final = reward_best = "n/a"
        if curve.exists():
            df = pd.read_csv(curve)
            if "reward" in df.columns and not df.empty:
                reward_final = round(float(df["reward"].iloc[-1]), 6)
                reward_best = round(float(df["reward"].max()), 6)
                df["reward_ma20"] = df["reward"].rolling(20, min_periods=1).mean()
                reward_chart = _svg_line_plot(df, "reward", f"{m}: reward vs episode", "#1f77b4")
                reward_ma_chart = _svg_line_plot(df, "reward_ma20", f"{m}: moving avg reward", "#ff7f0e")
                success_chart = _svg_line_plot(df, "success", f"{m}: success vs episode", "#2ca02c")
                chart_blocks.append(f"<section><h3>{m}</h3>{reward_chart}{reward_ma_chart}{success_chart}</section>")
        reward_drop = "n/a" if reward_final == "n/a" else round(float(reward_best) - float(reward_final), 6)
        warn = False if reward_final == "n/a" else (float(reward_final) < 0.95 * float(reward_best))
        rows.append({"model": m, "trained": m in trained, "checkpoint": str(p), "checkpoint_exists": p.exists(), "best_episode": int(df['episode'][df['reward'].idxmax()]) if curve.exists() and not df.empty and 'reward' in df.columns else "n/a", "final_episode": int(df['episode'].iloc[-1]) if curve.exists() and not df.empty else "n/a", "best_reward": reward_best, "final_reward": reward_final, "reward_drop": reward_drop, "warning_final_below_5pct": warn})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(base_dir / "training_summary.csv", index=False)
    html = (
        "<html><body><h1>Training Report</h1>"
        "<p>n/a indicates reward metrics unavailable for that training run.</p>"
        + sdf.to_html(index=False)
        + "<h2>Training Progress Charts</h2>"
        + ("".join(chart_blocks) if chart_blocks else "<p>No training curves found.</p>")
        + "</body></html>"
    )
    (base_dir / "training_report.html").write_text(html)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train/tune EAF control policies")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_training"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--algorithm", choices=["heuristic", "q_learning", "dqn", "ppo", "safe_ppo_agentic_mpc", "safe_ppo_agentic_sac", "safe_ppo_agentic_td3", "safe_ppo_agentic_bc", "behavior_cloning", "all"], default="heuristic")
    parser.add_argument("--max-steps", type=int, default=610)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.15)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=610)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    episodes = min(args.episodes, 5) if args.fast_dev_run else args.episodes
    train_map = {
        "trainable_adaptive_controller": lambda: train_trainable_adaptive_controller(base_cfg, args.iterations, args.seed, args.output_dir / "trainable_adaptive_controller", args.max_steps),
        "behavior_cloning": lambda: train_behavior_cloning(base_cfg, episodes, args.seed, args.output_dir / "behavior_cloning", args.max_steps),
        "q_learning": lambda: train_q_learning(base_cfg, episodes, args.seed, args.output_dir / "q_learning", args.max_steps),
        "dqn": lambda: train_dqn(base_cfg, episodes, args.seed, args.output_dir / "dqn", args.max_steps),
        "ppo": lambda: train_ppo(base_cfg, episodes, args.seed, args.output_dir / "ppo", args.max_steps, safe_hybrid=False, learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, rollout_steps=args.rollout_steps, epochs=args.epochs, batch_size=args.batch_size),
        "safe_ppo_agentic_mpc": lambda: train_ppo(base_cfg, episodes, args.seed, args.output_dir / "safe_ppo_agentic_mpc", args.max_steps, safe_hybrid=True, hybrid_name="safe_ppo_agentic_mpc", learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, rollout_steps=args.rollout_steps, epochs=args.epochs, batch_size=args.batch_size),
        "safe_ppo_agentic_sac": lambda: train_ppo(base_cfg, episodes, args.seed, args.output_dir / "safe_ppo_agentic_sac", args.max_steps, safe_hybrid=True, hybrid_name="safe_ppo_agentic_sac", learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, rollout_steps=args.rollout_steps, epochs=args.epochs, batch_size=args.batch_size),
        "safe_ppo_agentic_td3": lambda: train_ppo(base_cfg, episodes, args.seed, args.output_dir / "safe_ppo_agentic_td3", args.max_steps, safe_hybrid=True, hybrid_name="safe_ppo_agentic_td3", learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, rollout_steps=args.rollout_steps, epochs=args.epochs, batch_size=args.batch_size),
        "safe_ppo_agentic_bc": lambda: train_safe_ppo_agentic_bc(base_cfg, episodes, args.seed, args.output_dir / "safe_ppo_agentic_bc", args.max_steps, learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, rollout_steps=args.rollout_steps, epochs=args.epochs, batch_size=args.batch_size),
    }
    order=["trainable_adaptive_controller","behavior_cloning","q_learning","dqn","ppo","safe_ppo_agentic_mpc","safe_ppo_agentic_sac","safe_ppo_agentic_td3","safe_ppo_agentic_bc"]
    if args.algorithm == "heuristic":
        train_map["trainable_adaptive_controller"](); trained=["trainable_adaptive_controller"]
    elif args.algorithm == "all":
        trained=[]
        for name in order:
            train_map[name](); trained.append(name)
    else:
        train_map[args.algorithm](); trained=[args.algorithm]
    if (args.output_dir / "trainable_adaptive_controller" / "best_policy.json").exists():
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoints" / "best_policy.json").write_text((args.output_dir / "trainable_adaptive_controller" / "best_policy.json").read_text())
    _collect_training_report(args.output_dir, trained)


if __name__ == "__main__":
    main()
