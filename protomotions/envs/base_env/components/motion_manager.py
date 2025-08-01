import torch

from protomotions.envs.base_env.components.base_component import BaseComponent
from pathlib import Path

class MotionManager(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.motion_ids = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        self.motion_times = torch.zeros(self.env.num_envs, device=self.env.device)

        # Sampling vectors
        self.init_start_probs = (
            torch.ones(self.env.num_envs, dtype=torch.float, device=self.env.device)
            * self.config.motion_sampling.init_start_prob
        )

        # build deck
        full_deck = list(range(self.env.motion_lib.num_motions()))
        fail_file = Path(config.root_dir) / env.config.experiment_name / f"failed_motions_0.txt"
        print(f"Loading failed motions from {fail_file}")
        if fail_file.exists():
            with open(fail_file, "r") as f:
                failed = [int(line.strip()) for line in f if line.strip()]
        else:
            failed = []
        self._deck_success = [m for m in full_deck if m not in failed]
        self._deck_failure = failed.copy()
        self._deck = full_deck
        print(self._deck_success, self._deck_failure, self._deck)

        self._deck_ptr = 0
        print(f"Deck sizes — all: {len(self._deck)}, success: {len(self._deck_success)}, failure: {len(self._deck_failure)}")

        self.motion_weights = self.env.motion_lib.state.motion_weights.clone().to(device=self.env.device)

    def reset_envs(self, env_ids):
        self.sample_motions(env_ids)

    def get_respawn_info(self, env_ids):
        return self.motion_ids[env_ids], self.motion_times[env_ids]

    def sample_motions(self, env_ids, new_motion_ids=None):
        """
        Reset the motion and scene for a set of environments.
        This method handles the process of resetting the motion and scene for a specified set of environments.
        It ensures that the reset process is correctly handled based on the current configuration.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional): New motion IDs for the reset environments.
        Returns:
            Tuple[Tensor, Tensor]: New motion IDs and times for the reset environments.
        """
        if self.config.fixed_motion_per_env:
            # We typically use this for recording.
            motion_index_offset = self.config.motion_index_offset
            if motion_index_offset is None:
                motion_index_offset = 0
            new_motion_ids = torch.fmod(
                env_ids + motion_index_offset,
                self.env.motion_lib.num_motions(),
            )
            new_times = torch.zeros_like(
                self.env.motion_lib.state.motion_lengths[new_motion_ids]
            )
        elif getattr(self.config.motion_sampling, "use_deck_sampling", False):
            mode = getattr(self.config.motion_sampling, "deck_sampling_mode", "all")
            assert mode in ["all", "success", "failure"], f"Invalid deck sampling mode: {mode}"
            print("Using deck sampling for motion sampling.", mode)

            if mode == "success":
                deck = self._deck_success
            elif mode == "failure":
                deck = self._deck_failure
            else:
                deck = self._deck

            picks = []
            for _ in env_ids:
                if self._deck_ptr >= len(deck):
                    self._deck_ptr = 0
                picks.append(deck[self._deck_ptr])
                self._deck_ptr += 1

            new_motion_ids = torch.tensor(picks, dtype=torch.long,
                                          device=self.env.device)
            new_times = self.env.motion_lib.sample_time(
                new_motion_ids, truncate_time=self.env.dt
            )

            # apply your init‐start‐prob logic if desired
            if self.config.motion_sampling.init_start_prob > 0:
                init_start = torch.bernoulli(
                    self.init_start_probs[: len(env_ids)])
                new_times = torch.where(
                    init_start == 1,
                    torch.zeros_like(new_times),
                    new_times,
                )
            print(f"Deck sampling: {new_motion_ids} with times {new_times}")

        else:
            if new_motion_ids is None:
                new_motion_ids = torch.multinomial(self.motion_weights, num_samples=len(env_ids), replacement=True)
            if self.config.fixed_motion_id is not None:
                new_motion_ids = (
                    torch.zeros_like(new_motion_ids) + self.config.fixed_motion_id
                )
            new_times = self.env.motion_lib.sample_time(
                new_motion_ids, truncate_time=self.env.dt
            )

            if self.config.motion_sampling.init_start_prob > 0:
                init_start = torch.bernoulli(self.init_start_probs[: len(env_ids)])
                new_times = torch.where(
                    init_start == 1,
                    torch.zeros_like(new_times),
                    new_times,
                )

        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times

    def update_sampling_weights(self, weights):
        self.motion_weights[:] = weights

    def get_state_dict(self):
        state_dict = {
            "motion_weights": self.motion_weights.cpu().clone(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if "motion_weights" in state_dict:
            self.motion_weights[:] = state_dict["motion_weights"].to(self.motion_weights.device)
