from gpu_hideseek_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from gpu_hideseek_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from gpu_hideseek_learn.rnn import LSTM

import math
import torch

def setup_obs(sim, num_worlds):
    seeker_idx = torch.tensor([False, False, True, True] * num_worlds, dtype=torch.bool)
    self_obs_tensor = sim.self_obs_tensor().to_torch()[seeker_idx]
    agent_obs_tensor = sim.agent_obs_tensor().to_torch()[seeker_idx]
    box_obs_tensor = sim.box_obs_tensor().to_torch()[seeker_idx]
    ramp_obs_tensor = sim.ramp_obs_tensor().to_torch()[seeker_idx]
    lidar_tensor = sim.lidar_tensor().to_torch()[seeker_idx]
    # steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    # # Add in an agent ID tensor
    # id_tensor = torch.arange(A).float()
    # if A > 1:
    #     id_tensor = id_tensor / (A - 1)

    # id_tensor = id_tensor.to(device=self_obs_tensor.device)
    # id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    obs_tensors = [
        self_obs_tensor,
        agent_obs_tensor,
        box_obs_tensor,
        ramp_obs_tensor,
        lidar_tensor,
        # steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
        # id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, num_obs_features

# def process_obs(self_obs, partner_obs, room_ent_obs,
#                 door_obs, lidar, steps_remaining, ids):
def process_obs(self_obs, agent_obs, box_obs, ramp_obs, lidar):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(agent_obs).any())
    assert(not torch.isinf(agent_obs).any())

    assert(not torch.isnan(box_obs).any())
    assert(not torch.isinf(box_obs).any())

    assert(not torch.isnan(ramp_obs).any())
    assert(not torch.isinf(ramp_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        agent_obs.view(agent_obs.shape[0], -1),
        box_obs.view(box_obs.shape[0], -1),
        ramp_obs.view(ramp_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
    ], dim=1)

def make_policy(num_obs_features, num_channels, separate_value):
    #encoder = RecurrentBackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 2,
    #    ),
    #    rnn = LSTM(
    #        in_channels = num_channels,
    #        hidden_channels = num_channels,
    #        num_layers = 1,
    #    ),
    #)

    encoder = BackboneEncoder(
        net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 3,
        ),
    )

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = RecurrentBackboneEncoder(
                net = MLP(
                    input_dim = num_obs_features,
                    num_channels = num_channels,
                    num_layers = 2,
                ),
                rnn = LSTM(
                    in_channels = num_channels,
                    hidden_channels = num_channels,
                    num_layers = 1,
                ),
            )
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs,
            encoder = encoder,
        )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [11, 11, 11, 2, 2],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )
