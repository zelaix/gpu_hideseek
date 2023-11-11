#include <iostream>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float deltaT = 1.f / 30.f;
constexpr inline CountT numPhysicsSubsteps = 4;
constexpr inline CountT numPrepSteps = 96;
constexpr inline CountT episodeLen = 240;

void Sim::registerTypes(ECSRegistry &registry,
                        const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);

    render::BatchRenderingSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<AgentPrepCounter>();
    registry.registerComponent<Action>();
    registry.registerComponent<OwnerTeam>();
    registry.registerComponent<AgentType>();
    registry.registerComponent<GrabData>();

    registry.registerComponent<SimEntity>();

    registry.registerComponent<AgentActiveMask>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<OtherAgentObservations>();
    registry.registerComponent<AllBoxObservations>();
    registry.registerComponent<AllRampObservations>();
    registry.registerComponent<AgentVisibilityMasks>();
    registry.registerComponent<BoxVisibilityMasks>();
    registry.registerComponent<RampVisibilityMasks>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<Seed>();


    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<GlobalDebugPositions>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<CameraAgent>();
    registry.registerArchetype<DynAgent>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<AgentInterface, AgentPrepCounter>(2);
    registry.exportColumn<AgentInterface, Action>(3);
    registry.exportColumn<AgentInterface, AgentType>(4);
    registry.exportColumn<AgentInterface, AgentActiveMask>(5);
    registry.exportColumn<AgentInterface, SelfObservation>(6);
    registry.exportColumn<AgentInterface, OtherAgentObservations>(7);
    registry.exportColumn<AgentInterface, AllBoxObservations>(8);
    registry.exportColumn<AgentInterface, AllRampObservations>(9);
    registry.exportColumn<AgentInterface, AgentVisibilityMasks>(10);
    registry.exportColumn<AgentInterface, BoxVisibilityMasks>(11);
    registry.exportColumn<AgentInterface, RampVisibilityMasks>(12);
    registry.exportColumn<AgentInterface, Lidar>(13);
    registry.exportColumn<AgentInterface, Seed>(14);
    registry.exportSingleton<GlobalDebugPositions>(15);
}

static inline void resetEnvironment(Engine &ctx)
{
    ctx.data().curEpisodeStep = 0;

    if (ctx.data().enableBatchRender) {
        render::BatchRenderingSystem::reset(ctx);
    }

    if (ctx.data().enableViewer) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().obstacles;
    for (CountT i = 0; i < ctx.data().numObstacles; i++) {
        Entity e = all_entities[i];
        ctx.destroyEntity(e);
    }
    ctx.data().numObstacles = 0;
    ctx.data().numActiveBoxes = 0;
    ctx.data().numActiveRamps = 0;

    auto destroyAgent = [&](Entity e) {
        auto grab_data = ctx.getSafe<GrabData>(e);

        if (grab_data.valid()) {
            auto constraint_entity = grab_data.value().constraintEntity;
            if (constraint_entity != Entity::none()) {
                ctx.destroyEntity(constraint_entity);
            }
        }

        ctx.destroyEntity(e);
    };

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        destroyAgent(ctx.data().hiders[i]);
    }
    ctx.data().numHiders = 0;

    for (CountT i = 0; i < ctx.data().numSeekers; i++) {
        destroyAgent(ctx.data().seekers[i]);
    }
    ctx.data().numSeekers = 0;

    for (int32_t i = 0; i < ctx.data().numActiveAgents; i++) {
        ctx.destroyEntity(ctx.data().agentInterfaces[i]);
    }
    ctx.data().numActiveAgents = 0;
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;

    if (ctx.data().autoReset && ctx.data().curEpisodeStep == episodeLen - 1) {
        level = 1;
    }

    if (level != 0) {
        resetEnvironment(ctx);

        reset.resetLevel = 0;

        int32_t num_hiders = reset.numHiders;
        int32_t num_seekers = reset.numSeekers;

        generateEnvironment(ctx, level, num_hiders, num_seekers);
    } else {
        ctx.data().curEpisodeStep += 1;
    }

    ctx.data().hiderTeamReward.store_relaxed(1.f);
}

#if 0
inline void sortDebugSystem(Engine &ctx, WorldReset &)
{
    if (ctx.worldID().idx != 0) {
        return;
    }

    auto state_mgr = mwGPU::getStateManager();

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<AgentInterface>());
        
        printf("AgentInterface num rows: %u %d\n",
               TypeTracker::typeID<AgentInterface>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<AgentInterface>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<CameraAgent>());
        
        printf("CameraAgent num rows: %u %d\n",
               TypeTracker::typeID<CameraAgent>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<CameraAgent>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<DynAgent>());
        
        printf("DynAgent num rows: %u %d\n",
               TypeTracker::typeID<DynAgent>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<DynAgent>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }
}
#endif

inline void movementSystem(Engine &ctx, Action &action, SimEntity sim_e,
                                 AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;

    if (agent_type == AgentType::Seeker &&
            ctx.data().curEpisodeStep < numPrepSteps - 1) {
        return;
    }

    constexpr CountT discrete_action_buckets = 11;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float move_discrete_action_max = 120 * 4;
    constexpr float move_delta_per_bucket = move_discrete_action_max / half_buckets;

    constexpr float turn_discrete_action_max = 40 * 4;
    constexpr float turn_delta_per_bucket = turn_discrete_action_max / half_buckets;

    Vector3 cur_pos = ctx.get<Position>(sim_e.e);
    Quat cur_rot = ctx.get<Rotation>(sim_e.e);

    float f_x = move_delta_per_bucket * (action.x - 5);
    float f_y = move_delta_per_bucket * (action.y - 5);
    float t_z = turn_delta_per_bucket * (action.r - 5);

    if (agent_type == AgentType::Camera) {
        ctx.get<Position>(sim_e.e) =
            cur_pos + 0.001f * cur_rot.rotateVec({f_x, f_y, 0});

        Quat delta_rot = Quat::angleAxis(t_z * 0.001f, math::up);
        ctx.get<Rotation>(sim_e.e) = (delta_rot * cur_rot).normalize();

        return;
    }

    ctx.get<ExternalForce>(sim_e.e) = cur_rot.rotateVec({ f_x, f_y, 0 });
    ctx.get<ExternalTorque>(sim_e.e) = Vector3 { 0, 0, t_z };
}

inline void actionSystem(Engine &ctx, Action &action, SimEntity sim_e,
                         AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;
    if (agent_type == AgentType::Camera) return;

    if (action.l == 1) {
        Vector3 cur_pos = ctx.get<Position>(sim_e.e);
        Quat cur_rot = ctx.get<Rotation>(sim_e.e);

        auto &bvh = ctx.singleton<broadphase::BVH>();
        float hit_t;
        Vector3 hit_normal;
        Entity lock_entity = bvh.traceRay(cur_pos + 0.5f * math::up,
            cur_rot.rotateVec(math::fwd), &hit_t, &hit_normal, 2.5f);

        if (lock_entity != Entity::none()) {
            auto &owner = ctx.get<OwnerTeam>(lock_entity);
            auto &response_type = ctx.get<ResponseType>(lock_entity);

            if (response_type == ResponseType::Static) {
                if ((agent_type == AgentType::Seeker &&
                        owner == OwnerTeam::Seeker) ||
                        (agent_type == AgentType::Hider &&
                         owner == OwnerTeam::Hider)) {
                    response_type = ResponseType::Dynamic;
                    owner = OwnerTeam::None;
                }
            } else {
                if (owner == OwnerTeam::None) {
                    response_type = ResponseType::Static;
                    owner = agent_type == AgentType::Hider ?
                        OwnerTeam::Hider : OwnerTeam::Seeker;
                }
            }
        }
    }

    if (action.g == 1) {
        Vector3 cur_pos = ctx.get<Position>(sim_e.e);
        Quat cur_rot = ctx.get<Rotation>(sim_e.e);

        auto &grab_data = ctx.get<GrabData>(sim_e.e);

        if (grab_data.constraintEntity != Entity::none()) {
            ctx.destroyEntity(grab_data.constraintEntity);
            grab_data.constraintEntity = Entity::none();
        } else {
            auto &bvh = ctx.singleton<broadphase::BVH>();
            float hit_t;
            Vector3 hit_normal;

            Vector3 ray_o = cur_pos + 0.5f * math::up;
            Vector3 ray_d = cur_rot.rotateVec(math::fwd);

            Entity grab_entity =
                bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.5f);

            if (grab_entity != Entity::none()) {
                auto &owner = ctx.get<OwnerTeam>(grab_entity);
                auto &response_type = ctx.get<ResponseType>(grab_entity);

                if (owner == OwnerTeam::None &&
                    response_type == ResponseType::Dynamic) {

                    Entity constraint_entity =
                        ctx.makeEntity<ConstraintData>();
                    grab_data.constraintEntity = constraint_entity;

                    Vector3 other_pos = ctx.get<Position>(grab_entity);
                    Quat other_rot = ctx.get<Rotation>(grab_entity);

                    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

                    Vector3 hit_pos = ray_o + ray_d * hit_t;
                    Vector3 r2 =
                        other_rot.inv().rotateVec(hit_pos - other_pos);

                    Quat attach1 = { 1, 0, 0, 0 };
                    Quat attach2 = (other_rot.inv() * cur_rot).normalize();

                    float separation = hit_t - 1.25f;

                    ctx.get<JointConstraint>(constraint_entity) =
                        JointConstraint::setupFixed(sim_e.e, grab_entity,
                                                    attach1, attach2,
                                                    r1, r2, separation);

                }
            }
        }
    }

    // "Consume" the actions. This isn't strictly necessary but
    // allows step to be called without every agent having acted
    action.x = 5;
    action.y = 5;
    action.r = 5;
    action.g = 0;
    action.l = 0;
}

// Original code make the agents easier to control by zeroing out their velocity
// after each step. We do not do this because in the original HnS environment 
// the velocity is none-zero and is part of the observation.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               viz::VizCamera &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

inline void collectObservationsSystem(Engine &ctx,
                                      Entity agent_e,
                                      SimEntity sim_e,
                                      AgentType agent_type,
                                      SelfObservation &self_obs,
                                      OtherAgentObservations &agent_obs,
                                      AllBoxObservations &box_obs,
                                      AllRampObservations &ramp_obs,
                                      AgentPrepCounter &prep_counter)
{
    if (sim_e.e == Entity::none() || agent_type == AgentType::Camera) {
        return;
    }

    CountT cur_step = ctx.data().curEpisodeStep;
    float prep_percent = 1.0;
    if (cur_step <= numPrepSteps) {
        prep_counter.numPrepStepsLeft = numPrepSteps - cur_step;
        prep_percent = float(cur_step) / numPrepSteps;
    }

    Vector3 agent_pos = ctx.get<Position>(sim_e.e);
    Quat agent_rot = ctx.get<Rotation>(sim_e.e);
    Vector3 agent_fwd = agent_rot.rotateVec(math::fwd);
    Vector3 agent_vel = ctx.get<Velocity>(sim_e.e).linear;
    Vector3 agent_ang_vel = ctx.get<Velocity>(sim_e.e).angular;

    self_obs.pos = { agent_pos.x, agent_pos.y, agent_pos.z };
    self_obs.fwd = { agent_fwd.x, agent_fwd.y };
    self_obs.vel = { agent_vel.x, agent_vel.y, agent_vel.z };
    self_obs.angVel = agent_ang_vel.z;
    self_obs.isHider = float(agent_type == AgentType::Hider);
    self_obs.prepPercent = prep_percent;
    self_obs.curStep = float(cur_step);

    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        auto &obs = box_obs.obs[box_idx];

        if (box_idx >= num_boxes) {
            obs= {};
            continue;
        }

        Entity box_e = ctx.data().boxes[box_idx];

        Vector3 box_pos = ctx.get<Position>(box_e);
        Quat box_rot = ctx.get<Rotation>(box_e);
        Vector3 box_fwd = box_rot.rotateVec(math::fwd);
        Vector3 box_vel = ctx.get<Velocity>(box_e).linear;
        Vector3 box_ang_vel = ctx.get<Velocity>(box_e).angular;

        auto &box_type = ctx.get<ResponseType>(box_e);
        bool box_locked = false;
        bool box_team_locked = false;
        if (box_type == ResponseType::Static) {
            box_locked = true;
            OwnerTeam box_owner = ctx.get<OwnerTeam>(box_e);
            if ((agent_type == AgentType::Seeker && box_owner == OwnerTeam::Seeker) || 
            (agent_type == AgentType::Hider && box_owner == OwnerTeam::Hider)) {
                box_team_locked = true;
            }
        }

        // Vector3 box_relative_pos =
        //     agent_rot.inv().rotateVec(box_pos - agent_pos);
        // Vector3 box_relative_vel =
        //     agent_rot.inv().rotateVec(box_vel);

        obs.pos = { box_pos.x, box_pos.y, box_pos.z };
        obs.fwd = { box_fwd.x, box_fwd.y };
        obs.vel = { box_vel.x, box_vel.y, box_vel.z };
        obs.angVel = box_ang_vel.z;
        obs.isLocked = float(box_locked);
        obs.teamLocked = float(box_team_locked);

        // obs.boxSize = ctx.data().boxSizes[box_idx];

        // Quat relative_rot = agent_rot * box_rot.inv();
        // obs.boxRotation = atan2f(
        //     2.f * (relative_rot.w * relative_rot.z +
        //            relative_rot.x * relative_rot.y),
        //     1.f - 2.f * (relative_rot.y * relative_rot.y +
        //                  relative_rot.z * relative_rot.z));
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        auto &obs = ramp_obs.obs[ramp_idx];

        if (ramp_idx >= num_ramps) {
            obs = {};
            continue;
        }

        Entity ramp_e = ctx.data().ramps[ramp_idx];

        Vector3 ramp_pos = ctx.get<Position>(ramp_e);
        Quat ramp_rot = ctx.get<Rotation>(ramp_e);
        Vector3 ramp_fwd = ramp_rot.rotateVec(math::fwd);
        Vector3 ramp_vel = ctx.get<Velocity>(ramp_e).linear;
        Vector3 ramp_ang_vel = ctx.get<Velocity>(ramp_e).angular;

        auto &ramp_type = ctx.get<ResponseType>(ramp_e);
        bool ramp_locked = false;
        bool ramp_team_locked = false;
        if (ramp_type == ResponseType::Static) {
            ramp_locked = true;
            OwnerTeam ramp_owner = ctx.get<OwnerTeam>(ramp_e);
            if ((agent_type == AgentType::Seeker && ramp_owner == OwnerTeam::Seeker) || 
            (agent_type == AgentType::Hider && ramp_owner == OwnerTeam::Hider)) {
                ramp_team_locked = true;
            }
        }

        // Vector3 ramp_relative_pos =
        //     agent_rot.inv().rotateVec(ramp_pos - agent_pos);
        // Vector3 ramp_relative_vel =
        //     agent_rot.inv().rotateVec(ramp_vel);

        obs.pos = { ramp_pos.x, ramp_pos.y, ramp_pos.z };
        obs.fwd = { ramp_fwd.x, ramp_fwd.y };
        obs.vel = { ramp_vel.x, ramp_vel.y, ramp_vel.z };
        obs.angVel = ramp_ang_vel.z;
        obs.isLocked = float(ramp_locked);
        obs.teamLocked = float(ramp_team_locked);

        // Quat relative_rot = agent_rot * ramp_rot.inv();
        // obs.rampRotation = atan2f(
        //     2.f * (relative_rot.w * relative_rot.z +
        //            relative_rot.x * relative_rot.y),
        //     1.f - 2.f * (relative_rot.y * relative_rot.y +
        //                  relative_rot.z * relative_rot.z));
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_obs.obs[num_other_agents++] = {};
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.get<SimEntity>(other_agent_e).e;

        auto &obs = agent_obs.obs[num_other_agents++];

        AgentType other_agent_type = ctx.get<AgentType>(other_agent_e);
        Vector3 other_agent_pos = ctx.get<Position>(other_agent_sim_e);
        Quat other_agent_rot = ctx.get<Rotation>(other_agent_sim_e);
        Vector3 other_agent_fwd = other_agent_rot.rotateVec(math::fwd);
        Vector3 other_agent_vel = ctx.get<Velocity>(other_agent_sim_e).linear;
        Vector3 other_agent_ang_vel = ctx.get<Velocity>(other_agent_sim_e).angular;

        obs.pos = { other_agent_pos.x, other_agent_pos.y, other_agent_pos.z };
        obs.fwd = { other_agent_fwd.x, other_agent_fwd.y };
        obs.vel = { other_agent_vel.x, other_agent_vel.y, other_agent_vel.z };
        obs.angVel = other_agent_ang_vel.z;
        obs.isHider = float(other_agent_type == AgentType::Hider);
        obs.prepPercent = prep_percent;
    }
}

inline void computeVisibilitySystem(Engine &ctx,
                                    Entity agent_e,
                                    SimEntity sim_e,
                                    AgentType agent_type,
                                    AgentVisibilityMasks &agent_vis,
                                    BoxVisibilityMasks &box_vis,
                                    RampVisibilityMasks &ramp_vis)
{
    if (sim_e.e == Entity::none() || agent_type == AgentType::Camera) {
        return;
    }

    Vector3 agent_pos = ctx.get<Position>(sim_e.e);
    Quat agent_rot = ctx.get<Rotation>(sim_e.e);
    Vector3 agent_fwd = agent_rot.rotateVec(math::fwd);
    const float cos_angle_threshold = cosf(toRadians(135.f / 2.f));

    auto &bvh = ctx.singleton<broadphase::BVH>();

    auto checkVisibility = [&](Entity other_e) {
        Vector3 other_pos = ctx.get<Position>(other_e);

        Vector3 to_other = other_pos - agent_pos;

        Vector3 to_other_norm = to_other.normalize();

        float cos_angle = dot(to_other_norm, agent_fwd);

        if (cos_angle < cos_angle_threshold) {
            return 0.f;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(agent_pos, to_other, &hit_t, &hit_normal, 1.f);

        return hit_entity == other_e ? 1.f : 0.f;
    };

#ifdef MADRONA_GPU_MODE
    constexpr int32_t num_total_vis =
        consts::maxBoxes + consts::maxRamps + consts::maxAgents;
    const int32_t lane_id = threadIdx.x % 32;
    for (int32_t global_offset = 0; global_offset < num_total_vis;
         global_offset += 32) {
        int32_t cur_idx = global_offset + lane_id;

        Entity check_e = Entity::none();
        float *vis_out = nullptr;

        bool checking_agent = cur_idx < consts::maxAgents;
        uint32_t agent_mask = __ballot_sync(mwGPU::allActive, checking_agent);
        if (checking_agent) {
            bool valid_check = true;
            if (cur_idx < ctx.data().numActiveAgents) {
                Entity other_agent_e = ctx.data().agentInterfaces[cur_idx];
                valid_check = other_agent_e != agent_e;

                if (valid_check) {
                    check_e = ctx.get<SimEntity>(other_agent_e).e;
                }
            }

            uint32_t valid_mask = __ballot_sync(agent_mask, valid_check);
            valid_mask <<= (32 - lane_id);
            uint32_t num_lower_valid = __popc(valid_mask);

            if (valid_check) {
                vis_out = &agent_vis.visible[num_lower_valid];
            }
        } else if (int32_t box_idx = cur_idx - consts::maxAgents;
                   box_idx < consts::maxBoxes) {
            if (cur_idx < ctx.data().numActiveBoxes) {
                check_e = ctx.data().boxes[cur_idx];
            }
            vis_out = &box_vis.visible[cur_idx];
        } else if (int32_t ramp_idx =
                       cur_idx - consts::maxAgents - consts::maxBoxes;
                   ramp_idx < consts::maxRamps) {
            if (ramp_idx < ctx.data().numActiveRamps) {
                check_e = ctx.data().ramps[ramp_idx];
            }
            vis_out = &ramp_vis.visible[ramp_idx];
        } 

        if (check_e == Entity::none()) {
           if (vis_out != nullptr) {
               *vis_out = 0.f;
           }
        } else {
            bool is_visible = checkVisibility(check_e);
            *vis_out = is_visible ? 1.f : 0.f;
        }
    }
#else
    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        if (box_idx < num_boxes) {
            Entity box_e = ctx.data().boxes[box_idx];
            box_vis.visible[box_idx] = checkVisibility(box_e);
        } else {
            box_vis.visible[box_idx] = 0.f;
        }
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        if (ramp_idx < num_ramps) {
            Entity ramp_e = ctx.data().ramps[ramp_idx];
            ramp_vis.visible[ramp_idx] = checkVisibility(ramp_e);
        } else {
            ramp_vis.visible[ramp_idx] = 0.f;
        }
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_vis.visible[num_other_agents++] = 0.f;
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.get<SimEntity>(other_agent_e).e;

        bool is_visible = checkVisibility(other_agent_sim_e);

        if (agent_type == AgentType::Seeker && is_visible) {
            AgentType other_type = ctx.get<AgentType>(other_agent_e);
            if (other_type == AgentType::Hider) {
                ctx.data().hiderTeamReward.store_relaxed(-1.f);
            }
        }

        agent_vis.visible[num_other_agents++] = is_visible;
    }
#endif
}

inline void lidarSystem(Engine &ctx,
                        SimEntity sim_e,
                        Lidar &lidar)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Vector3 pos = ctx.get<Position>(sim_e.e);
    Quat rot = ctx.get<Rotation>(sim_e.e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (float(idx) / float(30));
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos , ray_dir, &hit_t, &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.depth[idx] = 0.f;
        } else {
            lidar.depth[idx] = hit_t;
        }
    };


#ifdef MADRONA_GPU_MODE
    int32_t idx = threadIdx.x % 32;

    if (idx < 30) {
        traceRay(idx);
    }
#else
    for (int32_t i = 0; i < 30; i++) {
        traceRay(i);
    }
#endif
}

// FIXME: refactor this so the observation systems can reuse these raycasts
// (unless a reset has occurred)
inline void rewardsVisSystem(Engine &ctx,
                             SimEntity sim_e,
                             AgentType agent_type)
{
    const float cos_angle_threshold = cosf(toRadians(135.f / 2.f));

    if (sim_e.e == Entity::none() || agent_type != AgentType::Seeker) {
        return;
    }

    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 seeker_pos = ctx.get<Position>(sim_e.e);
    Quat seeker_rot = ctx.get<Rotation>(sim_e.e);
    Vector3 seeker_fwd = seeker_rot.rotateVec(math::fwd);

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        Entity hider_sim_e = ctx.data().hiders[i];

        Vector3 hider_pos = ctx.get<Position>(hider_sim_e);

        Vector3 to_hider = hider_pos - seeker_pos;

        Vector3 to_hider_norm = to_hider.normalize();

        float cos_angle = dot(to_hider_norm, seeker_fwd);

        if (cos_angle < cos_angle_threshold) {
            continue;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(seeker_pos, to_hider, &hit_t, &hit_normal, 1.f);

        if (hit_entity == hider_sim_e) {
            ctx.data().hiderTeamReward.store_relaxed(-1);
            break;
        }
    }
}

inline void outputRewardsDonesSystem(Engine &ctx,
                                    SimEntity sim_e,
                                    AgentType agent_type)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    // FIXME: allow loc to be passed in
    Loc l = ctx.loc(sim_e.e);
    float *reward_out = &ctx.data().rewardBuffer[l.row];
    uint8_t * const done_out = &ctx.data().doneBuffer[l.row];

    CountT cur_step = ctx.data().curEpisodeStep;

    if (cur_step == 0) {
        *done_out = 0;
    }

    if (cur_step < numPrepSteps - 1) {
        *reward_out = 0.f;
        return;
    } else if (cur_step == episodeLen - 1) {
        *done_out = 1;
    }

    float reward_val = ctx.data().hiderTeamReward.load_relaxed();
    if (agent_type == AgentType::Seeker) {
        reward_val *= -1.f;
    }

    Vector3 pos = ctx.get<Position>(sim_e.e);

    if (fabsf(pos.x) >= 18.f || fabsf(pos.y) >= 18.f) {
        reward_val -= 10.f;
    }

    *reward_out = reward_val;
}

inline void globalPositionsDebugSystem(Engine &ctx,
                                       GlobalDebugPositions &global_positions)
{
    auto getXY = [](Vector3 v) {
        return Vector2 {
            v.x,
            v.y,
        };
    };

    for (CountT i = 0; i < consts::maxBoxes; i++) {
        if (i >= ctx.data().numActiveBoxes) {
            global_positions.boxPositions[i] = Vector3 {0, 0, 0};
            continue;
        }

        global_positions.boxPositions[i] =
            ctx.get<Position>(ctx.data().boxes[i]);
    }

    for (CountT i = 0; i < consts::maxRamps; i++) {
        if (i >= ctx.data().numActiveRamps) {
            global_positions.rampPositions[i] = Vector3 {0, 0, 0};
            continue;
        }

        global_positions.rampPositions[i] =
            ctx.get<Position>(ctx.data().ramps[i]);
    }

    {
        CountT out_offset = 0;
        for (CountT i = 0; i < ctx.data().numHiders; i++) {
            global_positions.agentPositions[out_offset++] = 
                ctx.get<Position>(ctx.data().hiders[i]);
        }

        for (CountT i = 0; i < ctx.data().numSeekers; i++) {
            global_positions.agentPositions[out_offset++] = 
                ctx.get<Position>(ctx.data().seekers[i]);
        }

        for (; out_offset < consts::maxAgents; out_offset++) {
            global_positions.agentPositions[out_offset++] = Vector3 {0, 0, 0};
        }
    }
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraphBuilder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

void Sim::setupTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    auto move_sys = builder.addToGraph<ParallelForNode<Engine, movementSystem,
        Action, SimEntity, AgentType>>({});

    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
        {move_sys});

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, SimEntity, AgentType>>({broadphase_setup_sys});

    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {action_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, viz::VizCamera>>(
            {substep_sys});

    auto sim_done = agent_zero_vel;

    sim_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto rewards_vis = builder.addToGraph<ParallelForNode<Engine,
        rewardsVisSystem,
            SimEntity,
            AgentType
        >>({sim_done});

    auto output_rewards = builder.addToGraph<ParallelForNode<Engine,
        outputRewardsDonesSystem,
            SimEntity,
            AgentType
        >>({rewards_vis});

    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({output_rewards});

    auto clearTmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});

#ifdef MADRONA_GPU_MODE
    // FIXME: these 3 need to be compacted, but sorting is unnecessary
    auto sort_cam_agent = queueSortByWorld<CameraAgent>(builder, {clearTmp});
    auto sort_dyn_agent = queueSortByWorld<DynAgent>(builder, {sort_cam_agent});
    auto sort_objects = queueSortByWorld<DynamicObject>(builder, {sort_dyn_agent});
    auto sort_agent_iface =
        queueSortByWorld<AgentInterface>(builder, {sort_objects});
    auto reset_finish = sort_agent_iface;
#else
    auto reset_finish = clearTmp;
#endif

    if (cfg.enableBatchRender) {
        render::BatchRenderingSystem::setupTasks(builder,
            {reset_finish});
    }

    if (cfg.enableViewer) {
        viz::VizRenderingSystem::setupTasks(builder,
            {reset_finish});
    }

#if 0
    prep_finish = builder.addToGraph<ParallelForNode<Engine,
        sortDebugSystem, WorldReset>>({prep_finish});
#endif

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_finish});
    (void)recycle_sys;
#endif
    
    auto post_reset_broadphase = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {reset_finish});

    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Entity,
            SimEntity,
            AgentType,
            SelfObservation,
            OtherAgentObservations,
            AllBoxObservations,
            AllRampObservations,
            AgentPrepCounter
        >>({post_reset_broadphase});


#ifdef MADRONA_GPU_MODE
    auto compute_visibility = builder.addToGraph<CustomParallelForNode<Engine,
        computeVisibilitySystem, 32, 1,
#else
    auto compute_visibility = builder.addToGraph<ParallelForNode<Engine,
        computeVisibilitySystem,
#endif
            Entity,
            SimEntity,
            AgentType,
            AgentVisibilityMasks,
            BoxVisibilityMasks,
            RampVisibilityMasks
        >>({post_reset_broadphase});

#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            SimEntity,
            Lidar
        >>({reset_finish});

    auto global_positions_debug = builder.addToGraph<ParallelForNode<Engine,
        globalPositionsDebugSystem,
            GlobalDebugPositions
        >>({reset_finish});

    (void)lidar;
    (void)compute_visibility;
    (void)collect_observations;
    (void)global_positions_debug;
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      rewardBuffer(init.rewardBuffer),
      doneBuffer(init.doneBuffer)
{
    CountT max_total_entities =
        std::max(init.maxEntitiesPerWorld, uint32_t(3 + 3 + 9 + 2 + 6)) + 100;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         50 * 20, 10);

    if (cfg.enableViewer) {
        viz::VizRenderingSystem::init(ctx, init.vizBridge);
    }

    if (cfg.enableBatchRender) {
        render::BatchRenderingSystem::init(ctx, init.batchRenderBridge);
    }

    obstacles =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numObstacles = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    numHiders = 0;
    numSeekers = 0;
    numActiveAgents = 0;

    curEpisodeStep = 0;

    enableBatchRender = cfg.enableBatchRender;
    enableViewer = cfg.enableViewer;
    autoReset = cfg.autoReset;

    resetEnvironment(ctx);
    generateEnvironment(ctx, 1, 3, 2);
    ctx.singleton<WorldReset>() = {
        .resetLevel = 0,
        .numHiders = 3,
        .numSeekers = 2,
    };

    ctx.data().hiderTeamReward.store_relaxed(1.f);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
