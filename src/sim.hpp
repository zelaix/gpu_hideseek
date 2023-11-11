#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/mw.hpp>
#include <madrona/viz/system.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace GPUHideSeek {

using madrona::Entity;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

namespace consts {

static inline constexpr int32_t maxBoxes = 1;
static inline constexpr int32_t maxRamps = 1;
static inline constexpr int32_t maxAgents = 4;

}

struct Config {
    bool enableBatchRender;
    bool enableViewer;
    bool autoReset;
};

class Engine;

struct WorldReset {
    int32_t resetLevel;
    int32_t numHiders;
    int32_t numSeekers;
};

struct AgentPrepCounter {
    int32_t numPrepStepsLeft;
};

enum class OwnerTeam : uint32_t {
    None,
    Seeker,
    Hider,
    Unownable,
};

struct GrabData {
    Entity constraintEntity;
};

enum class AgentType : uint32_t {
    Seeker = 0,
    Hider = 1,
    Camera = 2,
};

struct DynamicObject : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OwnerTeam
> {};

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
    int32_t g;
    int32_t l;
};

struct SimEntity {
    Entity e;
};

struct AgentActiveMask {
    float mask;
};

struct GlobalDebugPositions {
    madrona::math::Vector3 boxPositions[consts::maxBoxes];
    madrona::math::Vector3 rampPositions[consts::maxRamps];
    madrona::math::Vector3 agentPositions[consts::maxAgents];
};

struct SelfObservation {
    madrona::math::Vector3 pos;
    madrona::math::Vector2 fwd;
    madrona::math::Vector3 vel;
    // float angle;
    float angVel;
    float isHider;
    float prepPercent;
    float curStep;
    // vector_door_obs_self doorObs;
};

struct AgentObservation {
    madrona::math::Vector3 pos;
    madrona::math::Vector2 fwd;
    madrona::math::Vector3 vel;
    // float angle;
    float angVel;
    float isHider;
    float prepPercent;
};

struct BoxObservation {
    madrona::math::Vector3 pos;
    madrona::math::Vector2 fwd;
    madrona::math::Vector3 vel;
    float angVel;
    float isLocked;
    // float youLocked;
    float teamLocked;
    // madrona::math::Vector2 boxSize;
    // float boxRotation;
};

struct RampObservation {
    madrona::math::Vector3 pos;
    madrona::math::Vector2 fwd;
    madrona::math::Vector3 vel;
    float angVel;
    float isLocked;
    // float youLocked;
    float teamLocked;
    // float rampRotation;
};

struct OtherAgentObservations {
    AgentObservation obs[consts::maxAgents - 1];
};

struct AllBoxObservations {
    BoxObservation obs[consts::maxBoxes];
};

struct AllRampObservations {
    RampObservation obs[consts::maxRamps];
};

struct AgentVisibilityMasks {
    float visible[consts::maxAgents - 1];
};

struct BoxVisibilityMasks {
    float visible[consts::maxBoxes];
};

struct RampVisibilityMasks {
    float visible[consts::maxRamps];
};

struct Lidar {
    float depth[30];
};

struct Seed {
    int32_t seed;
};

static_assert(sizeof(Action) == 5 * sizeof(int32_t));

struct AgentInterface : public madrona::Archetype<
    SimEntity,
    AgentPrepCounter,
    Action,
    AgentType,
    AgentActiveMask,
    SelfObservation,
    OtherAgentObservations,
    AllBoxObservations,
    AllRampObservations,
    AgentVisibilityMasks,
    BoxVisibilityMasks,
    RampVisibilityMasks,
    Lidar,
    Seed
> {};

struct CameraAgent : public madrona::Archetype<
    Position,
    Rotation,
    madrona::render::BatchRenderCamera,
    madrona::viz::VizCamera
> {};

struct DynAgent : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OwnerTeam,
    GrabData,
    madrona::render::BatchRenderCamera,
    madrona::viz::VizCamera
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    EpisodeManager *episodeMgr;
    float *rewardBuffer;
    uint8_t *doneBuffer;
    RNG rng;

    Entity *obstacles;
    CountT numObstacles;

    Entity hiders[3];
    CountT numHiders;
    Entity seekers[3];
    CountT numSeekers;

    Entity boxes[consts::maxBoxes];
    madrona::math::Vector2 boxSizes[consts::maxBoxes];
    float boxRotations[consts::maxBoxes];
    Entity ramps[consts::maxRamps];
    float rampRotations[consts::maxRamps];
    Entity agentInterfaces[consts::maxAgents];
    CountT numActiveBoxes;
    CountT numActiveRamps;
    CountT numActiveAgents;

    CountT curEpisodeStep;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;

    uint32_t curEpisodeSeed;
    bool enableBatchRender;
    bool enableViewer;
    bool autoReset;

    madrona::AtomicFloat hiderTeamReward {0};
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
