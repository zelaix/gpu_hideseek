#include "level_gen.hpp"
#include "geo_gen.hpp"

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

template <typename T>
static Entity makeAgent(Engine &ctx, AgentType agent_type)
{
    Entity agent_iface =
        ctx.data().agentInterfaces[ctx.data().numActiveAgents++] =
            ctx.makeEntity<AgentInterface>();

    Entity agent = ctx.makeEntity<T>();
    ctx.get<SimEntity>(agent_iface).e = agent;
    ctx.get<AgentActiveMask>(agent_iface).mask = 1.f;

    ctx.get<AgentType>(agent_iface) = agent_type;

    if (agent_type == AgentType::Seeker) {
        ctx.data().seekers[ctx.data().numSeekers++] = agent;
    } else {
        ctx.data().hiders[ctx.data().numHiders++] = agent;
    }

    ctx.get<Seed>(agent_iface).seed = ctx.data().curEpisodeSeed;

    // Zero out actions
    ctx.get<Action>(agent_iface) = {
        .x = 5,
        .y = 5,
        .r = 5,
        .g = 0,
        .l = 0,
    };

    return agent;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, 1, ResponseType::Static,
                         OwnerTeam::Unownable);
}

// Emergent tool use configuration:
// 1 - 3 Hiders
// 1 - 3 Seekers
// 3 - 9 Movable boxes (at least 3 elongated)
// 2 movable ramps

static void generateTrainingEnvironment(Engine &ctx,
                                        CountT num_hiders,
                                        CountT num_seekers)
{
    auto &rng = ctx.data().rng;

    // CountT total_num_boxes = CountT(rng.rand() * 6) + 3;
    CountT total_num_boxes = CountT(rng.rand() * 2) + 3;
    // CountT total_num_boxes = 1;
    assert(total_num_boxes <= consts::maxBoxes);

    CountT num_elongated = 
    CountT(ctx.data().rng.rand() * (total_num_boxes - 3)) + 3;
    // CountT num_elongated = 0;

    CountT num_cubes = total_num_boxes - num_elongated;

    assert(num_elongated + num_cubes == total_num_boxes);

    const Vector2 bounds { -18.f, 18.f };
    float bounds_diff = bounds.y - bounds.x;

    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    CountT num_entities =
        populateStaticGeometry(ctx, rng, {bounds.y, bounds.y});

    Entity *all_entities = ctx.data().obstacles;

    auto checkOverlap = [&obj_mgr, &ctx,
                         &all_entities, &num_entities](const AABB &aabb) {
        for (int i = 0; i < num_entities; ++i) {
            ObjectID obj_id = ctx.get<ObjectID>(all_entities[i]);
            AABB other = obj_mgr.rigidBodyAABBs[obj_id.idx];

            Position p = ctx.get<Position>(all_entities[i]);
            Rotation r = ctx.get<Rotation>(all_entities[i]);
            Scale s = ctx.get<Scale>(all_entities[i]);
            other = other.applyTRS(p, r, Diag3x3(s));

            if (aabb.overlaps(other)) {
                return false;
            }
        }

        return true;
    };

    const CountT max_rejections = 20;

    for (CountT i = 0; i < num_elongated; i++) {
        CountT rejections = 0;
        // Choose a random room and put the entity in a random position in that room
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

            Vector3 pos {
                bounds.x + rng.rand() * bounds_diffx,
                bounds.x + rng.rand() * bounds_diffy,
                1.0f,
            };

            float box_rotation = rng.rand() * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[6];
            aabb = aabb.applyTRS(pos, rot, scale);

            // Check overlap with all other entities
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().boxes[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 7);

                ctx.data().boxSizes[i] = { 8, 1.5 };
                ctx.data().boxRotations[i] = box_rotation;
                break;
            }

            rejections++;
        }
    }

    for (CountT i = 0; i < num_cubes; i++) {
        CountT rejections = 0;
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

#if 0
            Vector3 pos {
                room->low.x + rng.rand() * bounds_diffx,
                room->low.y + rng.rand() * bounds_diffy,
                1.0f,
            };
#endif

            Vector3 pos {
                bounds.x + rng.rand() * bounds_diffx,
                bounds.x + rng.rand() * bounds_diffy,
                1.0f,
            };

            float box_rotation = rng.rand() * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[2];
            aabb = aabb.applyTRS(pos, rot, scale);

            if (checkOverlap(aabb) || rejections == max_rejections) {
                CountT box_idx = i + num_elongated;

                ctx.data().boxes[box_idx] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 2);

                ctx.data().boxSizes[box_idx] = { 2, 2 };
                ctx.data().boxRotations[box_idx] = box_rotation;
                break;
            }

            ++rejections;
        }
    }

    ctx.data().numActiveBoxes = total_num_boxes;

    const CountT num_ramps = consts::maxRamps;
    for (CountT i = 0; i < num_ramps; i++) {
        CountT rejections = 0;
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

            Vector3 pos {
                bounds.x + rng.rand() * bounds_diffx,
                bounds.x + rng.rand() * bounds_diffy,
                1.0f,
            };

            float ramp_rotation = rng.rand() * math::pi;
            const auto rot = Quat::angleAxis(ramp_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[5];
            aabb = aabb.applyTRS(pos, rot, scale);

            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().ramps[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 6);
                ctx.data().rampRotations[i] = ramp_rotation;
                break;
            }

            ++rejections;
        }
    }
    ctx.data().numActiveRamps = num_ramps;

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        if (ctx.data().enableBatchRender) {
            ctx.get<render::BatchRenderCamera>(agent) =
                render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, view_idx);
        }

        if (ctx.data().enableViewer) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                        Vector3 { 0, 0, 1.5 }, view_idx);
        }

        // ObjectID agent_obj_id = ObjectID { 4 };
        ObjectID agent_obj_id = is_hider ? ObjectID { 4 } : ObjectID { 5 };
        ctx.get<ObjectID>(agent) = agent_obj_id;
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.get<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();
        ctx.get<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    for (CountT i = 0; i < num_hiders; i++) {
        CountT rejections = 0;
        while (true) {
            Vector3 pos {
                bounds.x + rng.rand() * bounds_diff,
                    bounds.x + rng.rand() * bounds_diff,
                    0.f,
            };

            const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, true, i);
                break;
            }

            rejections++;
        }
    }

    for (CountT i = 0; i < num_seekers; i++) {
        CountT rejections = 0;
        while (true) {
            Vector3 pos {
                bounds.x + rng.rand() * bounds_diff,
                    bounds.x + rng.rand() * bounds_diff,
                    0.f,
            };

            const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);

            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, false, num_hiders + i);
                break;
            }

            rejections++;
        }
    }

    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, 0, 100}, Quat::angleAxis(pi, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {-100, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {100, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, -100, 0}, Quat::angleAxis(-pi_d2, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, 100, 0}, Quat::angleAxis(pi_d2, {1, 0, 0}));

    ctx.data().numObstacles = num_entities;
}

static void generateDebugEnvironment(Engine &ctx, CountT level_id);

static void generateQuadrantEnvironment(Engine &ctx,
                                        CountT num_hiders,
                                        CountT num_seekers)
{
    // Set quadrant length
    float length = 7.f;
    float door_size = 3.f;
    auto &rng = ctx.data().rng;
    CountT num_entities = 0;
    Entity *all_entities = ctx.data().obstacles;

    // Init object manager to check overlaps
    const CountT max_rejections = 20;
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    auto checkOverlap = [&obj_mgr, &ctx,
                         &all_entities, &num_entities](const AABB &aabb) {
        for (int i = 0; i < num_entities; ++i) {
            ObjectID obj_id = ctx.get<ObjectID>(all_entities[i]);
            AABB other = obj_mgr.rigidBodyAABBs[obj_id.idx];

            Position p = ctx.get<Position>(all_entities[i]);
            Rotation r = ctx.get<Rotation>(all_entities[i]);
            Scale s = ctx.get<Scale>(all_entities[i]);
            other = other.applyTRS(p, r, Diag3x3(s));

            if (aabb.overlaps(other)) {
                return false;
            }
        }
        return true;
    };    

    // Make plane
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    // Make quadrant
    all_entities[num_entities++] = makeDynObject(
        ctx, {0, -2 * length, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
        ResponseType::Static, OwnerTeam::Unownable, {2 * length, 0.2f, 1.f} );
    all_entities[num_entities++] = makeDynObject(
        ctx, {0, 2 * length, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
        ResponseType::Static, OwnerTeam::Unownable, {2 * length, 0.2f, 1.f} );
    all_entities[num_entities++] = makeDynObject(
        ctx, {-2 * length, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
        ResponseType::Static, OwnerTeam::Unownable, {0.2f, 2 * length, 1.f} );
    all_entities[num_entities++] = makeDynObject(
        ctx, {2 * length, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
        ResponseType::Static, OwnerTeam::Unownable, {0.2f, 2 * length, 1.f} );

    // Make room
    bool horizontal_door = (bool)(rng.u32Rand() % 2);
    if (horizontal_door) {
        float left_length = 1 + rng.rand() * (2 * length - door_size - 2);
        float right_length = 2 * length - door_size - left_length;
        all_entities[num_entities++] = makeDynObject(
            ctx, {-length, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {length, 0.2f, 1.f} );
        all_entities[num_entities++] = makeDynObject(
            ctx, {0, -left_length / 2, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {0.2f, left_length / 2, 1.f} );
        all_entities[num_entities++] = makeDynObject(
            ctx, {0, -(2 * length - right_length / 2), 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {0.2f, right_length / 2, 1.f} );   
    } else {
        float top_length = 1 + rng.rand() * (2 * length - door_size - 2);
        float bottom_length = 2 * length - door_size - top_length;
        all_entities[num_entities++] = makeDynObject(
            ctx, {0, -length, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {0.2f, length, 1.f} );
        all_entities[num_entities++] = makeDynObject(
            ctx, {-top_length / 2, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {top_length / 2, 0.2f, 1.f} );
        all_entities[num_entities++] = makeDynObject(
            ctx, {-(2 * length - bottom_length / 2), 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {bottom_length / 2, 0.2f, 1.f} );
    }

    // Make boxes
    const CountT num_boxes = consts::maxBoxes;
    for (CountT i = 0; i < num_boxes; i++) {
        CountT rejections = 0;
        while (true) {
            // boxes are spawned inside the room
            Vector3 pos {
                -2 * length + 2 * rng.rand() * length,
                -2 * length + 2 * rng.rand() * length,
                1.0f,
            };
            float box_rotation = rng.rand() * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[2];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().boxes[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 2);
                ctx.data().boxSizes[i] = { 2, 2 };
                ctx.data().boxRotations[i] = box_rotation;
                break;
            }
            ++rejections;
        }
    }
    ctx.data().numActiveBoxes = num_boxes;

    // Make ramps
    const CountT num_ramps = consts::maxRamps;
    for (CountT i = 0; i < num_ramps; i++) {
        CountT rejections = 0;
        while (true) {
            // ramps are spawned randomly in the environment
            Vector3 pos {
                -2 * length + 4 * rng.rand() * length,
                -2 * length + 4 * rng.rand() * length,
                1.0f,
            };
            float ramp_rotation = rng.rand() * math::pi;
            const auto rot = Quat::angleAxis(ramp_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[5];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().ramps[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 6);
                ctx.data().rampRotations[i] = ramp_rotation;
                break;
            }
            ++rejections;
        }
    }
    ctx.data().numActiveRamps = num_ramps;

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        if (ctx.data().enableBatchRender) {
            ctx.get<render::BatchRenderCamera>(agent) =
                render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, view_idx);
        }

        if (ctx.data().enableViewer) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                        Vector3 { 0, 0, 1.5 }, view_idx);
        }

        // ObjectID agent_obj_id = ObjectID { 4 };
        ObjectID agent_obj_id = is_hider ? ObjectID { 4 } : ObjectID { 5 };
        ctx.get<ObjectID>(agent) = agent_obj_id;
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.get<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();
        ctx.get<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    // Make hiders
    for (CountT i = 0; i < num_hiders; i++) {
        CountT rejections = 0;
        while (true) {
            // Hiders are spawned randomly in the environment
            Vector3 pos {
                -2 * length + 4 * rng.rand() * length,
                -2 * length + 4 * rng.rand() * length,
                0.f,
            };
            const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, true, i);
                break;
            }
            rejections++;
        }
    }

    // Make seekers
    for (CountT i = 0; i < num_seekers; i++) {
        CountT rejections = 0;
        while (true) {
            // Seekers are only spawned outside the room
            Vector3 pos {
                -length + 2 * rng.rand() * length,
                -length + 2 * rng.rand() * length,
                0.f,
            };
            CountT quadrant = CountT(rng.u32Rand() % 3) + 1;
            switch (quadrant) {
                case 1: {
                    pos.x += length;
                    pos.y += length;
                } break;
                case 2: {
                    pos.x -= length;
                    pos.y += length;
                } break;
                case 3: {
                    pos.x += length;
                    pos.y -= length;
                } break;
            }
            const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, false, num_hiders + i);
                break;
            }
            rejections++;
        }
    }

    ctx.data().numObstacles = num_entities;
}

static void generateDebugQuadrantEnvironment(Engine &ctx)
{
    // Set quadrant length
    float length = 7.f;
    float door_size = 3.f;
    auto &rng = ctx.data().rng;
    CountT num_entities = 0;
    Entity *all_entities = ctx.data().obstacles;

    // Init object manager to check overlaps
    const CountT max_rejections = 20;
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    auto checkOverlap = [&obj_mgr, &ctx,
                         &all_entities, &num_entities](const AABB &aabb) {
        for (int i = 0; i < num_entities; ++i) {
            ObjectID obj_id = ctx.get<ObjectID>(all_entities[i]);
            AABB other = obj_mgr.rigidBodyAABBs[obj_id.idx];

            Position p = ctx.get<Position>(all_entities[i]);
            Rotation r = ctx.get<Rotation>(all_entities[i]);
            Scale s = ctx.get<Scale>(all_entities[i]);
            other = other.applyTRS(p, r, Diag3x3(s));

            if (aabb.overlaps(other)) {
                return false;
            }
        }
        return true;
    };    

    // Make plane
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    // Make boxes
    const CountT num_boxes = consts::maxBoxes;
    // const CountT num_boxes = 0;
    for (CountT i = 0; i < num_boxes; i++) {
        CountT rejections = 0;
        while (true) {
            // boxes are spawned inside the room
            Vector3 pos {
                -10.f,
                -8.f,
                1.0f,
            };
            float box_rotation = 0 * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[2];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().boxes[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 2);
                ctx.data().boxSizes[i] = { 2, 2 };
                ctx.data().boxRotations[i] = box_rotation;
                break;
            }
            ++rejections;
        }
    }
    ctx.data().numActiveBoxes = num_boxes;

    // Make ramps
    const CountT num_ramps = consts::maxRamps;
    // const CountT num_ramps = 0;
    for (CountT i = 0; i < num_ramps; i++) {
        CountT rejections = 0;
        while (true) {
            // ramps are spawned randomly in the environment
            Vector3 pos {
                10.f,
                0.f,
                1.0f,
            };
            float ramp_rotation = 0 * math::pi;
            const auto rot = Quat::angleAxis(ramp_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[5];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().ramps[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, 6);
                ctx.data().rampRotations[i] = ramp_rotation;
                break;
            }
            ++rejections;
        }
    }
    ctx.data().numActiveRamps = num_ramps;

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        if (ctx.data().enableBatchRender) {
            ctx.get<render::BatchRenderCamera>(agent) =
                render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, view_idx);
        }

        if (ctx.data().enableViewer) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                        Vector3 { 0, 0, 1.5 }, view_idx);
        }

        // ObjectID agent_obj_id = ObjectID { 4 };
        ObjectID agent_obj_id = is_hider ? ObjectID { 4 } : ObjectID { 5 };
        ctx.get<ObjectID>(agent) = agent_obj_id;
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.get<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();
        ctx.get<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    // Make hiders
    for (CountT i = 0; i < 2; i++) {
        CountT rejections = 0;
        while (true) {
            // Hiders are spawned randomly in the environment
            Vector3 pos {
                20.f * i - 10.f,
                -10.f,
                0.f,
            };
            const auto rot = Quat::angleAxis(0, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, true, i);
                break;
            }
            rejections++;
        }
    }

    // Make seekers
    for (CountT i = 0; i < 2; i++) {
        CountT rejections = 0;
        while (true) {
            // Seekers are only spawned outside the room
            Vector3 pos {
                20.f * i - 10.f,
                10.f,
                0.f,
            };
            const auto rot = Quat::angleAxis(0 * math::pi, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[4];
            aabb = aabb.applyTRS(pos, rot, scale);
            if (checkOverlap(aabb) || rejections == max_rejections) {
                makeDynAgent(pos, rot, false, 2 + i);
                break;
            }
            rejections++;
        }
    }

    ctx.data().numObstacles = num_entities;
}

void generateEnvironment(Engine &ctx,
                         CountT level_id,
                         CountT num_hiders,
                         CountT num_seekers)
{
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);

    ctx.data().curEpisodeSeed = episode_idx;

    // generateDebugQuadrantEnvironment(ctx);
    generateQuadrantEnvironment(ctx, 2, 2);

    // if (level_id == 1) {
    //     generateTrainingEnvironment(ctx, num_hiders, num_seekers);
    // } else {
    //     generateDebugEnvironment(ctx, level_id);
    // }
}

static void singleCubeLevel(Engine &ctx, Vector3 pos, Quat rot)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    const Quat agent_rot =
        Quat::angleAxis(toRadians(-45), {0, 0, 1});

    ctx.data().numObstacles = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    if (ctx.data().enableBatchRender) {
        ctx.get<render::BatchRenderCamera>(agent) =
            render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }
    if (ctx.data().enableViewer) {
        ctx.get<viz::VizCamera>(agent) =
            viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }
    ctx.get<Position>(agent) = Vector3 { -5, -5, 0 };
    ctx.get<Rotation>(agent) = agent_rot;
}

static void level2(Engine &ctx)
{
    Quat cube_rotation = (Quat::angleAxis(atanf(1.f/sqrtf(2.f)), {0, 1, 0}) *
        Quat::angleAxis(toRadians(45), {1, 0, 0})).normalize().normalize();
    singleCubeLevel(ctx, { 0, 0, 5 }, cube_rotation);
}

static void level3(Engine &ctx)
{
    singleCubeLevel(ctx, { 0, 0, 5 }, Quat::angleAxis(0, {0, 0, 1}));
}

static void level4(Engine &ctx)
{
    Vector3 pos { 0, 0, 5 };

    Quat rot = (
        Quat::angleAxis(toRadians(45), {0, 1, 0})).normalize();
#if 0
        Quat::angleAxis(toRadians(45), {0, 1, 0}) *
        Quat::angleAxis(toRadians(40), {1, 0, 0})).normalize();
#endif

    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    //Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    //all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makeDynObject(ctx, pos + Vector3 {0, 0, 5}, rot, 7,
                      ResponseType::Dynamic, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    const Quat agent_rot =
        Quat::angleAxis(toRadians(-45), {0, 0, 1});

    ctx.data().numObstacles = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    if (ctx.data().enableBatchRender) {
        ctx.get<render::BatchRenderCamera>(agent) =
            render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }

    if (ctx.data().enableViewer) {
        ctx.get<viz::VizCamera>(agent) =
            viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }

    ctx.get<Position>(agent) = Vector3 { -7.5, -7.5, 0.5 };
    ctx.get<Rotation>(agent) = agent_rot;
}

static void level5(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;
    CountT num_entities_range =
        ctx.data().maxEpisodeEntities - ctx.data().minEpisodeEntities;

    CountT num_dyn_entities =
        CountT(ctx.data().rng.rand() * num_entities_range) +
        ctx.data().minEpisodeEntities;

    const Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < num_dyn_entities; i++) {
        Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        const auto rot = Quat::angleAxis(0, {0, 0, 1});

        all_entities[i] = makeDynObject(ctx, pos, rot, 2);
    }

    CountT total_entities = num_dyn_entities;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 40}, Quat::angleAxis(pi, {1, 0, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {0, -20, 0}, Quat::angleAxis(-pi_d2, {1, 0, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {0, 20, 0}, Quat::angleAxis(pi_d2, {1, 0, 0}));

    const Quat agent_rot =
        Quat::angleAxis(-pi_d2, {1, 0, 0});

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    if (ctx.data().enableBatchRender) {
        ctx.get<render::BatchRenderCamera>(agent) =
            render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                                    up * 0.5f, 0);
    }

    if (ctx.data().enableViewer) {
        ctx.get<viz::VizCamera>(agent) =
            viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                                    up * 0.5f, 0);
    }

    ctx.get<Position>(agent) = Vector3 { 0, 0, 35 };
    ctx.get<Rotation>(agent) = agent_rot;

    ctx.data().numObstacles = total_entities;
}

static void level6(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT num_entities = 0;
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    all_entities[num_entities++] =
        makeDynObject(
            ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), 3,
            ResponseType::Static, OwnerTeam::Unownable, {10.f, 0.2f, 1.f} );

    all_entities[num_entities++] =
        makeDynObject(
            ctx, {0, -5, 1}, Quat::angleAxis(0, {1, 0, 0}), 2,
            ResponseType::Dynamic, OwnerTeam::None, {1.f, 1.f, 1.f} );

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        if (ctx.data().enableBatchRender) {
            ctx.get<render::BatchRenderCamera>(agent) =
                render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, view_idx);
        }

        if (ctx.data().enableViewer) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, view_idx);
        }

        // ObjectID agent_obj_id = ObjectID { 4 };
        ObjectID agent_obj_id = is_hider ? ObjectID { 4 } : ObjectID { 5 };
        ctx.get<ObjectID>(agent) = agent_obj_id;
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.get<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();
        ctx.get<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    makeDynAgent({ -15, -15, 1.5 },
        Quat::angleAxis(toRadians(-45), {0, 0, 1}), true, 0);

    makeDynAgent({ -15, -10, 1.5 },
        Quat::angleAxis(toRadians(45), {0, 0, 1}), false, 1);

    ctx.data().numObstacles = num_entities;
}

static void level7(Engine &ctx)
{
    Vector3 pos { 0, 0, 5 };

    Quat rot = (
        Quat::angleAxis(toRadians(45), {0, 1, 0}) *
        Quat::angleAxis(toRadians(40), {1, 0, 0})).normalize();

    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    all_entities[total_entities++] =  makeDynObject(ctx, pos, rot, 2);

    all_entities[total_entities++] =
        makeDynObject(ctx, pos + Vector3 {0, 0, 5}, rot, 2,
                      ResponseType::Dynamic, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    const Quat agent_rot =
        Quat::angleAxis(toRadians(-45), {0, 0, 1});

    ctx.data().numObstacles = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    if (ctx.data().enableBatchRender) {
        ctx.get<render::BatchRenderCamera>(agent) =
            render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }
    if (ctx.data().enableViewer) {
        ctx.get<viz::VizCamera>(agent) =
            viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }
    ctx.get<Position>(agent) = Vector3 { -5, -5, 0.5 };
    ctx.get<Rotation>(agent) = agent_rot;
}

static void level8(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    Vector3 ramp_pos { 0, 0, 10 };

    Quat ramp_rot = (
        Quat::angleAxis(toRadians(25), {0, 1, 0}) *
        Quat::angleAxis(toRadians(90), {0, 0, 1}) *
        Quat::angleAxis(toRadians(45), {1, 0, 0})).normalize();

    Entity ramp_dyn = all_entities[total_entities++] =
        makeDynObject(ctx, ramp_pos, ramp_rot, 6);

    ctx.get<Velocity>(ramp_dyn).linear = {0, 0, -30};

    all_entities[total_entities++] =
        makeDynObject(ctx,
                      {-0.5, -0.5, 1},
                      (Quat::angleAxis(toRadians(-90), {1, 0, 0 }) *
                          Quat::angleAxis(pi, {0, 1, 0 })).normalize(),
                      6,
                      ResponseType::Static, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    const Quat agent_rot =
        Quat::angleAxis(toRadians(-45), {0, 0, 1});

    ctx.data().numObstacles = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    if (ctx.data().enableBatchRender) {
        ctx.get<render::BatchRenderCamera>(agent) =
            render::BatchRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                                    up * 0.5f, 0);
    }
    if (ctx.data().enableViewer) {
        ctx.get<viz::VizCamera>(agent) =
            viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                                               up * 0.5f, 0);
    }
    ctx.get<Position>(agent) = Vector3 { -5, -5, 0.5 };
    ctx.get<Rotation>(agent) = agent_rot;
}

static void generateDebugEnvironment(Engine &ctx, CountT level_id)
{
    switch (level_id) {
    case 2: {
        level2(ctx);
    } break;
    case 3: {
        level3(ctx);
    } break;
    case 4: {
        level4(ctx);
    } break;
    case 5: {
        level5(ctx);
    } break;
    case 6: {
        level6(ctx);
    } break;
    case 7: {
        level7(ctx);
    } break;
    case 8: {
        level8(ctx);
    } break;
    }
}

}
