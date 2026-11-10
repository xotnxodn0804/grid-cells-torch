--[[ Paper's "square arena" open-field task (Fig. 2b,c; Methods "Custom environment: square
arena"). No public source -- reconstructed here from the Methods text, which the RL-agent
roadmap plan flags as a substantial task in its own right, not glue code around an existing
level.

Methods, verbatim spec this implements:
  "This comprised a 10x10 square arena, corresponding to a 2.5x2.5m arena... The arena
  contained a single, randomly changed, intra-maze cue whose position and colour changed each
  episode, as did the texture of the floor, the texture of the walls and the goal location...
  The agent always started at the central 6x6 grids (that is, 1.5x1.5m) of the environment...
  Upon reaching the goal the agent was teleported to another random location and continued to
  navigate with the aim of maximising the number of times it reached the goal before the
  episode ended."
  Episode length: Supplementary Results 1a states 5,400 steps (~90s) per episode.

GOAL POSITION IS FIXED FOR THE WHOLE EPISODE (2026-08-14 fix). "cue ... changed each episode, as
did [floor texture], [wall texture] and [the goal location]" is a list of things that all change
on the SAME cadence -- once per 90s episode, not once per goal-touch. This is the paper's Morris
water maze framing: the agent must relearn a path back to one fixed hidden goal from repeated
random starting points within an episode, not chase a freshly-relocated goal every touch. Only
the agent's own spawn point re-randomizes on every touch ("teleported to another random
location"). An earlier version of this file re-randomized the goal position (and, worse,
recompiled the entire map via q3map2) on every touch -- both a task-design bug and, with 32
concurrent RL actors, the dominant cause of a >14x training-throughput collapse (see the
RL-agent roadmap plan's "Throughput fix" section). Fixed by splitting `api:start()` (true
episode boundary: pick new goal/cue positions + a new floor/wall texture, full map rebuild) from
`api:nextMap()` (may fire many times per episode, on every goal touch: only the agent's spawn
point changes, via a `''`-mapName "fast restart" -- see
`lab/game_scripts/factories/random_goal_factory.lua` for the same pattern -- so no recompile).

Deliberately NOT `require`-ing a shared factory module: the paper's task (single unmarked goal,
no scattered reward pickups, a hard central-6x6 spawn constraint) doesn't match
`factories.random_goal_factory`'s maze-topology/scattered-fruit assumptions closely enough for a
thin wrapper, and any project-owned `rl/factories/*.lua` file couldn't be `require`d from here
anyway -- `require` inside an externally-loaded level resolves against the `lab/` submodule's
own game_scripts root, not this level's own directory (confirmed empirically via
`rl/smoke_test.py`). So this file is self-contained, `require`ing only submodule modules.

Intra-maze cue (2026-08-28 fix): now a wall-mounted texture, not a floating prop -- Fig. 2b's
actual picture shows a bright patch flush with the wall, and Methods says the cue "changed on
each episode" in colour and position, same cadence as floor/wall texture and goal location.
Implemented via the engine's `theme:placeWallDecals` callback (see
`deepmind/engine/lua_text_level_maker.cc`'s header comment for the exact contract): the engine
hands the callback every valid wall-art location on the compiled map, and the callback -- using
the same per-episode-seeded `randomMap` RNG as the goal/floor/wall randomization above -- picks
one at random and paints it with one of six distinct colours (`CUE_DECAL_PALETTE`, below) drawn
from `game_scripts/themes/texture_sets.lua`'s existing colour-tinted wall textures. Because
`placeWallDecals` only fires when `make_map.makeMap` actually recompiles the map -- i.e. only on
the `_needFullRebuild` true-episode-boundary path, never on the fast-restart path -- the cue
naturally changes once per episode and not on every mid-episode goal touch, matching the
floor/wall texture's own cadence for free. `decalFrequency = 0` is passed to
`themes.fromTextureSet` to suppress its own random decorative wall decals (which would otherwise
add unrelated visual noise indistinguishable from the one reward-relevant cue).
Unverified against an actual DMLab build/render (this workspace has no dmlab_module.so built --
see `grid-cells-torch/README.md`'s RL environment setup) -- the decal texture's UV fit at a wall
face has only been checked by reading `lua_text_level_maker.cc`'s C++ side, not by looking at
rendered output.

Distal cues ("buildings rendered at infinity"): CONFIRMED already implemented, no changes
needed. An earlier version of this file's header claimed no distant-buildings skybox asset
existed anywhere in `lab/`, based only on grepping asset filenames (`lg_sky_01/02/03`,
`desert_day/night_sky` -- none of which *sound* like buildings). That check never opened the
actual texture images and was wrong: `lg_sky_01`, `lg_sky_02` and `lg_sky_03` (converted from
.tga and inspected directly, 2026-08-28) are all grey high-rise skyline silhouettes against a
blue sky -- exactly the paper's "buildings ... rendered at infinity" distal cue, and exactly
what a live DMLab render shows (confirmed against a user-provided screenshot of this same
skybox). This file already sets `useSkybox = true`, which `make_map.lua` wires to the constant
`SKYBOX_TEXTURE_NAME = 'map/lab_games/sky/lg_sky_03'` -- unconditionally, not re-randomized per
episode. A skybox is by construction rendered at infinite distance (always centred on the
camera, so it never exhibits parallax as the agent moves) and this file never varies which one
is loaded, so both of the paper's requirements -- "directional but not distance information" and
"consistent across episodes" -- are already satisfied with no code change.
]]

local tensor = require 'dmlab.system.tensor'
local custom_observations = require 'decorators.custom_observations'
local debug_observations = require 'decorators.debug_observations'
local make_map = require 'common.make_map'
local map_maker = require 'dmlab.system.map_maker'
local pickups = require 'common.pickups'
local random = require 'common.random'
local themes = require 'themes.themes'
local texture_sets = require 'themes.texture_sets'
local timeout = require 'decorators.timeout'
local randomMap = random(map_maker:randomGen())

-- 10x10 interior grid = the paper's "10x10 square arena"; each cell is 0.25m (2.5m / 10).
local GRID = 10
-- "central 6x6 grids": cells 3..8 (inclusive, 1-indexed) of the 10-wide interior.
local CENTER_MIN, CENTER_MAX = 3, 8
local EPISODE_LENGTH_SECONDS = 90  -- Supplementary Results 1a: 5,400 steps @ 60fps

-- Intra-maze cue palette: six visually distinct colours, one texture per colour, drawn from
-- MISHMASH's own existing wall textures (see header note) rather than inventing new assets.
-- "_bright" variants preferred for salience against the base wall texture.
local CUE_DECAL_PALETTE = {
    {tex = 'map/lab_games/lg_style_01_wall_green_bright'},
    {tex = 'map/lab_games/lg_style_01_wall_red_bright'},
    {tex = 'map/lab_games/lg_style_02_wall_yellow_bright'},
    {tex = 'map/lab_games/lg_style_02_wall_blue_bright'},
    {tex = 'map/lab_games/lg_style_03_wall_orange_bright'},
    {tex = 'map/lab_games/lg_style_03_wall_gray_bright'},
}

-- Builds a bordered GRIDxGRID entity layer with a goal cell and a player-start cell (the
-- default text-level action for 'P', per docs/developers/creating_levels/text_level.md --
-- without a 'P' character, make_map never creates an info_player_start entity at all, and
-- `updateSpawnVars` below -- which only overrides existing spawnVars -- has nothing to act on),
-- plus a variation layer that's uniformly non-default so `themes.lua`'s per-call
-- `randomMap:choice` path fires (the unnamed "default" variation always picks texture index
-- [1], never randomizing -- confirmed by reading themes.lua).
local function buildLayers(goalRow, goalCol, spawnRow, spawnCol)
  local n = GRID + 2  -- +1-cell wall border on each side
  local entityRows, variationRows = {}, {}
  for row = 1, n do
    local eChars, vChars = {}, {}
    for col = 1, n do
      local isWall = (row == 1 or row == n or col == 1 or col == n)
      local isGoal = (not isWall) and (row - 1 == goalRow) and (col - 1 == goalCol)
      local isSpawn = (not isWall) and (row - 1 == spawnRow) and (col - 1 == spawnCol)
      eChars[col] = isWall and '*' or (isGoal and 'G' or (isSpawn and 'P' or ' '))
      vChars[col] = isWall and ' ' or 'A'
    end
    entityRows[row] = table.concat(eChars)
    variationRows[row] = table.concat(vChars)
  end
  return table.concat(entityRows, '\n') .. '\n', table.concat(variationRows, '\n') .. '\n'
end

-- Grid cell (1-indexed, interior-relative) -> world-unit origin string "x y z".
-- 100 world units per cell, matching make_map's own text-level convention; +1 for the border.
local function cellOrigin(row, col, z)
  return string.format('%d %d %d', col * 100 + 50, row * 100 + 50, z or 30)
end

local tensor = require 'dmlab.system.tensor'

local api = {}

-- Level setting `goalVisible` (Lab(config={goalVisible='false'})): the paper's goal is
-- "unmarked" (Methods p.6), but the text-level 'G' cell spawns DMLab's visible `goal` pickup.
-- With goalVisible=false the pickup keeps its GOAL type, reward and collision box (pickup
-- collision is the engine's fixed item bounding box, not the model) but is drawn with a
-- zero-area model, so the agent can only find it by touching it. Default true (the original
-- behaviour) until the switch is decided -- see docs/paper/deviations.md, item I.
api._goalVisible = true

function api:init(params)
  if params.goalVisible ~= nil then
    api._goalVisible = params.goalVisible ~= 'false'
  end
end

local INVISIBLE_MODEL = 'invisible_goal_model'

function api:createModel(modelName)
  if modelName == INVISIBLE_MODEL then
    return {
        surfaces = {
            empty = {
                vertices = tensor.FloatTensor{
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                },
                indices = tensor.Int32Tensor{{1, 2, 3}},
                shaderName = 'textures/model/beam_d',
            },
        },
    }
  end
end

-- True episode boundary (~90s, per timeout.decorate below): the goal/cue positions and the
-- floor/wall texture all change here, together, matching Methods' "changed each episode" list
-- -- and nowhere else. `nextMap()` (which also fires on every goal touch, many times per
-- episode) reads and consumes `_needFullRebuild` to know whether *this* call is that boundary
-- or just an in-episode agent-relocation.
function api:start(episode, seed)
  random:seed(seed)
  randomMap:seed(random:mapGenerationSeed())
  api._mapCount = 0
  api._goalRow = random:uniformInt(1, GRID)
  api._goalCol = random:uniformInt(1, GRID)
  api._needFullRebuild = true
  -- DEBUG.CAMERA.TOP_DOWN defaults to a fixed camera at world origin (0,0,500) -- fine for
  -- maze_gen-based levels, which call debug_observations.setMaze() to recenter it, but this
  -- level builds via make_map.makeMap{} directly, so that never fires and the camera is left
  -- pointed at whatever happens to sit near the origin, not this arena (centered at
  -- ((GRID+2)*100/2, same) in world units, per cellOrigin's cell->world convention). Recenter
  -- it explicitly; look angle is left at its default ({90,90,0}, already straight down).
  local center = (GRID + 2) * 100 / 2
  debug_observations.setCameraPos{center, center, 1200}
end

function api:nextMap()
  -- The agent's own spawn point re-randomizes on every call (including every goal touch --
  -- Methods: "Upon reaching the goal the agent was teleported to another random location").
  -- Resample until it doesn't land on the fixed goal cell.
  repeat
    api._spawnRow = random:uniformInt(CENTER_MIN, CENTER_MAX)
    api._spawnCol = random:uniformInt(CENTER_MIN, CENTER_MAX)
  until api._spawnRow ~= api._goalRow or api._spawnCol ~= api._goalCol

  if not api._needFullRebuild then
    -- Fast restart (goal touch mid-episode): same compiled map, same texture, same goal --
    -- only `updateSpawnVars` below repositions the player. No q3map2 recompile. This is the
    -- fix for the throughput collapse: see this file's header note.
    return ''
  end
  api._needFullRebuild = false

  api._mapCount = api._mapCount + 1
  local entityLayer, variationLayer = buildLayers(
      api._goalRow, api._goalCol, api._spawnRow, api._spawnCol)
  -- Built explicitly (not left to make_map.makeMap's own default) so a custom
  -- placeWallDecals can be attached below for the intra-maze cue -- see header note.
  -- decalFrequency = 0 suppresses fromTextureSet's own random decorative wall decals; without
  -- this it would also create a theme:placeWallDecals that scatters unrelated decorative
  -- images, which the engine (deepmind/engine/lua_text_level_maker.cc) only lets one function
  -- per theme own, so ours would silently replace/conflict with it otherwise.
  local theme = themes.fromTextureSet{
      textureSet = texture_sets.MISHMASH,
      decalFrequency = 0,
  }
  -- One cue per real episode, at a random valid wall location, in a random colour -- both
  -- drawn from the same per-episode-seeded `randomMap` RNG the goal/floor/wall randomization
  -- above uses, so re-running with the same seed reproduces the same cue. `wallLocs` is every
  -- wall-art location on this compiled map (see lua_text_level_maker.cc's contract comment);
  -- picking one at random needs no knowledge of which cells border a wall on our part.
  function theme:placeWallDecals(wallLocs)
    if #wallLocs == 0 then return {} end
    local index = randomMap:uniformInt(1, #wallLocs)
    local colour = randomMap:choice(CUE_DECAL_PALETTE)
    return {{index = index, decal = colour}}
  end
  return make_map.makeMap{
      mapName = 'square_arena_' .. api._mapCount,
      mapEntityLayer = entityLayer,
      mapVariationsLayer = variationLayer,
      theme = theme,
      useSkybox = true,
  }
end

function api:createPickup(className)
  local pickup = pickups.defaults[className]
  if className == 'goal' and not api._goalVisible then
    pickup = {name = pickup.name, classname = pickup.classname, model = INVISIBLE_MODEL,
              quantity = pickup.quantity, type = pickup.type}
  end
  return pickup
end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == 'info_player_start' then
    local x = api._spawnCol * 100 + random:uniformReal(0, 100)
    local y = api._spawnRow * 100 + random:uniformReal(0, 100)
    spawnVars.origin = string.format('%d %d %d', x, y, 40)
    spawnVars.randomAngleRange = '180'
  end
  return spawnVars
end

-- Debug-only exact goal-cell centre, in metres. DO NOT feed to the policy
-- before first goal arrival. Original square_arena.lua remains unchanged.
function api:customObservationSpec()
  return {{name='ORACLE.GOAL.XY', type='Doubles', shape={2}}}
end

function api:customObservation(name)
  if name == 'ORACLE.GOAL.XY' then
    -- DMLab arena: grid cells 100 world units, 10 cells across 2.5m.
    -- Grid cell centres in metres lie at -1.125,-0.875,...,1.125.
    return tensor.DoubleTensor{
      (api._goalCol - (GRID + 1) / 2) * (2.5 / GRID),
      (api._goalRow - (GRID + 1) / 2) * (2.5 / GRID)}
  end
  return nil
end

timeout.decorate(api, EPISODE_LENGTH_SECONDS)
custom_observations.decorate(api)

return api
