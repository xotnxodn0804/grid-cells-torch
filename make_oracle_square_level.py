#!/usr/bin/env python3
"""Make a separate oracle-enabled copy of square_arena.lua without editing the original."""
from pathlib import Path

src = Path('levels/square_arena.lua')
dst = Path('levels/square_arena_oracle.lua')
if not src.is_file():
    raise SystemExit('Run this from ~/grid-cells-torch; missing '+str(src))
if dst.exists():
    raise SystemExit(f'{dst} already exists. Refusing to overwrite it.')
text = src.read_text(encoding='utf-8')
a = "local custom_observations = require 'decorators.custom_observations'"
if text.count(a) != 1:
    raise SystemExit('Unexpected square_arena.lua structure: decorator import not unique')
text = text.replace(a, "local tensor = require 'dmlab.system.tensor'\n"+a, 1)
needle = 'timeout.decorate(api, EPISODE_LENGTH_SECONDS)'
if text.count(needle)!=1:
    raise SystemExit('Unexpected square_arena.lua structure: timeout call not unique')
insert = '''-- Debug-only exact goal-cell centre, in metres. DO NOT feed to the policy
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

'''
text = text.replace(needle, insert+needle, 1)
dst.write_text(text,encoding='utf-8')
print('Created',dst,'(source preserved)')
