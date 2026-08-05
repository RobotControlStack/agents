## v0.3.0 (2026-08-05)

### Feat

- resizing now happens transperantly in the transport layer

### Fix

- goal image handling

### Refactor

- **envs**: factory with from_id, gym with make_gym
- register env into init
- fix import paths
- **envs**: split into seperate envs
- move envs interface into envs folder
- **policies**: extracted into own files
- deleted unused code
- env interface includes action chunking
- adapt policies to the new act/obs interface
- new observation and action interface

## v0.2.0 (2026-06-20)

### Feat

- high res cameras in videos
- **client**: allow reconnection of the client
- parallelize over envs
- multi processing over envs
- xvla and remap key
- video as mp4
- add duobench envs
- add config to wandb
- **lerobot**: correct resizing and option for temporal ensemble
- **rcs**: adapt multi robot space
- **policies**: added support for lerobot policies

### Fix

- fan out for same eval id
- removed multi processing on single server level
- env reset in jpeg case
- multiprocessing of single tasks
- respect n_action_steps for non act policies
- left over fixes
- change reset options default
- seeding by gym reset and video recording
- rcs duobench integration and lerobot policies

### Refactor

- remove xvla import and return chunk

## v0.1.0 (2026-01-16)

### Feat

- shm and jpeg configurable
- libero fixes
- added support for libero
- add transparent jpeg support
- save videos
- **openpi**: configurable execution horizon for action chunks
- added openpi model
- python command/path to start eval envs in
- add rcs pick up env

### Fix

- **openpi**: gripper definition
- openpi checkpoint path
- client error log not shown
- wrong access to outdated cached envs
- usage of undefined variable
- agent config refactor leftovers and rcs reset
- agent config in write results function
- old obs dataclass format
- shared memory in dist models

### Refactor

- added agents config


- rename agents -> vlagents
