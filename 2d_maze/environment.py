import jax
import jax.numpy as jnp
import jax.ops as jops
from jax import jit
from functools import partial

class Mazes2d():
    # class attributes
    num_envs = 10
    num_rows = 10
    num_cols = 10
    
    # action[0]: going back to initial location
    actions = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.int32)
    observation_space = None
    
    mazes = None
    init_locs = None
    reward_locs = None
    

    def __init__(self, key, envs=10, rows=10, cols=10):

        self.num_envs = envs
        self.num_rows = rows
        self.num_cols = cols

        # not relevant in this application
        self.reward_locs = jnp.zeros((self.num_envs, 2), dtype=jnp.int32)
        #_, self.key = jax.random.split(self.key)
        self.reward_locs = jax.ops.index_update(self.reward_locs, jax.ops.index[:, 0], jax.random.randint(key, (1,), 1, self.num_rows+1))
        key, _ = jax.random.split(key)
        self.reward_locs = jax.ops.index_update(self.reward_locs, jax.ops.index[:, 1], jax.random.randint(key, (1,), 1, self.num_cols+1))

        
        self.mazes = jnp.ones((self.num_envs, self.num_rows+2, self.num_cols+2)) 
        # additional two rows and two cols as border, setup as follows
        self.mazes = jax.ops.index_update(self.mazes, jax.ops.index[:, :, 0], jnp.zeros(self.num_rows+2))
        self.mazes = jax.ops.index_update(self.mazes, jax.ops.index[:, :, self.num_cols+1], jnp.zeros(self.num_rows+2))
        self.mazes = jax.ops.index_update(self.mazes, jax.ops.index[:, 0, :], jnp.zeros(self.num_cols+2))
        self.mazes = jax.ops.index_update(self.mazes, jax.ops.index[:, self.num_rows+1, :], jnp.zeros(self.num_cols+2))
        # additional walls
        # |-------------|
        # |--------     |
        # |    ---------|
        # |--------     |
        # |-------------|
        wall_length = int(0.8*self.num_cols)
        self.mazes = jops.index_update(self.mazes, jops.index[:, int(0.2*self.num_rows)+1, 1:wall_length+1], jnp.zeros(wall_length))
        self.mazes = jops.index_update(self.mazes, jops.index[:, int(0.6*self.num_rows)+1, 1:wall_length+1], jnp.zeros(wall_length))
        self.mazes = jops.index_update(self.mazes, jops.index[:, int(0.4*self.num_rows)+1, self.num_cols+1-wall_length:self.num_cols+1], jnp.zeros(wall_length))
        self.mazes = jops.index_update(self.mazes, jops.index[:, int(0.8*self.num_rows)+1, self.num_cols+1-wall_length:self.num_cols+1], jnp.zeros(wall_length))

        # starting at the top left
        self.init_locs = jnp.ones((self.num_envs, 2), dtype=jnp.int32)


    @partial(jit, static_argnums=(0,), inline=True)
    def step(self, a, state):

        # update the states with actions
        state_new = jax.ops.index_add(state, jax.ops.index[:,0], self.actions[a,0])
        state_new = jax.ops.index_add(state_new, jax.ops.index[:,1], self.actions[a,1])
        
        # collision with borders or walls
        tmp = jax.ops.index_add(self.mazes, jax.ops.index[jnp.arange(self.num_envs), state_new[:,0], state_new[:,1]], 1)
        tmp = tmp.reshape((self.num_envs, -1))
        state_new = jnp.where((jnp.amax(tmp, axis=1)<1.5)[:,jnp.newaxis], state, state_new)
        
        # Death Zone computation
        in_zone = jnp.logical_or(state[:, 1] == 15, state[:, 1] == 15)
        in_zone = in_zone[:, None]
        wrong_act = jnp.logical_and(a != 1, a != 2)
        #wrong_act = jnp.logical_and(wrong_act, a != 0)
        #wrong_act = a == 0
        wrong_act = wrong_act[:, None]

        state_new = jnp.where(jnp.logical_and(in_zone, wrong_act),
                              jnp.ones(state_new.shape, jnp.int32),
                              state_new)
        

        # special action that takes the agents to initial locations 
        #state_new = jnp.where((a<0.5)[:,jnp.newaxis], self.init_locs, state_new)

        # not relevant for this application
        #done = (state_new[:,0]==self.reward_locs[:,0]) & (state_new[:,1]==self.reward_locs[:,1])
        #reward = jnp.where(done==True, 1, 0)

        return state_new, jnp.logical_and(in_zone, wrong_act)#, reward, done

    # not relevant for this application
    def init_state(self, key):
        state = jnp.zeros((self.num_envs, 2), dtype=jnp.int32)
        state = jax.ops.index_update(state, jax.ops.index[:, 0], jax.random.randint(key, (self.num_envs,), 1, self.num_rows+1))
        key, _ = jax.random.split(key)
        state = jax.ops.index_update(state, jax.ops.index[:, 1], jax.random.randint(key, (self.num_envs,), 1, self.num_cols+1))
        
        return state

    # not relevant for this application
    def restart(self, key, done, state):
        state_new = self.init_state(key)
        state_new = jnp.where((done==True)[:,jnp.newaxis], state_new, state)
          
        return state_new