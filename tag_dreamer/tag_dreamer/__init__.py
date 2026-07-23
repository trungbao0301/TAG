from gym.envs.registration import register


register(
    id="tag-ros-v0",
    entry_point="tag_dreamer.env_tcp:TagGym",
)

register(
    id="tag-ros-shaped-v0",
    entry_point="tag_dreamer.env_tcp_shaped:TagGym",
)
