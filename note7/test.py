import numpy as np
import paddle.fluid as fluid
import random
import gym
from collections import deque


def QNetWork(ipt):
    fc1 = fluid.layers.fc(input=ipt, size=1024, act='relu')
    fc2 = fluid.layers.fc(input=fc1, size=4096, act='relu')
    fc3 = fluid.layers.fc(input=fc2, size=4096, act='relu')
    out = fluid.layers.fc(input=fc3, size=2)
    return out


state_data = fluid.layers.data(name='state', shape=[4], dtype='float32')
action_data = fluid.layers.data(name='action', shape=[1], dtype='int64')
reward_data = fluid.layers.data(name='reward', shape=[], dtype='float32')
next_state_data = fluid.layers.data(name='next_state', shape=[4], dtype='float32')
done_data = fluid.layers.data(name='done', shape=[], dtype='bool')

batch_size = 32
env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=10000)
state_model = QNetWork(state_data)

predict_program = fluid.default_main_program().clone()

action_onehot = fluid.layers.one_hot(action_data, 2)
action_onehot = fluid.layers.cast(action_onehot, dtype='float32')
pred_action_value = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(action_onehot, state_model), dim=1)

targetQ_predict_value = QNetWork(next_state_data)
best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
target = reward_data + (1.0 - best_v) * (1.0 - fluid.layers.cast(done_data, dtype='float32'))

cost = fluid.layers.square_error_cost(pred_action_value, target)
avg_cost = fluid.layers.reduce_mean(cost)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opt = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for pass_id in range(500):
    state = env.reset()
    for t in range(490):
        #         env.render()
        if random.random() < 0.01:
            state = np.expand_dims(state, axis=0)
            action = env.action_space.sample()
        else:
            state = np.expand_dims(state, axis=0)
            action = exe.run(predict_program,
                             feed={'state': state.astype('float32')},
                             fetch_list=[state_model])[0]
            action = np.squeeze(action, axis=0)
            action = np.argmax(action)

        next_state, reward, done, info = env.step(action)

        reward = -5 if done else reward
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print('分数：', t)
            break

        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in zip(*random.sample(replay_buffer, batch_size))]

            batch_action = np.expand_dims(batch_action, -1)
            batch_next_state = np.expand_dims(batch_next_state, axis=1)

            exe.run(fluid.default_main_program(),
                    feed={'state': batch_state.astype('float32'),
                          'action': batch_action.astype('int64'),
                          'reward': batch_reward,
                          'next_state': batch_next_state.astype('float32'),
                          'done': batch_done})
