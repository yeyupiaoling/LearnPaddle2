import numpy as np
import paddle.fluid as fluid
import random
import gym
from collections import deque


# 定义一个深度神经网络
def QNetWork(ipt):
    fc1 = fluid.layers.fc(input=ipt, size=1024, act='relu')
    fc3 = fluid.layers.fc(input=fc1, size=4096, act='relu')
    out = fluid.layers.fc(input=fc3, size=2)
    return out


# 定义输入数据
state_data = fluid.layers.data(name='state', shape=[4], dtype='float32')
action_data = fluid.layers.data(name='action', shape=[1], dtype='int64')
reward_data = fluid.layers.data(name='reward', shape=[], dtype='float32')
next_state_data = fluid.layers.data(name='next_state', shape=[4], dtype='float32')
done_data = fluid.layers.data(name='done', shape=[], dtype='bool')

# 定义训练的参数
batch_size = 32
num_episodes = 350
num_exploration_episodes = 100
max_len_episode = 1000
learning_rate = 1e-3
gamma = 1.0
initial_epsilon = 1.0
final_epsilon = 0.01

# 实例化一个游戏环境，参数为游戏名称
env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=10000)

# 获取网络
state_model = QNetWork(state_data)

# 克隆预测程序
predict_program = fluid.default_main_program().clone()

action_onehot = fluid.layers.one_hot(action_data, 2)
action_onehot = fluid.layers.cast(action_onehot, dtype='float32')
pred_action_value = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(action_onehot, state_model), dim=1)

targetQ_predict_value = QNetWork(next_state_data)
best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
target = reward_data + (gamma - best_v) * (1.0 - fluid.layers.cast(done_data, dtype='float32'))

# 定义损失函数
cost = fluid.layers.square_error_cost(pred_action_value, target)
avg_cost = fluid.layers.reduce_mean(cost)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opt = optimizer.minimize(avg_cost)

# 创建执行器并进行初始化
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 开始玩游戏
for epsilon_id in range(num_episodes):
    # 初始化环境，获得初始状态
    state = env.reset()
    epsilon = max(initial_epsilon * (num_exploration_episodes - epsilon_id) /
                  num_exploration_episodes, final_epsilon)
    for t in range(max_len_episode):
        env.render()
        # epsilon-greedy 探索策略
        if random.random() < epsilon:
            # 以 epsilon 的概率选择随机下一步动作
            state = np.expand_dims(state, axis=0)
            action = env.action_space.sample()
        else:
            # 使用模型预测作为结果下一步动作
            state = np.expand_dims(state, axis=0)
            action = exe.run(predict_program,
                             feed={'state': state.astype('float32')},
                             fetch_list=[state_model])[0]
            action = np.squeeze(action, axis=0)
            action = np.argmax(action)

        # 让游戏执行动作，获得执行完 动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        next_state, reward, done, info = env.step(action)

        # 如果游戏结束，就进行惩罚
        reward = -5 if done else reward
        # 记录游戏输出的结果，作为之后训练的数据
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        # 如果游戏结束，就重新玩游戏
        if done:
            print('Pass: %d, score：%d' % (epsilon_id, t))
            break

        # 如果收集的数据大于Batch的大小，就开始训练
        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in zip(*random.sample(replay_buffer, batch_size))]

            batch_action = np.expand_dims(batch_action, -1)
            batch_next_state = np.expand_dims(batch_next_state, axis=1)

            # 执行训练
            exe.run(fluid.default_main_program(),
                    feed={'state': batch_state.astype('float32'),
                          'action': batch_action.astype('int64'),
                          'reward': batch_reward,
                          'next_state': batch_next_state.astype('float32'),
                          'done': batch_done})
