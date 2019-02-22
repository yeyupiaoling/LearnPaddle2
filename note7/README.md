@[TOC]

# 前言
本章介绍使用PaddlePaddle实现强化学习，通过自我学习，完成一个经典控制类的游戏，相关游戏介绍可以在[Gym官网](https://gym.openai.com/envs/#classic_control)上了解。我们这次玩的是一个`CartPole-v1`游戏，操作就是通过控制滑块的左右移动，不让竖着的柱子掉下来。利用强化学习的方法，不断自我学习，通过在玩游戏的过程中获取到奖励或者惩罚，学习到一个模型。在王者荣耀中的超强人机使用的AI技术也类似这样。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190131173040228.gif)

# PaddlePaddle程序
创建一个`DQN.py`的Python文件。导入项目所需的依赖库，如果还没安装gym的话，可以通过命令`pip3 install gym`安装。
```python
import numpy as np
import paddle.fluid as fluid
import random
import gym
from collections import deque
from paddle.fluid.param_attr import ParamAttr
```

定义一个简单的网络，这个网络只是由4个全连接层组成，并为每个全连接层指定参数的名称。指定参数的作用是为了之后更新模型参数使用的，因为之后会通过这个网络生成两个模型，而且没有模型参数更新不一样。
```python
# 定义一个深度神经网络，通过指定参数名称，用于之后更新指定的网络参数
def DQNetWork(ipt, variable_field):
    fc1 = fluid.layers.fc(input=ipt,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc1'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc1_b'.format(variable_field)))
    fc2 = fluid.layers.fc(input=fc1,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc2'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc2_b'.format(variable_field)))
    out = fluid.layers.fc(input=fc2,
                          size=2,
                          param_attr=ParamAttr(name='{}_fc3'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc3_b'.format(variable_field)))
    return out
```

定义一个更新参数的函数，这个函数是通过指定的参数名称，通过修剪参数的方式完成模型更新。
```python
# 定义更新参数程序
def _build_sync_target_network():
    # 获取所有的参数
    vars = list(fluid.default_main_program().list_vars())
    # 把两个网络的参数分别过滤出来
    policy_vars = list(filter(lambda x: 'GRAD' not in x.name and 'policy' in x.name, vars))
    target_vars = list(filter(lambda x: 'GRAD' not in x.name and 'target' in x.name, vars))
    policy_vars.sort(key=lambda x: x.name)
    target_vars.sort(key=lambda x: x.name)

    # 从主程序中克隆一个程序用于更新参数
    sync_program = fluid.default_main_program().clone()
    with fluid.program_guard(sync_program):
        sync_ops = []
        for i, var in enumerate(policy_vars):
            sync_op = fluid.layers.assign(policy_vars[i], target_vars[i])
            sync_ops.append(sync_op)
    # 修剪第二个玩了个的参数，完成更新参数
    sync_program = sync_program._prune(sync_ops)
    return sync_program
```

定义5个数据输出层，`state_data`是当前游戏状态的数据输入层，`action_data`是对游戏操作动作的数据输入层，只有两个动作0和1，`reward_data`是当前游戏给出的奖励的数据输入层，`next_state_data`是游戏下一个状态的数据输入层，`done_data`是游戏是否结束的数据输入层。
```python
# 定义输入数据
state_data = fluid.layers.data(name='state', shape=[4], dtype='float32')
action_data = fluid.layers.data(name='action', shape=[1], dtype='int64')
reward_data = fluid.layers.data(name='reward', shape=[], dtype='float32')
next_state_data = fluid.layers.data(name='next_state', shape=[4], dtype='float32')
done_data = fluid.layers.data(name='done', shape=[], dtype='float32') 
```

定义一些必要的训练参数，比如epsilon-greedy 探索策略参数。
```python
# 定义训练的参数
batch_size = 32
num_episodes = 300
num_exploration_episodes = 100
max_len_episode = 1000
learning_rate = 1e-3
gamma = 1.0
initial_epsilon = 1.0
final_epsilon = 0.01
```

创建一个游戏，通过指定游戏的名称`CartPole-v1`就可以获取前言部分所说的游戏。也可以创建其他更多的有些，具体可以参照官方的游戏名称。
```python
# 实例化一个游戏环境，参数为游戏名称
env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=10000)
```

获取第一个网络模型，并指定参数名称内包含`policy`字符串。
```python
# 获取网络
state_model = DQNetWork(state_data, 'policy')
```

这里从主程序中克隆一个预测程序，这个预测程序是之后预测游戏的下一个动作的，也就是说它在操作游戏。
```python
# 克隆预测程序
predict_program = fluid.default_main_program().clone()
```

这里定义损失函数，强化学习中的损失函数跟之后我们使用的损失函数有点不一样。虽然最终还是使用平方差损失函数，但是输入的不只是普通的输入数据和标签。
```python
# 定义损失函数
action_onehot = fluid.layers.one_hot(action_data, 2)
action_value = fluid.layers.elementwise_mul(action_onehot, state_model)
pred_action_value = fluid.layers.reduce_sum(action_value, dim=1)

targetQ_predict_value = DQNetWork(next_state_data, 'target')
best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
best_v.stop_gradient = True
target = reward_data + gamma * best_v * (1.0 - done_data)

cost = fluid.layers.square_error_cost(pred_action_value, target)
avg_cost = fluid.layers.reduce_mean(cost)
```

这里获取一个更新参数的程序，用于之后执行更新参数。
```python
# 获取更新参数程序
_sync_program = _build_sync_target_network()
```

定义一个优化方法，这里还是用AdamOptimizer，笔者也是比较喜欢使用这个优化方法。
```python
# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-3)
opt = optimizer.minimize(avg_cost)
```

开始创建执行器
```python
# 创建执行器并进行初始化
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
epsilon = initial_epsilon
```

这个循环有点大，不过因为是一个整体，不好拆分出来，所以就一起介绍吧。

 - 在每次循环开始，就开始获取游戏的状态，这个是游戏结束之后再执行的。
 - 定义一个epsilon-greedy探索策略，这个策略是根据训练的进度，开始选择自动操作的动作或者是模型预测的动作的概率。
 - 接下来就是一局游戏的的循环，在这里可以显示游戏的界面
 - 下面就是通过使用epsilon-greedy探索策略，决定使用随机生成动作，还是预测生成动作，使用随机动作可以增加数据的多样性，通过使用模型预测就是让模型根据当前的游戏状态来预测下一步动作是怎么才是正确的，随着模型的不断训练，这个预测也是越来越正确。
 - 然后更加随机生成的动作，或者模型预测的动作，传递个游戏，得到游戏的相关输出，比如游戏的下一个状态，游戏的奖励，游戏是否结束等等。
 - 如果游戏结束了，应当给游戏一个负奖励，惩罚它做出了一个错误的操作。
 - 然后把这些数据存储起来，用于之后训练使用。
 - 当存储的数据大于或等于Batch size，就可以开始训练。
```python
update_num = 0
# 开始玩游戏
for epsilon_id in range(num_episodes):
    # 初始化环境，获得初始状态
    state = env.reset()
    epsilon = max(initial_epsilon * (num_exploration_episodes - epsilon_id) /
                  num_exploration_episodes, final_epsilon)
    for t in range(max_len_episode):
        # 显示游戏界面
        # env.render()
        state = np.expand_dims(state, axis=0)
        # epsilon-greedy 探索策略
        if random.random() < epsilon:
            # 以 epsilon 的概率选择随机下一步动作
            action = env.action_space.sample()
        else:
            # 使用模型预测作为结果下一步动作
            action = exe.run(predict_program,
                             feed={'state': state.astype('float32')},
                             fetch_list=[state_model])[0]
            action = np.squeeze(action, axis=0)
            action = np.argmax(action)

        # 让游戏执行动作，获得执行完 动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        next_state, reward, done, info = env.step(action)

        # 如果游戏结束，就进行惩罚
        reward = -10 if done else reward
        # 记录游戏输出的结果，作为之后训练的数据
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        # 如果游戏结束，就重新玩游戏
        if done:
            print('Pass:%d, epsilon:%f, score:%d' % (epsilon_id, epsilon, t))
            break

        # 如果收集的数据大于Batch的大小，就开始训练
        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [np.array(a, np.float32) for a in zip(*random.sample(replay_buffer, batch_size))]

            # 更新参数
            if update_num % 200 == 0:
                exe.run(program=_sync_program)
            update_num += 1

            # 调整数据维度
            batch_action = np.expand_dims(batch_action, axis=-1)
            batch_next_state = np.expand_dims(batch_next_state, axis=1)

            # 执行训练
            exe.run(program=fluid.default_main_program(),
                    feed={'state': batch_state,
                          'action': batch_action.astype('int64'),
                          'reward': batch_reward,
                          'next_state': batch_next_state,
                          'done': batch_done})
```

输出训练信息：
```
......
Pass:70, epsilon:0.300000, score:234
Pass:71, epsilon:0.290000, score:272
Pass:72, epsilon:0.280000, score:254
Pass:73, epsilon:0.270000, score:148
Pass:74, epsilon:0.260000, score:147
Pass:75, epsilon:0.250000, score:342
Pass:76, epsilon:0.240000, score:295
Pass:77, epsilon:0.230000, score:290
Pass:78, epsilon:0.220000, score:276
Pass:79, epsilon:0.210000, score:279
......
```

关于通过使用PaddlePaddle实现强化学习，并玩一个小游戏就介绍完成了。强化学习还有很多好玩的地方，比如应用于机器人的避障等一些智能控制上。

同步到百度AI Studio平台：http://aistudio.baidu.com/aistudio/#/projectdetail/31310
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5c3eaac54223d9002bfef5ae
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note7

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. https://github.com/PaddlePaddle/models/blob/develop/fluid/DeepQNetwork/README_cn.md
2. https://github.com/snowkylin/TensorFlow-cn
