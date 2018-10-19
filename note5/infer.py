import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np

word_dict = imdb.word_dict()

# 是否使用GPU
place = fluid.CPUPlace()
# 生成调试器
exe = fluid.Executor(place)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model("model/", exe)
# 定义预测数据
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
# 把每个句子拆成一个个单词
reviews = [c.split() for c in reviews_str]

# 获取结束符号的标签
UNK = word_dict['<unk>']
# 获取每句话对应的标签
lod = []
for c in reviews:
    lod.append([word_dict.get(words, UNK) for words in c])

# 获取每句话的单词数量
base_shape = [[len(c) for c in lod]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

# 预测获取预测结果,因为输入的是3个数据，所以要模拟3个label的输入
results = exe.run(program=inference_program,
                  feed={feed_target_names[0]: tensor_words},
                  fetch_list=fetch_targets)

# 打印每句话的正负面概率
for i, r in enumerate(results[0]):
    print("\'%s\'的预测结果为：正面概率为：%0.5f，负面概率为：%0.5f\n" % (reviews_str[i], r[0], r[1]))
