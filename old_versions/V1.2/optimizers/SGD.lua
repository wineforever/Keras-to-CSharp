-- 类库中默认提供的优化器接口参数有:
-- learning_rate : 学习率
-- error : 当前神经元误差
-- y : 当前神经元输出
-- d_activation : 激励器导数
-- loss : 损失值
-- rand_error : 随机样本误差
-- SGD : 随机梯度下降值

d_w = rand_error * d_activation * learning_rate