# CS336 Spring 2025 Assignment 5: Alignment

## 对齐与强化学习

### 1.1任务定义与基准测试
基础预训练（pre-train）：构建语言的底层概率分别，让模型学习语法、常识、推理雏形

持续预训练(continuous-pre-train): 垂类领域的知识灌输

后训练(Post-training):

(1) 监督微调（SFT：supervised Fine-Tuning）:以“指令-回答”对（（q,a）pairs）让模型模仿标准答案--泛化能力差 （根据指令回答，好用）

(2) 强化学习：RLHF（Reinforcement Learning from Human Feekback） 与PPO（可控：符合人类偏好）

    流程：采用与排序 -> 奖励建模 -> 策略优化(PPO)
    局限性：PPO工程及其复杂，显存开销极大，奖励函数不稳定

(3)直接偏好微调(DPO,Direct Preference Optimization)

    训练数据格式：(prompt, good_answer, bad_answer)
    特点：DPO不用奖励模型，直接利用数学转化，在好答案和坏答案的对比上进行优化
    优势：将RL问题简化为分类函数损失优化，提升训练的稳定性和效率
    局限性：在具有“绝对对错”的问题中，只看相对好坏，不看客观真理，且在没有在线采用的过程，无法有效激发模型的自主搜索能力

所以对于推理任务，需要“在线强化学习”

(4)推理强化学习（Reasoning RL/GRPO）

    训练数学格式:(q,a)
    定义：针对有客观标准的问题，不依赖人工排序，而是直接利用结果对错作为奖励
    GRPO：通过组里相对打分取消了Critic网络

总结

Post-train:通过模仿/反馈，把pre-train中习得的知识压缩到人类正确的表达路径上

| 阶段                | 当前缺失                     | 解决方案         | 算法局限性                         |
| ------------------- | ---------------------------- | ---------------- | ---------------------------------- |
| Pre-train           | 语言基本规律                 | 预测下一个词     | 专业逻辑知识匮乏                   |
| Continue Pre-train  | 专业知识密度                 | 垂直语料训练     | 不懂指令，不会交流                 |
| Post-train: SFT     | 格式与指令意识               | 模仿学习         | 仅表面模仿，逻辑死板               |
| Post-train: PPO     | RLHF复杂度高             | 奖励模型+RL      | 极度不稳定，显存吞噬者             |
| Post-train: DPO     | 无法量化多维偏好              | 离线对比学习     | 缺乏自主探索推理路径的能力         |
| Post-train: GRPO    | 推理准确率瓶颈               | 在线验证奖励     | 仅适用于有客观标准的任务           |
### 1.2提示词模板和规则奖励函数
#### 1.2.1 结构化推理：特殊的标签设计

1.关键标签定义：

    ·<think>   :思维链开始
    ·</think>  :思维链结束
    ·<answer>  :结果开始
    ·</answer> :结果结束

#### 1.2.2.Prompt模板解析

#### 1.2.3.规则奖励函数
1.奖励逻辑的双重判定
    
    1.格式奖励：模型严格遵守</think><answer> </answer>
    2.答案奖励

2.数学等价性的判断难点

    归一化(Normalization)      ： mathd_normalize_answer 去除latex噪音，多余空格和单位
    符号计算(symbolic Equality)： symbolic_equal 利用sympy库--如果A-B=0代数上成立，认为答案等价
    数值近似(Numeric Equality):  isclose 函数，允许误差

### 1.3vllm与基线测试
```
理解vllm的核心优势
部署推理服务器
实现并发和评估流水线
理解基准测试逻辑
```
#### 1.3.1 why vllm？

    PagedAttention：提高显存利用率
    Continuous Batching： 动态地将请求合并为批次，提高吞吐
    OpenAI兼容接口

#### 1.3.2 vLLM的部署与配置


