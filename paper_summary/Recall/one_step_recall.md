# 召回新框架-突破双塔

传统的召回模型基于双塔匹配式网络，即Query/User侧特征和Item侧特征分别通过两个神经网络的到表征，再最小化两个表征之间的距离，线上检索时则采用ANN框架进行检索。这种方法的局限在于，User侧的Embedding表示和Item侧的Embedding表示，只能进行内积/余弦距离/欧氏距离这种简单的交互计算，而没有办法进行复杂的交互，从而限制模型的效果。

# 1 阿里TDM

阿里2018年的工作《Learning Tree-based Deep Model for Recommender Systems》提出了深度树匹配方法来突破双塔的后交互限制，从而提升召回的效果。

## 1.1 算法原理

TDM基本原理是使用树结构对全库item进行索引，然后训练深度模型以支持树上的逐层检索，从而将大规模推荐中全库检索的复杂度由O(n)（n为所有item的量级）下降至O(log n)。

## 1.2 模型架构

### 1.2.1 树结构索引

TDM索引采取树结构，树中的每一个叶节点对应库中的一个item；非叶节点表示item的集合。这样的一种层次化结构，体现了粒度从粗到细的item架构。此时，推荐任务转换成了如何从树中检索一系列叶节点，作为用户最感兴趣的item返回。

检索采用的是 Beam Search 的方法从根节点（root node）开始逐层挑选 Top K 节点，而挑选的依据正是用户对每个节点的偏好 ，然后将这些 Top K 节点的子节点作为下一层的候选节点，一直到最后一层。

### 1.2.2 判别器

深度网络在TDM框架中承担的是用户兴趣判别器的角色，其输出的（用户，节点）对的兴趣度，将被用于检索过程作为寻找每层Top K的评判指标。

上述网络结构中，在用户特征方面仅使用了用户历史行为，并对历史行为根据其发生时间，进行了时间窗口划分。在节点特征方面，使用的是节点经过embedding后的向量作为输入，借助DIN结构建模概率。

### 1.2.3 联合训练

树索引结构在TDM框架中起到了两方面的作用，一是在训练过程提供了上溯正采样样本和平层负采样样本；二是在检索过程中决定了选择与剪枝方案。TDM使用学习得到的叶节点（即item）embedding向量进行层次化聚类，来生成新的树索引结构。联合训练过程如下图所示：

# 2 字节DR

字节跳动 2020 年的工作《Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations》。

## 2.1 算法原理

阿里提出了基于树的检索算法 TDM/JTM。TDM/JTM 将索引建模成一棵树结构，候选集的每个 Item 为树中的叶子结点，并将模型的参数学习和树结构的参数学习结合起来训练。这种很好的提高了检索的精度，但是基于树的检索算法也有很明显的问题：

* 首先，树结构本身很难学习，而且树结构的部分叶子结点可以会因为稀疏数据而导致学习不充分；
* 其次，候选集只属于一个叶子节点，不符合常理。同时这也限制了模型只能从一个角度来刻画候选集，影响了模型的表达。

DR模型使用 D*K 维的矩阵来作为索引结构（如下图所示）。每个 Item 都需要走 D 步，每一步有 K 种选择。走到最后会有 K^D 可能的路径（也可以叫做编码），每一条路径可以代表一个包含多个 item。每个 Item 可以被这样编码一次或多次，所以 item 也可以属于多条路径。

## 2.2 模型架构

### 2.2.1 总体概述

DR有D层网络，每一层是一个K维softmax的MLP，每一层的输入是原始的user emb与前面每一层的Embedding进行concat。因此这里有K^D条path，每一条path都可以理解成一个cluster。

DR的目的就是： 给定一个user，把训练集这个用户互动过的所有item(比如点击item等) 映射到这K^D里面的某几个cluster中，每个cluster可以被表示成一个D维的向量，其中可以包括多个item，每个item也可以属于多个cluster。所有的user共享中间的网络，线上serve的时候，输入user信息(比如user id)即可自动找到与其相关的cluster，然后把每个cluster中的item当做召回的item candidate.

另外， 经过上述流程最后的召回候选集可能比较大，因此DR在后面又加了一个rerank模块，这里的rerank不是我们通常说的 精排后面的重排/混排， 而且用来对前面DR模型召回的item进行进一步过滤，如上图所示。也就是把前面DR召回的候选集当做新的训练集，送给后面的rerank模型， 论文里公开数据集离线实验使用的是softmax去学习，真正字节业务上使用的是LR，softmax分类的loss如下：

这里softmax输出的size是V(所有items的数量)，然后使用sample softmax进行优化，选出Top N送入下面流程。

### **2.2.2 目标函数**

DR不但要学习每一层网络的参数，也要学习给定user如何把一个item映射到某一个cluster/path里面(item-to-path mapping)。每条路径的概率是每层经过的节点概率的连乘：

首先固定path的学习，也就是假设知道每个item最后属于哪几条path。给定一个N条样本的训练集, 对于其中一条path的最大似然函数为:

这里作者认为将每个item分类到一个cluster其实是不够的，比如巧克力可以是“food”，也可以是“gift”，因此作者将一个item映射到J个path里面，因此对于多条path的似然函数表示为：

属于多条路径的概率是属于每条路径的概率之和。但是直接优化这个目标有一个很严重的问题，就是直接把所有的item都分类到某一个path即可，这样对于每个user属于这个path的概率都是1，因此所有的item都在一个类别了，召回也就失效了。因此对上述函数加了惩罚：

其中惩罚项一般为N阶范数，字节采用的4阶。联合之前的rerank学习，最后的loss就是：

预估的时候，用Beam Search找到J条路径，合并每条路径召回的item即可。

### 2.2.3 参数学习

# 3 阿里二向箔

## 3.1 模型架构

### 3.1.1 Post-training索引构建

为了使模型训练不再依赖于索引，作者选择在模型训练后再构建没有任何虚拟节点的索引。一种很自然的想法是，将 TDM 树索引构建方式修改为通过层次 k-medoids 聚类的方式构建，其中叶节点依然为所有 item，但中间节点不再是虚拟节点，而是类簇的中心 item。但是，这种索引构建方式要求 item embedding 有层次化的类簇结构，实际上模型在训练时并没有加入相关约束，因此学出的 item embedding 并没有层次化类簇结构。从离线实验结果来看，这种索引构建方案的效果也比较差。

k-mediods层次聚类构建树索引

因此，作者的目光转向了对模型和 item embedding 都没有任何约束的图检索。具体来说，选用检索精度较高的 HNSW 检索图来根据所有候选项的 embedding 构建索引（构建算法见原始论文）。如下图所示，HNSW 检索图是一个层次化的图检索结构，第0层包含所有节点，上层节点为下层节点的随机采样，每层采样比固定（如32、64）；每层都是一个近似 Delaunay 图 [12]，图中每个节点都有节点与之相邻，且相邻节点距离相近（如欧式距离）。

### 3.1.2 检索实现

检索过程如下，从上往下逐层检索(通过 SEARCH-LAYER 函数)得到每层打分最高的候选项 ，作为下一层的检索起始点；最后，将最底层的检索结果中打分Top K 个候选项作为该用户的最终检索结果。

![](https://xjs3ti3gwc.feishu.cn/space/api/box/stream/download/asynccode/?code=YTNhZTNjZmUyOGJkOTVjMDZmNDUyMDEwOWVhYTczMDhfODJHVVdOdjlyaWdwSTdNVGV2TFlOWGtVN1JBek1YUU1fVG9rZW46S1kyOWJhWm5Xb3JETmt4RnZhZmNBUzlXbjdjXzE2OTk1MDExMDA6MTY5OTUwNDcwMF9WNA)

每层的检索算法如下，给定用户 u 、检索起始点 ep。使用 S 记录本层访问过的节点，C 记录待拓展近邻的候选集， W 动态记录本层检索结果。

### 3.1.3 模型结构

采用的打分模型结构如下图，其主要由以下四部分构成：

* **用户聚合特征提取** ： 用户聚合特征包含用户性别、nickname 等聚合特征。在获取原始 embedding 后采用 transformer 提取 user 侧深度特征。
* **用户行为序列特征提取** ： 用户行为序列特征与目标 target 间采用了 target attention 的方式进行交互，以获取与目标 target 最相关的行为特征。该部分模型结构见下图右边，将序列特征中的 item 和 target 都通过 MLP 进行特征提取后，利用 Scaled Dot-Product Attention 得到最终用户行为序列特征。
* **Target 特征提取** ： 使用多层 MLP 进一步提取 target 侧的特征。这是与 TDM 等一段式召回方案中模型结构区别最大的部分。由于 TDM 中模型训练与索引强耦合，且索引中存在虚拟中间节点，因此 item 侧 side information 特征比较难以直接应用，需要一些类似 top 特征上溯的机制来支持。在二向箔算法体系对模型结构没有任何限制，因此可以很自然的加入各种 item 侧特征。
* **MLP** ** 打分** ： 将上述三路深度特征合并后，经由多轮 MLP 得到最终打分。

二向箔模型结构

# 参考文献

[深度树匹配模型(TDM)](https://github.com/alibaba/x-deeplearning/wiki/%E6%B7%B1%E5%BA%A6%E6%A0%91%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B(TDM))

[阿里深度树匹配召回体系演进](https://zhuanlan.zhihu.com/p/417643436)

[TDM到二向箔：阿里妈妈展示广告Match底层技术架构演进](https://zhuanlan.zhihu.com/p/443113850)

[一文详解深度树检索技术（TDM）三部曲（与Deep Retrieval）_模型_物品_用户](https://it.sohu.com/a/586811949_121119001)
