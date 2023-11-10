# 生成式模型的解码方法

## 1 确定性方法（Deterministic Methods）

确定性方法，如贪心搜索和波束搜索，通过在语言模型输出的所有候选补全词中选择概率最高的词来生成最终文本。然而，正如之前研究 [[3]](https://huggingface.co/blog/zh/introducing-csearch#references)[[4]](https://huggingface.co/blog/zh/introducing-csearch#references) 指出的，确定性方法通常会导致 _模型退化_，即生成的文本不自然且包含不必要的重复。

### 1.1 贪心搜索（Greedy Search）

贪心搜索在每个时间步 $t$ 都简单地选择概率最高的词作为当前输出词: $w_t = argmax_{w}P(w | w_{1:t-1})$ ，如下图所示。

![greedy search](https://huggingface.co/blog/assets/02_how-to-generate/greedy_search.png)

从单词 $\text{“The”}$ 开始，算法在第一步贪心地选择条件概率最高的词 $\text{“nice”}$ 作为输出，依此往后。最终生成的单词序列为 $(\text{“The”}, \text{“nice”}, \text{“woman”})$，其联合概率为 $0.5 \times 0.4 = 0.2$。

贪心搜索的主要缺点是它错过了隐藏在低概率词后面的高概率词，如上图所示:

条件概率为 $0.9$ 的单词 $\text{“has”}$ 隐藏在单词 $\text{“dog”}$ 后面，而 $\text{“dog”}$ 因为在 `t=1` 时条件概率值只排第二所以未被选择，因此贪心搜索会错过序列 $\text{“The”}, \text {“dog”}, \text{“has”}$ 。

### 1.2 波束搜索（Beam Search）

波束搜索通过在每个时间步保留最可能的 `num_beams` 个词，并从中最终选择出概率最高的序列来降低丢失潜在的高概率序列的风险。以 `num_beams=2` 为例:

![beam search](https://huggingface.co/blog/assets/02_how-to-generate/beam_search.png)

在时间步 1，除了最有可能的假设 $(\text{“The”}, \text{“nice”})$，波束搜索还跟踪第二可能的假设 $(\text{“The”}, \text{“dog”})$。在时间步 2，波束搜索发现序列 $(\text{“The”}, \text{“dog”}, \text{“has”})$ 概率为$0.36$，比 $(\text{“The”}, \text{“nice”}, \text{“woman”})$ 的 $0.2$ 更高。

波束搜索一般都会找到比贪心搜索概率更高的输出序列，但仍不保证找到全局最优解。虽然结果比贪心搜索更流畅，但输出中仍然包含重复。一个简单的补救措施是引入 *n-grams* (即连续 n 个词的词序列) 惩罚，该方法是由 [Paulus 等人 (2017)](https://arxiv.org/abs/1705.04304) 和 [Klein 等人 (2017)](https://arxiv.org/abs/1701.02810) 引入的。最常见的 *n-grams* 惩罚是确保每个 *n-gram* 都只出现一次，方法是如果看到当前候选词与其上文所组成的 *n-gram* 已经出现过了，就将该候选词的概率设置为 0。波束搜索的另一个重要特性是我们能够比较概率最高的几个波束，并选择最符合我们要求的波束作为最终生成文本。

### 1.3 对比搜索（Contrastive Search）

给定前缀文本 $x_{< t}$，我们按如下公式选择输出词元 $x_{t}$:

![](https://huggingface.co/blog/assets/115_introducing_contrastive_search/formulation.png)

上式中， $V^{(k)}$ 是语言模型输出概率分布 $p_{\theta}(v|x_{< t})$ 中 k 个概率最大的候选词元的集合。第一项，即  *模型置信度 (model confidence)*，是语言模型预测的每个候选词元 $v$ 的概率。第二项，  *退化惩罚 (degeneration penalty) *，用于度量 $v$ 与上文 $x* {< t}$ 中每个词元的相异度，其中函数 $s(\cdot, \cdot)$ 用于计算每两个词元间的余弦相似度。更具体地说，退化惩罚被定义为 $v$ 的向量表征 $h* {v}$ 与其上文 $x* {< t}$ 中每个词元的向量表征间余弦相似度的最大值。这里，候选词元的向量表征 $h* {v}$ 是在给定 $x_{< t}$ 和 $v$ 的条件下将二者连接起来输入给语言模型，然后由语言模型计算出来的。直观上，如果 $v$ 的退化惩罚较大意味着它与上文更相似 (在表示空间中)，因此更有可能导致模型退化问题。超参数 $\alpha$ 用于在这两项中折衷。当 $\alpha=0$ 时，对比搜索退化为纯贪心搜索。

**[备注]** 在生成输出时，对比搜索同时考虑 (i) 语言模型预测的概率，以保持生成文本和前缀文本之间的语义连贯性; (ii) 与上文的相似性以避免模型退化。

## 2 随机方法（Stochastic Methods）

为了解决确定性方法带来的问题，随机方法通过在解码过程中引入随机性来生成文本。常用的两种随机方法是 (i) top-k 采样 [[3]](https://huggingface.co/blog/zh/introducing-csearch#references) 和 (ii) 核采样 (也称为 top-p 采样) [[4]](https://huggingface.co/blog/zh/introducing-csearch#references)。

### 2.1 Top-K 采样

[Fan 等人 (2018)](https://arxiv.org/pdf/1805.04833.pdf) 的论文介绍了一种简单但非常强大的采样方案，称为 ***Top-K*** 采样。在 *Top-K* 采样中，概率最大的 *K* 个词会被选出，然后这 *K* 个词的概率会被重新归一化，最后就在这重新被归一化概率后的 *K* 个词中采样。 GPT2 采用了这种采样方案，这也是它在故事生成这样的任务上取得成功的原因之一。

将上文例子中的候选单词数从 3 个单词扩展到 10 个单词，以更好地说明 *Top-K* 采样。

![Top K sampling](https://huggingface.co/blog/assets/02_how-to-generate/top_k_sampling.png)

设 $K = 6$，即我们将在两个采样步的采样池大小限制为 6 个单词。我们定义 6 个最有可能的词的集合为 $V_{\text{top-K}}$。在第一步中，$V_{\text{top-K}}$ 仅占总概率的大约三分之二，但在第二步，它几乎占了全部的概率。同时，我们可以看到在第二步该方法成功地消除了那些奇怪的候选词 $(\text{“not”}, \text{“the”}, \text{“small”}, \text{“told”})$。

*Top-K* 采样不会动态调整从需要概率分布 $P(w|w_{1:t-1})$ 中选出的单词数。这可能会有问题，因为某些分布可能是非常尖锐 (上图中右侧的分布)，而另一些可能更平坦 (上图中左侧的分布)，所以对不同的分布使用同一个绝对数 *K* 可能并不普适。

在 $t=1$ 时，*Top-K* 将 $(\text{“people”}, \text{“big”}, \text{“house”}, \text{“cat”})$ 排出了采样池，而这些词似乎是合理的候选词。另一方面，在$t=2$ 时，该方法却又把不太合适的 $(\text{“down”}, \text{“a”})$ 纳入了采样池。因此，将采样池限制为固定大小 *K* 可能会在分布比较尖锐的时候产生胡言乱语，而在分布比较平坦的时候限制模型的创造力。这一发现促使 [Ari Holtzman 等人 (2019)](https://arxiv.org/abs/1904.09751) 发明了  **Top-p** - 或  **核** - 采样。

### 2.2 Top-p (核) 采样

在 *Top-p* 中，采样不只是在最有可能的 *K* 个单词中进行，而是在累积概率超过概率 *p* 的最小单词集中进行。然后在这组词中重新分配概率质量。这样，词集的大小 (*又名* 集合中的词数) 可以根据下一个词的概率分布动态增加和减少。好吧，说的很啰嗦，一图胜千言。

![Top p sampling](https://huggingface.co/blog/assets/02_how-to-generate/top_p_sampling.png)

假设 $p=0.92$，*Top-p* 采样对单词概率进行降序排列并累加，然后选择概率和首次超过 $p=92%$ 的单词集作为采样池，定义为 $V_{\text{top-p}}$。在 $t=1$ 时 $V_{\text{top-p}}$ 有 9 个词，而在 $t=2$ 时它只需要选择前 3 个词就超过了 92%。其实很简单吧！可以看出，在单词比较不可预测时，它保留了更多的候选词，*如* $P(w | \text{“The”})$，而当单词似乎更容易预测时，只保留了几个候选词，*如* $P(w | \text{“The”}, \text{“car”})$。

虽然从理论上讲， *Top-p* 似乎比 *Top-K* 更优雅，但这两种方法在实践中都很有效。 *Top-p* 也可以与 *Top-K* 结合使用，这样可以避免排名非常低的词，同时允许进行一些动态选择。

## 3 总结

在开放域语言生成场景中，作为最新的解码方法， *top-p* 和 *top-K* 采样于传统的 *贪心* 和 *波束* 搜索相比，似乎能产生更流畅的文本。但，最近有更多的证据表明 *贪心* 和 *波束* 搜索的明显缺陷 - 主要是生成重复的单词序列 - 是由模型 (特别是模型的训练方式) 引起的，而不是解码方法， *参见* [Welleck 等人 (2019)](https://arxiv.org/pdf/1908.04319.pdf) 的论文。此外，如 [Welleck 等人 (2020)](https://arxiv.org/abs/2002.02492) 的论文所述，看起来 *top-K* 和 *top-p* 采样也会产生重复的单词序列。

在 [Welleck 等人 (2019)](https://arxiv.org/pdf/1908.04319.pdf) 的论文中，作者表明，根据人类评估，在调整训练目标后，波束搜索相比 *Top-p* 采样能产生更流畅的文本。

开放域语言生成是一个快速发展的研究领域，而且通常情况下这里没有放之四海而皆准的方法，因此必须了解哪种方法最适合自己的特定场景。
