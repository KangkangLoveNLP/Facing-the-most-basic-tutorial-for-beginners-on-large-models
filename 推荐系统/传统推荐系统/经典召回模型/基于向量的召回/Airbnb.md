# Airbnb

这是 Airbnb 于2018年发表的一篇论文，主要介绍了 Airbnb 在 Embedding 技术上的应用，并获得了 KDD 2018 的 Best Paper。Airbnb 是全球最大的短租平台，包含了数百万种不同的房源。这篇论文介绍了 Airbnb 如何使用 Embedding 来实现相似房源推荐以及实时个性化搜索。在本文中，Airbnb 在用户和房源的 Embedding 上的生成都是基于谷歌的 Word2Vec 模型，

## Embeddding 方法


+ 用于描述短期实时性的个性化特征 Embedding：listing Embeddings
listing 表示房源的意思。
通过用户点击行为，将房源映射到向量空间，实现 item-to-item 的召回；

+ 用于描述长期的个性化特征 Embedding：user-type & listing type Embeddings
通过预订行为学习用户与房源的长期偏好 embedding，实现 user-to-item 的召回



短期（session 内行为）：记录用户在当前会话内的 click-sequence；

长期偏好：基于历史 booking data，划分 user-type 和 listing-type embedding

### listing Embeddings

Listing Embeddings 是基于用户的点击 session 学习得到的，用于表示房源的短期实时性特征。给定数据集 $ \mathcal{S} $ ，其中包含了 $ N $ 个用户的 $ S $ 个点击 session（序列）。

+ 正样本集构建

使用 booked listing 作为全局上下文，booked listing 表示用户在 session 中最终预定的房源，一般只会出现在结束的 session 中。

Airbnb 将最终预定的房源，始终作为滑窗的上下文，即全局上下文。

+ 负样本的选择

对于每个滑窗中的中心 lisitng，其负样本的选择新增了与其位于同一个 market 的 listing。

