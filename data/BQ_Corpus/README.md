---
license: Apache License 2.0
text:
  sentence-similarity:
    language:
      - zh
---

介绍:

作为语义匹配任务，句子语义等价识别（SSEI）是自然语言处理（NLP）在问答（QA）、自动客户服务和聊天机器人中的一项基础任务。在客户服务系统中，如果两个问题传达相同的意图或可以由相同的答案回答，则它们被定义为语义等价。我们介绍了银行问题（BQ）语料库，这是一个用于 SSEI 的大规模特定领域中文语料库。 BQ 语料库包含来自网上银行自定义服务日志的 120,000 个问题对。它分为三部分：100,000 对用于训练，10,000 对用于验证，10,000 对用于测试。我们在我们的语料库上展示了五个 SSEI 基准性能，包括最先进的算法。作为银行领域最大的人工标注公共中文 SSEI 语料库，BQ 语料不仅可用于中文问题语义匹配研究，也是跨语言、跨领域 SSEI 研究的重要资源。


```bib
@inproceedings{chen-etal-2018-bq,
    title = "The {BQ} Corpus: A Large-scale Domain-specific {C}hinese Corpus For Sentence Semantic Equivalence Identification",
    author = "Chen, Jing  and
      Chen, Qingcai  and
      Liu, Xin  and
      Yang, Haijun  and
      Lu, Daohe  and
      Tang, Buzhou",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1536",
    doi = "10.18653/v1/D18-1536",
    pages = "4946--4951",
    abstract = "This paper introduces the Bank Question (BQ) corpus, a Chinese corpus for sentence semantic equivalence identification (SSEI). The BQ corpus contains 120,000 question pairs from 1-year online bank custom service logs. To efficiently process and annotate questions from such a large scale of logs, this paper proposes a clustering based annotation method to achieve questions with the same intent. First, the deduplicated questions with the same answer are clustered into stacks by the Word Mover{'}s Distance (WMD) based Affinity Propagation (AP) algorithm. Then, the annotators are asked to assign the clustered questions into different intent categories. Finally, the positive and negative question pairs for SSEI are selected in the same intent category and between different intent categories respectively. We also present six SSEI benchmark performance on our corpus, including state-of-the-art algorithms. As the largest manually annotated public Chinese SSEI corpus in the bank domain, the BQ corpus is not only useful for Chinese question semantic matching research, but also a significant resource for cross-lingual and cross-domain SSEI research. The corpus is available in public.",
}
```
### Clone with HTTP
* http://www.modelscope.cn/datasets/DAMO_NLP/BQ_Corpus.git