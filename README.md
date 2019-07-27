# A PyTorch Implementation of Cross-Modal Interaction Networks for Query-Based Moment Retrieval in Videos

**The paper is available [here](https://arxiv.org/abs/1906.02497).**

**The pretrained word vectors can be download [here](https://nlp.stanford.edu/projects/glove/)**

**The original ActivityNet dataset can be download [here](http://activity-net.org/download.html).**
+ Train:
```
python main.py
--word2vec-path
XXX/glove_model.bin
--dataset
ActivityNet
--feature-path
XXX/activity-c3d
--train-data
data/activity/train_data_gcn.json
--val-data
data/activity/val_data_gcn.json
--test-data
data/activity/test_data_gcn.json
--max-num-epochs
10
--dropout
0.2
--warmup-updates
300
--warmup-init-lr
1e-06
--lr
8e-4
--num-heads
4
--num-gcn-layers
2
--num-attn-layers
2
--weight-decay
1e-7
--train
--model-saved-path
models_activity
```

**The original TACoS dataset can be download [here](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos).**
+ The features are extracted through pre-trained C3D networks.
+ Train:
```
python main.py
--word2vec-path
XXX/glove_model.bin
--dataset
TACOS
--feature-path
XXX
--train-data
data/tacos/TACOS_train_gcn.json
--val-data
data/tacos/TACOS_val_gcn.json
--test-data
data/tacos/TACOS_test_gcn.json
--max-num-epochs
40
--dropout
0.2
--warmup-updates
300
--warmup-init-lr
1e-07
--lr
4e-4
--num-heads
4
--num-gcn-layers
2
--num-attn-layers
2
--weight-decay
1e-8
--train
--model-saved-path
models_tacos
--batch-size
64
```

