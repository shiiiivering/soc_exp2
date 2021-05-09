### 依赖
- networkx
- scikit-learn
- matplotlib

### 运行
- `python main.py --path=path/to/dataset`, 如：`python main.py --path="D:\\learning\\社会计算\\exp2\\data"`


### 说明
- datapreprocess.py 读取数据，原始数据读取完毕之后存储为json文件，若有json文件直接读取json文件
- 数据集特点

| 类型   | 数目  |
| :----: | :---: |
| Group  | 482   |
| Event  | 55396 |
| Member | 73685 |
| Topic  | 17128 |

### 正确率
- 使用用户共同topic计算相似度：约 0.76
- 使用用户共同参与事件计算相似度：约 0.84
- 初步使用级联（基于选择或者评分矩阵）：约 0.90到0.91之间
