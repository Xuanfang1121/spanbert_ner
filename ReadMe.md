### Span ner baseline
基于span的实体抽取baseline模型，这里采用的是bert预训练模型. 这里是预测实体的开始和结束位置处的实体类型 <br>
备注：  
span抽取实体的时候，根据开始位置的实体类型找相同尾部实体类型，这里选择最近的实体<br>
span实体抽取无法处理实体嵌套的情况    

#### 依赖
```
transformers==4.6.0
torch==1.8.1
numpy==1.19.2
```

#### 代码目录
```
   .
   |__common
   |__config
   |__dataset
   |__data
   |__models
   |__utils
   |__main.py  主代码
   |__predict.py
```

#### 数据结构
```
浙	B-company
商	I-company
银	I-company
行	I-company
企	O
业	O
信	O
贷	O
部	O
叶	B-name
老	I-name
桂	I-name
博	O
士	O
则	O
从	O
另	O
一	O
个	O
角	O
度	O
对	O
五	O
道	O
门	O
槛	O
进	O
行	O
了	O
解	O
读	O
。	O
叶	O
老	O
桂	O
认	O
为	O
，	O
对	O
目	O
前	O
国	O
内	O
商	O
业	O
银	O
行	O
而	O
言	O
，	O

生	O
生	O
不	O
息	O
C	B-game
S	I-game
O	I-game
L	I-game
生	O
化	O
狂	O
```

#### 模型测试结果
在cluener 数据集上测试span bert 实体抽取，模型结果如下：
```
precision: 0.7952 - recall: 0.7695 - f1: 0.7821
******* address results ********
precision: 0.6230 - recall: 0.6381 - f1: 0.6305 
******* book results ********
precision: 0.8722 - recall: 0.7532 - f1: 0.8084 
******* company results ********
precision: 0.7659 - recall: 0.7963 - f1: 0.7808 
******* game results ********
precision: 0.8272 - recall: 0.8441 - f1: 0.8356 
******* government results ********
precision: 0.8196 - recall: 0.8462 - f1: 0.8327 
******* movie results ********
precision: 0.7901 - recall: 0.8477 - f1: 0.8179 
******* name results ********
precision: 0.8559 - recall: 0.8817 - f1: 0.8686 
******* organization results ********
precision: 0.8615 - recall: 0.7629 - f1: 0.8092 
******* position results ********
precision: 0.8228 - recall: 0.7506 - f1: 0.7850 
******* scene results ********
precision: 0.7297 - recall: 0.5167 - f1: 0.6050
```