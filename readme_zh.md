# 基于[Atlas](./README.md)的检索增强语言模型
原始atlas使用Contriever初始化检索器并使用T5以FID的方式初始化语言模型。在此基础上做了一些改动。
## 环境变动
```
pip install -r requirements_new.txt
```
## 更新日志
[2023.9.27] 将atlas中的语言模型从encoder-decoder架构更改为decoder-only架构，从而可以使用大语言模型如llama来代替原来的T5模型生成文本。
```
bash run.sh # 运行原始atlas
bash run_gpt.sh # 更改其中--model_path参数即可使用decoder-only的语言模型进行训练
```
