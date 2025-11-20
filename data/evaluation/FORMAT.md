
# 评测数据格式说明

每条数据包含以下字段：

1. id: 数据编号
2. instruction: 主要问题/指令
3. input: 补充信息（可能为空）
4. output: 标准答案
5. full_question: 完整问题（instruction + input组合）

## 使用方法

### 零样本推理
使用 full_question 作为输入，直接提问模型。

### CoT推理
将 full_question 嵌入CoT prompt模板中。

## 注意事项

- 如果input为空，full_question = instruction
- 如果input不为空，full_question = instruction + "\n\n补充信息：\n" + input
