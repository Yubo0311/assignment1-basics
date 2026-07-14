# Tokenization

## BPE training example

### Dataset

读取数据

### Vocabulary

初始字典：256个byte加上一个special token `<|endoftext|>`

### Pre-Tokenization

通过regex进行pre-tokenization

- 输入一段长文本
- 输出字典frequency table`dict[tuple[bytes, ....], int]` 用python的`bytes`类

```Python
{low: 5, lower: 2, widest: 3, newest: 6}
```

**tie breaker: 按照字典序**

#### 可以并行

通过`<|endoftext|>` 进行chunking然后并行pre-tokenize

#### 去掉 special tokens

一开始split并为special token预留id然后把它扔掉进行tokenization

- `re.split`
- `"|".join(special_tokens)`
- `re.escape`
  ```Python

  print(re.escape('https://www.python.org'))

  legal_chars = string.ascii_lowercase + string.digits + "!#$%&'*+-.^_`|~:"
  print('[%s]+' % re.escape(legal_chars))

  operators = ['+', '-', '*', '/', '**']
  print('|'.join(map(re.escape, sorted(operators, reverse=True))))
  ```

### Merges

通过frequency table找出需要被合并的pair

#### 优化
