# 渊
> 渊 - A project for Classical Chinese

一个文言诗词的NLP项目们。🌼

## 翻译
### 现代文到文言文翻译
可以去[🤗 模型主页](https://huggingface.co/raynardj/wenyanwen-chinese-translate-to-ancient)体验或下载这个模型。使用了这个[翻译句对的数据集](https://github.com/BangBOOM/Classical-Chinese)


在python中推荐使用以下的代码进行inference：
```python
from transformers import (
  EncoderDecoderModel,
  AutoTokenizer
)
PRETRAINED = "raynardj/wenyanwen-chinese-translate-to-ancient"
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
model = EncoderDecoderModel.from_pretrained(PRETRAINED)

def inference(text):
    tk_kwargs = dict(
      truncation=True,
      max_length=128,
      padding="max_length",
      return_tensors='pt')
   
    inputs = tokenizer([text,],**tk_kwargs)
    with torch.no_grad():
        return tokenizer.batch_decode(
            model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=3,
            bos_token_id=101,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ), skip_special_tokens=True)
```

目前版本， 按照上述通道的翻译案例：
```python
>>> inference('你连一百块都不肯给我')
['不 肯 与 我 百 钱 。']
>>> inference("他不能做长远的谋划")
['不 能 为 远 谋 。']
>>> inference("我们要干一番大事业")
['吾 属 当 举 大 事 。']
>>> inference("这感觉，已经不对，我努力，在挽回")
['此 之 谓 也 ， 已 不 可 矣 ， 我 勉 之 ， 以 回 之 。']
>>> inference("轻轻地我走了， 正如我轻轻地来， 我挥一挥衣袖，不带走一片云彩")
['轻 我 行 ， 如 我 轻 来 ， 挥 袂 不 携 一 片 云 。']
```

感兴趣的可以参考[训练的笔记](nbs/zh2cc_translate.ipynb),其中可改进处颇多。

* [ ] 目前，模型最长语句是128，可以通过修改tokenizer的max_length参数来调整。也就是会忽略一些现代文的语句。
* [ ] 可以通过去除pad token 的标签设置为-100，这样就不需要传eos token id了。
* [ ] 目前使用现代文预训练的bert-base-chinese作为encoder, 现代文预训练的 gpt2作为decoder。我们完全可以使用文言文+诗词预训练的gpt2作为decoder, 提升效果几乎是肯定的。
* [ ] 算力有限，许多调参细节，基本都没有试过。
