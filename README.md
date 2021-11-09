# 渊
> 渊 - A project for Classical Chinese

一个文言诗词的NLP项目们。🌼

* [翻译](#翻译)
    * [现代文到文言文翻译器](#现代文到文言文翻译器)
    * [文言文到现代文翻译器](#文言文到现代文翻译器)


## 翻译
### 现代文到文言文翻译器
* 可以去[🤗 模型主页](https://huggingface.co/raynardj/wenyanwen-chinese-translate-to-ancient)体验或下载这个模型。
* 使用了这个[翻译句对的数据集](https://github.com/BangBOOM/Classical-Chinese)
* 感兴趣的可以参考[训练的笔记](nbs/zh2cc_translate.ipynb)

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
            max_length=128,
            bos_token_id=101,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ), skip_special_tokens=True)
```
#### 目前版本的案例
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

其中可改进处颇多:

* [ ] 目前，模型最长语句是128，可以通过修改tokenizer的max_length参数来调整。也就是会忽略一些现代文的语句。
* [ ] 可以通过去除pad token 的标签设置为-100，这样就不需要传eos token id了。
* [ ] 目前使用现代文预训练的bert-base-chinese作为encoder, 现代文预训练的 gpt2作为decoder。我们完全可以使用文言文+诗词预训练的gpt2作为decoder, 提升效果几乎是肯定的。
* [ ] 算力有限，许多调参细节，基本都没有试过。

### 文言文到现代文翻译器
> 输入文言文， 可以是**断句** 或者 **未断句**的文言文， 模型会预测现代文的表述。

* 欢迎前往[🤗 文言文（古文）到现代文的翻译器模型主页](https://huggingface.co/raynardj/wenyanwen-ancient-translate-to-modern)
* 训练语料是就是九十多万句句对， [数据集链接📚](https://github.com/BangBOOM/Classical-Chinese)。 训练时source序列（古文序列）， 按照50%的概率整句去除所有标点符号。
* 感兴趣的可以参考[训练的笔记](nbs/cc2zh_translate.ipynb),其中可改进处颇多。

#### 推荐的inference 通道
**注意**
* 你必须将```generate```函数的```eos_token_id```设置为102就可以翻译出完整的语句， 不然翻译完了会有残留的语句(因为做熵的时候用pad标签=-100导致)。
目前huggingface 页面上compute按钮会有这个问题， 推荐使用以下代码来得到翻译结果
* 请设置```generate```的参数```num_beams>=3```, 以达到较好的翻译效果
* 请设置```generate```的参数```max_length```256， 不然结果会吃掉句子
```python
from transformers import (
  EncoderDecoderModel,
  AutoTokenizer
)
PRETRAINED = "raynardj/wenyanwen-ancient-translate-to-modern"
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
            max_length=256,
            bos_token_id=101,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ), skip_special_tokens=True)
```
#### 目前版本的案例
> 当然， 拿比较熟知的语句过来， 通常会有些贻笑大方的失误， 大家如果有好玩的调戏案例， 也欢迎反馈
```python
>>> inference('非我族类其心必异')
['不 是 我 们 的 族 类 ， 他 们 的 心 思 必 然 不 同 。']
>>> inference('肉食者鄙未能远谋')
['吃 肉 的 人 鄙 陋 ， 不 能 长 远 谋 划 。']
# 这里我好几批模型都翻不出这个**输**字（甚至有一个版本翻成了秦始皇和汉武帝）， 可能并不是很古朴的用法， 
>>> inference('江山如此多娇引无数英雄竞折腰惜秦皇汉武略输文采唐宗宋祖稍逊风骚')
['江 山 如 此 多 ， 招 引 无 数 的 英 雄 ， 竞 相 折 腰 ， 可 惜 秦 皇 、 汉 武 ， 略 微 有 文 采 ， 唐 宗 、 宋 祖 稍 稍 逊 出 风 雅 。']
>>> inference("清风徐来水波不兴")
['清 风 慢 慢 吹 来 ， 水 波 不 兴 。']
>>> inference("无他唯手熟尔")
['没 有 别 的 事 ， 只 是 手 熟 罢 了 。']
>>> inference("此诚危急存亡之秋也")
['这 实 在 是 危 急 存 亡 的 时 候 。']
```