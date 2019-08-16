import pinyin

text = "必须树立公共交通优先发展的理念"
text = pinyin.get(text, format="numerical", delimiter=" ")
print(text)
