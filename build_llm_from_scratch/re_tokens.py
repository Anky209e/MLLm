import re

text = """Hello, This is a sentence."""
print(text)

text1 = re.split(r"(\s)", text)

print(text1)

text2 = re.split(r"([,.]|\s)", text)

print(text2)

result = [item for item in text2 if item.strip()]

print(result)
