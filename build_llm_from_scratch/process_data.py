import re

FILE = "Data.txt"

with open(FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(raw_text[:55])

result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
print(result[:100])
