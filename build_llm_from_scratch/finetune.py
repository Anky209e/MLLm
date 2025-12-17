# TODO: ad finetune functions

from download_gpt import download_and_load_gpt2

s, p = download_and_load_gpt2("124M", "downloaded_models")
print(s, p)
