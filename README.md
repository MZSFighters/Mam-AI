# mam-ai

## Reproduction instructions

This is a rough sketch of how you could reproduce what we created in this project.
It is divided into 

## Remote resources 

We host copies of various models on a temporary VPS. The app fetches these the first time it
launches. This is just done for simplicity's sake.

They are as follows:
- `Gecko_1024_quant.tflite`: embedding model from [litert-community/Gecko-110m-en](https://huggingface.co/litert-community/Gecko-110m-en)
- `sentencepiece.model`: tokenizer from [litert-community/Gecko-110m-en])(https://huggingface.co/litert-community/Gecko-110m-en)
- `gemma-3n-E4B-it-int4.task`: Gemma3n E4B
- `embeddings.sqlite`: pre-computed document embeddings
