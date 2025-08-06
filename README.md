# mam-ai

## Reproduction instructions

This is a rough sketch of how you could reproduce what we created in this project.

**Prepare RAG docs:**
   1. Curate documents you want to include (or use the Google Drive link from the writeup)
   2. Run [MMORE](https://github.com/swiss-ai/mmore) over these documents to extract their text
   3. Chunk the documents using the scripts in `rag/`
   4. Copy the chunks to the `mamai_trim.txt` in the `assets` folder of the Android app and uncomment the `memorizeChunks()` call
   5. Run the app and wait for it to incorporate the chunks into the sqlite db
   6. Re-comment the `memorizeChunks`
   7. Use `adb` to pull the `embeddings.sqlite` for redistribution

**How we developed the app:**
- Flutter frontend with regular Flutter <-> Android FFI/bridging
  - Built this out to meet our needs
- Adapted the [Google AI Edge RAG](ai.google.dev/edge/mediapipe/solutions/genai/rag/android) for the Android backend which runs the LLM

**Serving the remote files to the users:**
- Start an nginx server with a self-signed cert and the files (see below) in `/var/www/html/`


## Remote resources 

We host copies of various models on a temporary VPS. The app fetches these the first time it
launches. This is just done for simplicity's sake.

They are as follows:
- `Gecko_1024_quant.tflite`: embedding model from [litert-community/Gecko-110m-en](https://huggingface.co/litert-community/Gecko-110m-en)
- `sentencepiece.model`: tokenizer from [litert-community/Gecko-110m-en])(https://huggingface.co/litert-community/Gecko-110m-en)
- `gemma-3n-E4B-it-int4.task`: Gemma3n E4B
- `embeddings.sqlite`: pre-computed document embeddings
