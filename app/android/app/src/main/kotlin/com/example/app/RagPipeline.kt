package com.example.app

import android.app.Application
import android.content.Context
import android.util.Log
import kotlinx.coroutines.channels.Channel
import com.google.ai.edge.localagents.rag.chains.ChainConfig
import com.google.ai.edge.localagents.rag.chains.RetrievalAndInferenceChain
import com.google.ai.edge.localagents.rag.memory.DefaultSemanticTextMemory
import com.google.ai.edge.localagents.rag.memory.SqliteVectorStore
import com.google.ai.edge.localagents.rag.models.AsyncProgressListener
import com.google.ai.edge.localagents.rag.models.Embedder
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.GeminiEmbedder
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import com.google.ai.edge.localagents.rag.models.MediaPipeLlmBackend
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.ai.edge.localagents.rag.prompt.PromptBuilder
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig.TaskType
import com.google.ai.edge.localagents.rag.retrieval.RetrievalRequest
import com.google.common.collect.ImmutableList
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.asCoroutineDispatcher
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.Executors
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.guava.await
import kotlinx.coroutines.withContext
import java.util.Optional
import kotlin.jvm.optionals.getOrNull

/** The RAG pipeline for LLM generation. */
class RagPipeline(private val application: Application) {
    private val mediaPipeLanguageModelOptions: LlmInferenceOptions =
        LlmInferenceOptions.builder().setModelPath(
            GEMMA_MODEL_PATH
        ).setPreferredBackend(LlmInference.Backend.GPU).setMaxTokens(1024).build()
    private val mediaPipeLanguageModelSessionOptions: LlmInferenceSession.LlmInferenceSessionOptions =
        LlmInferenceSession.LlmInferenceSessionOptions.builder().setTemperature(1.0f)
            .setTopP(0.95f).setTopK(64).build()
    private val mediaPipeLanguageModel: MediaPipeLlmBackend =
        MediaPipeLlmBackend(
            application.applicationContext, mediaPipeLanguageModelOptions,
            mediaPipeLanguageModelSessionOptions
        )

    private val embedder: Embedder<String> = if (COMPUTE_EMBEDDINGS_LOCALLY) {
        GeckoEmbeddingModel(
            GECKO_MODEL_PATH,
            Optional.of(TOKENIZER_MODEL_PATH),
            USE_GPU_FOR_EMBEDDINGS,
        )
    } else {
        GeminiEmbedder(
            GEMINI_EMBEDDING_MODEL,
            GEMINI_API_KEY
        )
    }

    private val config = ChainConfig.create(
        mediaPipeLanguageModel, PromptBuilder(PROMPT_TEMPLATE),
        DefaultSemanticTextMemory(
            // Gecko embedding model dimension is 768
            SqliteVectorStore(768), embedder
        )
    )
    private val retrievalAndInferenceChain = RetrievalAndInferenceChain(config)

    private var llmReady = false

    // rendezvous channel for the LLM being ready - allows coroutine to wait for the llm to be ready
    // https://stackoverflow.com/a/55421973
    private val onLlmReady = Channel<Unit>(0)

    init {
        Futures.addCallback(
            mediaPipeLanguageModel.initialize(),
            object : FutureCallback<Boolean> {
                override fun onSuccess(result: Boolean) {
                    Log.i("mam-ai", "LLM initialized!")

                    // TODO minimise # of chunks
//                    memorizeChunks(application.applicationContext, "mamai_context.txt")
//                    Log.i("mam-ai", "Chunks loaded!")

                    llmReady = true
                    onLlmReady.trySend(Unit)
                    // no-op
                }

                override fun onFailure(t: Throwable) {
                }
            },
            Executors.newSingleThreadExecutor(),
        )
    }

    fun memorizeChunks(context: Context, filename: String) {
        // BufferedReader is needed to read the *.txt file
        // Create and Initialize BufferedReader
        val reader = BufferedReader(InputStreamReader(context.assets.open(filename)))

        val sb = StringBuilder()
        val texts = mutableListOf<String>()
        generateSequence { reader.readLine() }
            .forEach { line ->
                for (prefix in CHUNK_SEPARATORS) {
                    if (line.startsWith(prefix)) {
                        if (sb.isNotEmpty()) {
                            val chunk = sb.toString()
                            texts.add(chunk)
                        }
                        sb.clear()
                        sb.append(line.removePrefix(prefix).trim())
                        return@forEach
                    }
                }

                sb.append(" ")
                sb.append(line)
            }
        if (sb.isNotEmpty()) {
            texts.add(sb.toString())
        }
        reader.close()
        Log.i("mam-ai", "Texts is empty!!!")
        if (texts.isNotEmpty()) {
            Log.i("mam-ai", "Texts is " + texts.size)
            return memorize(texts)
        }
    }

    /** Stores input texts in the semantic text memory. */
    private fun memorize(facts: List<String>) {
        val future = config.semanticMemory.getOrNull()?.recordBatchedMemoryItems(ImmutableList.copyOf(facts))
        future?.get()
    }

    /** Generates the response from the LLM. */
    suspend fun generateResponse(
        prompt: String,
        callback: AsyncProgressListener<LanguageModelResponse>?,
    ): String =
        coroutineScope {
            // Wait for llm to be ready via rendezvous channel
            if (!llmReady) {
                onLlmReady.receive()
            }

            val retrievalRequest =
                RetrievalRequest.create(
                    prompt,
                    RetrievalConfig.create(3, 0.0f, TaskType.QUESTION_ANSWERING)
                )
            retrievalAndInferenceChain.invoke(retrievalRequest, callback).await().text
        }

    companion object {
        private const val COMPUTE_EMBEDDINGS_LOCALLY = true
        private const val USE_GPU_FOR_EMBEDDINGS = true
        private val CHUNK_SEPARATORS = listOf("<sep>", "<doc_sep>")

        private const val GEMMA_MODEL_PATH = "/data/local/tmp/llm/gemma-3-1b-it-int4.task"
        private const val TOKENIZER_MODEL_PATH = "/data/local/tmp/sentencepiece.model"
        private const val GECKO_MODEL_PATH = "/data/local/tmp/gecko.tflite"
        private const val GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
        private const val GEMINI_API_KEY = "..."

        // The prompt template for the RetrievalAndInferenceChain. It takes two inputs: {0}, which is the retrieved context, and {1}, which is the user's query.
        private const val PROMPT_TEMPLATE: String =
            "You are a kind and helpful summarising assistant designed to support pregnant women and new mothers. Speak in simple, clear English. Give accurate, medically grounded information from reliable sources. Keep answers short and easy to understand.\n" +
                    "\n" +
                    "Always be calm, supportive, and non-judgmental.\n" +
                    "\n" +
                    "If a user describes symptoms that could be urgent or dangerous—such as bleeding, severe pain, trouble breathing, fever in a newborn, or anything that sounds serious—tell them to contact a doctor, nurse, or emergency service right away. Do not try to diagnose emergencies.\n" +
                    "\n" +
                    "If you're unsure or don’t have enough information, say: “I'm not sure. It's best to ask a doctor or nurse.”\n" +
                    "\n" +
                    "Use the context provided to answer questions. If no context is available, give general advice based on trusted health guidelines.\n" +
                    "\n" +
                    "Never make guesses. Always prioritize safety and clarity. Remember that you have no physical body as an LLM and therefore in an emergency you need to ask the mother to ask a doctor or nurse.\n" +
                    "\n" +
                    "You can ONLY REPLY ONCE. The user therefore CANNOT ask clarifying questions. Include ALL info in your first AND ONLY answer. In this way, you act like a more intelligent Google search.\n" +
                    "\n" +
                    "============================" +
                    "\n" +
                    "Here is the context: {0}" +
                    "\n" +
                    "=============================\n" +
                    "Here is the user's question: {1}"
    }
}
