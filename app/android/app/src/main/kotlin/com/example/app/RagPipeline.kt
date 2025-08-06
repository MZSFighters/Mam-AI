package com.example.app

import android.app.Application
import android.content.Context
import android.util.Log
import com.google.ai.edge.localagents.rag.chains.ChainConfig
import com.google.ai.edge.localagents.rag.chains.RetrievalAndInferenceChain
import com.google.ai.edge.localagents.rag.memory.DefaultSemanticTextMemory
import com.google.ai.edge.localagents.rag.memory.SqliteVectorStore
import com.google.ai.edge.localagents.rag.models.AsyncProgressListener
import com.google.ai.edge.localagents.rag.models.Embedder
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import com.google.ai.edge.localagents.rag.models.MediaPipeLlmBackend
import com.google.ai.edge.localagents.rag.prompt.PromptBuilder
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig
import com.google.ai.edge.localagents.rag.retrieval.RetrievalConfig.TaskType
import com.google.ai.edge.localagents.rag.retrieval.RetrievalRequest
import com.google.common.collect.ImmutableList
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.guava.await
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Optional
import java.util.concurrent.Executors
import kotlin.jvm.optionals.getOrNull

/** The RAG pipeline for LLM generation. */
class RagPipeline(application: Application) {
    private val baseFolder = application.getExternalFilesDir(null).toString() + "/"
    private val mediaPipeLanguageModelOptions: LlmInferenceOptions =
        LlmInferenceOptions.builder().setModelPath(
            baseFolder + GEMMA_MODEL
        ).setPreferredBackend(LlmInference.Backend.GPU).setMaxTokens(4096).build()
    private val mediaPipeLanguageModelSessionOptions: LlmInferenceSession.LlmInferenceSessionOptions =
        LlmInferenceSession.LlmInferenceSessionOptions.builder().setTemperature(1.0f)
            .setTopP(0.95f).setTopK(64).build()
    private val mediaPipeLanguageModel: MediaPipeLlmBackend =
        MediaPipeLlmBackend(
            application.applicationContext, mediaPipeLanguageModelOptions,
            mediaPipeLanguageModelSessionOptions
        )

    private val embedder: Embedder<String> = GeckoEmbeddingModel(
        baseFolder + GECKO_MODEL,
        Optional.of(baseFolder + TOKENIZER_MODEL),
        USE_GPU_FOR_EMBEDDINGS,
    )


    private val textMemory = DefaultSemanticTextMemory(
        // Gecko embedding model dimension is 768
        SqliteVectorStore(768, baseFolder + "embeddings.sqlite"), embedder
    )

    private val config = ChainConfig.create(
        mediaPipeLanguageModel, PromptBuilder(PROMPT_TEMPLATE),
        textMemory
    )
    private val retrievalAndInferenceChain = RetrievalAndInferenceChain(config)

    var llmReady = false

    // rendezvous channel for the LLM being ready - allows coroutine to wait for the llm to be ready
    // https://stackoverflow.com/a/55421973
    val onLlmReady = Channel<Unit>(0)

    /// Load the model
    init {
        Futures.addCallback(
            mediaPipeLanguageModel.initialize(),
            object : FutureCallback<Boolean> {
                override fun onSuccess(result: Boolean) {
                    Log.i("mam-ai", "LLM initialized!")
                    // UNCOMMENT IF YOU WANT TO ADD MORE CONTEXT
                    // memorizeChunks(application.applicationContext, "mamai_trim.txt")
                    // Log.i("mam-ai", "Chunks loaded!")

                    llmReady = true
                    onLlmReady.trySend(Unit)
                }

                override fun onFailure(t: Throwable) {
                }
            },
            Executors.newSingleThreadExecutor(),
        )
    }

    // Memorise the given file (from inside the app context)
    // Unused at the moment, since we ship a pre-memorised sqlite DB, but this is the code that
    // could be used to memorise more documents on the fly
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
        retrievalListener: (docs: List<String>) -> Unit,
        generationListener: AsyncProgressListener<LanguageModelResponse>?,
    ): String =
        coroutineScope {
            // Wait for llm to be ready via rendezvous channel
            if (!llmReady) {
                onLlmReady.receive()
            }

            // Get relevant docs - this is actually duplicated by the retrievalAndInferenceChain.invoke
            // call but we also want to show the docs themselves to the user, so we have to do it
            // twice to get the actual output (usually it's passed to the model only)
            val retrievalRequest =
                RetrievalRequest.create(
                    prompt,
                    RetrievalConfig.create(3, 0.0f, TaskType.RETRIEVAL_QUERY)
                )
            val retrievalResults = textMemory.retrieveResults(retrievalRequest).await().getEntities().map { e -> e.data }.toList()
            retrievalListener(retrievalResults);

            // Get the model output
            retrievalAndInferenceChain.invoke(retrievalRequest, generationListener).await().text
        }

    companion object {
        private const val USE_GPU_FOR_EMBEDDINGS = true
        private val CHUNK_SEPARATORS = listOf("<sep>", "<doc_sep>")

        private const val GEMMA_MODEL = "gemma-3n-E4B-it-int4.task"
        private const val TOKENIZER_MODEL = "sentencepiece.model"
        private const val GECKO_MODEL = "Gecko_1024_quant.tflite"

        // The prompt template for the RetrievalAndInferenceChain. It takes two inputs: {0}, which is the retrieved context, and {1}, which is the user's query.
        private const val PROMPT_TEMPLATE: String =
            "CONTEXT FOR THE QUERY BELOW: {0}.\n" +
                    "<separator>\n" +
                    "You are a smart search engine designed to support nurses and midwives in neonatal care. Speak in simple, clear English suitable for a second-language speaker. Be impartial and impersonal - you are creating a summary for the user, not chatting with them. Give accurate, medically grounded information from reliable sources, orienting toward actionable steps for practical care. Keep answers short (prioritise conciseness and  and easy to understand, making use of bullet points.\n" +
                    "\n" +
                    "THE CONTEXT MAY OR MAY NOT BE RELEVANT, ONLY THE 3 MOST SIMILAR DOCUMENTS ARE RETRIEVED. IF NONE ARE RELEVANT, 3 IRRELEVANT DOCUMENTS WILL BE RETRIEVED. IGNORE THEM IN THIS CASE AND SAY THAT NOTHING RELEVANT WAS FOUND!\n" +
                    "\n" +
                    "If a user describes symptoms that could be urgent or dangerous—such as bleeding, severe pain, trouble breathing, fever in a newborn, or anything that sounds serious—tell them to contact a doctor, nurse, or emergency service right away. Do not try to diagnose emergencies.\n" +
                    "\n" +
                    "If you're unsure or don’t have enough information, say: “I'm not sure. It's best to ask a doctor or nurse.”\n" +
                    "\n" +
                    "Use the context provided to answer questions. If no context is available, give general advice based on trusted health guidelines.\n" +
                    "\n" +
                    "Never make guesses. Always prioritize safety and clarity. Remember that you have no physical body as an LLM and therefore in an emergency you need to ask the mother to ask a doctor or nurse.\n" +
                    "\n" +
                    "You can ONLY REPLY ONCE with a FULL summary. The user therefore CANNOT ask clarifying questions. Include ALL info in your first AND ONLY answer. In this way, you act like a more intelligent Google search.\n" +
                    "\n" +
                    "<separator>\n" +
                    "Here is the user's question: {1}.\n" +
                    "<separator>\n" +
                    "INCLUDE ALL INFORMATION IN YOUR CONCISE, MINIMAL SUMMARY, SIMILAR TO A SEARCH ENGINE SUMMARY.\n"
    }
}
