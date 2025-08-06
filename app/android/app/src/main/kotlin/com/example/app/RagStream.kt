package com.example.app

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.LifecycleCoroutineScope
import com.google.ai.edge.localagents.rag.models.LanguageModelResponse
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.collections.hashMapOf

class RagStream(application: Application, val lifecycleScope: LifecycleCoroutineScope): EventChannel.StreamHandler {
    private val backgroundExecutor: Executor = Executors.newSingleThreadExecutor()
    private var currentJob: Job? = null
    private var currentPrompt: String? = null

    val ragPipeline: RagPipeline by lazy { RagPipeline(application) }
    var latestGeneration: EventChannel.EventSink? = null

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        requestLlmPreinit()
        this.latestGeneration = events
    }

    override fun onCancel(arguments: Any?) {}

    suspend fun waitForLlmInit() {
        // Wait for llm to be ready via rendezvous channel
        if (!ragPipeline.llmReady) {
            ragPipeline.onLlmReady.receive()
        }
    }

    fun requestLlmPreinit() {
        ragPipeline // Force lazy initialization to happen now
    }

    val isReady: Boolean
        get() = this.ragPipeline.llmReady;

    fun generateResponse(prompt: String) {
        synchronized(this) {
            // We are already generating for this prompt
            if (currentJob != null && currentPrompt == prompt) {
                return
            }

            currentJob?.cancel()

            currentPrompt = prompt
            currentJob = lifecycleScope.launch {
                withContext(backgroundExecutor.asCoroutineDispatcher()) {
                    fun onGenerate(response: LanguageModelResponse, done: Boolean) {
                        // We are off the ui thread at the moment. You can only send
                        // messages through event channels while on ui thread.
                        // The `post` puts the sending part back onto the ui thread
                        Handler(Looper.getMainLooper()).post {
                            latestGeneration?.success(
                                hashMapOf(
                                    "response" to response.text
                                )
                            )

                            if (done) {
                                synchronized(this) {
                                    currentJob = null
                                    currentPrompt = null
                                }
                            }
                        }
                    }

                    fun onRetrieve(documents: List<String>) {
                        Handler(Looper.getMainLooper()).post {
                            latestGeneration?.success(
                                hashMapOf(
                                    "results" to documents
                                )
                            )
                        }
                    }

                    ragPipeline.generateResponse(
                        prompt,
                        { results -> onRetrieve(results) },
                        { response, done -> onGenerate(response, done) }
                    );
                }
            }
        }
    }
}