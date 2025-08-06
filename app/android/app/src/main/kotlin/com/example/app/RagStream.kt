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

/**
 * The eventchannel stream from Android backend <-> Flutter. Flutter subscribes to this to get
 * generation output as it is being generated
 */
class RagStream(application: Application, val lifecycleScope: LifecycleCoroutineScope): EventChannel.StreamHandler {
    private val backgroundExecutor: Executor = Executors.newSingleThreadExecutor()
    // Currently executing prompt future - we can't execute two at once else the mediapipe
    // backend crashes
    private var currentJob: Job? = null
    // Current prompt that is executing - we store this to avoid re-creating the current execution
    // if there is already one running with the exact same prompt (e.g. if user clicks search twice
    // for the same query)
    private var currentPrompt: String? = null

    val ragPipeline: RagPipeline by lazy { RagPipeline(application) }

    // Channel sink that sends to flutter
    var latestGeneration: EventChannel.EventSink? = null

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        requestLlmPreinit()
        this.latestGeneration = events
    }

    override fun onCancel(arguments: Any?) {}

    // Wait for the LLM to initialise
    suspend fun waitForLlmInit() {
        // Wait for llm to be ready via rendezvous channel
        if (!ragPipeline.llmReady) {
            ragPipeline.onLlmReady.receive()
        }
    }

    fun requestLlmPreinit() {
        ragPipeline // Force lazy initialization to happen now
    }

    // Generate a response to the prompt, sending updates to Flutter as it is being generated
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