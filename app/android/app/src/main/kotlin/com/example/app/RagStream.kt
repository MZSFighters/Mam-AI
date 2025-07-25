package com.example.app

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.LifecycleCoroutineScope
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class RagStream(application: Application, val lifecycleScope: LifecycleCoroutineScope): EventChannel.StreamHandler {
    private val backgroundExecutor: Executor = Executors.newSingleThreadExecutor()
    private var currentJob: Job? = null
    private var currentPrompt: String? = null

    val ragPipeline: RagPipeline by lazy { RagPipeline(application) }
    var events: EventChannel.EventSink? = null

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        ensureLlmInit()
        this.events = events
    }

    override fun onCancel(arguments: Any?) {}

    fun ensureLlmInit() {
        ragPipeline // Force lazy initialization to happen now
    }

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
                    ragPipeline
                        .generateResponse(prompt) { response, done ->
                            // We are off the ui thread at the moment. You can only send
                            // messages through event channels while on ui thread.
                            // The `post` puts the sending part back onto the ui thread
                            Handler(Looper.getMainLooper()).post {
                                events?.success(
                                    response.text
                                )

                                if (done) {
                                    synchronized(this) {
                                        currentJob = null
                                        currentPrompt = null
                                    }
                                }
                            }
                        }
                }
            }
        }
    }
}