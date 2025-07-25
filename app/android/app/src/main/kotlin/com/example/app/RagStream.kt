package com.example.app

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.LifecycleCoroutineScope
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class RagStream(application: Application, val lifecycleScope: LifecycleCoroutineScope): EventChannel.StreamHandler {
    private val backgroundExecutor: Executor = Executors.newSingleThreadExecutor()
    private val executorService = Executors.newSingleThreadExecutor()

    val ragPipeline: RagPipeline by lazy { RagPipeline(application) }
    var events: EventChannel.EventSink? = null

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        ragPipeline // Force lazy initialization to happen now
        this.events = events
    }

    override fun onCancel(arguments: Any?) {}

    fun generateResponse(prompt: String) {
        executorService.submit {
            lifecycleScope.launch {
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
                            }
                        }
                }
            }
        }
    }
}