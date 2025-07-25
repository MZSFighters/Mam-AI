package com.example.app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import androidx.lifecycle.lifecycleScope
import io.flutter.plugin.common.EventChannel

class MainActivity : FlutterActivity() {
    private val channel = "io.github.mzsfighters.mam_ai/request_generation"
    private val latestMessageEventChannel = "io.github.mzsfighters.mam_ai/latest_message"
    private lateinit var ragStream: RagStream

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        ragStream = RagStream(application, lifecycleScope)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channel).setMethodCallHandler {
                call, result ->
            when (call.method) {
                "ensureInit" -> {
                    ragStream.ensureLlmInit()
                    result.success(0)
                }
                "generateResponse" -> {
                    ragStream.generateResponse(call.arguments<String>()!!)
                    result.success(0)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, latestMessageEventChannel).setStreamHandler(ragStream)
    }
}

