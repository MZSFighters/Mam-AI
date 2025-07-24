import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  String? _latestMessage;
  static const platform = MethodChannel("io.github.mzsfighters.mam_ai/request_generation");
  static const latestMessageStream = EventChannel("io.github.mzsfighters.mam_ai/latest_message");
  StreamSubscription? _latestMessageSubscription;

  final TextEditingController _textController = TextEditingController();

  Future<void> _generateResponse(String prompt) async {
    try {
      await platform.invokeMethod<int>("generateResponse", prompt);
    } on PlatformException catch (e) {
      print("Error: $e");
    }
  }

  void _startListeningForLatestMessage() {
    _latestMessageSubscription = latestMessageStream.receiveBroadcastStream().listen(_onLatestMessageUpdate);
  }

  void _stopListeningForLatestMessage() {
    _latestMessageSubscription?.cancel();
    setState(() {
      _latestMessage = '<channel broken>';
    });
  }

  void _onLatestMessageUpdate(value) {
    setState(() {
      _latestMessage = value;
    });
  }

  @override
  void dispose() {
    super.dispose();
    _stopListeningForLatestMessage();
  }


  @override
  Widget build(BuildContext context) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_latestMessageSubscription == null) {
        _startListeningForLatestMessage();
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Medical Chat',
          style: TextStyle(color: Colors.white),
        ),
        centerTitle: true,

        backgroundColor: Colors.blueAccent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Chatbot welcome message
              Column(
                children: const [
                  Icon(
                    Icons.account_circle,
                    size: 60,
                    color: Colors.blueAccent,
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Welcome to Mam AI!',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                      color: Colors.blueAccent,
                    ),
                  ),
                ],
              ),
              // Medical chatbot themed input field, centered
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.lightBlue[50],
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(color: Colors.blueAccent, width: 2),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.blueAccent.withValues(alpha: 0.1),
                      blurRadius: 8,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                width: 400, // Fixed width for centering
                child: Row(
                  children: [
                    const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 8.0),
                      child: Icon(
                        Icons.medical_services,
                        color: Colors.blueAccent,
                      ),
                    ),
                    Expanded(
                      child: Center(
                        child: TextField(
                          controller: _textController,
                          textAlign: TextAlign.center,
                          decoration: const InputDecoration(
                            hintText: 'Ask your medical question...',
                            hintStyle: TextStyle(
                              color: Colors.blueGrey,
                              fontStyle: FontStyle.italic,
                            ),
                            border: InputBorder.none,
                          ),
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(
                        Icons.send_rounded,
                        color: Colors.blueAccent,
                      ),
                      onPressed: () {
                        _generateResponse(_textController.text);
                      },
                    ),
                  ],
                ),
              ),
              // Suggested prompts
              Container(
                margin: const EdgeInsets.symmetric(vertical: 16),

                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 9,
                        vertical: 6,
                      ),
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      decoration: BoxDecoration(
                        color: Colors.blue[100],
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: const Text(
                        'What should I do when the baby is crying?',
                        style: TextStyle(
                          color: Colors.blueAccent,
                          fontSize: 14,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              Text(_latestMessage ?? "<Not yet initialized>"),
            ],
          ),
        ),
      ),
    );
  }
}
