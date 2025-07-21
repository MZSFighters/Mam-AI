import 'package:flutter/material.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mam AI Medical Chat'),
        backgroundColor: Colors.blueAccent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
        // Search-like input field
        Container(
          decoration: BoxDecoration(
            color: Colors.grey[200],
            borderRadius: BorderRadius.circular(30),
          ),
          child: Row(
            children: [
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 12.0),
            child: Icon(Icons.search, color: Colors.blueAccent),
          ),
          Expanded(
            child: TextField(
              decoration: const InputDecoration(
            hintText: 'Ask a medical question...',
            border: InputBorder.none,
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.send, color: Colors.blueAccent),
            onPressed: () {
              // Handle send action
            },
          ),
            ],
          ),
        ),
        const SizedBox(height: 20),
        // Chat messages area
        Expanded(
          child: ListView(
            children: [
          // Example user message
          Align(
            alignment: Alignment.centerRight,
            child: Container(
              margin: const EdgeInsets.symmetric(vertical: 4),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
            color: Colors.teal[100],
            borderRadius: BorderRadius.circular(16),
              ),
              child: const Text('What are the symptoms of flu?'),
            ),
          ),
          // Example bot response
          Align(
            alignment: Alignment.centerLeft,
            child: Container(
              margin: const EdgeInsets.symmetric(vertical: 4),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
            color: Colors.grey[300],
            borderRadius: BorderRadius.circular(16),
              ),
              child: const Text(
            'Common symptoms of flu include fever, cough, sore throat, muscle aches, and fatigue.',
              ),
            ),
          ),
          // Add more messages here
            ],
          ),
        ),
          ],
        ),
      ),
    );
  }
}