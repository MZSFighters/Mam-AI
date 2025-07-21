import 'package:app/screens/chat_screen.dart';
import 'package:app/screens/intro_page.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const ChatApp());
}

class ChatApp extends StatelessWidget {
  const ChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Mam AI Chat',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const IntroPage(),
      routes: {
        '/chat': (context) => const ChatPage(),
      },
    );
  }
}
