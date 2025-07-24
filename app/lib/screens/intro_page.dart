import 'package:flutter/material.dart';

class IntroPage extends StatelessWidget {
  const IntroPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28.0, vertical: 24.0),
            child: Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: 400),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 32),
                    // Logo or illustration for the health chatbot
                    Center(
                      child: Column(
                        children: [
                          CircleAvatar(
                            radius: 56,
                            backgroundColor: Colors.blue.shade50,
                            child: Icon(
                              Icons.health_and_safety,
                              color: Colors.blueAccent,
                              size: 64,
                            ),
                          ),
                          const SizedBox(height: 24),
                          Text(
                            'Welcome to MAM-Ai',
                            style: TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                              color: Colors.blueAccent,
                              letterSpacing: 1.2,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'An edge-based AI assistant designed to support midwives in Zanzibar, offering offline functionality and Swahili language support for reliable, context-aware care.',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey[700],
                              height: 1.5,
                            ),
                            textAlign: TextAlign.left,
                          ),
                        ],
                      ),
                    ),
                    // Partners section
                    SizedBox(height: 20),
                    Column(
                      children: [
                        Text(
                          'In partnership with',
                          style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 5),
                        Wrap(
                          alignment: WrapAlignment.center,
                          spacing: 20.0,
                          runSpacing: 5.0,
                          children: [
                            // Partner logos
                            Image.asset('images/epfl.png', height: 15),
                            Image.asset('images/light.png', height: 20),
                            Image.asset('images/swiss_tph.png', height: 20),
                            Image.asset('images/d-tree.jpg', height: 20),
                            // Add more partners as needed
                          ],
                        ),
                        const SizedBox(height: 20),
                        Text(
                          'Supported by',
                          style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 2),
                        Image.asset('images/swiss_dev_coop.webp', height: 100),
                      ],
                    ),
                    const SizedBox(height: 36),
                    // Next button
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pushNamed(context, '/chat');
                      },
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(20),
                        backgroundColor: Colors.blueAccent,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(5.0),
                        ),
                        elevation: 2,
                      ),
                      child: const Text(
                        'Start Chat',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.1,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
