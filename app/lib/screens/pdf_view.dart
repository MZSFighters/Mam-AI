import 'package:flutter/material.dart';
import 'package:flutter_pdfview/flutter_pdfview.dart';

class PdfView extends StatelessWidget {
  const PdfView({super.key});

  @override
  Widget build(BuildContext context) {
    PdfViewArguments args = ModalRoute.of(context)!.settings.arguments as PdfViewArguments;

    print(args.path);
    print(args.page);

    return Material(
      child: Scaffold(
        appBar: AppBar(
          toolbarHeight: 64,
          title: Text(
              args.title,
              style: const TextStyle(color: Colors.white),
          ),
          centerTitle: true,
          backgroundColor: Colors.deepOrange,
        ),
        body: PDFView(
          filePath: args.path,
          // enableSwipe: true,
          // swipeHorizontal: true,
          autoSpacing: false,
          pageFling: false,
          defaultPage: args.page,
          backgroundColor: Colors.grey,
        ),
      ),
    );
  }

}

class PdfViewArguments {
  const PdfViewArguments({required this.path, required this.title, required this.page});
  final String path;
  final String title;
  final int page;
}
