import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:markdown_widget/markdown_widget.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();

  static void requestLlmPreinit() {
    print("init LLM");
    _ChatPageState.platform.invokeMethod("ensureInit");
  }
}

class _ChatPageState extends State<ChatPage> {
  String? _latestMessage;
  static const platform = MethodChannel("io.github.mzsfighters.mam_ai/request_generation");
  static const latestMessageStream = EventChannel("io.github.mzsfighters.mam_ai/latest_message");
  StreamSubscription? _latestMessageSubscription;
  SearchController controller = SearchController();
  bool _searchedBefore = false;

  Future<void> _generateResponse(String prompt) async {
    try {
      setState(() {
        _searchedBefore = true;
        _latestMessage = null;
      });

      await platform.invokeMethod<int>("generateResponse", prompt);
    } on PlatformException catch (e) {
      print("Error: $e");
    }
  }

  void _startListeningForLatestMessage() {
    _latestMessageSubscription = latestMessageStream.receiveBroadcastStream().listen(_onLatestMessageUpdate);
  }

  void _onLatestMessageUpdate(value) {
    setState(() {
      _latestMessage = value;
    });
  }

  @override
  void dispose() {
    super.dispose();
    _latestMessageSubscription?.cancel();
  }

  void onSubmit(String text) {
    // For some reason, if we close an already closed view, it will make the
    // entire screen black - I think that this is probably poorly-isolated code
    // given that it affects the entire _screen_ and not only the widget it is
    // supposed to control
    if (controller.isOpen) {
      controller.closeView(text);
    } else {
      // We still want to set the text even if the search view is closed
      controller.text = text;
    }

    _generateResponse(text);
  }

  @override
  Widget build(BuildContext context) {
    var examples = ["Baby continuous crying", "Preparing for home birth", "Infection risks childbirth"];
    var history = ["I searched this once", "I also searched this"];

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_latestMessageSubscription == null) {
        _startListeningForLatestMessage();
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'MAM-AI clinical search',
          style: TextStyle(color: Colors.white),
        ),
        centerTitle: true,
        backgroundColor: Colors.blueAccent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SearchAnchor(
              shrinkWrap: true,
              searchController: controller,
              viewOnSubmitted: onSubmit,
              builder: (BuildContext context, SearchController controller) {
                return SearchBar(
                  constraints: const BoxConstraints(minWidth: 360.0, minHeight: 56.0),
                  leading: Padding(
                    padding: const EdgeInsets.only(left: 8.0, top: 12.0, right: 0.0, bottom: 8.0),
                    child: Icon(Icons.search),
                  ),
                  controller: controller,
                  hintText: "Search in medical guidelines...",
                  onSubmitted: onSubmit,
                  onTap: controller.openView,
                  onChanged: (_) => controller.openView(),
                );
              },
              suggestionsBuilder: (BuildContext context, SearchController controller) {
                RegExp regex = RegExp(RegExp.escape(controller.text.toLowerCase()));
                return history
                    .map((text) => SearchSuggestionTile(text, SuggestionType.history, onPressed: onSubmit))
                    .followedBy(examples.map((text) => SearchSuggestionTile(text, SuggestionType.example, onPressed: onSubmit)))
                    .where((tile) => regex.hasMatch(tile.text.toLowerCase()))
                    .toList();
              },
            ),
            const SizedBox(height: 16),

            Expanded(
              child: SingleChildScrollView(
                child:
                  (_searchedBefore)
                    ? SearchOutput(summary: _latestMessage)
                    : Container(
                      margin: const EdgeInsets.only(bottom: 20),
                      child: Wrap(
                          alignment: WrapAlignment.center,
                          spacing: 5,
                          runSpacing: 5,
                          children: history
                              .map((text) => SearchSuggestionChip(text, SuggestionType.history, onPressed: onSubmit))
                              .followedBy(examples.map((text) => SearchSuggestionChip(text, SuggestionType.example, onPressed: onSubmit)))
                              .toList()
                      ),
                    ),
              ),
            )
          ],
        ),
      ),
    );
  }
}

enum SuggestionType {
  example,
  history,
}

class SearchSuggestionTile extends StatelessWidget {
  const SearchSuggestionTile(this.text, this.type, {super.key, required this.onPressed});

  final String text;
  final Function(String) onPressed;
  final SuggestionType type;

  @override
  Widget build(BuildContext context) {
    Icon icon;
    Color? textColor;

    switch(type) {
      case SuggestionType.example:
        // !!!IMPORTANT!!! If you want to modify any colours, please run them
        // through WCAG contrast checker first:
        // https://webaim.org/resources/contrastchecker/
        // Spare a thought for legibility and accessibility

        textColor = Color(0xFF0041B3);
        icon = Icon(Icons.auto_awesome, color: textColor);
        break;

      case SuggestionType.history:
        icon = Icon(Icons.history);
        break;
    }

    return ListTile(
      leading: icon,
      title: Text(
        text,
        style: TextStyle(color: textColor)
      ),
      onTap: () => onPressed(text),
    );
  }
}

class SearchSuggestionChip extends StatelessWidget {
  const SearchSuggestionChip(this.text, this.type, {super.key, required this.onPressed});

  final String text;
  final Function(String) onPressed;
  final SuggestionType type;

  @override
  Widget build(BuildContext context) {
    Icon icon;
    Color? bgColor;
    Color? textColor;
    Color borderColor;

    switch(type) {
      case SuggestionType.example:
        icon = Icon(Icons.auto_awesome);
        bgColor = Colors.blue[50];

        // Passes WCAG
        textColor = Color(0xFF0041B3);
        borderColor = Colors.blue[300]!;
        break;

      case SuggestionType.history:
        // Greyest grey that still passed WCAG AAA (7:1 contrast ratio)
        textColor = Colors.black.withAlpha(166);
        icon = Icon(Icons.history, color: textColor);
        bgColor = null;
        borderColor = Colors.grey;
        break;
    }

    return ChipTheme(
      data: ChipThemeData(
        labelStyle: TextStyle(color: textColor, fontWeight: FontWeight.w500),
        backgroundColor: bgColor,
        shape: RoundedRectangleBorder(
          side: BorderSide(color: borderColor),
          borderRadius: BorderRadiusGeometry.circular(12),
        ),
      ),
      child: ActionChip(
        avatar: icon,
        label: Text(
          text,
        ),
        onPressed: () => onPressed(text),
      ),
    );
  }
}

class SearchOutput extends StatelessWidget {
  const SearchOutput({super.key, required this.summary});

  final String? summary;

  @override
  Widget build(BuildContext context) {
    if (summary == null) {
      return Center(child: SizedBox(width: 75, height: 75, child: CircularProgressIndicator()));
    }

    return Column(
      children: [
        Card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ListTile(
                leading: const Icon(Icons.auto_awesome, color: Colors.purple),
                title: const Text('Generated summary'),
                subtitle: RichText(text: TextSpan(
                    text: '⚠️ Read with care. ',
                    style: DefaultTextStyle.of(context).style,
                    children: [
                      TextSpan(
                          text: 'AI can make serious mistakes! ⚠️',
                          style: TextStyle(fontWeight: FontWeight.bold)
                      )
                    ])
                ),
                contentPadding: const EdgeInsetsDirectional.only(start: 16.0, end: 24.0),
              ),
              Padding(
                padding: EdgeInsetsDirectional.only(start: 16.0, end: 24.0, bottom: 16.0),
                child: MarkdownBlock(data: summary!),
              )
            ],
          )
        ),
        Card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const ListTile(
                  leading: Icon(Icons.book),
                  title: Text('Title of some guideline 1'),
                  contentPadding: EdgeInsetsDirectional.only(start: 16.0, end: 24.0),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.only(start: 16.0, end: 24.0, bottom: 16.0),
                  child: Text("... some text from the guideline (the original source used in RAG) ..."),
                )
              ],
            )
        ),
        Card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const ListTile(
                  leading: Icon(Icons.book),
                  title: Text('Title of some guideline 2'),
                  contentPadding: EdgeInsetsDirectional.only(start: 16.0, end: 24.0),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.only(start: 16.0, end: 24.0, bottom: 16.0),
                  child: Text("... some text from the guideline (the original source used in RAG) ..."),
                )
              ],
            )
        ),
      ],
    );
  }
}