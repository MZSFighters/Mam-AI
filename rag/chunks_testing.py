import PyPDF2
from mmore.process.processors.pdf_processor import PDFProcessor
from mmore.process.processors.base import ProcessorConfig
from mmore.type import MultimodalSample
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from rich import print
import re
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from langchain_core.embeddings import Embeddings
from unstructured.partition.md import partition_md
from objectbox import Id, String, Float32Vector, VectorDistanceType, HnswIndex, Entity, Store, Int32
import os
from typing import List


@Entity()
class Chunk:
    id = Id()
    text = String()
    title = String()
    page = Int32()
    embeddings = Float32Vector(index=HnswIndex(dimensions=768, distance_type=VectorDistanceType.COSINE))


class GeckoEmbeddings(Embeddings):
    """Custom embeddings class for local Gecko TensorFlow Lite model"""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.interpreter = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the TensorFlow Lite model and SentencePiece tokenizer"""
        print(f"Loading Gecko TFLite model: {self.model_path}")
        print(f"Loading SentencePiece tokenizer: {self.tokenizer_path}")

        try:
            # Load TensorFlow Lite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=24)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"Model input shape: {self.input_details[0]['shape']}")
            print(f"Model output shape: {self.output_details[0]['shape']}")

            # Load SentencePiece tokenizer
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.tokenizer_path)

            print("Gecko model and tokenizer loaded successfully")

        except Exception as e:
            print(f"Error loading Gecko model: {e}")
            raise

    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using SentencePiece tokenizer"""
        # Get the expected input length from model
        max_length = self.input_details[0]['shape'][1]

        # Tokenize text
        token_ids = self.tokenizer.encode_as_ids(text)

        # Pad or truncate to expected length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids = token_ids + [0] * (max_length - len(token_ids))

        return np.array([token_ids], dtype=np.int32)

    def _count_tokens(self, text: str) -> int:
        """Count tokens for a given text"""
        token_ids = self.tokenizer.encode_as_ids(text)
        return len(token_ids)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:

            # ----------------------------------------------------------------------
            # Count tokens and characters
            char_count = len(text)
            token_count = self._count_tokens(text)

            # Print side by side
            print(f"Chars: {char_count:4d} | Tokens: {token_count:4d}")
            # ----------------------------------------------------------------------

            # Tokenize input
            input_tokens = self._tokenize_text(text)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tokens)

            # Run inference
            self.interpreter.invoke()

            # Get output
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Return as list (flatten if needed)
            return embedding.flatten().tolist()

        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            output_shape = self.output_details[0]['shape']
            embedding_dim = np.prod(output_shape[1:])  # Calculate total embedding dimension
            return [0.0] * embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        print(f"Embedding {len(texts)} documents...")
        embeddings = []

        for i, text in enumerate(texts):
            if i % 10 == 0:  # Progress indicator
                print(f"Embedding document {i + 1}/{len(texts)}")

            embedding = self._get_embedding(text)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)


def create_vector_store(chunks: List[Document], title_page_list):
    """Add chunks to vector store"""
    print(f"Adding {len(chunks)} chunks to vector store...")

    objectbox_store = Store(directory="db")
    box = objectbox_store.box(Chunk)

    embeddings = GeckoEmbeddings(model_path="Gecko_1024_quant.tflite", tokenizer_path="sentencepiece.model")

    for meta, chunk in zip(title_page_list, chunks):
        box.put(Chunk(text=chunk.page_content, embeddings=embeddings._get_embedding(chunk.page_content), title=meta[0],
                      page=int(meta[1])))


# Split and Save the pages of each pdf in the output folder
def split_and_save_pages(pdf_path, output_folder=None):
    """
    Split the PDF into its individual page files and save them in the output directory.

    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save individual pages (optional)
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            pages = list(range(num_pages))

            # If output folder is specified, save individual pages into the output folder
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]

                for i, page in enumerate(pdf_reader.pages):
                    pdf_writer = PyPDF2.PdfWriter()
                    pdf_writer.add_page(page)

                    output_path = os.path.join(output_folder, f"{base_name}_page_{i + 1}.pdf")
                    with open(output_path, 'wb') as output_file:
                        pdf_writer.write(output_file)

                print(f"Saved {num_pages} individual page files to {output_folder}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []


def process_text_file(input_file_path, output_file_path=None):
    """
    Process a text file by applying regex transformations and removing <attachment> tags.

    Args:
        input_file_path (str): Path to the input text file
        output_file_path (str, optional): Path for the output file. If None, overwrites input file.

    Returns:
        str: The processed text content
    """
    # Read the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        txt = file.read()

    # Apply regex transformations
    # 1. Replace multiple newlines (with optional spaces) with double newlines
    txt = re.sub(r"\s?\n(\s?\n)+", "\n\n", txt)

    # 2. Replace multiple spaces with single space
    txt = re.sub(r"\s\s(\s+)", " ", txt)

    # 3. Replace multiple dashes after pipe with single dash
    txt = re.sub(r"\|--+", "|-", txt)

    # 4. Remove all instances of <attachment>
    txt = re.sub(r"<attachment>", "", txt)

    txt = re.sub(r"<br>", " ", txt)

    txt = re.sub(r"<br/>", " ", txt)

    # Write to output file
    if output_file_path is None:
        output_file_path = input_file_path

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(txt)

    print(f"File processed successfully. Output saved to: {output_file_path}")
    return txt


def add_meta_tags(input_file, output_file):
    """
    Processes a JSONL file to extract PDF text and add formatted meta tags.

    Args:
        input_file (str): Path to input JSONL file containing PDF data
        output_file (str): Path to output text file with combined content
    """
    combined_text = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            # Skip if text is empty or contains only whitespace
            text_content = data.get('text', '').strip()
            if not text_content:
                continue

            # Get file path and format metadata
            file_path = data['metadata']['file_path']
            filename = os.path.basename(file_path).replace('.pdf', '')

            # Extract document name and page number
            parts = filename.split('_page_')
            if len(parts) == 2:
                doc_name = parts[0]
                page_num = parts[1]
                meta = f"<meta>{doc_name}; Page: {page_num}</meta>"
            else:
                meta = f"<meta>{filename}</meta>"

            # Combine metadata and text
            combined_text.append(meta + data['text'])

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write('\n\n'.join(combined_text))

    print(f"Processed {len(combined_text)} files -> {output_file}")


def is_table(text: str) -> bool:
    """Check if text contains a markdown table by counting pipes in lines."""
    lines = text.strip().split('\n')

    for line in lines:
        pipe_count = line.count('|')
        if pipe_count >= 2:
            return True

    return False


def load_and_chunk_text(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    print(f"Loading text file: {file_path}")

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]

    loader = TextLoader(file_path, encoding='utf-8')
    text_content = loader.load()[0].page_content

    print("Markdown Aware Text Splitting...")
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(text_content)

    print("Processing chunks with table awareness...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = []

    for doc in md_header_splits:
        if is_table(doc.page_content):
            # --- Start of Implemented Logic ---
            lines = doc.page_content.split('\n')
            table_start_idx, table_end_idx = -1, -1

            # Find the start of the first table
            for i, line in enumerate(lines):
                if line.strip().count('|') >= 2:
                    table_start_idx = i
                    break

            # Find the end of the same contiguous table
            if table_start_idx != -1:
                table_end_idx = table_start_idx
                for i in range(table_start_idx + 1, len(lines)):
                    if lines[i].strip().count('|') >= 2:
                        table_end_idx = i
                    else:
                        break  # End of the table block

            # Define the three parts: text before, the table, and text after
            text_before = "\n".join(lines[:table_start_idx])
            table_content = "\n".join(lines[table_start_idx: table_end_idx + 1])
            text_after = "\n".join(lines[table_end_idx + 1:])

            # Helper to process each part
            def add_part(content: str, metadata: dict):
                if content.strip():  # Only process if there is content
                    part_doc = Document(page_content=content, metadata=metadata)
                    if len(content) > chunk_size:
                        # If part is too big, split it
                        chunks = text_splitter.split_documents([part_doc])
                        final_chunks.extend(chunks)
                    else:
                        # Otherwise, add it as one chunk
                        final_chunks.append(part_doc)

            # Process the three parts in order
            add_part(text_before, doc.metadata)
            add_part(table_content, doc.metadata)
            add_part(text_after, doc.metadata)
            # --- End of Implemented Logic ---
        else:
            # If no table is detected, chunk the document normally
            chunks = text_splitter.split_documents([doc])
            final_chunks.extend(chunks)

    # Final cleanup and report
    final_chunks = [chunk for chunk in final_chunks if chunk.page_content.strip()]
    if not final_chunks:
        print("Warning: No chunks were created.")
        return []

    chunk_sizes = [len(chunk.page_content) for chunk in final_chunks]
    print(f"Created {len(final_chunks)} chunks")
    print(
        f"Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes) / len(chunk_sizes):.1f}")

    return final_chunks

def parse_meta_tag(meta_tag_text):
    # Remove the <meta> tags
    content = meta_tag_text.replace('<meta>', '').replace('</meta>', '')

    # Split by semicolon to separate title and page info
    parts = content.split(';')

    # Extract title (first part, stripped of whitespace)
    title = parts[0].strip()

    # Extract page number (second part, remove "Page:" and strip whitespace)
    page_number = parts[1].replace('Page:', '').strip()

    return title, page_number


# save the chunks into a .txt file and meta tag handling post chunking
def save_chunks(chunks, title_page_list, output_chunks_path):
    processed_chunks = []

    for chunk in chunks:
        # Convert chunk to string if it's not already
        # chunk_text = str(chunk)
        chunk_text = chunk.page_content
        if '<meta>' in chunk_text:
            # Case 1: Chunk contains meta tags
            # Find first meta tag
            start = chunk_text.find('<meta>')
            end = chunk_text.find('</meta>') + 7
            first_meta = chunk_text[start:end]

            # Remove all meta tags from the text
            text = chunk_text
            # print(text)
            while '<meta>' in text:
                start = text.find('<meta>')
                end = text.find('</meta>') + 7
                text = text[:start] + text[end:]

            # Put first meta tag at the front

            processed_chunk = first_meta + text
            relevant_meta_tag = first_meta
        else:

            # Case 2: No meta tags, use previous chunk's meta tag
            processed_chunk = relevant_meta_tag + chunk_text

        title, page = parse_meta_tag(relevant_meta_tag)
        inst = (title, page)
        title_page_list.append(inst)
        # print(inst)
        processed_chunks.append(processed_chunk)

    # Save the tagged docs to file
    with open(output_chunks_path, 'w', encoding='utf-8') as f:
        f.write('<SEP>'.join(processed_chunks))

    print(f"Saved {len(processed_chunks)} chunks to {output_chunks_path}")


# --------------------------------------------------- End-To-End Pipeline -------------------------------------------------------

def main():
    # Step 1: Split the pages of each PDF file and save them in output_pages folder

    # pdf_folder = "./data"       # folder with all the guideline pdf files
    # pdfs = glob.glob(os.path.join(pdf_folder, "**", "*.pdf"), recursive=True)

    # for pdf_file in pdfs:
    #     if os.path.exists(pdf_file):
    #         print(f"Processing and saving pages from: {pdf_file}")
    #         split_and_save_pages(pdf_file, "output_pages")

    # # Step 2: Use Mmore to extract text from each pdf file in the output_pages and save them to example.jsonl file

    # pages_folder = "./output_pages"

    # # Collect all PDF pages from output_pages and sort them in order for consistency
    # pdf_file_paths = glob.glob(os.path.join(pages_folder, "*.pdf"))
    # # print(pdf_file_paths)
    # out_file = "./example.jsonl"

    # pdf_processor_config = ProcessorConfig(custom_config={"output_path": "examples/process/outputs"})
    # pdf_processor = PDFProcessor(config=pdf_processor_config)
    # result_pdf = pdf_processor.process_batch(pdf_file_paths, False, 16) # args: file_paths, fast mode (True/False), num_workers

    # MultimodalSample.to_jsonl(out_file, result_pdf)

    # Step 3: Add meta tags to the extracted text of each page and then merge all the text into a single .txt file

    add_meta_tags('example.jsonl', 'combined_pdf_texts.txt')

    process_text_file("./combined_pdf_texts.txt")

    # Step 4: Chunking and Tagged Docs
    text_file_path = "./combined_pdf_texts.txt"
    chunks = load_and_chunk_text(file_path=text_file_path, chunk_size=2000, chunk_overlap=400)

    output_chunks_path = "chunks_tagged_docs.txt"  # Output file for chunks
    title_page_list = []
    # Save chunks to file with <sep> separator
    # print(chunks)
    save_chunks(chunks, title_page_list, output_chunks_path)

    # create_vector_store(chunks, title_page_list)


if __name__ == "__main__":
    main()