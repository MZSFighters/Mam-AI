import os
from typing import List, Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from rich import print
from rich.console import Console
from rich.table import Table
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from langchain_core.embeddings import Embeddings


console = Console()

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    device: str = "cuda"
    top_k: int = 3
    gecko_model_path: str = "Gecko_1024_quant.tflite"
    tokenizer_path: str = "sentencepiece.model"


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
        console.print(f"Loading Gecko TFLite model: {self.model_path}")
        console.print(f"Loading SentencePiece tokenizer: {self.tokenizer_path}")
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=24)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            console.print(f"Model input shape: {self.input_details[0]['shape']}")
            console.print(f"Model output shape: {self.output_details[0]['shape']}")
            
            # Load SentencePiece tokenizer
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.tokenizer_path)
            
            console.print("Gecko model and tokenizer loaded successfully")
            
        except Exception as e:
            console.print(f"Error loading Gecko model: {e}")
            raise
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using SentencePiece tokenizer"""
        max_length = self.input_details[0]['shape'][1]
        
        # tokenizing text
        token_ids = self.tokenizer.encode_as_ids(text)
        
        # pad or truncate to expected length
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
            
            #----------------------------------------------------------------------
            # Count tokens and characters
            char_count = len(text)
            token_count = self._count_tokens(text)
            console.print(f"Chars: {char_count:4d} | Tokens: {token_count:4d}")
            #----------------------------------------------------------------------
            input_tokens = self._tokenize_text(text)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tokens)
            
            self.interpreter.invoke()
            
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])

            return embedding.flatten().tolist()
            
        except Exception as e:
            console.print(f"Error getting embedding: {e}")
            output_shape = self.output_details[0]['shape']
            embedding_dim = np.prod(output_shape[1:])
            return [0.0] * embedding_dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        console.print(f"Embedding {len(texts)} documents...")
        embeddings = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0:  # Progress indicator
                console.print(f"Embedding document {i+1}/{len(texts)}")
            
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)


class LLMInterface(ABC):
    """Abstract base class for LLM models to ensure model agnosticity"""
    
    @abstractmethod
    def generate(self, prompt: str, context: List[Document]) -> str:
        """Generate response given prompt and context"""
        pass


class HuggingFaceLLM(LLMInterface):
    """HuggingFace LLM implementation with actual model loading"""
    
    def __init__(self, model_name: str = "google/gemma-3n-E4B-it", device: str = "cuda"):
        self.model_name = model_name 
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the HuggingFace model and tokenizer"""
        console.print(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it does not exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            console.print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            console.print(f"Error loading model: {e}")
            
        
    def _create_prompt(self, question: str, context: List[Document]) -> str:
        """Create a well-formatted prompt for the model"""
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

        Context:
        {context_text}

        Question: {question}

        Answer:"""
        
        return prompt
        
    def generate(self, prompt: str, context: List[Document]) -> str:
        """Generate response using the loaded HuggingFace model"""
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded properly"
            
        # Create the full prompt
        full_prompt = self._create_prompt(prompt, context)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            answer = response[len(full_prompt):].strip()
            return answer if answer else "I couldn't generate a proper response based on the given context."
            
        except Exception as e:
            console.print(f"Error during generation: {e}")
            return f"Error generating response: {str(e)}"



class FixedSizeTextSplitter:
    """Custom text splitter that creates exactly sized chunks with overlap"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into exactly sized chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
   
            if end < len(text):
                search_start = max(start, end - 100)
                sentence_endings = ['.', '!', '?', '\n\n']
                
                best_break = end
                for ending in sentence_endings:
                    last_pos = text.rfind(ending, search_start, end)
                    if last_pos != -1 and last_pos > search_start:
                        best_break = last_pos + 1
                        break
                
                chunk = text[start:best_break]
            else:
                chunk = text[start:]
            
            chunks.append(chunk)
            
            # Move start position considering overlap
            if end >= len(text):
                break
                
            start = end - self.chunk_overlap
            
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into fixed-size chunks"""
        chunks = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(text_chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'chunk_size': len(chunk)
                    }
                )
                chunks.append(new_doc)
                
        return chunks


# State definition for LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAGPipeline:
    """Main RAG Pipeline class with fixed-size chunking"""
    
    def __init__(self, config: RAGConfig, llm: Optional[LLMInterface] = None):
        self.config = config
        self.llm = llm or HuggingFaceLLM(device=config.device)
        self.embeddings = None
        self.vector_store = None
        self.graph = None
        self._setup_embeddings()
        self._setup_vector_store()
        
    def _setup_embeddings(self):
        """Initialize embedding model"""
        console.print(f"Setting up Gecko embeddings from: {self.config.gecko_model_path}")
        self.embeddings = GeckoEmbeddings(
            model_path=self.config.gecko_model_path,
            tokenizer_path=self.config.tokenizer_path
        )
        
    def _setup_vector_store(self):
        """Initialize vector store"""
        console.print("Setting up vector store...")
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def load_and_chunk_text(self, file_path: str) -> List[Document]:
        """Load and chunk text file with exactly sized chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        console.print(f"Loading text file: {file_path}")
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

      
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_content = documents[0].page_content
        
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(text_content)

        print(md_header_splits)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)

        # text_splitter = FixedSizeTextSplitter(
        #         chunk_size=self.config.chunk_size,
        #         chunk_overlap=self.config.chunk_overlap
        #     )

        chunks = text_splitter.split_documents(md_header_splits)
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        console.print(f"Created {len(chunks)} chunks")
        console.print(f"Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        return chunks
        
    def save_chunks_to_file(self, chunks: List[Document], output_path: str, separator: str = "<docsep>"):
        """Save all chunks to a text file separated by a specified token"""
        console.print(f"Saving {len(chunks)} chunks to {output_path}...")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    # Write the chunk content
                    f.write(chunk.page_content)
                    
                    
                    if i < len(chunks) - 1:
                        f.write(f"\n{separator}\n")
                        
            console.print(f"[green]Successfully saved chunks to {output_path}[/green]")
            
            file_size = os.path.getsize(output_path)
            console.print(f"Output file size: {file_size:,} bytes")
            
        except Exception as e:
            console.print(f"[red]Error saving chunks to file: {e}[/red]")
            raise    
        
    def create_vector_store(self, chunks: List[Document]):
        """Add chunks to vector store"""
        console.print(f"Adding {len(chunks)} chunks to vector store...")
        self.vector_store.add_documents(documents=chunks)
        console.print("Vector store created successfully")
        
    def setup_rag_graph(self):
        """Setup the RAG pipeline graph"""
        def retrieve(state: State):
            """Retrieve relevant documents"""
            retrieved_docs = self.vector_store.similarity_search(
                state["question"], 
                k=self.config.top_k
            )
            return {"context": retrieved_docs}
            
        def generate_answer(state: State):
            """Generate answer using LLM"""
            answer = self.llm.generate(state["question"], state["context"])
            return {"answer": answer}
        
        # the graph
        graph_builder = StateGraph(State)
        graph_builder.add_sequence([retrieve, generate_answer])
        graph_builder.add_edge(START, "retrieve")
        
        self.graph = graph_builder.compile()
        console.print("RAG pipeline graph setup complete")
        
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        if self.graph is None:
            raise ValueError("RAG graph not setup. Call setup_rag_graph() first.")
            
        console.print(f"Processing query: {question}")
        response = self.graph.invoke({"question": question})
        
        return response
        
    def get_retrieved_contexts(self, question: str) -> List[Document]:
        """Get only the retrieved contexts without generating answer"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
            
        retrieved_docs = self.vector_store.similarity_search(
            question, 
            k=self.config.top_k
        )
        
        return retrieved_docs
        
    def display_contexts(self, contexts: List[Document]):
        """Display retrieved contexts in a formatted way"""
        table = Table(title="Retrieved Contexts")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Content", style="white", width=60)
        table.add_column("Size", style="yellow", no_wrap=True)
        table.add_column("Headers", style="green", width=30)
        
        for idx, doc in enumerate(contexts):
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            chunk_size = doc.metadata.get('chunk_size', len(doc.page_content))
            
            # Extract header information
            headers = []
            for key, value in doc.metadata.items():
                if key.startswith('Header') and value:
                    headers.append(f"{key}: {value}")
            header_text = "\n".join(headers) if headers else "No headers"
            
            table.add_row(str(idx + 1), content_preview, str(chunk_size), header_text)
            
        console.print(table)


def main():
   
    # Configuration - Updated to use Gecko
    config = RAGConfig(
        chunk_size=1000, 
        chunk_overlap=200,
        device="cuda",
        top_k=3,
        gecko_model_path="Gecko_1024_quant.tflite",
        tokenizer_path="sentencepiece.model"
    )
    
    # Initialising the RAG pipeline
    rag_pipeline = RAGPipeline(config, HuggingFaceLLM("google/gemma-3n-E2B-it", device=config.device))      # using gemma-3n-E4B-it instead of gemma-3n-E4B-it-litert-preview
    
    try:
        # Load and process text file
        text_file_path = "WHO_PositiveBirth_2018_extracted.txt"
        
        console.print("[bold blue]Chunking using Markdown-Aware Fixed Splitter with Gecko Embeddings:[/bold blue]")
        chunks = rag_pipeline.load_and_chunk_text(text_file_path)
        
        
        output_chunks_path = "chunks_output.txt"  # Output file for chunks
        
        # Save chunks to file with <sep> separator
        rag_pipeline.save_chunks_to_file(chunks, output_chunks_path, separator="<sep>")
        
        
        rag_pipeline.create_vector_store(chunks)
        rag_pipeline.setup_rag_graph()
        
        # Example questions/queries
        questions = [
            "Given a newborn presenting with respiratory distress and murmurs at birth, how can we differentiate between transient tachypnea of the newborn (TTN), neonatal sepsis, and meconium aspiration using clinical signs and a basic pulse oximeter, given that we do not have access to chest x-rays and blood cultures?",
            "What immediate steps should be taken after the baby is born to ensure breathing, warmth, and safe cord care without advanced equipment?",
            "How can I help a mother start breastfeeding within the first hour, and what are the signs that the baby is breastfeeding well?",
            "What basic hygiene rules should be followed to prevent infections in newborns during and after delivery at home or in the clinic?",
            "What are the main warning signs in a newborn that require urgent referral or emergency care in the first 7 days?",
            "How can I tell if a newborn is too cold or too hot, and what simple steps can I take to stabilize their temperature?",
            "If a newborn is not breathing at birth, what step-by-step steps should I take in the first minute without special tools?"
        ]
        
        for question in questions:
            console.print(f"\n[bold yellow]Question: {question}[/bold yellow]")
            
            # Get only retrieved contexts
            contexts = rag_pipeline.get_retrieved_contexts(question)
            console.print(f"[blue]Retrieved {len(contexts)} contexts:[/blue]")
            rag_pipeline.display_contexts(contexts)
            
            response = rag_pipeline.query(question)
            console.print(f"[green]Answer: {response['answer']}[/green]")
            
    except FileNotFoundError as e:
        console.print(f"Error: {e}")
    except Exception as e:
        console.print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()