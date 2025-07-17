cd import os
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


console = Console()

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "intfloat/multilingual-e5-small"
    device: str = "cuda"
    top_k: int = 4


class LLMInterface(ABC):
    """Abstract base class for LLM models to ensure model agnosticity"""
    
    @abstractmethod
    def generate(self, prompt: str, context: List[Document]) -> str:
        """Generate response given prompt and context"""
        pass


class HuggingFaceLLM(LLMInterface):
    """HuggingFace LLM implementation with actual model loading"""
    
    def __init__(self, model_name: str = "OpenMeditron/Meditron3-8B", device: str = "cuda"):
        self.model_name = model_name    # default is Meditron
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
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with appropriate settings
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
                max_length=2048,  # Adjust based on the model's context length
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Maximum tokens to generate
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            answer = response[len(full_prompt):].strip()
            
            return answer if answer else "I couldn't generate a proper response based on the given context."
            
        except Exception as e:
            console.print(f"Error during generation: {e}")
            return f"Error generating response: {str(e)}"


# State definition for LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


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
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
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


class MarkdownAwareFixedSplitter:
    """Markdown-aware splitter that creates exactly sized chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, strict_size: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strict_size = strict_size
        self.fixed_splitter = FixedSizeTextSplitter(chunk_size, chunk_overlap)
        
        # Headers to preserve structure
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        
    def _extract_headers_from_text(self, text: str) -> Dict[str, str]:
        """Extract the most recent headers from text for context"""
        headers = {}
        lines = text.split('\n')
        
        for line in lines:
            for header_marker, header_name in self.headers_to_split_on:
                if line.strip().startswith(header_marker + " "):
                    level = len(header_marker)
                    header_text = line.strip()[level+1:].strip()
                    headers[header_name] = header_text
                    break
        
        return headers
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with markdown awareness but enforcing fixed chunk sizes"""
        all_chunks = []
        
        for doc in documents:
            if self.strict_size:
                # Always apply fixed-size splitting regardless of structure
                section_chunks = self.fixed_splitter.split_documents([doc])
                
                # Add header information to each chunk based on content
                for chunk in section_chunks:
                    headers = self._extract_headers_from_text(chunk.page_content)
                    chunk.metadata.update(headers)
                
                all_chunks.extend(section_chunks)
            else:
                # Use the original logic (allows smaller chunks)
                sections = self._extract_section_with_headers(doc.page_content)
                
                for section in sections:
                    if len(section["content"]) <= self.chunk_size:
                        # Section fits in one chunk
                        chunk_doc = Document(
                            page_content=section["content"],
                            metadata={
                                **doc.metadata,
                                **section["headers"],
                                'chunk_size': len(section["content"])
                            }
                        )
                        all_chunks.append(chunk_doc)
                    else:
                        # Section is too large, apply fixed-size splitting
                        temp_doc = Document(
                            page_content=section["content"],
                            metadata={**doc.metadata, **section["headers"]}
                        )
                        
                        section_chunks = self.fixed_splitter.split_documents([temp_doc])
                        
                        # Add section headers to metadata
                        for chunk in section_chunks:
                            chunk.metadata.update(section["headers"])
                        
                        all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _extract_section_with_headers(self, text: str) -> List[Dict]:
        """Extract sections with their headers for context preservation"""
        sections = []
        lines = text.split('\n')
        current_section = {"content": "", "headers": {}}
        
        for line in lines:
            # Check if line is a header
            is_header = False
            for header_marker, header_name in self.headers_to_split_on:
                if line.strip().startswith(header_marker + " "):
                    level = len(header_marker)
                    header_text = line.strip()[level+1:].strip()
                    current_section["headers"][header_name] = header_text
                    is_header = True
                    break
            
            current_section["content"] += line + '\n'
            
            # If we hit a major header (# or ##), start a new section
            if is_header and line.strip().startswith(("# ", "## ")):
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"content": line + '\n', "headers": current_section["headers"].copy()}
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
            
        return sections


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
        console.print(f"Setting up embeddings with model: {self.config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.device},
            encode_kwargs={"device": self.config.device}
        )
        
    def _setup_vector_store(self):
        """Initialize vector store"""
        console.print("Setting up vector store...")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
    def load_and_chunk_text(self, file_path: str, use_markdown_aware: bool = True, strict_size: bool = True) -> List[Document]:
        """Load and chunk text file with exactly sized chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        console.print(f"Loading text file: {file_path}")
        
        # Load text file
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Choose splitter based on preference
        if use_markdown_aware:
            if strict_size:
                console.print("Using markdown-aware fixed-size splitter (STRICT mode - all chunks exactly 1000 chars)...")
            else:
                console.print("Using markdown-aware fixed-size splitter (FLEXIBLE mode - allows smaller chunks)...")
            text_splitter = MarkdownAwareFixedSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strict_size=strict_size
            )
        else:
            console.print("Using simple fixed-size splitter...")
            text_splitter = FixedSizeTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        chunks = text_splitter.split_documents(documents)
        
        # Verify chunk sizes
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        console.print(f"Created {len(chunks)} chunks")
        console.print(f"Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        # Show distribution of chunk sizes
        exact_1000 = sum(1 for size in chunk_sizes if size == 1000)
        less_than_1000 = sum(1 for size in chunk_sizes if size < 1000)
        console.print(f"Chunks exactly 1000 chars: {exact_1000}, Chunks < 1000 chars: {less_than_1000}")
        
        return chunks
        
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
        
        # Build the graph
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
    """Example usage of the RAG pipeline with real LLM"""
    
    # Configuration
    config = RAGConfig(
        chunk_size=1000, 
        chunk_overlap=200,
        embedding_model="intfloat/multilingual-e5-small",
        device="cuda",
        top_k=3
    )
    
    
    # # Debugging
    # simple_llm = HuggingFaceLLM(device=config.device)
    # test_context = [Document(page_content="The capital of France is Paris.")]
    # test_question = "What is the capital of France?"
    # direct_answer = simple_llm.generate(test_question, test_context)
    # console.print(f"Direct LLM test answer: {direct_answer}")
    
    # Initialize RAG pipeline with LLM
    rag_pipeline = RAGPipeline(config, HuggingFaceLLM("OpenMeditron/Meditron3-8B", device=config.device))
    
    try:
        # Load and process text file
        text_file_path = "out.txt"
        
        
        console.print("[bold blue]Testing Markdown-Aware Fixed Splitter (STRICT - all chunks exactly 1000):[/bold blue]")
        chunks = rag_pipeline.load_and_chunk_text(text_file_path, use_markdown_aware=True, strict_size=True)
        
        # Create vector store
        rag_pipeline.create_vector_store(chunks)
        
        # Setup RAG graph
        rag_pipeline.setup_rag_graph()
        
        # Example queries
        # questions = [
        #     "How do I take care of the mother after birth?",
        #     "What to do if my baby is crying? His face is turning Blue! What do I do?",
        #     "How do I successfully breastfeed my baby?",
        #     "How do I take care of the woman after birth?"
        # ]
        
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
            
            # Get full RAG response with actual LLM
            response = rag_pipeline.query(question)
            console.print(f"[green]Answer: {response['answer']}[/green]")
            
    except FileNotFoundError as e:
        console.print(f"Error: {e}")
    except Exception as e:
        console.print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()