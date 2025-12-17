"""
PDF Processor Module
Handles PDF file text extraction and chunking for vector storage
"""

import pdfplumber
import io
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processes PDF files and extracts text for vector storage
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the PDF processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract all text from a PDF file
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            text_content = []
            
            # Open PDF from bytes
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                logger.info(f"📄 Processing PDF with {len(pdf.pages)} pages")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"✅ Extracted {len(full_text)} total characters from PDF")
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF. The PDF might be image-based or encrypted.")
            
            return full_text
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, title: str = "document") -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage
        
        Args:
            text: Full text to chunk
            title: Document title for chunk IDs
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # First, try to split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'id': f"{title.lower().replace(' ', '_').replace('.', '_')}_chunk_{chunk_index}",
                    'text': current_chunk.strip(),
                    'title': title,
                    'chunk_index': chunk_index,
                    'source_type': 'pdf_document'
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"{title.lower().replace(' ', '_').replace('.', '_')}_chunk_{chunk_index}",
                'text': current_chunk.strip(),
                'title': title,
                'chunk_index': chunk_index,
                'source_type': 'pdf_document'
            })
        
        logger.info(f"✅ Created {len(chunks)} chunks from PDF text")
        return chunks
    
    def process_pdf(self, pdf_bytes: bytes, title: str, category: str = "general", 
                    metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Complete PDF processing pipeline: extract text and chunk it
        
        Args:
            pdf_bytes: PDF file as bytes
            title: Document title
            category: Document category
            metadata: Additional metadata to include
            
        Returns:
            List of document chunks ready for embedding and storage
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_bytes)
        
        # Chunk text
        chunks = self.chunk_text(text, title)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk['category'] = category
            chunk['file_type'] = 'pdf'
            if metadata:
                chunk.update(metadata)
        
        return chunks


if __name__ == "__main__":
    # Test the PDF processor
    print("PDF Processor Test")
    print("Note: This requires a PDF file to test")
    
    # Example usage:
    # processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    # with open("test.pdf", "rb") as f:
    #     pdf_bytes = f.read()
    # chunks = processor.process_pdf(pdf_bytes, title="Test Document", category="test")
    # print(f"Created {len(chunks)} chunks")
