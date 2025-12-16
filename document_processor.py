"""
Document Processor Module
Handles parsing and chunking of generic documents (text, markdown, CSV)
"""

import pandas as pd
import logging
import uuid
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes generic documents for the knowledge base
    """

    def __init__(self):
        pass

    def process_text(self, text: str, source_name: str, doc_type: str = "general_knowledge") -> List[Dict[str, Any]]:
        """
        Process a text string into chunks

        Args:
            text: The content of the document
            source_name: Name/Title of the source document
            doc_type: Type of document (e.g., 'org_knowledge', 'general_knowledge')

        Returns:
            List of document chunks
        """
        # Simple chunking by paragraphs for now
        # In a real system, we'd use a more sophisticated chunker (e.g., langchain)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            # Check if paragraph is too long, split if necessary (naive splitting)
            max_len = 1000
            if len(para) > max_len:
                sub_chunks = [para[j:j+max_len] for j in range(0, len(para), max_len)]
            else:
                sub_chunks = [para]

            for j, chunk_text in enumerate(sub_chunks):
                chunk = {
                    'id': str(uuid.uuid4()),
                    'text': chunk_text,
                    'source': source_name,
                    'source_type': doc_type,
                    'category': doc_type, # Using doc_type as category for now
                    'chunk_index': i * 100 + j
                }
                chunks.append(chunk)

        logger.info(f"Processed {source_name} into {len(chunks)} chunks")
        return chunks

    def process_csv(self, file_path: str, doc_type: str = "org_knowledge") -> List[Dict[str, Any]]:
        """
        Process a CSV file.
        Note: This is different from PhishingDataProcessor which does specific analytics.
        This treats the CSV as a knowledge source (row by row).

        Args:
            file_path: Path to CSV file
            doc_type: Type of document

        Returns:
            List of document chunks
        """
        try:
            df = pd.read_csv(file_path)
            chunks = []

            for index, row in df.iterrows():
                # Convert row to text representation
                text_parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                row_text = ". ".join(text_parts)

                chunk = {
                    'id': str(uuid.uuid4()),
                    'text': row_text,
                    'source': file_path,
                    'source_type': doc_type,
                    'category': doc_type,
                    'chunk_index': index
                }
                chunks.append(chunk)

            logger.info(f"Processed {file_path} into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            raise

    def process_file(self, file_path: str, doc_type: str = "general_knowledge") -> List[Dict[str, Any]]:
        """
        Process a file based on extension
        """
        if file_path.endswith('.csv'):
            return self.process_csv(file_path, doc_type)
        else:
            # Assume text or markdown
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.process_text(content, file_path, doc_type)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                raise
