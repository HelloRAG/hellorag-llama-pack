import os
import re
import uuid
import zipfile
from typing import Any
from typing import Dict

from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from lxml import etree
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

TABLE_INDEX = "table_index"
ALL_TEXT_INDEX = "all_text_index"


def delete_file(file_path):
    """
    Deletes the file specified by the file path.

    Parameters:
    file_path (str): The path of the file to be deleted.

    Returns:
    No return value.
    """
    try:
        os.remove(file_path)  # Attempt to delete the file specified by file_path
    except Exception as e:
        print(f"delete temp file error:{file_path}")  # Print an error message if the deleting of the file fails


def sort_text_node(elem):
    """
    Sorts text node based on its page number.

    Parameters:
    elem: An element containing text node information, which should have a 'metadata' dictionary attribute with a 'page_no' key representing the page number.

    Returns:
    An integer indicating the page number where this element is located.
    """
    return int(elem.metadata['page_no'])


class BetterTablesHelloragPack(BaseLlamaPack):
    """
        A better tables retriever pack —— By HelloRAG.ai

    """

    def __init__(
            self,
            base_path: str = None,
            need_refresh: bool = False,
            index_path: str = None,
            storage_context: StorageContext = None,
            font_path: str = None,
            chunk_size: int = 512,
            chunk_overlap: int = 200,
            top_k: int = 3,
            **kwargs: Any,
    ) -> None:
        """
        Initialization function.

        Args:
        - base_path: The base path used when building the index. If `need_refresh` is True, this parameter cannot be None.
        - need_refresh: A flag indicating whether to refresh the index. If set to True, the index will be rebuilt using `base_path`.
        - index_path: The path to load the index from. If provided, the index will be loaded from this path.
        - storage_context: The storage context used to access vector stores. If `index_path` is not provided, the index will be built or loaded based on this parameter.
        - font_path: An optional font path for settings during index construction.
        - chunk_size: The size of chunks used when building the index.
        - chunk_overlap: The overlap size of chunks when building the index.
        - top_k: The number of results to return during retrieval.
        - **kwargs: Additional keyword arguments,but not used yet

        Returns:
        - None

        """
        if need_refresh and base_path is None:
            raise ValueError("base_path can not be None")
        if index_path is None and storage_context is None:
            raise ValueError("index_path and storage_context can not be both None")
        if index_path is None and storage_context is not None and storage_context.vector_stores is None:
            raise ValueError("vector_stores can not be None")

        if need_refresh:
            # Build the index with provided parameters
            self.build_index(base_path, chunk_overlap, chunk_size, font_path, index_path, storage_context)

        # Load the index
        self._index = None
        if index_path:
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            self._index = load_index_from_storage(storage_context)
        elif storage_context:
            self._index = VectorStoreIndex.from_vector_store(
                storage_context.vector_store, storage_context=storage_context
            )

        if self._index is None:
            raise ValueError("Can not initiate index")

        # Initialize retrievers, query engine, and chat engine
        self.retriever = self._index.as_retriever(top_k=top_k)
        self.query_engine = self._index.as_query_engine()
        self.chat_engine = self._index.as_chat_engine()

    def build_index(self, base_path, chunk_overlap, chunk_size, font_path, index_path, storage_context):
        """
        Builds the index by processing files in the specified base path or in the vector store in storage_context

        :param base_path: The root directory to traverse and process files.
        :param chunk_overlap: Size of overlap between chunks during sentence splitting.
        :param chunk_size: Size of each chunk for sentence splitting.
        :param font_path: Path to a font file used when generating PDFs.
        :param index_path: Directory where the index should be persisted; if provided.
        :param storage_context: Storage context used to store node information.

        :return: No return value.
        """
        all_nodes = []  # List to hold all nodes information

        # Initialize the SentenceSplitter
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Go through all files and directories in the base path
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if filename.endswith('.zip'):
                    all_text_nodes_map = {}  # Map to hold text pages and their content
                    zip_path = os.path.join(root, filename)
                    content_file_name = filename[:-4]  # Remove '.zip' extension

                    # Process and extract content from the ZIP file
                    with zipfile.ZipFile(zip_path, 'r') as myzip:
                        zip_content = myzip.infolist()
                        for member in zip_content:
                            if member.is_dir():
                                continue
                            member_name = member.filename
                            if member_name.find('/') == -1:
                                continue
                            member_path_split = member_name.split('/')
                            page_no = member_path_split[0]
                            base_name, ext = os.path.splitext(member_name)
                            if ext == '.txt':
                                # Handle text files
                                content = myzip.read(member).decode('utf-8')
                                all_text_nodes_map[page_no] = content
                            elif ext == '.html':
                                # Handle HTML files
                                content = myzip.read(member).decode('utf-8')
                                html_root = etree.HTML(content)
                                title = html_root.xpath('//h1/text()')[0]
                                description = html_root.xpath('//p/text()')[0]
                                first_table = html_root.xpath('//table')[0]
                                table_html = etree.tostring(first_table, method='html', encoding='unicode')
                                soup = BeautifulSoup(table_html, 'html.parser')
                                texts = [''.join(c for c in td.stripped_strings if not re.match(r'^-?\d+(\.\d+)?$', c)) for tr in soup.find_all('tr') for td in tr.find_all('td')]
                                html_core_content='\n'.join(texts).replace("\n\n","\n")
                                print(page_no,html_core_content)
                                print('''---------------------------''')
                                node = TextNode(text=f"{title}\n{description}\n{html_core_content}",
                                                id_=f"{uuid.uuid4()}")
                                node.metadata['page_no'] = page_no
                                node.metadata['file_name'] = content_file_name
                                node.metadata['table_html'] = table_html
                                node.metadata['typeaz'] = 'table'
                                all_nodes.append(node)

                    # After processing all text within the ZIP, generate a PDF
                    text_nodes_len = len(all_text_nodes_map)
                    if text_nodes_len > 0:
                        font_name = "CustomFontA"
                        temp_file = f"./{uuid.uuid4()}.pdf"
                        if font_path:
                            pdfmetrics.registerFont(TTFont(font_name, font_path))
                        c = canvas.Canvas(temp_file)
                        current_page = 1
                        sorted_pages = sorted([int(k) for k in all_text_nodes_map.keys()])
                        for page_number in sorted_pages:
                            while current_page < int(page_number):
                                c.showPage()
                                current_page += 1
                            if font_path:
                                c.setFont(font_name, 12)
                            c.drawString(0, 0, all_text_nodes_map[str(page_number)])
                            c.showPage()
                            current_page += 1
                        c.save()

                        # Define a callback function for PDF metadata
                        def _file_metadata_call_back(str_):
                            return {"file_name": content_file_name}

                        documents = SimpleDirectoryReader(input_files=[temp_file],
                                                          file_metadata=_file_metadata_call_back).load_data()
                        # Extract nodes from processed documents
                        all_nodes.extend(parser.get_nodes_from_documents(documents))
                        delete_file(temp_file)  # Delete the temporary file

        # Build page text index
        if index_path:
            index = VectorStoreIndex(all_nodes, show_progress=True, store_nodes_override=True)
            index.storage_context.persist(persist_dir=index_path)
        elif storage_context:
            VectorStoreIndex(nodes=all_nodes, show_progress=True,
                             store_nodes_override=True,
                             storage_context=storage_context
                             )

    def get_index(self) -> VectorStoreIndex:
        """
        Retrieves the index.

        Returns:
            VectorStoreIndex: The stored index object.
        """
        return self._index

    def get_modules(self) -> Dict[str, Any]:
        """
        Retrieves the modules dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing 'retriever', 'chat_engine', and 'query_engine'.
        """
        return {
            "retriever": self.retriever,
            "chat_engine": self.query_engine,
            "query_engine": self.query_engine,
        }

    def retrieve(self, query_str: str) -> Any:
        """
        Retrieves information based on the query string.

        Parameters:
            query_str (str): The query string to retrieve data from.

        Returns:
            Any: The retrieved result.
        """
        return self.retriever.retrieve(query_str)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the pipeline.

        Parameters:
            *args (Any): Variable positional arguments.
            **kwargs (Any): Variable keyword arguments.

        Returns:
            Any: The result of the pipeline query.
        """
        return self.query_engine.query(*args, **kwargs)

