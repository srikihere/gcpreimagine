import functions_framework
import os
import pandas as pd
from typing import Optional
from typing import Sequence
import time
import vertexai

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
from google.cloud import storage
from google.cloud.documentai_toolbox import document
from vertexai.language_models import TextGenerationModel


# TODO(developer): Uncomment these variables before running the sample.
project_id = "gcds-oht33425u9-2023"
location = "us" # Format is "us" or "eu"
processor_id = "565e10119d1e0bdc" # Create processor before running sample
file_path = "/tmp"
mime_type = "application/pdf" # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
# field_mask = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.
# processor_version_id = "YOUR_PROCESSOR_VERSION_ID" # Optional. Processor version to use
bucket_name = 'escrow-doc-bucket'

@functions_framework.http
def process_document_sample(request) -> None:
    # You must set the `api_endpoint` if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    name = client.processor_path(project_id, location, processor_id)
    print('name of processor is:')
    print(name)

    local_path = "/tmp/" + 'Sample_Statement_Page.pdf'
    txt_file_path = "/tmp/" + 'Summary.txt'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('Sample_Statement_Page.pdf')
    blob.download_to_filename(local_path)

    # Read the file into memory
    with open(local_path, "rb") as image:
        image_content = image.read()

    # Load binary data
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name, raw_document=raw_document
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    #print("The document contains the following text:")
    #print(document.text)
    text = document.text
    text = text.replace('\n', ' ')
    print(f"Full document text: {repr(text)}\n")
    print(f"There are {len(document.pages)} page(s) in this document.")
    with open (txt_file_path, 'w') as text_file:
        text_file.write(text)
    dest_blob_name = "Summary.txt"
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(txt_file_path)
    print(
        f"File {txt_file_path} uploaded to {dest_blob_name}."
    )

    vertexai.init(project="gcds-oht33425u9-2023", location="us-central1")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 400,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response_json = {}

    for page in document.pages:
        print(f"\n\n**** Page {page.page_number} ****")
        print(f"\nFound {len(page.tables)} table(s):")
        for table in page.tables:
            num_columns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f"Table with {num_columns} columns and {num_rows} rows:")

            # Print header rows
            print("Columns:")
            #print_table_rows(table.header_rows, text)
            df = pd.DataFrame()
            selected_table, list_values, df = print_table_csv_headers(table.header_rows, text)
            print("DataFrame for headers:")
            print(df)
            # Print body rows
            print("Table body data:")
            if selected_table:
            #print_table_rows(table.body_rows, text)
                print("List of column names: ")
                print(list_values)
                df_total = print_table_csv_rows(table.body_rows, text, df)
                #print("List of row values: ")
                #print(tmp_list_values)
                print("DataFrame for rows:")
                print(df_total)
                source_file_name = "/tmp/output.csv" 
                df_total.to_csv(source_file_name)
                time.sleep(2)
                destination_blob_name = "output.csv"
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(source_file_name)
                print(
                    f"File {source_file_name} uploaded to {destination_blob_name}."
                )
                print("Calling Vertext AI for Long Summary")
                long_summary_text = invoke_vertexai_for_long_summary(text, model)
                response_json['long_summary_text'] = long_summary_text
                print("Calling Vertext AI for Short Summary")
                short_summary_text = invoke_vertexai_for_short_summary(text, model)
                response_json['short_summary_text'] = short_summary_text
                print("Calling Vertext AI for Table Summary")
                table_summary_text = invoke_vertexai_for_table_summary(df_total, model)
                response_json['table_summary_text'] = table_summary_text
                print("Calling Vertext AI for Factual Summary")
                factual_summary_text = invoke_vertexai_for_factual_summary(text, model)
                response_json['factual_summary_text'] = factual_summary_text
                print("Calling Vertext AI for Action Items")
                action_items_text = invoke_vertexai_for_action_items(text, model)
                response_json['action_items_text'] = action_items_text


        '''print(f"\nFound {len(page.form_fields)} form field(s):")
        for field in page.form_fields:
            name = layout_to_text(field.field_name, text)
            value = layout_to_text(field.field_value, text)
            print(f"    * {repr(name.strip())}: {repr(value.strip())}")'''

    return response_json

def print_table_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> None:
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} | "
        print(row_text)

def print_table_csv_headers(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> None:
    selected_table = False
    header_list_values = []
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            if cell_text.strip() == 'Current Monthly Payment':
                print("Column name matching")
                selected_table = True
            header_list_values.append(cell_text.strip())
            row_text += f"{repr(cell_text.strip())} , "
        df = pd.DataFrame(columns=header_list_values)
        print(row_text)
    return selected_table, header_list_values, df
    
def print_table_csv_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str, df
) -> None:
    selected_table = False
    for table_row in table_rows:
        row_list_values = []
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} , "
            row_list_values.append(cell_text.strip())
        tmp_column_list = df.columns.values.tolist()
        print("column list in dataframe on rows:")
        print(tmp_column_list)
        print("row values in table:")
        print(row_list_values)
        #for entry in tmp_column_list:
        df.loc[len(df)] = row_list_values
    return df

def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document"s text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response


def invoke_vertexai_for_short_summary(content: str, model) -> str:
    input_prompt = """Summarize the following text in about 100 words: {}""".format(content)
    response = model.predict(prompt = input_prompt)
    print(f"Response from Model for Short Summary: {response.text}")    
    return 'Response: {}'.format(response.text)

def invoke_vertexai_for_long_summary(content: str, model) -> str:
    input_prompt = """Summarize the following text in about 300 words: {}""".format(content)
    response = model.predict(prompt = input_prompt)
    print(f"Response from Model for Long Summary: {response.text}")    
    return 'Response: {}'.format(response.text)

def invoke_vertexai_for_factual_summary(content: str, model) -> str:
    input_prompt = """By using the numerical values given, summarize the following text in about 300 words: {}""".format(content)
    response = model.predict(prompt = input_prompt)
    print("Response from Model for Factual Summary:")
    print(response.text)
    return 'Response: {}'.format(response.text)

def invoke_vertexai_for_action_items(content: str, model) -> str:
    input_prompt = """What are the action items for the client mentioned in the following text. Think step by step: {}""".format(content)
    response = model.predict(prompt = input_prompt)
    print(f"Response from Model for Action Items: {response.text}")    
    return 'Response: {}'.format(response.text)


def invoke_vertexai_for_table_summary(df_table, model) -> str:
    input_text='''Write a detailed summary for the given table and do not miss any values from the table while writing summary'''
    input_prompt = """Please regard the following data:\n {}. Answer the following question: {}""".format(df_table, input_text)
    response = model.predict(prompt = input_prompt)
    print(f"Response from Model for Table Summary: {response.text}")    
    return 'Response: {}'.format(response.text)
