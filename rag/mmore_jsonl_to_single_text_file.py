import json
import os

def extract_text(jsonl_file_path, filename, output_file_path):
    """
    Extract text from JSONL and save to text file.
    
    Args:
        jsonl_file_path (str): Path to the input JSONL file
        output_file_path (str): Path to the output text file
    """
    
    extracted_text = []
    found_file = False
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    
                    if 'metadata' in data and 'file_path' in data['metadata']:
                        file_path = data['metadata']['file_path']
                        
                        if filename in file_path:
                            found_file = True
                            print(f"File path: {file_path}")
                            
                            # Extract text content
                            if 'text' in data:
                                extracted_text.append(data['text'])
                                print(f"Extracted text length: {len(data['text'])} characters")
                            
                            if 'antml:document_content' in data:
                                extracted_text.append(data['antml:document_content'])
                                print(f"Extracted document content length: {len(data['antml:document_content'])} characters")
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_number}: {e}")
                    continue
        
        if not found_file:
            print("Available files in the dataset:")
            
            # Show available files for reference
            with open(jsonl_file_path, 'r', encoding='utf-8') as file:
                file_paths = set()
                for line in file:
                    try:
                        data = json.loads(line.strip())
                        if 'metadata' in data and 'file_path' in data['metadata']:
                            file_paths.add(data['metadata']['file_path'])
                    except:
                        continue
                
                for fp in sorted(file_paths):
                    print(f"  - {fp}")
            
            return False
        

        if extracted_text:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # joining all extracted text with double newlines
                full_text = '\n\n'.join(extracted_text)
                output_file.write(full_text)
            
            print(f"\nSuccessfully extracted text to: {output_file_path}")
            print(f"Total text length: {len(full_text)} characters")
            print(f"Number of text sections: {len(extracted_text)}")
            return True
        else:
            return False
            
    except FileNotFoundError:
        print(f"Error: Could not find the file {jsonl_file_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    input_jsonl_file = "merged_results.jsonl"
    
    filename = "WHO_PositiveBirth_2018"
    
    output_text_file = f"{filename}_extracted.txt"
   
    success = extract_text(input_jsonl_file, filename, output_text_file)
    
    if success:
        print("Extraction completed successfully!")
    else:
        print("Extraction failed. Please check the file paths and content.")