from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from agent import Agent, AgentConfig # Import Agent logic
from docx import Document
from pypdf import PdfReader
import logging
import json
import re
import asyncio
import models
import os
import uuid
import shutil

# navigate to directory
# add API endpoints to example.env and change from example.env to .env
# add FLASK_APP=app.py to .env
# type 'virtualenv venv' into terminal
# then 'source venv/bin/activate'
# then 'conda activate a0'
# then 'pip install -r requirements.txt' do
# then 'flask run'

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mapping of contract types

guidelines_mapping = {
    'tc': 'termsAndConditionsGuidelines.txt',
    'msaMpa': 'msaMpaGuidelines.txt',
    'nda': 'ndaConfidentialityGuidelines.txt',
    'dsRp': 'distributorRepresentativeGuidelines.txt',
    'other': 'other.txt'
}
template_mapping = {
    'tc': 'termsAndConditionsAssessment.txt',
    'msaMpa': 'msaMpaAssessment.txt',
    'nda': 'ndaConfidentialityAssessment.txt',
    'dsRp': 'distributorRepresentativeAssessment.txt',
    'other': 'other.txt'
}

# Models for API Requests
class ExecuteRequest(BaseModel):
    task: str

class ExecuteResponse(BaseModel):
    status: str
    data: dict

@app.route('/api/test', methods=['POST'])
def test_api():
    try:
        # Get the question from the request
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Initialize models as in main.py
        chat_llm = models.get_openai_chat(model_name="gpt-4o", temperature=0)
        utility_llm = chat_llm
        embedding_llm = models.get_openai_embedding(model_name="text-embedding-3-small")

        # Configure the Agent as in main.py
        config = AgentConfig(
            chat_model=chat_llm,
            utility_model=utility_llm,
            embeddings_model=embedding_llm,
            auto_memory_count=0,
            rate_limit_requests=10,
            rate_limit_input_tokens=0,
            rate_limit_output_tokens=0,
            rate_limit_seconds=60,
            max_tool_response_length=150000,
            code_exec_docker_enabled=True,
            code_exec_ssh_enabled=True,
        )

        # Initialize the Agent
        agent = Agent(number=1, config=config)

        # Use the asyncio event loop to run the monologue coroutine
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(agent.monologue(question))

        # Return the agent's response
        return jsonify({'message': result}), 200

    except Exception as e:
        # Log the exception
        print(f'Error: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/evaluate-contract', methods=['POST'])
async def evaluate_contract():
    logger.debug('Received request to evaluate contract')

    if 'files' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')  # Get all files
    contract_type = request.form.get('contract_type')
    logger.debug(f'Contract type: {contract_type}')

    if not files:
        logger.error('No files provided')
        return jsonify({'error': 'No files provided'}), 400

    # Process all files into one combined text
    combined_text = []
    for file in files:
        if not file or not file.filename:  # Add this check
            continue
            
        if file.filename.endswith('.docx'):
            document = Document(file.stream)
            combined_text.append('\n'.join([para.text for para in document.paragraphs]))
        elif file.filename.endswith('.pdf'):
            pdf_reader = PdfReader(file.stream)
            combined_text.append('\n'.join(page.extract_text() for page in pdf_reader.pages))
        else:
            combined_text.append(file.read().decode('utf-8'))

    contract_text = "\n\n=== NEW DOCUMENT ===\n\n".join(combined_text)
    logger.debug(f'Processed {len(files)} files')

    if contract_type not in guidelines_mapping:
        logger.error('Unsupported contract type')
        return jsonify({'error': 'Unsupported contract type'}), 400

    # Load the guidelines for risk assessment
    guidelines_file = guidelines_mapping[contract_type]
    guidelines_path = f'knowledge/custom/main/{guidelines_file}'
    logger.debug(f'Loading guidelines from {guidelines_path}')
    with open(guidelines_path, 'r') as f:
        guidelines = f.read()

    # Load the output template
    template_file = template_mapping[contract_type]
    template_path = f'knowledge/custom/main/{template_file}'
    logger.debug(f'Loading template from {template_path}')
    with open(template_path, 'r') as f:
        template = f.read()

    risk_assessment = await assess_risk(contract_text, guidelines, template)

    logger.debug('Returning assessment output')
    return jsonify({'assessment': risk_assessment}), 200

def chunk_text(text, chunk_size=15000):
    """Split text into smaller chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

async def assess_risk(contract_text, guidelines, template):
    try:
        chat_llm = models.get_openai_chat(model_name="gpt-3.5-turbo", temperature=0)
        utility_llm = chat_llm
        embedding_llm = models.get_openai_embedding(model_name="text-embedding-3-small")

        config = AgentConfig(
            chat_model=chat_llm,
            utility_model=utility_llm,
            embeddings_model=embedding_llm,
            auto_memory_count=0,
            rate_limit_requests=0,
            rate_limit_input_tokens=0,
            rate_limit_output_tokens=0,
            rate_limit_seconds=0,
            max_tool_response_length=15000,
            code_exec_docker_enabled=True,
            code_exec_ssh_enabled=True,
        )

        # Calculate sections based on content length and token limits
        content_length = len(contract_text)
        total_sections = min(max(3, content_length // 8000), 6)  # Smaller chunks
        logger.debug(f'Using {total_sections} agents for {content_length} characters')

        # Create agents
        agents = {
            f'section_{i}': Agent(number=i, config=config) for i in range(1, total_sections + 1)
        }
        agents['master'] = Agent(number=total_sections + 1, config=config)

        # Create temp directory in knowledge base
        temp_dir = 'knowledge/default/temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Save contract to knowledge base
        temp_contract_path = f'{temp_dir}/contract.txt'
        with open(temp_contract_path, 'w') as f:
            f.write(contract_text)

        # Split and save sections
        sections = chunk_text(contract_text, len(contract_text) // total_sections)
        analysis_tasks = []
        
        # Process each section with simpler prompts
        for i, section in enumerate(sections, 1):
            section_path = f'{temp_dir}/section_{i}.txt'
            with open(section_path, 'w') as f:
                f.write(section)
                
            prompt = (
                f"Read and analyze section {i} from knowledge/default/temp/section_{i}.txt\n\n"
                f"Use these contract guidelines for your analysis:\n"
                f"{guidelines}\n\n"
                f"Report your findings based on these guidelines."
            )
            analysis_tasks.append(agents[f'section_{i}'].monologue(prompt))

        analyses = await asyncio.wait_for(asyncio.gather(*analysis_tasks), timeout=300)

        # Simpler synthesis prompt
        synthesis_prompt = (
            f"Review these analyses and complete the risk assessment form:\n\n"
            f"GUIDELINES:\n{guidelines[:2000]}\n\n"
            f"ANALYSES:\n"
            f"{chr(10).join(f'Section {i+1}: {analysis[:1000]}' for i, analysis in enumerate(analyses))}\n\n"
            f"ASSESSMENT FORM:\n{template}"
        )
        
        final_result = await agents['master'].monologue(synthesis_prompt)
        
        # Cleanup
        try:
            os.remove(temp_contract_path)
            for i in range(1, total_sections + 1):
                os.remove(f'{temp_dir}/section_{i}.txt')
        except Exception as e:
            logger.error(f'Cleanup error: {e}')

        logger.debug(f'Created {len(sections)} sections')
        logger.debug(f'Received {len(analyses)} section analyses')
        logger.debug(f'Final result length: {len(final_result)}')

        return final_result

    except Exception as e:
        logger.error(f'Error in assess_risk: {e}')
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)
