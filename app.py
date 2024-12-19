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
    temp_files = []  # Initialize at start of function

    if 'files' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400

    try:
        # Create temp directory if it doesn't exist
        temp_dir = 'knowledge/default/temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        files = request.files.getlist('files')
        contract_type = request.form.get('contract_type')
        logger.debug(f'Contract type: {contract_type}')

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

        # Save uploaded files to temp directory
        combined_text = []
        for i, file in enumerate(files, 1):
            if not file or not file.filename:
                continue

            # Save original file
            temp_path = f'{temp_dir}/input_{i}_{file.filename}'
            file.save(temp_path)
            temp_files.append(temp_path)
            
            # Process file content
            if file.filename.endswith('.docx'):
                document = Document(temp_path)
                combined_text.append('\n'.join([para.text for para in document.paragraphs]))
            elif file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(temp_path)
                combined_text.append('\n'.join(page.extract_text() for page in pdf_reader.pages))
            else:
                with open(temp_path, 'r') as f:
                    combined_text.append(f.read())

        contract_text = "\n\n=== NEW DOCUMENT ===\n\n".join(combined_text)
        
        # Rest of the function...
        result = await assess_risk(contract_text, guidelines, template)

        # Cleanup temp files
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f'Removed temp file: {path}')

        return jsonify({'assessment': result}), 200

    except Exception as e:
        logger.error(f'Error in evaluate_contract: {e}')
        # Cleanup on error
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({'error': str(e)}), 500

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
    # Calculate sections with larger chunks for GPT-4
    content_length = len(contract_text)
    total_sections = min(max(3, content_length // 8000), 8)  # Increased from 3000 to 8000
    logger.debug(f'Starting analysis with {total_sections} sections')
    
    try:
        chat_llm = models.get_openai_chat(model_name="gpt-4o", temperature=0)  # Changed to GPT-4
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
            code_exec_ssh_enabled=True
        )

        # Create agents
        agents = {
            f'section_{i}': Agent(number=i, config=config) for i in range(1, total_sections + 1)
        }
        agents['master'] = Agent(number=total_sections + 1, config=config)

        # Split content into smaller chunks
        sections = chunk_text(contract_text, len(contract_text) // total_sections)
        
        # Save sections to temp files and add to knowledge base
        for i, section in enumerate(sections, 1):
            # Save section to temp file
            section_path = f'knowledge/default/temp/section_{i}.txt'
            with open(section_path, 'w') as f:
                f.write(section)
            logger.debug(f'Saved section {i} to {section_path}')

        # Process sections with larger content
        analyses = []
        for i, section in enumerate(sections, 1):
            logger.debug(f'Processing section {i} of {total_sections}')
            
            prompt = (
                f"Section {i}/{total_sections} Analysis:\n\n"
                f"CONTRACT TEXT:\n{section}\n\n"  # Include section content directly
                f"GUIDELINES:\n{guidelines}\n\n"
                f"Analyze this section against the guidelines. For each relevant point:\n"
                f"1. Quote specific contract language\n"
                f"2. Explain implications (2-3 sentences)\n"
                f"3. Assign risk level (Red/Orange/Yellow/Green)\n"
                f"4. Justify risk assessment"
            )
            
            try:
                analysis = await asyncio.wait_for(
                    agents[f'section_{i}'].monologue(prompt),
                    timeout=180  # Increased timeout
                )
                analyses.append(analysis[:2000])  # Increased from 500
                logger.debug(f'Completed section {i} analysis')
            except asyncio.TimeoutError:
                logger.warning(f"Section {i} analysis timed out")
                analyses.append(f"Section {i}: Analysis timed out")

        # Fuller synthesis prompt with better combination instructions
        synthesis_prompt = (
            f"Complete this risk assessment form by combining all section analyses.\n\n"
            f"SECTION ANALYSES:\n"
            f"{chr(10).join(f'SECTION {i+1}:\n{analysis[:1000]}' for i, analysis in enumerate(analyses))}\n\n"
            f"TEMPLATE TO COMPLETE:\n{template}\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"1. Review ALL section analyses before answering each question\n"
            f"2. Combine relevant information from different sections\n"
            f"3. Only mark as 'missing information' if NO section contains relevant details\n"
            f"4. For each template question:\n"
            f"   - Quote specific contract language found in ANY section\n"
            f"   - Explain implications comprehensively\n"
            f"   - Assign appropriate risk level based on ALL available information\n"
            f"   - Provide detailed justification\n"
            f"5. If truly no information exists, mark as Red risk\n\n"
            f"Begin template completion with 'Contract Risk Assessment'"
        )

        try:
            final_result = await asyncio.wait_for(
                agents['master'].monologue(synthesis_prompt),
                timeout=180
            )
            logger.debug('Synthesis completed')
            return final_result
        except asyncio.TimeoutError:
            logger.error("Synthesis timed out")
            return {"error": "Synthesis timed out - please try again"}

    except Exception as e:
        logger.error(f'Error in assess_risk: {e}')
        return {'error': str(e)}

    finally:
        # Cleanup in finally block
        try:
            for i in range(1, total_sections + 1):
                path = f'knowledge/default/temp/section_{i}.txt'
                if os.path.exists(path):
                    os.remove(path)
        except Exception as e:
            logger.error(f'Cleanup error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
