from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from agent import Agent, AgentConfig # Import Agent logic
from docx import Document
import logging
import asyncio
import models
import os

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
            max_tool_response_length=6500,
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
def evaluate_contract():
    logger.debug('Received request to evaluate contract')

    if 'contract_file' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['contract_file']
    contract_type = request.form.get('contract_type')
    logger.debug(f'Contract type: {contract_type}')

    if contract_type not in guidelines_mapping:
        logger.error('Unsupported contract type')
        return jsonify({'error': 'Unsupported contract type'}), 400

    # Check the file extension
    if file.filename.endswith('.docx'):
        logger.debug('Processing .docx file')
        document = Document(file)
        contract_text = '\n'.join([para.text for para in document.paragraphs])
    else:
        logger.debug('Processing .txt file')
        contract_text = file.read().decode('utf-8')

    logger.debug('Loaded contract text')

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

    risk_assessment = assess_risk(contract_text, guidelines)

    output = format_output(template, risk_assessment)

    logger.debug('Returning assessment output')
    return jsonify({'assessment': output}), 200


def assess_risk(contract_text, guidelines):
    try:
        chat_llm = models.get_openai_chat(model_name="gpt-4o", temperature=0)
        utility_llm = chat_llm
        embedding_llm = models.get_openai_embedding(model_name="text-embedding-3-small")

        config = AgentConfig(
            chat_model=chat_llm,
            utility_model=utility_llm,
            embeddings_model=embedding_llm,
            auto_memory_count=0,
            rate_limit_requests=10,
            rate_limit_input_tokens=0,
            rate_limit_output_tokens=0,
            rate_limit_seconds=60,
            max_tool_response_length=6500,
            code_exec_docker_enabled=True,
            code_exec_ssh_enabled=True,
        )

        agent = Agent(number=1, config=config)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        prompt = f"Using the guidelines: {guidelines}, assess the contract: {contract_text}"
        result = loop.run_until_complete(agent.monologue(prompt))

        return result

    except Exception as e:
        return {'error': str(e)}


def format_output(template, risk_assessment):
    filled_assessment = template
    for key, value in risk_assessment.items():
        filled_assessment = filled_assessment.replace(f'{{{key}}}', value)
    return filled_assessment

if __name__ == '__main__':
    app.run(debug=True)
