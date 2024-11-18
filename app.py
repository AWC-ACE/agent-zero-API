from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from agent import Agent, AgentConfig # Import Agent logic
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

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({'message': 'API is working!'}), 200

@app.route('/api/evaluate-contract', methods=['POST'])
def evaluate_contract():
    # Check if the contract file is included in the request
    if 'contract_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['contract_file']
    contract_type = request.form.get('contract_type')

    # Validate the contract type
    if contract_type not in guidelines_mapping:
        return jsonify({'error': 'Unsupported contract type'}), 400

    # Read the contract file content
    contract_text = file.read().decode('utf-8')

    # Load the guidelines for risk assessment
    guidelines_file = guidelines_mapping[contract_type]
    guidelines_path = f'knowledge/custom/main/guidelines/{guidelines_file}'
    with open(guidelines_path, 'r') as f:
        guidelines = f.read()

    # Perform risk assessment
    risk_assessment = assess_risk(contract_text, guidelines)

    # Load the output template
    template_file = template_mapping[contract_type]
    template_path = f'knowledge/custom/main/guidelines/{template_file}'
    
    # Format the output based on the assessment
    output = format_output(template, risk_assessment)

    # Return the assessment as a JSON response
    return jsonify({'assessment': output}), 200


def assess_risk(contract_text, guidelines):
    try:
        # Initialize models as in main.py
        chat_llm = models.get_openai_chat(model_name="gpt-4o-mini", temperature=0)
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
            max_tool_response_length=3000,
            code_exec_docker_enabled=True,
            code_exec_ssh_enabled=True,
        )

        # Initialize the Agent
        agent = Agent(number=1, config=config)

        ### Expand upon this to include a chat prompt template for 
        result = agent.message.loop(f'Assess the following contract: {contract_text} with {guidelines}')
        return {'risk-level': result['risk_level'], 'details': result['details']}

    except Exception as e:
        return {'risk_level': 'unknown', 'details': str(e)}
           
def format_output(template, risk_assessment):
    # Format the output using the template and risk assessment results
    return template.replace('{risk_level}', risk_assessment['risk_level'])


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
