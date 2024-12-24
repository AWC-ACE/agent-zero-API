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
import datetime

# navigate to directory
# add API endpoints to example.env and change file name from example.env to .env
# add FLASK_APP=app.py to .env
# download and install python 3.12.1 and conda
# use agentzero docs for assistance to get started based on your environment
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

@app.route('/api/question', methods=['POST'])
def question_api():
    try:
        data = request.get_json()
        question = data.get('question')
        previous_analysis = data.get('previous_analysis', '')  # Get previous analysis if provided

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Initialize models
        chat_llm = models.get_openai_chat(model_name="gpt-4o-mini", temperature=0)
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
            max_tool_response_length=15000,
            code_exec_docker_enabled=False,
            code_exec_ssh_enabled=False
        )

        # Initialize the Agent
        agent = Agent(number=1, config=config)

        # Create prompt that includes previous analysis if available
        prompt = (
            f"Previous Contract Analysis:\n{previous_analysis}\n\n"
            f"Question about the analysis:\n{question}\n\n"
            f"Instructions:\n"
            f"1. Answer based ONLY on information in the previous analysis\n"
            f"2. If the answer cannot be found in the analysis, say so\n"
            f"3. Do not make assumptions beyond what's stated\n"
            f"4. Quote relevant parts of the analysis when possible\n"
        ) if previous_analysis else question

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(agent.monologue(prompt))

        return jsonify({'message': result}), 200

    except Exception as e:
        logger.error(f'Error in test_api: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate-contract', methods=['POST'])
async def evaluate_contract():
    logger.debug('Received request to evaluate contract')
    temp_files = []  # Keep track for error handling only

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
        result = await assess_risk(contract_text, guidelines, template, files)

        return jsonify({'assessment': result}), 200

    except Exception as e:
        logger.error(f'Error in evaluate_contract: {e}')
        # Cleanup only on error
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

async def assess_risk(contract_text, guidelines, template, files):
    # Calculate smaller sections for GPT-4o mini's token limit
    content_length = len(contract_text)
    total_sections = min(max(4, content_length // 3000), 10)  # Back to smaller chunks
    logger.debug(f'Starting analysis with {total_sections} sections')
    
    try:
        chat_llm = models.get_openai_chat(model_name="gpt-4o-mini", temperature=0)  # Changed to mini
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

        # Process sections with adjusted content sizes
        analyses = []
        for i, section in enumerate(sections, 1):
            logger.debug(f'Processing section {i} of {total_sections}')
            
            # Extract relevant guidelines to save tokens
            relevant_guidelines = "\n".join(
                line for line in guidelines.split('\n')
                if any(keyword in line.lower() for keyword in ['risk', 'assess'])
            )[:1000]  # Limit guidelines size
            
            prompt = (
                f"Section {i}/{total_sections} Analysis:\n\n"
                f"CONTRACT TEXT:\n{section[:2000]}\n\n"
                f"GUIDELINES:\n{relevant_guidelines}\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Analyze content that CLOSELY matches guidelines\n"
                f"2. For each relevant match found:\n"
                f"   - Quote: '[relevant contract language]' or 'Missing Information'\n"
                f"   - Location: [exact section title from contract] or 'Not Specified'\n"
                f"     Example good: 'Article 5: Confidentiality Terms'\n"
                f"     Example bad: 'Section 5' or 'Section 5/10'\n"
                f"   - Explanation: [how this relates to guideline requirement]\n"
                f"   - Risk Level: [Red/Orange/Yellow/Green per guideline]\n"
                f"   - Justification: [reason based on guidelines]\n"
                f"3. Look for content that addresses the spirit of each guideline\n"
                f"4. Make reasonable connections but avoid speculation\n"
            )
            
            try:
                analysis = await asyncio.wait_for(
                    agents[f'section_{i}'].monologue(prompt),
                    timeout=120
                )
                analyses.append(analysis[:800])  # Reduced analysis size
                logger.debug(f'Completed section {i} analysis')
            except asyncio.TimeoutError:
                logger.warning(f"Section {i} analysis timed out")
                analyses.append(f"Section {i}: Analysis timed out")

        # Get contract title from filename(s), remove extension and clean up
        contract_titles = [os.path.splitext(f.filename)[0].replace('_', ' ').strip() for f in files if f.filename]
        if len(contract_titles) > 1:
            contract_title = f"Multiple Contracts: {', '.join(contract_titles)}"
        else:
            contract_title = contract_titles[0] if contract_titles else "Untitled Contract"
            
        # Remove common suffixes like "Template" or "Assessment"
        contract_title = re.sub(r'\b(Template|Assessment|Risk)\b', '', contract_title, flags=re.IGNORECASE).strip()
        analysis_date = datetime.datetime.now().strftime("%B %d, %Y")

        # Synthesis prompt with better title extraction instructions
        synthesis_prompt = (
            f"Complete risk assessment form by combining analyses:\n\n"
            f"ANALYSES:\n"
            f"{chr(10).join(f'SECTION {i+1}:\n{analysis[:500]}' for i, analysis in enumerate(analyses))}\n\n"
            f"GUIDELINES:\n{guidelines}\n\n"
            f"TEMPLATE:\n{template}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. First, extract the exact contract title:\n"
            f"   - Look at the very beginning of Section 1's analysis\n"
            f"   - Find text like 'Agreement', 'Contract', 'NDA' with any company names\n"
            f"   - Example: If you see 'Georgia-Pacific(GP) Non-Disclosure Agreement (NDA)', use that\n"
            f"   - If no formal title found, use '{contract_title}'\n\n"
            f"2. Start the assessment with:\n"
            f"Contract Risk Assessment\n"
            f"Contract ID: [EXACT contract title from document]\n"
            f"Analysis Date: {analysis_date}\n\n"
            f"3. For each template question:\n"
            f"   - Find relevant content that addresses the question\n"
            f"   - Quote: Must be EXACT text from contract (2-3 sentences)\n"
            f"   - Location: [exact section title from contract] or 'Not Specified'\n"
            f"   - Explanation: [how this affects AWC's obligations]\n"
            f"   - Risk Level: [Red/Orange/Yellow/Green]\n"
            f"   - Justification: Quote the relevant guideline AND explain how the contract aligns or conflicts\n\n"
            f"4. Example format:\n"
            f"   1. Does the contract contain mutual confidentiality obligations?\n\n"
            f"   Quote: 'Each party agrees to maintain the confidentiality of all information received from the other party. The receiving party shall protect such information with at least the same degree of care as it protects its own confidential information.'\n"
            f"   Location: Article 7: Confidentiality Obligations\n"
            f"   Explanation: Establishes mutual confidentiality requirements\n"
            f"   Risk Level: Green\n"
            f"   Justification: Per AWC guidelines: 'Mutual obligations on both parties - Low risk.' This contract establishes mutual obligations, which aligns with the preferred risk profile. The reciprocal nature of the obligations provides balanced protection for both parties.\n\n"
            f"5. If no exact match exists:\n"
            f"   Quote: 'WARNING: Potential Missing Information'\n"
            f"   Location: 'Not included or not specified in contract.'\n"
            f"   Explanation: 'No specific language addressing this requirement was found.'\n"
            f"   Risk Level: Red\n"
            f"   Justification: Per AWC guidelines: 'Missing critical terms creates significant risk.' The absence of these terms leaves AWC exposed without necessary protections.\n\n"
            f"6. IMPORTANT:\n"
            f"   - Quotes must be exact contract language\n"
            f"   - Justifications must quote relevant AWC guidelines\n"
            f"7. Complete ALL questions with ALL format elements\n\n"
            f"Begin with 'Contract Risk Assessment'"
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

@app.route('/api/end-session', methods=['POST'])
async def complete_analysis():
    """Cleanup endpoint to remove temporary files after analysis is done"""
    try:
        temp_dir = 'knowledge/default/temp'
        files_removed = 0
        
        # Clean up all files in temp directory
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
                logger.debug(f'Removed temp file: {file_path}')
        
        return jsonify({
            'message': 'Analysis completed and cleaned up',
            'files_removed': files_removed
        }), 200

    except Exception as e:
        logger.error(f'Error in complete_analysis: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
