from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import os
from flask import Response, json
from collections import OrderedDict
import ast
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from typing import List, Optional, Dict, Any

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['JSON_SORT_KEYS'] = False

GROQ_API_KEY = "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"

# Global variables
initialized = False
tfidf_vectorizer = None
tfidf_matrix = None
workers = []



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.before_request
def initialize_app():
    global initialized, df_metadata, tfidf_vectorizer, tfidf_matrix ,workers
    if not initialized:
        try:
            METADATA_PATH = "metadata.parquet"
            
            # Load metadata
            df_metadata = pd.read_parquet(METADATA_PATH)
            
            # Precompute TF-IDF features
            corpus = df_metadata['Individual Test Solutions'] + " " + df_metadata['Test Type']+ " " + df_metadata['Description']
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            
            # Initialize workers
            workers = initialize_workers()  # Properly store the result of the function
            
            initialized = True
            
        except Exception as e:
            app.logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize application resources")

class Supervisor:
    def __init__(self, supervisor_name: str, supervisor_prompt: str, model: Any):
        self.name = supervisor_name
        self.prompt_template = supervisor_prompt
        self.model = model

    def format_prompt(self, team_members: List[str]) -> str:
        return self.prompt_template.format(team_members=", ".join(team_members))

class Worker:
    def __init__(self, worker_name: str, worker_prompt: str, supervisor: Supervisor, tools: Optional[List[Any]] = None):
        self.name = worker_name
        self.prompt_template = worker_prompt
        self.supervisor = supervisor
        self.tools = tools or []
        
    def clean_response(self, response: str) -> Any:
        if '</think>' in response:
            response = response.split('</think>')[-1]
        response = response.replace('**', '').strip()
        response = response.split(':')[-1].strip()
        
        if self.name == 'TestTypeAnalyst':
            return ''.join([c for c in response if c.isupper() or c == ','])
        elif self.name == 'Skill Extractor':
            return '\n'.join([s.split('. ')[-1] for s in response.split('\n')])
        elif self.name == 'Time Limit Identifier':
            return response.split()[0]
        elif self.name == 'Testing Type Identifier':
            response = response.strip('[]')
            parts = [part.strip().lower() for part in response.split(',')]
            return [part if part in ('yes', 'no') else 'no' for part in parts]
        
        return response.split('\n')[0].strip('"').strip()
        
    def process_input(self, user_input: str) -> str:
        prompt = f"{self.prompt_template}\n\nUser Input: {user_input}"
        messages = [HumanMessage(content=prompt)]
        response = self.supervisor.model.invoke(messages)
        return self.clean_response(response.content)

def initialize_workers():
    groq_model = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0,
        streaming=True,
        api_key=GROQ_API_KEY
    )

    supervisor = Supervisor(
        supervisor_name="AssessmentCoordinator",
        supervisor_prompt="You manage these specialists: {team_members}. Coordinate assessment creation workflow. Select next worker strategically. FINISH when complete.",
        model=groq_model
    )

    return [
        Worker(
        worker_name="TestTypeAnalyst",
        worker_prompt='''You are an AI classifier that maps user inputs to test type codes from this taxonomy:

Test Types (Code: Description)

A: Ability & Aptitude (cognitive skills, problem-solving)

B: Biodata & Situational Judgement (past behavior, hypothetical scenarios)

C: Competencies (job-specific skills like leadership)

D: Development & 360 (growth feedback, multi-rater reviews)

E: Assessment Exercises (role-plays, case studies)

K: Knowledge & Skills (technical/domain expertise)

P: Personality & Behavior (traits, motivations)

S: Simulations (realistic job-task replicas)

Rules:

Return only the relevant letter codes (e.g., K, A,S).

Use commas for multiple matches (no spaces).

Prioritize specificity (e.g., "Python coding test" → K, not A).

Default to B for biographical/historical scenarios.

Examples:

Input: "Quiz on Java and cloud architecture" → K

Input: "Test how someone leads a team during a crisis" → C,S

Input: "Evaluate agreeableness and reaction to feedback" → P,D

Output Format:
Return only the letter code(s) as a comma-separated string (e.g., P or B,S).

''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Skill Extractor",
        worker_prompt='''You are a skill extractor for assessment design. Identify both hard and soft skills explicitly mentioned in the user’s input that are relevant to the test’s purpose.

Rules:
Focus: Extract hard skills (technical) and soft skills (non-technical):

✅ Hard Skills:

Tools: Python, SQL, AWS

Frameworks: TensorFlow, React

Domains: cybersecurity, CAD, data analysis

✅ Soft Skills:

communication, leadership, teamwork, problem-solving

🚫 Exclude:

Generic terms: "experience," "knowledge," "proficiency"

Job roles: "developer," "engineer"

Test Type Context: Use the test type code (A/B/C/D/E/K/P/S) to refine extraction:

Example: Test type K (Knowledge & Skills) → Prioritize hard skills like Python.

Example: Test type C (Competencies) → Include both hard skills (CAD) and soft skills (leadership).

Example: Test type P (Personality) → Extract only soft skills if mentioned (e.g., adaptability).

Normalization:

Standardize terms: JS → JavaScript, ML → machine learning.

Merge equivalents: CAD → Computer-Aided Design.

Output:

Return a comma-separated list (e.g., Python, leadership, CAD).

If no skills are found, return [].

Examples:
Input	Test Type	Output
“Test Python coding and teamwork.”	K	Python, teamwork
“Assess problem-solving and cloud architecture.”	A	problem-solving, cloud architecture
“Evaluate leadership and CAD proficiency.”	C	leadership, CAD
“Behavioral test focusing on communication.”	P	communication
“No skills mentioned.”	S	[]''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Job Level Identifier",
        worker_prompt='''You are an AI assistant tasked with identifying the job level for which a test is intended. Given input that may include job titles, responsibilities, or descriptions, determine the most appropriate job level from the following list:

Director

Entry Level

Executive

Frontline Manager

General Population

Graduate

Manager

Mid-Professional

Professional

Professional Individual Contributor

Supervisor

Use contextual clues in the input to make an accurate classification. Respond only with the job level.
 ''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Language Preference Identifier",
        worker_prompt='''You are a language detector for assessments. Identify spoken (natural) languages (e.g., English, Mandarin, Spanish) explicitly mentioned in the user’s input.

Rules:

Focus:

Extract only natural languages (e.g., "French", "Japanese").

Ignore programming languages (Python, Java), tools (SQL), or frameworks (React).

Defaults:

Return English if no spoken language is mentioned.

For multi-language requests (e.g., "English and Spanish"), return a comma-separated list: English, Spanish.

Output:

Use full language names (e.g., "German" not "Deutsch").

Case-insensitive (e.g., "spanish" → Spanish).

Examples:

Input: "Test must be in Portuguese." → Output: Portuguese

Input: "Python coding test with instructions in Arabic." → Output: Arabic

Input: "Math exam for Spanish-speaking students." → Output: Spanish

Input: "Timed Java assessment." → Output: English

Respond only with the language name(s). No explanations.

'''
,
        supervisor=supervisor
    ),
    Worker(
        worker_name="Time Limit Identifier",
        worker_prompt='''You are an AI that extracts explicit test durations from user input.

Rules
Extract:

Return exact phrases with a number + time unit (e.g., 90 minutes, 2.5 hrs, no more than 45 mins).

Include comparative phrasing (e.g., under 1 hour, at least 20 minutes).

Ignore:

Deadlines (e.g., submit by Friday).

Experience durations (e.g., 5 years of experience).

Vague terms (e.g., timed test, time-sensitive).

Output:

For valid durations: Return them as a comma-separated list (e.g., 1 hour, 30 mins).

For no valid durations: Return no time specified.

Examples
Input	Output
"Complete the test in 45 mins."	45 mins
"Section A: 1 hour; Section B: 30 mins."	1 hour, 30 mins
"Timed exam with no duration mentioned."	no time specified
"Submit by 5 PM and allow up to 2 hrs."	2 hrs
"Requires 3+ years of experience."	no time specified
Strict Constraints
Never return explanations, formatting, or placeholders.

Only return extracted durations or no time specified.

''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Testing Type Identifier",
        worker_prompt='''You are an AI classifier that detects mentions of remote testing or adaptive testing/IRT in user inputs and returns a structured response.

Rules
Detection Logic:

Remote Testing: yes if the exact phrase "remote testing" is present.

Adaptive Testing: yes if "adaptive testing" or "IRT" (case-insensitive) is present.

Default to no for missing terms.

Output Format:

Return [yes,yes] if both terms are present.

Return [yes,no] if only remote testing is mentioned.

Return [no,yes] if only adaptive testing/IRT is mentioned.

Return [no,no] if neither is mentioned.

Constraints:

NO explanations, NO deviations from the format.

Exact matches only (e.g., "remote" ≠ "remote testing").

Examples
Input	Output
"Conduct remote testing with IRT."	[yes,yes]
"Use adaptive testing."	[no,yes]
"Remote testing required."	[yes,no]
"Timed onsite exam."	[no,no]
Command:
Return ONLY the structured list ([yes,yes], [no,yes], etc.). No other text!''',
        supervisor=supervisor
    )
      
        # ... (other workers with their prompts unchanged) ...
    ]

@app.route('/recommend', methods=['POST'])
def api_recommendations():
    try:
        data = request.get_json()
        user_input = data.get('query')
        
        if not user_input:
            return jsonify({"error": "Missing 'query' field"}), 400

        input_list = process_user_input(user_input, workers)
        recommendations = recommend_assessments(input_list, df_metadata)
        
        results = []
        if not recommendations.empty:
            for record in recommendations.to_dict('records'):
                results.append(OrderedDict([
                    ("url", record.get('URL', '')),
                    ("adaptive_support", "Yes" if str(record.get('Adaptive/IRT', 'No')).lower() == 'yes' else "No"),
                    ("duration", int(record.get('Assessment Length', 0))),
                    ("remote_support", "Yes" if str(record.get('Remote Testing', 'No')).lower() == 'yes' else "No"),
                    ("test_type", ast.literal_eval(record.get('test_type', '[]'))),
                    ("description", record.get('Description', ''))
                ]))
        
        return Response(
            json.dumps({"recommended_assessments": results[:10]}, ensure_ascii=False),
            mimetype='application/json',
            status=200
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def recommend_assessments(input_list: List[str], df_metadata: pd.DataFrame) -> pd.DataFrame:
    try:
        remote_support, adaptive_support, test_type, skills_query, language, duration = input_list
        
        # Language filtering
        def language_filter(lang_entry: str) -> bool:
            primary_lang = re.split(r'[^a-zA-Z]', language.lower())[0]
            return any(
                primary_lang == re.split(r'[^a-zA-Z]', l.strip())[0]
                for l in re.split(r'[,/]', str(lang_entry).lower())
            )

        # Test type filtering
        def test_type_filter(test_entry: str) -> bool:
            query_terms = set(re.split(r'[\s-]+', test_type.lower()))
            entry_terms = set(re.split(r'[\s-]+', test_entry.lower()))
            return len(query_terms & entry_terms) > 0

        # Apply filters
        filtered = df_metadata[
            df_metadata['Language'].apply(language_filter) &
            df_metadata['Test Type'].apply(test_type_filter)
        ]
        
        # TF-IDF similarity calculation
        query_text = f"{skills_query} {test_type}"
        query_vec = tfidf_vectorizer.transform([query_text])
        filtered_indices = filtered.index
        subset_matrix = tfidf_matrix[filtered_indices]
        similarities = cosine_similarity(query_vec, subset_matrix)
        
        filtered['similarity'] = similarities[0]
        return filtered.sort_values('similarity', ascending=False).head(10)

    except Exception as e:
        return pd.DataFrame()

# ... (keep process_user_input and route functions unchanged) ...
# [Keep process_user_input() function]
def process_user_input(user_input: str, workers: List[Any]) -> list:
    """Process input through workers and format for recommendation system"""
    output_order = [
        "Testing Type Identifier",
        "TestTypeAnalyst",
        "Skill Extractor",
        "Job Level Identifier",
        "Language Preference Identifier",
        "Time Limit Identifier"
    ]
    
    # Get worker results
    results = {}
    for worker in workers:
        result = worker.process_input(user_input)
        results[worker.name] = result
    
    # Create ordered list
    original_list = [results[key] for key in output_order]
    
    # Transform to input_list format
    input_list = [
        *original_list[0],  # Flatten testing type identifiers
        original_list[1].replace(',', ' '),  # Test types
        f"{original_list[2]}, {original_list[3]}",  # Skills + Job Level
        *original_list[4:]  # Language and Duration
    ]
    
    return input_list

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            user_input = request.form.get('job_description', '')
            
            if not user_input:
                return render_template('index.html', 
                                    error="Please enter a job description first!")
            
            # Process input
            input_list = process_user_input(user_input, workers)
            
            
            # Get recommendations
            # Fix: Remove unused parameters
            recommendations = recommend_assessments(input_list, df_metadata)
            
            # Prepare results
            results = recommendations[[
                'Individual Test Solutions', 'Test Type', 
                'Language', 'Remote Testing', 
                'Adaptive/IRT', 'Assessment Length','URL', 'Description' 
            ]].to_dict('records') if not recommendations.empty else []
            
            return render_template('index.html', 
                                 results=results,
                                 input_text=user_input)
            
        except Exception as e:
            app.logger.error(f"Processing error: {str(e)}")
            return render_template('index.html', 
                                error=f"Error processing request: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)

